mod progress;
mod eth_walk;
mod matcher;
use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use clap::Parser;
use ed25519_dalek::SigningKey;
use num_format::{Locale, ToFormattedString};
use pbkdf2::pbkdf2_hmac;
use rand::RngCore;
use sha2::Sha256;
use sha3::{Digest, Keccak256};
use secp256k1::{Secp256k1, SecretKey};
#[cfg(feature = "gpu")]
use secp256k1::PublicKey;
use std::{
    env,
    fs,
    str::FromStr,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::{Duration, Instant},
    path::PathBuf,
};
#[cfg(feature = "gpu")]
use std::array;

// ========================================
// Chain Type Definition
// ========================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainType {
    Ethereum,
    Solana,
}

impl std::str::FromStr for ChainType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "eth" | "ethereum" | "evm" => Ok(ChainType::Ethereum),
            "sol" | "solana" | "svm" => Ok(ChainType::Solana),
            _ => Err(format!("Unknown chain \"{}\". Try --chain eth or --chain sol.", s)),
        }
    }
}

impl std::fmt::Display for ChainType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChainType::Ethereum => write!(f, "eth"),
            ChainType::Solana => write!(f, "sol"),
        }
    }
}

// ========================================
// CLI Arguments
// ========================================

#[derive(Debug, Parser)]
#[clap(name = "vanity")]
#[clap(about = "CPU/GPU Vanity Address Generator for SVM (Solana) and EVM (Ethereum) blockchains")]
#[clap(after_help = "\
EXAMPLES:
    # Ethereum: simple prefix
    vanity --chain eth -p cafe

    # Solana: case-insensitive prefix
    vanity --chain sol -p test -i

    # Save as JSON
    vanity --chain eth -p dead -o json --save

    # Save encrypted result using passphrase from environment
    vanity --chain sol -p test --save ./results --encrypt

    # Decrypt a previously encrypted result file
    vanity --decrypt ./results/vanity-sol-xxxx.txt.enc --decrypt-out ./restored.txt

    # CPU only with 8 threads (must set --gpus 0)
    vanity --chain sol -p sol --cpus 8 --gpus 0

    # ASCII output with timeout
    vanity --chain eth -p 00 -o plain --max-runtime 30

GPU vs CPU MODE:
    When --gpus > 0: Only GPU is used (CPU threads are ignored)
    When --gpus = 0: Only CPU is used with specified threads
    No hybrid mode available - GPU and CPU cannot run simultaneously

GPU TUNING (ETH and SOL):
    --gpu-blocks, --gpu-threads, --gpu-iters
      Override launch params to push throughput.
    --gpu-target-ms
      Adaptive batch time target (ms).
    Note: These flags are available only in GPU builds.

    # ETH with tuning
    vanity --chain eth -s 1010 --gpus 1 --gpu-blocks 512 --gpu-threads 512 --gpu-target-ms 1200

SOLANA NOTES:
    Solana GPU starts with a small batch and supports the same tuning flags.
    Suffix patterns benefit from a fast check to avoid full Base58 encodes.

    # SOL suffix example
    vanity --chain sol -s Jozef --gpus 1

    # CPU only (requires --gpus 0)
    vanity --chain eth -p cafe --cpus 16 --gpus 0
")]
pub struct Args {
    /// Chain type: eth|sol (accepts ethereum|evm|solana|svm)
    #[clap(long, value_parser = parse_chain)]
    pub chain: Option<ChainType>,

    /// Target prefix for the address
    #[clap(short = 'p', long)]
    pub prefix: Option<String>,

    /// Target suffix for the address
    #[clap(short = 's', long)]
    pub suffix: Option<String>,

    /// Case-insensitive matching (Solana only)
    #[clap(short = 'i', long)]
    pub ignore_case: bool,

    /// Number of GPUs to use (default: 1 if GPU detected, else 0)
    #[cfg(feature = "gpu")]
    #[clap(long)]
    pub gpus: Option<u32>,

    /// Override GPU blocks per launch
    #[cfg(feature = "gpu")]
    #[clap(long)]
    pub gpu_blocks: Option<u32>,

    /// Override GPU threads per block
    #[cfg(feature = "gpu")]
    #[clap(long)]
    pub gpu_threads: Option<u32>,

    /// Override GPU iterations per thread
    #[cfg(feature = "gpu")]
    #[clap(long)]
    pub gpu_iters: Option<u64>,

    /// Target kernel time per batch in milliseconds
    #[cfg(feature = "gpu")]
    #[clap(long)]
    pub gpu_target_ms: Option<u64>,

    /// Number of CPU threads (default: 0 = auto)
    #[clap(long, default_value = "0")]
    pub cpus: u32,

    /// Number of matches to find (default: 1)
    #[clap(long, default_value = "1")]
    pub count: u32,

    /// Maximum runtime in seconds (default: unlimited)
    #[clap(long)]
    pub max_runtime: Option<u64>,

    /// Output format: card|plain|json (default: card)
    #[clap(short = 'o', long, default_value = "card")]
    pub output: OutputFormat,

    /// Save result to file (optional path, defaults to current directory)
    #[clap(long)]
    pub save: Option<Option<PathBuf>>,

    /// Encrypt saved result using a passphrase from the configured environment variable
    #[clap(long)]
    pub encrypt: bool,

    /// Environment variable name containing the encryption passphrase
    #[clap(long, default_value = "VANITY_ENCRYPTION_PASSWORD")]
    pub encrypt_passphrase_env: String,

    /// Decrypt a previously encrypted result file
    #[clap(long)]
    pub decrypt: Option<PathBuf>,

    /// Output path for decrypted content; prints to stdout when omitted
    #[clap(long)]
    pub decrypt_out: Option<PathBuf>,

    /// Progress display on stderr: auto (interactive), always, or never
    #[clap(long, value_enum, default_value = "auto")]
    pub progress: progress::ProgressMode,

    /// Extra diagnostics to stderr
    #[clap(long)]
    pub debug: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Card,
    Plain,
    Json,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "card" => Ok(OutputFormat::Card),
            "plain" => Ok(OutputFormat::Plain),
            "json" => Ok(OutputFormat::Json),
            _ => Err(format!("Unknown output format \"{}\". Try card, plain, or json.", s)),
        }
    }
}

fn parse_chain(s: &str) -> Result<ChainType, String> {
    ChainType::from_str(s)
}

impl Args {
    fn generation_chain(&self) -> ChainType {
        self.chain.expect("chain should be validated before generation")
    }
}

// ========================================
// Result Structure
// ========================================

#[derive(Debug, Clone)]
pub struct VanityResult {
    pub chain: ChainType,
    pub address: String,
    pub public_key: String,
    pub secret_type: &'static str,
    pub secret: String,
    pub pattern_prefix: String,
    pub pattern_suffix: String,
    pub ignore_case: bool,
    pub backend: String,
    pub gpus: u32,
    pub cpus: u32,
    pub elapsed_sec: f64,
    pub iterations: u64,
    pub rate_per_sec: f64,
    pub encodes: u64,
    pub encode_rate_per_sec: f64,
}

// ========================================
// Helper Functions
// ========================================

fn format_seed_as_array(seed_bytes: &[u8]) -> String {
    let numbers: Vec<String> = seed_bytes.iter()
        .map(|b| b.to_string())
        .collect();
    format!("[{}]", numbers.join(","))
}

// ========================================
// EIP-55 Checksum for Ethereum
// ========================================

fn eip55_checksum(address_hex: &str) -> String {
    // address_hex should be 40 hex chars without 0x prefix
    let mut hasher = Keccak256::new();
    hasher.update(address_hex.to_lowercase().as_bytes());
    let hash = hasher.finalize();

    let mut checksummed = String::with_capacity(42);
    checksummed.push_str("0x");

    for (i, ch) in address_hex.chars().enumerate() {
        if ch.is_ascii_digit() {
            checksummed.push(ch);
        } else {
            // Check if the i/2-th byte (since 2 hex chars = 1 byte) has high bit set
            let byte_index = i / 2;
            let nibble = if i % 2 == 0 { hash[byte_index] >> 4 } else { hash[byte_index] & 0xf };
            if nibble >= 8 {
                checksummed.push(ch.to_ascii_uppercase());
            } else {
                checksummed.push(ch.to_ascii_lowercase());
            }
        }
    }

    checksummed
}

// ========================================
// Output Formatters
// ========================================

fn print_result(result: &VanityResult, format: OutputFormat) {
    match format {
        OutputFormat::Plain => print_result_plain(result),
        OutputFormat::Card => print_result_card(result),
        OutputFormat::Json => print_result_json(result),
    }
}

fn print_result_plain(result: &VanityResult) {
    // ASCII mode
    println!("* Vanity Match");
    println!("- Chain: {} ({})", result.chain.to_string().to_uppercase(),
             if result.chain == ChainType::Ethereum { "EVM" } else { "SVM" });
    println!("- Address: {}", result.address);
    println!("- Private Key: {}", result.secret);
    println!("- Method: {}", result.backend);
    println!("- Elapsed: {:.3}s", result.elapsed_sec);
    println!("- Iterations: {}", result.iterations.to_formatted_string(&Locale::en));
    println!("- Rate: {} / sec", (result.rate_per_sec as u64).to_formatted_string(&Locale::en));
    if result.encodes > 0 {
        println!("- Encodes: {}", result.encodes.to_formatted_string(&Locale::en));
        println!("- Encodes/sec: {}", (result.encode_rate_per_sec as u64).to_formatted_string(&Locale::en));
    }
    let pattern_parts = [
        if !result.pattern_prefix.is_empty() { format!("prefix=\"{}\"", result.pattern_prefix) } else { String::new() },
        if !result.pattern_suffix.is_empty() { format!("suffix=\"{}\"", result.pattern_suffix) } else { String::new() },
    ].iter().filter(|s| !s.is_empty()).cloned().collect::<Vec<_>>().join(", ");
    if result.chain == ChainType::Ethereum {
        println!("- Pattern: {} (matched on lowercase, displayed with EIP-55 checksum)", pattern_parts);
    } else {
        let ignore_info = if result.ignore_case { " (case-insensitive)" } else { "" };
        println!("- Pattern: {}{}", pattern_parts, ignore_info);
    }
}

fn print_result_card(result: &VanityResult) {
    // Unicode mode
    println!("✨ Vanity Match");
    println!("🔗 Chain: {} ({})", result.chain.to_string().to_uppercase(),
             if result.chain == ChainType::Ethereum { "EVM" } else { "SVM" });
    println!("🏷️ Address: {}", result.address);
    println!("🔑 Private Key: {}", result.secret);
    println!("⚙️ Method: {}", result.backend);
    println!("⏱️ Elapsed: {:.3}s", result.elapsed_sec);
    println!("🔁 Iterations: {}", result.iterations.to_formatted_string(&Locale::en));
    println!("🚀 Rate: {} / sec", (result.rate_per_sec as u64).to_formatted_string(&Locale::en));
    if result.encodes > 0 {
        println!("🧮 Encodes: {}", result.encodes.to_formatted_string(&Locale::en));
        println!("⚡ Encodes/sec: {}", (result.encode_rate_per_sec as u64).to_formatted_string(&Locale::en));
    }
    let pattern_parts = [
        if !result.pattern_prefix.is_empty() { format!("prefix=\"{}\"", result.pattern_prefix) } else { String::new() },
        if !result.pattern_suffix.is_empty() { format!("suffix=\"{}\"", result.pattern_suffix) } else { String::new() },
    ].iter().filter(|s| !s.is_empty()).cloned().collect::<Vec<_>>().join(", ");
    if result.chain == ChainType::Ethereum {
        println!("🎯 Pattern: {} (matched on lowercase, displayed with EIP-55 checksum)", pattern_parts);
    } else {
        let ignore_info = if result.ignore_case { " (case-insensitive)" } else { "" };
        println!("🎯 Pattern: {}{}", pattern_parts, ignore_info);
    }
}

fn print_result_json(result: &VanityResult) {
    // Single-line JSON
    let json = build_result_json(result);
    println!("{}", serde_json::to_string(&json).unwrap());
}

// ========================================
// File Saving
// ========================================

const PBKDF2_ITERATIONS: u32 = 600_000;

fn build_result_json(result: &VanityResult) -> serde_json::Value {
    serde_json::json!({
        "chain": result.chain.to_string(),
        "address": result.address,
        "public_key": result.public_key,
        "secret_type": result.secret_type,
        "secret": result.secret,
        "pattern": {
            "prefix": result.pattern_prefix,
            "suffix": result.pattern_suffix,
            "ignore_case": result.ignore_case,
        },
        "metrics": {
            "backend": result.backend,
            "gpus": result.gpus,
            "cpus": result.cpus,
            "elapsed_sec": result.elapsed_sec,
            "iterations": result.iterations,
            "rate_per_sec": result.rate_per_sec,
            "encodes": result.encodes,
            "encode_rate_per_sec": result.encode_rate_per_sec,
        },
        "created_at": chrono::Utc::now().to_rfc3339(),
        "version": env!("CARGO_PKG_VERSION"),
    })
}

fn build_result_content(result: &VanityResult, format: OutputFormat) -> String {
    match format {
        OutputFormat::Json => serde_json::to_string_pretty(&build_result_json(result)).unwrap(),
        OutputFormat::Plain => {
            let mut output = String::new();
            output.push_str("* Vanity Match\n");
            output.push_str(&format!("- Chain: {} ({})\n", result.chain.to_string().to_uppercase(),
                                    if result.chain == ChainType::Ethereum { "EVM" } else { "SVM" }));
            output.push_str(&format!("- Address: {}\n", result.address));
            output.push_str(&format!("- Private Key: {}\n", result.secret));
            output.push_str(&format!("- Method: {}\n", result.backend));
            output.push_str(&format!("- Elapsed: {:.3}s\n", result.elapsed_sec));
            output.push_str(&format!("- Iterations: {}\n", result.iterations.to_formatted_string(&Locale::en)));
            output.push_str(&format!("- Rate: {} / sec\n", (result.rate_per_sec as u64).to_formatted_string(&Locale::en)));
            let pattern_parts = [
                if !result.pattern_prefix.is_empty() { format!("prefix=\"{}\"", result.pattern_prefix) } else { String::new() },
                if !result.pattern_suffix.is_empty() { format!("suffix=\"{}\"", result.pattern_suffix) } else { String::new() },
            ].iter().filter(|s| !s.is_empty()).cloned().collect::<Vec<_>>().join(", ");
            if result.chain == ChainType::Ethereum {
                output.push_str(&format!("- Pattern: {} (matched on lowercase, displayed with EIP-55 checksum)\n", pattern_parts));
            } else {
                let ignore_info = if result.ignore_case { " (case-insensitive)" } else { "" };
                output.push_str(&format!("- Pattern: {}{}\n", pattern_parts, ignore_info));
            }
            output
        }
        OutputFormat::Card => {
            let mut output = String::new();
            output.push_str("✨ Vanity Match\n");
            output.push_str(&format!("🔗 Chain: {} ({})\n", result.chain.to_string().to_uppercase(),
                                    if result.chain == ChainType::Ethereum { "EVM" } else { "SVM" }));
            output.push_str(&format!("🏷️ Address: {}\n", result.address));
            output.push_str(&format!("🔑 Private Key: {}\n", result.secret));
            output.push_str(&format!("⚙️ Method: {}\n", result.backend));
            output.push_str(&format!("⏱️ Elapsed: {:.3}s\n", result.elapsed_sec));
            output.push_str(&format!("🔁 Iterations: {}\n", result.iterations.to_formatted_string(&Locale::en)));
            output.push_str(&format!("🚀 Rate: {} / sec\n", (result.rate_per_sec as u64).to_formatted_string(&Locale::en)));
            let pattern_parts = [
                if !result.pattern_prefix.is_empty() { format!("prefix=\"{}\"", result.pattern_prefix) } else { String::new() },
                if !result.pattern_suffix.is_empty() { format!("suffix=\"{}\"", result.pattern_suffix) } else { String::new() },
            ].iter().filter(|s| !s.is_empty()).cloned().collect::<Vec<_>>().join(", ");
            if result.chain == ChainType::Ethereum {
                output.push_str(&format!("🎯 Pattern: {} (matched on lowercase, displayed with EIP-55 checksum)\n", pattern_parts));
            } else {
                let ignore_info = if result.ignore_case { " (case-insensitive)" } else { "" };
                output.push_str(&format!("🎯 Pattern: {}{}\n", pattern_parts, ignore_info));
            }
            output
        }
    }
}

fn encrypt_content(content: &str, passphrase: &str, format: OutputFormat) -> Result<String, String> {
    if passphrase.is_empty() {
        return Err("Encryption passphrase is empty".to_string());
    }

    let mut salt = [0u8; 16];
    let mut nonce_bytes = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut salt);
    rand::thread_rng().fill_bytes(&mut nonce_bytes);

    let mut key = [0u8; 32];
    pbkdf2_hmac::<Sha256>(passphrase.as_bytes(), &salt, PBKDF2_ITERATIONS, &mut key);

    let cipher = Aes256Gcm::new_from_slice(&key).map_err(|e| e.to_string())?;
    let ciphertext = cipher
        .encrypt(Nonce::from_slice(&nonce_bytes), content.as_bytes())
        .map_err(|e| format!("Encryption failed: {}", e))?;

    let payload_format = match format {
        OutputFormat::Card => "card",
        OutputFormat::Plain => "plain",
        OutputFormat::Json => "json",
    };
    let envelope = serde_json::json!({
        "encrypted": true,
        "algorithm": "AES-256-GCM",
        "kdf": {
            "name": "PBKDF2-HMAC-SHA256",
            "iterations": PBKDF2_ITERATIONS,
            "salt_base64": BASE64.encode(salt),
        },
        "nonce_base64": BASE64.encode(nonce_bytes),
        "ciphertext_base64": BASE64.encode(ciphertext),
        "payload_format": payload_format,
        "created_at": chrono::Utc::now().to_rfc3339(),
        "version": env!("CARGO_PKG_VERSION"),
    });

    serde_json::to_string_pretty(&envelope).map_err(|e| e.to_string())
}

fn decrypt_content(content: &str, passphrase: &str) -> Result<String, String> {
    let envelope: serde_json::Value =
        serde_json::from_str(content).map_err(|e| format!("Invalid encrypted file: {}", e))?;

    let algorithm = envelope
        .get("algorithm")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "Encrypted file is missing algorithm".to_string())?;
    if algorithm != "AES-256-GCM" {
        return Err(format!("Unsupported algorithm: {}", algorithm));
    }

    let iterations = envelope
        .get("kdf")
        .and_then(|v| v.get("iterations"))
        .and_then(|v| v.as_u64())
        .ok_or_else(|| "Encrypted file is missing PBKDF2 iterations".to_string())?;
    let salt = envelope
        .get("kdf")
        .and_then(|v| v.get("salt_base64"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| "Encrypted file is missing salt".to_string())?;
    let nonce = envelope
        .get("nonce_base64")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "Encrypted file is missing nonce".to_string())?;
    let ciphertext = envelope
        .get("ciphertext_base64")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "Encrypted file is missing ciphertext".to_string())?;

    let salt_bytes = BASE64.decode(salt).map_err(|e| format!("Invalid salt: {}", e))?;
    let nonce_bytes = BASE64.decode(nonce).map_err(|e| format!("Invalid nonce: {}", e))?;
    let ciphertext_bytes = BASE64.decode(ciphertext).map_err(|e| format!("Invalid ciphertext: {}", e))?;
    if nonce_bytes.len() != 12 {
        return Err("Invalid nonce length; expected 12 bytes".to_string());
    }

    let mut key = [0u8; 32];
    pbkdf2_hmac::<Sha256>(passphrase.as_bytes(), &salt_bytes, iterations as u32, &mut key);

    let cipher = Aes256Gcm::new_from_slice(&key).map_err(|e| e.to_string())?;
    let plaintext = cipher
        .decrypt(Nonce::from_slice(&nonce_bytes), ciphertext_bytes.as_ref())
        .map_err(|_| "Decryption failed. Check that the passphrase matches.".to_string())?;

    String::from_utf8(plaintext).map_err(|e| format!("Decrypted content is not valid UTF-8: {}", e))
}

fn resolve_save_path(result: &VanityResult, save_path: &PathBuf, format: OutputFormat, encrypted: bool) -> PathBuf {
    let path = if save_path.is_dir() {
        // Generate filename based on format
        let filename = format!(
            "vanity-{}-{}.{}",
            result.chain.to_string(),
            filename_address_hint(result.chain, &result.address),
            match format {
                OutputFormat::Json => "json",
                OutputFormat::Plain | OutputFormat::Card => "txt",
            }
        );
        save_path.join(filename)
    } else {
        save_path.clone()
    };

    if encrypted {
        let mut encrypted_path = path.clone().into_os_string();
        encrypted_path.push(".enc");
        PathBuf::from(encrypted_path)
    } else {
        path
    }
}

fn save_result(result: &VanityResult, save_path: &PathBuf, format: OutputFormat, encryption_passphrase: Option<&str>) -> Result<PathBuf, String> {
    let encrypted = encryption_passphrase.is_some();
    let path = resolve_save_path(result, save_path, format, encrypted);
    let raw_content = build_result_content(result, format);
    let content = if let Some(passphrase) = encryption_passphrase {
        encrypt_content(&raw_content, passphrase, format)?
    } else {
        raw_content
    };

    // Write with secure permissions
    #[cfg(target_os = "windows")]
    {
        fs::write(&path, content).map_err(|e| e.to_string())?;
        eprintln!("⚠️  Warning: File permissions may not be restricted on Windows. Please verify manually.");
    }

    #[cfg(not(target_os = "windows"))]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::write(&path, content).map_err(|e| e.to_string())?;
        let metadata = fs::metadata(&path).map_err(|e| e.to_string())?;
        let mut permissions = metadata.permissions();
        permissions.set_mode(0o600);
        fs::set_permissions(&path, permissions).map_err(|e| e.to_string())?;
    }

    Ok(path)
}

fn filename_address_hint(chain: ChainType, address: &str) -> String {
    let s = if chain == ChainType::Ethereum && address.starts_with("0x") {
        &address[2..]
    } else {
        address
    };
    s.chars().take(8).collect::<String>()
}

// ========================================
// Pattern Validation
// ========================================

fn validate_pattern(chain: ChainType, pattern: &str, pattern_type: &str, ignore_case: bool) -> Result<(), String> {
    match chain {
        ChainType::Ethereum => {
            // Ethereum: hex chars only (after lowercasing and removing 0x if present)
            let check_pattern = if pattern.starts_with("0x") || pattern.starts_with("0X") {
                &pattern[2..]
            } else {
                pattern
            };

            // Collect all invalid characters
            let invalid_chars: Vec<char> = check_pattern.chars()
                .filter(|ch| !ch.is_ascii_hexdigit())
                .collect();

            if !invalid_chars.is_empty() {
                // Remove duplicates while preserving order
                let mut unique_invalid: Vec<char> = Vec::new();
                for ch in invalid_chars {
                    if !unique_invalid.contains(&ch) {
                        unique_invalid.push(ch);
                    }
                }

                let invalid_list = unique_invalid.iter()
                    .map(|c| format!("'{}'", c))
                    .collect::<Vec<_>>()
                    .join(", ");

                return Err(format!(
                    "Invalid {} '{}' for Ethereum:\n\n\
                    ❌ Invalid character(s): {}\n\
                    ✅ Allowed characters: 0-9, a-f, A-F (hexadecimal)\n\
                    💡 Tip: Remove these characters from your {}: {}",
                    pattern_type, pattern, invalid_list, pattern_type, invalid_list
                ));
            }
        }
        ChainType::Solana => {
            // Solana: base58 chars only
            const BS58_CHARS: &str = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

            // Collect all invalid characters
            let invalid_chars: Vec<char> = pattern.chars()
                .filter(|ch| !BS58_CHARS.contains(*ch) && !(ignore_case && ch.is_ascii_alphabetic()
                    && (BS58_CHARS.contains(ch.to_ascii_lowercase()) || BS58_CHARS.contains(ch.to_ascii_uppercase()))))
                .collect();

            if !invalid_chars.is_empty() {
                // Remove duplicates while preserving order
                let mut unique_invalid: Vec<char> = Vec::new();
                for ch in invalid_chars {
                    if !unique_invalid.contains(&ch) {
                        unique_invalid.push(ch);
                    }
                }

                let invalid_list = unique_invalid.iter()
                    .map(|c| format!("'{}'", c))
                    .collect::<Vec<_>>()
                    .join(", ");

                // Provide helpful suggestions for commonly confused characters
                let mut suggestions = Vec::new();
                for ch in &unique_invalid {
                    match ch {
                        '0' => suggestions.push("'0' (zero) → try '1' or remove it"),
                        'O' => suggestions.push("'O' (letter O) → try 'P' or remove it"),
                        'I' => suggestions.push("'I' (letter I) → try 'J' or remove it"),
                        'l' => suggestions.push("'l' (lowercase L) → try 'k' or 'm', or remove it"),
                        _ => {}
                    }
                }

                let mut error_msg = format!(
                    "Invalid {} '{}' for Solana:\n\n\
                    ❌ Invalid character(s): {}\n\
                    ✅ Allowed characters: Base58 (1-9, A-Z, a-z, excluding 0, O, I, l)\n",
                    pattern_type, pattern, invalid_list
                );

                if !suggestions.is_empty() {
                    error_msg.push_str(&format!("💡 Suggestions:\n   {}\n", suggestions.join("\n   ")));
                }

                error_msg.push_str(&format!("💡 Tip: Remove these characters from your {}: {}",
                    pattern_type, invalid_list));

                return Err(error_msg);
            }
        }
    }
    Ok(())
}

// ========================================
// Pattern Normalization
// ========================================

fn normalize_pattern(pattern: &str, ignore_case: bool, chain: ChainType) -> String {
    // Remove 0x prefix if present for Ethereum
    let pattern = if chain == ChainType::Ethereum && (pattern.starts_with("0x") || pattern.starts_with("0X")) {
        &pattern[2..]
    } else {
        pattern
    };

    match chain {
        ChainType::Ethereum => {
            // Ethereum addresses are always matched on lowercase (before EIP-55 checksum)
            pattern.to_lowercase()
        }
        ChainType::Solana => {
            // Solana: case-sensitive by default, unless ignore_case is set
            if ignore_case {
                pattern.to_lowercase()
            } else {
                pattern.to_string()
            }
        }
    }
}

// ========================================
// GPU/CPU Detection & Config
// ========================================

#[cfg(feature = "gpu")]
fn detect_gpu_count() -> u32 {
    // Try to detect available GPUs
    unsafe {
        let ret = cuda_init();
        if ret == 0 {
            1 // Assume at least 1 GPU if init succeeds
        } else {
            0
        }
    }
}

// ========================================
// Main Logic
// ========================================

#[cfg(feature = "gpu")]
static EXIT: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "gpu")]
extern "C" {
    fn cuda_init() -> i32;

    // Solana optimized kernel
    fn sol_vanity_round_optimized(
        gpu_id: i32,
        seed: *const u8,
        prefix: *const u8,
        suffix: *const u8,
        prefix_len: u64,
        suffix_len: u64,
        out: *mut u8,
        case_insensitive: bool,
        num_blocks: i32,
        num_threads: i32,
        iterations_per_thread: u64,
    ) -> i32;
    fn sol_vanity_cleanup(gpu_id: i32);

    // Ethereum ultra-optimized kernel with auto-tuning
    fn eth_vanity_round_ultra(
        gpu_id: i32,
        seed: *const u8,
        prefix: *const u8,
        suffix: *const u8,
        prefix_len: u64,
        suffix_len: u64,
        out: *mut u8,
        case_insensitive: bool,
        num_blocks: i32,
        num_threads: i32,
        iterations_per_thread: u64,
    ) -> i32;
    fn eth_vanity_cleanup_ultra(gpu_id: i32);

    // GPU configuration and auto-tuning
    fn get_gpu_config(gpu_id: i32, config: *mut GpuConfig) -> i32;
    fn print_gpu_config(config: *const GpuConfig);
}

#[cfg(feature = "gpu")]
#[repr(C)]
struct GpuConfig {
    device_id: i32,
    sm_count: i32,
    max_threads_per_sm: i32,
    max_threads_per_block: i32,
    max_blocks_per_sm: i32,
    total_global_mem: usize,
    shared_mem_per_block: usize,
    compute_capability_major: i32,
    compute_capability_minor: i32,
    optimal_threads_per_block: i32,
    optimal_blocks: i32,
    optimal_iterations_per_thread: i32,
}

#[cfg(feature = "gpu")]
fn new_gpu_seed(_gpu_index: u32, _iteration: u64) -> [u8; 32] {
    let mut seed = [0u8; 32];
    // Use fully random seed on every iteration for better search space coverage
    // This ensures GPU threads explore truly independent regions of the keyspace
    rand::thread_rng().fill_bytes(&mut seed);
    seed
}

fn resolve_passphrase(var_name: &str) -> Result<String, String> {
    match env::var(var_name) {
        Ok(value) if !value.is_empty() => Ok(value),
        Ok(_) => Err(format!("Environment variable {} is set but empty", var_name)),
        Err(_) => Err(format!("Environment variable {} is not set", var_name)),
    }
}

fn decrypt_file(args: &Args) {
    let input = args.decrypt.as_ref().expect("decrypt path must exist");
    let passphrase = match resolve_passphrase(&args.encrypt_passphrase_env) {
        Ok(value) => value,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    let encrypted_content = match fs::read_to_string(input) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error reading encrypted file {}: {}", input.display(), e);
            std::process::exit(1);
        }
    };

    let decrypted = match decrypt_content(&encrypted_content, &passphrase) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error decrypting file: {}", e);
            std::process::exit(1);
        }
    };

    if let Some(output_path) = &args.decrypt_out {
        if let Err(e) = fs::write(output_path, &decrypted) {
            eprintln!("Error writing decrypted file {}: {}", output_path.display(), e);
            std::process::exit(1);
        }
    } else {
        print!("{}", decrypted);
    }
}

fn main() {
    // Parse command line arguments
    let args = Args::parse();

    if args.decrypt.is_some() {
        decrypt_file(&args);
        return;
    }

    // Run the generation
    generate(args);
}

fn generate(args: Args) {
    let chain = match args.chain {
        Some(chain) => chain,
        None => {
            eprintln!("Error: --chain is required unless you are using --decrypt");
            std::process::exit(1);
        }
    };

    // Validate that prefix or suffix is provided
    if args.prefix.is_none() && args.suffix.is_none() {
        eprintln!("Error: Must provide at least one of --prefix or --suffix");
        std::process::exit(1);
    }

    // Validate patterns
    if let Some(ref prefix) = args.prefix {
        if let Err(e) = validate_pattern(chain, prefix, "prefix", args.ignore_case) {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }

    if let Some(ref suffix) = args.suffix {
        if let Err(e) = validate_pattern(chain, suffix, "suffix", args.ignore_case) {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }

    if args.encrypt && args.save.is_none() {
        eprintln!("Error: --encrypt requires --save");
        std::process::exit(1);
    }

    if args.decrypt_out.is_some() {
        eprintln!("Error: --decrypt-out can only be used together with --decrypt");
        std::process::exit(1);
    }

    // Normalize patterns
    let prefix = args.prefix.as_ref()
        .map(|p| normalize_pattern(p, args.ignore_case, chain))
        .unwrap_or_default();
    let suffix = args.suffix.as_ref()
        .map(|s| normalize_pattern(s, args.ignore_case, chain))
        .unwrap_or_default();

    let max_len = if chain == ChainType::Ethereum { 40 } else { 44 };
    if prefix.len() > max_len || suffix.len() > max_len {
        eprintln!("Error: patterns cannot exceed {} address characters", max_len);
        std::process::exit(1);
    }

    let encryption_passphrase = if args.encrypt {
        match resolve_passphrase(&args.encrypt_passphrase_env) {
            Ok(value) => Some(value),
            Err(e) => {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        None
    };

    // Detect/configure GPUs
    #[cfg(feature = "gpu")]
    let num_gpus = match args.gpus {
        Some(n) => n,
        None => detect_gpu_count(),
    };

    #[cfg(not(feature = "gpu"))]
    let num_gpus = 0;

    #[cfg(feature = "gpu")]
    if num_gpus > 16 || args.gpu_blocks.is_some_and(|n| n == 0 || n > i32::MAX as u32)
        || args.gpu_threads.is_some_and(|n| n == 0 || n > 1024)
        || args.gpu_iters == Some(0) || args.gpu_target_ms == Some(0) {
        eprintln!("Error: invalid GPU configuration (1..16 GPUs, positive blocks/iterations/time, 1..1024 threads)");
        std::process::exit(1);
    }

    // Configure CPUs
    let num_cpus = if args.cpus == 0 {
        if num_gpus == 0 {
            rayon::current_num_threads().max(1) as u32
        } else {
            0 // Default to GPU only if GPU available
        }
    } else {
        args.cpus
    };

    // Apply Rayon thread pool size if using CPU
    if num_gpus == 0 && num_cpus > 0 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus as usize)
            .build_global();
    }

    // Initialize CUDA if using GPU
    #[cfg(feature = "gpu")]
    if num_gpus > 0 {
        if args.debug {
            eprintln!("[debug] Initializing CUDA...");
        }
        unsafe {
            let ret = cuda_init();
            if ret != 0 {
                eprintln!("Error: Failed to initialize CUDA (error code {})", ret);
                std::process::exit(1);
            }
        }
        if args.debug {
            eprintln!("[debug] CUDA initialized successfully (GPU: {})", num_gpus);
        }
    }

    if args.debug {
        eprintln!("[debug] Chain: {:?}", chain);
        eprintln!("[debug] Prefix: '{}'", prefix);
        eprintln!("[debug] Suffix: '{}'", suffix);
        eprintln!("[debug] Ignore case: {}", args.ignore_case);
        eprintln!("[debug] Count: {}", args.count);
        eprintln!("[debug] Max runtime: {:?}", args.max_runtime);
    }

    // Run the generation
    let start_time = Instant::now();
    let deadline = args.max_runtime.map(|s| start_time + Duration::from_secs(s));

    let progress = progress::Progress::new(args.progress, args.output == OutputFormat::Json,
        chain, &prefix, &suffix, args.ignore_case, args.count);

    // For now, we'll use GPU in main thread (simpler for Windows)
    #[cfg(feature = "gpu")]
    if num_gpus > 0 {
        run_gpu_generation(&args, &prefix, &suffix, num_gpus, num_cpus, start_time, deadline, encryption_passphrase.as_deref(), &progress);
        return;
    }

    // CPU fallback
    run_cpu_generation(&args, &prefix, &suffix, num_cpus, start_time, deadline, encryption_passphrase.as_deref(), &progress);
}

#[cfg(feature = "gpu")]
fn run_gpu_generation(
    args: &Args,
    prefix: &str,
    suffix: &str,
    num_gpus: u32,
    num_cpus: u32,
    start_time: Instant,
    deadline: Option<Instant>,
    encryption_passphrase: Option<&str>,
    progress: &progress::Progress,
) {
    let chain = args.generation_chain();
    match chain {
        ChainType::Ethereum => {
            let mut out = [0u8; 149]; // 32 privkey + 65 pubkey + 40 address + 8 count + 4 done
            let mut iteration = 0_u64;
            let mut last_gpu_index = 0i32;
            let mut matches_found: u32 = 0;

            // Get GPU configuration and auto-tune parameters
            let mut gpu_config = GpuConfig {
                device_id: 0,
                sm_count: 0,
                max_threads_per_sm: 0,
                max_threads_per_block: 0,
                max_blocks_per_sm: 0,
                total_global_mem: 0,
                shared_mem_per_block: 0,
                compute_capability_major: 0,
                compute_capability_minor: 0,
                optimal_threads_per_block: 256,
                optimal_blocks: 8192,
                optimal_iterations_per_thread: 10_000,  // Default 10K iterations per thread
            };

            unsafe {
                if get_gpu_config(0, &mut gpu_config as *mut GpuConfig) == 0 {
                    if args.debug {
                        print_gpu_config(&gpu_config as *const GpuConfig);
                    }
                } else {
                    eprintln!("Warning: Could not get GPU config, using defaults");
                }
            }
            // Initial config from device, then apply overrides and OS-safe defaults
            let mut blocks = gpu_config.optimal_blocks.max(1);
            let mut threads = gpu_config.optimal_threads_per_block.max(128);
            let mut iters = (gpu_config.optimal_iterations_per_thread as u64).max(1000);

            // Apply user overrides if provided
            if let Some(b) = args.gpu_blocks { blocks = b as i32; }
            if let Some(t) = args.gpu_threads { threads = t as i32; }
            if let Some(i) = args.gpu_iters { iters = i; }

            // Safe defaults per-OS
            #[cfg(target_os = "windows")]
            {
                let sm = gpu_config.sm_count.max(1);
                // Start reasonably high; autotuner will adjust towards target_ms
                if args.gpu_blocks.is_none() { blocks = (sm * 4).max(blocks); }
                if args.gpu_threads.is_none() { threads = 256; }
                if args.gpu_iters.is_none() { iters = 256; }
            }
            #[cfg(not(target_os = "windows"))]
            {
                let sm = gpu_config.sm_count.max(1);
                if args.gpu_blocks.is_none() { blocks = (sm * 8).max(blocks); }
                if args.gpu_threads.is_none() { threads = 512; }
                if args.gpu_iters.is_none() { iters = 256; }
            }
            if args.debug {
                let keys_per_batch = (blocks as u64) * (threads as u64) * (iters as u64);
                eprintln!(
                    "[debug] GPU batch config → blocks={}, threads={}, iters/thread={}, keys/batch={}",
                    blocks, threads, iters, keys_per_batch
                );
            }

            let mut total_iters: u64 = 0;
            let target_ms: u64 = args.gpu_target_ms.unwrap_or_else(|| {
                if cfg!(target_os = "windows") { 900 } else { 1500 }
            });
            loop {
                if EXIT.load(Ordering::SeqCst) {
                    unsafe { eth_vanity_cleanup_ultra(last_gpu_index); }
                    break;
                }

                if let Some(d) = deadline {
                    if Instant::now() >= d {
                        unsafe { eth_vanity_cleanup_ultra(last_gpu_index); }
                        break;
                    }
                }

                out.fill(0);
                let gpu_index = (iteration as u32 % num_gpus) as i32;
                last_gpu_index = gpu_index;
                let seed = new_gpu_seed(gpu_index as u32, iteration);
                iteration += 1;

                let batch_start = Instant::now();
                let status = unsafe {
                    eth_vanity_round_ultra(
                        gpu_index,
                        seed.as_ptr(),
                        prefix.as_ptr(),
                        suffix.as_ptr(),
                        prefix.len() as u64,
                        suffix.len() as u64,
                        out.as_mut_ptr(),
                        args.ignore_case,
                        blocks,
                        threads,
                        iters,
                    )
                };

                if status != 0 {
                    // Launch or memcpy error: back off parameters and retry next loop
                    if args.debug {
                        eprintln!("[debug] GPU launch failed (status={}), backing off...", status);
                    }
                    if threads > 256 { threads /= 2; }
                    else if blocks > gpu_config.sm_count { blocks = (blocks / 2).max(gpu_config.sm_count); }
                    else { iters = (iters / 2).max(64); }
                    continue;
                }

                let count = u64::from_le_bytes(array::from_fn(|i| out[137 + i]));
                total_iters = total_iters.saturating_add(count);
                progress.record_batch(total_iters, count, batch_start.elapsed());
                let done = i32::from_le_bytes(array::from_fn(|i| out[145 + i]));
                let elapsed_ms = batch_start.elapsed().as_millis() as u64;
                if args.debug {
                    eprintln!("[debug] GPU batch complete: {} iterations, done={} (total={}), ~{} ms", count, done, total_iters, elapsed_ms);
                }

                // Simple autotuner: adjust iters to approach target_ms on next batch (Windows focus)
                if args.gpu_iters.is_none() && done == 0 {
                    if elapsed_ms < target_ms.saturating_div(2) {
                        iters = (iters.saturating_mul(2)).min(200_000);
                    } else if elapsed_ms > target_ms.saturating_mul(3).saturating_div(2) {
                        iters = (iters / 2).max(64);
                    }
                }

                if done == 1 {
                    let elapsed = start_time.elapsed().as_secs_f64();

                    // Extract results
                    let private_key_bytes: [u8; 32] = array::from_fn(|i| out[i]);
                    let public_key_bytes: [u8; 65] = array::from_fn(|i| out[32 + i]);
                    let address_hex = String::from_utf8_lossy(&out[97..137]).to_string();

                    let verified_key = SecretKey::from_slice(&private_key_bytes)
                        .expect("GPU returned invalid private key");
                    let verified_public = PublicKey::from_secret_key(&Secp256k1::new(), &verified_key).serialize_uncompressed();
                    let verified_address = hex::encode(&Keccak256::digest(&verified_public[1..])[12..]);
                    assert!(public_key_bytes == verified_public && address_hex == verified_address
                        && address_hex.starts_with(prefix) && address_hex.ends_with(suffix),
                        "GPU result failed independent CPU verification");

                    // Apply EIP-55 checksum
                    let address = eip55_checksum(&address_hex);

                    let result = VanityResult {
                        chain,
                        address: address.clone(),
                        public_key: format!("0x{}", hex::encode(&public_key_bytes)),
                        secret_type: "private_key",
                        secret: format!("0x{}", hex::encode(&private_key_bytes)),
                        pattern_prefix: prefix.to_string(),
                        pattern_suffix: suffix.to_string(),
                        ignore_case: args.ignore_case,
                        backend: format!("GPU (CUDA) x{}", num_gpus),
                        gpus: num_gpus,
                        cpus: num_cpus,
                        elapsed_sec: elapsed,
                        iterations: total_iters,
                        rate_per_sec: total_iters as f64 / elapsed,
                        encodes: total_iters,
                        encode_rate_per_sec: total_iters as f64 / elapsed,
                    };

                    // Output to screen
                    progress.found();
                    progress.output(|| print_result(&result, args.output));

                    // Save if requested
                    if let Some(ref path_opt) = args.save {
                        let path = path_opt.clone().unwrap_or_else(|| PathBuf::from("."));
                        if let Err(e) = save_result(&result, &path, args.output, encryption_passphrase) {
                            eprintln!("Error saving result: {}", e);
                        }
                    }

                    matches_found += 1;
                    if matches_found >= args.count {
                        unsafe { eth_vanity_cleanup_ultra(gpu_index); }
                        break;
                    }
                }
            }
            // Cleanup all GPUs used
            for i in 0..num_gpus {
                unsafe { eth_vanity_cleanup_ultra(i as i32); }
            }
        }
        ChainType::Solana => {
            let mut out = [0u8; 192]; // 32 seed + 64 privkey + 32 pubkey + 44 address + 8 count + 8 encodes + 4 done
            let mut iteration = 0_u64;
            let mut total_iters = 0u64;
            let mut total_encodes = 0u64;
            let mut last_gpu_index = 0i32;
            let mut matches_found: u32 = 0;
            // Start small enough to measure before choosing a larger batch.
            let mut blocks = args.gpu_blocks.unwrap_or(128) as i32;
            let mut threads = args.gpu_threads.unwrap_or(128) as i32;
            let mut iters = args.gpu_iters.unwrap_or(64);
            let target_ms = args.gpu_target_ms.unwrap_or(500).max(1);

            loop {
                if EXIT.load(Ordering::SeqCst) {
                    unsafe { sol_vanity_cleanup(last_gpu_index); }
                    break;
                }

                if let Some(d) = deadline {
                    if Instant::now() >= d {
                        unsafe { sol_vanity_cleanup(last_gpu_index); }
                        break;
                    }
                }

                out.fill(0);
                let gpu_index = (iteration as u32 % num_gpus) as i32;
                last_gpu_index = gpu_index;
                let seed = new_gpu_seed(gpu_index as u32, iteration);
                iteration += 1;

                let batch_start = Instant::now();
                let status = unsafe {
                    sol_vanity_round_optimized(
                        gpu_index,
                        seed.as_ptr(),
                        prefix.as_ptr(),
                        suffix.as_ptr(),
                        prefix.len() as u64,
                        suffix.len() as u64,
                        out.as_mut_ptr(),
                        args.ignore_case,
                        blocks,
                        threads,
                        iters,
                    )
                };
                if status != 0 {
                    if args.debug { eprintln!("[debug] SOL GPU launch failed (status={}), backing off...", status); }
                    if threads > 256 { threads /= 2; }
                    else if blocks > 256 { blocks /= 2; }
                    else { iters = (iters / 2).max(1); }
                    continue;
                }

                let count = u64::from_le_bytes(array::from_fn(|i| out[172 + i]));
                let encodes = u64::from_le_bytes(array::from_fn(|i| out[180 + i]));
                total_iters += count;
                progress.record_batch(total_iters, count, batch_start.elapsed());
                total_encodes += encodes;
                let done = i32::from_le_bytes(array::from_fn(|i| out[188 + i]));
                let elapsed_ms = batch_start.elapsed().as_millis() as u64;
                if args.debug {
                    eprintln!("[debug] SOL GPU batch: {} iterations, done={}, ~{} ms", count, done, elapsed_ms);
                }
                if args.gpu_iters.is_none() && elapsed_ms > 0 && done == 0 {
                    iters = ((iters as u128 * target_ms as u128 / elapsed_ms as u128) as u64)
                        .clamp((iters / 2).max(1), iters.saturating_mul(2).min(200_000));
                }

                if done == 1 {
                    let elapsed = start_time.elapsed().as_secs_f64();

                    // Extract results
                    let seed_bytes: [u8; 32] = array::from_fn(|i| out[i]);
                    let pubkey_bytes: [u8; 32] = array::from_fn(|i| out[96 + i]);
                    let address_bytes = &out[128..172];
                    let address_end = address_bytes.iter().position(|&b| b == 0).unwrap_or(44);
                    let address = String::from_utf8_lossy(&address_bytes[..address_end]).to_string();

                    let verified_public = SigningKey::from_bytes(&seed_bytes).verifying_key().to_bytes();
                    let verified_address = bs58::encode(verified_public).into_string();
                    let check = if args.ignore_case { address.to_ascii_lowercase() } else { address.clone() };
                    assert!(pubkey_bytes == verified_public && address == verified_address
                        && check.starts_with(prefix) && check.ends_with(suffix),
                        "GPU result failed independent CPU verification");

                    // Solana keypair format: private key (32 bytes) + public key (32 bytes) = 64 bytes
                    let mut keypair_bytes = [0u8; 64];
                    // For the GPU path, the first 32 bytes returned are the private key bytes
                    keypair_bytes[0..32].copy_from_slice(&seed_bytes);
                    keypair_bytes[32..64].copy_from_slice(&pubkey_bytes);

                    let result = VanityResult {
                        chain,
                        address: address.clone(),
                        public_key: bs58::encode(&pubkey_bytes).into_string(),
                        secret_type: "private_key",
                        secret: format_seed_as_array(&keypair_bytes),
                        pattern_prefix: prefix.to_string(),
                        pattern_suffix: suffix.to_string(),
                        ignore_case: args.ignore_case,
                        backend: format!("GPU (CUDA) x{}", num_gpus),
                        gpus: num_gpus,
                        cpus: num_cpus,
                        elapsed_sec: elapsed,
                        iterations: total_iters,
                        rate_per_sec: total_iters as f64 / elapsed,
                        encodes: total_encodes,
                        encode_rate_per_sec: total_encodes as f64 / elapsed,
                    };

                    // Output to screen
                    progress.found();
                    progress.output(|| print_result(&result, args.output));

                    // Save if requested
                    if let Some(ref path_opt) = args.save {
                        let path = path_opt.clone().unwrap_or_else(|| PathBuf::from("."));
                        if let Err(e) = save_result(&result, &path, args.output, encryption_passphrase) {
                            eprintln!("Error saving result: {}", e);
                        }
                    }

                    matches_found += 1;
                    if matches_found >= args.count {
                        unsafe { sol_vanity_cleanup(gpu_index); }
                        break;
                    }
                }
            }
            // Cleanup all GPUs used
            for i in 0..num_gpus {
                unsafe { sol_vanity_cleanup(i as i32); }
            }
        }
    }
}

fn run_cpu_generation(
    args: &Args,
    prefix: &str,
    suffix: &str,
    num_cpus: u32,
    start_time: Instant,
    deadline: Option<Instant>,
    encryption_passphrase: Option<&str>,
    progress: &progress::Progress,
) {
    use rayon::prelude::*;
    let chain = args.generation_chain();
    let iterations = progress.attempts();
    let encodes = AtomicU64::new(0);
    let secp = Secp256k1::new();
    let eth_matcher = matcher::EthMatcher::new(
        if chain == ChainType::Ethereum { prefix } else { "" },
        if chain == ChainType::Ethereum { suffix } else { "" },
    );
    let sol_matcher = matcher::SolMatcher::new(prefix, suffix, args.ignore_case);
    for _ in 0..args.count {
        let stopped = AtomicBool::new(false);
        let hit = (0..num_cpus).into_par_iter().find_map_any(|_| {
            let mut rng = rand::thread_rng();
            let mut pending = 0u64;
            let mut encoded_count = 0u64;
            let mut walk = if chain == ChainType::Ethereum {
                Some(eth_walk::EthWalk::new(&secp, SecretKey::new(&mut rng)))
            } else { None };
            let mut first = true;
            loop {
                if pending == 256 {
                    iterations.fetch_add(pending, Ordering::Relaxed);
                    pending = 0;
                }
                if stopped.load(Ordering::Relaxed)
                    || (pending == 0 && deadline.is_some_and(|d| Instant::now() >= d)) {
                    iterations.fetch_add(pending, Ordering::Relaxed);
                    encodes.fetch_add(encoded_count, Ordering::Relaxed);
                    return None;
                }
                let mut secret = [0u8; 32];
                pending += 1;
                let candidate = match chain {
                    ChainType::Ethereum => {
                        let walk = walk.as_mut().unwrap();
                        if !first { walk.advance(); }
                        first = false;
                        let (sk, public) = walk.current();
                        secret = sk;
                        encoded_count += 1;
                        let hash = Keccak256::digest(&public[1..]);
                        if !eth_matcher.matches(&hash[12..]) { continue; }
                        let address = eip55_checksum(&hex::encode(&hash[12..]));
                        (address, format!("0x{}", hex::encode(public)), format!("0x{}", hex::encode(secret)))
                    }
                    ChainType::Solana => {
                        rng.fill_bytes(&mut secret);
                        let key = SigningKey::from_bytes(&secret);
                        let public = key.verifying_key().to_bytes();
                        let mut encoded = [0; 44];
                        let Some(len) = sol_matcher.matches_counted(&public, &mut encoded, &mut encoded_count) else { continue; };
                        let address = std::str::from_utf8(&encoded[..len]).unwrap().to_owned();
                        (address.clone(), address, format_seed_as_array(&key.to_keypair_bytes()))
                    }
                };
                iterations.fetch_add(pending, Ordering::Relaxed);
                encodes.fetch_add(encoded_count, Ordering::Relaxed);
                // Exactly one winner per round, including empty/easy patterns.
                return if !stopped.swap(true, Ordering::Relaxed) { Some(candidate) } else { None };
            }
        });
        let Some((address, public_key, secret)) = hit else { break; };
        let elapsed = start_time.elapsed().as_secs_f64();
        let total = iterations.load(Ordering::Relaxed);
        let rate = total as f64 / elapsed;
        let result = VanityResult {
            chain, address, public_key, secret_type: "private_key", secret,
            pattern_prefix: prefix.to_owned(), pattern_suffix: suffix.to_owned(),
            ignore_case: args.ignore_case, backend: "CPU".to_owned(), gpus: 0, cpus: num_cpus,
            elapsed_sec: elapsed, iterations: total, rate_per_sec: rate,
            encodes: encodes.load(Ordering::Relaxed),
            encode_rate_per_sec: encodes.load(Ordering::Relaxed) as f64 / elapsed,
        };
        progress.found();
        progress.output(|| print_result(&result, args.output));
        if let Some(ref path_opt) = args.save {
            let path = path_opt.clone().unwrap_or_else(|| PathBuf::from("."));
            if let Err(e) = save_result(&result, &path, args.output, encryption_passphrase) {
                eprintln!("Error saving result: {}", e);
            }
        }
    }
    if args.debug {
        let total = iterations.load(Ordering::Relaxed);
        eprintln!("[debug] CPU complete: {} candidates, {:.0} candidates/sec", total,
            total as f64 / start_time.elapsed().as_secs_f64());
    }
}

#[cfg(all(test, feature = "gpu"))]
mod gpu_tests;

#[cfg(test)]
mod benchmarks;
