use sha3::{Digest, Keccak256};
use std::process::Command;

fn command() -> Command {
    #[allow(unused_mut)] // GPU builds add an explicit CPU-only option.
    let mut command = Command::new(env!("CARGO_BIN_EXE_vanity"));
    #[cfg(feature = "gpu")]
    command.args(["--gpus", "0"]);
    command
}

#[test]
fn cpu_count_and_wallet_compatibility() {
    for chain in ["eth", "sol"] {
        for (prefix, suffix, count) in [("", "", 20), ("a", "b", 3)] {
            let output = command()
                .args([
                    "--chain",
                    chain,
                    "--cpus",
                    "4",
                    "--prefix",
                    prefix,
                    "--suffix",
                    suffix,
                    "--ignore-case",
                    "--count",
                    &count.to_string(),
                    "--max-runtime",
                    "10",
                    "--output",
                    "json",
                ])
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let text = String::from_utf8(output.stdout).unwrap();
            let rows: Vec<serde_json::Value> = text
                .lines()
                .map(|s| serde_json::from_str(s).unwrap())
                .collect();
            assert_eq!(rows.len(), count);
            let mut unique = std::collections::HashSet::new();
            for row in rows {
                let secret = row["secret"].as_str().unwrap();
                assert!(unique.insert(secret.to_owned()));
                let address = row["address"].as_str().unwrap();
                let derived = if chain == "eth" {
                    let secret = hex::decode(secret.strip_prefix("0x").unwrap()).unwrap();
                    let sk = secp256k1::SecretKey::from_slice(&secret).unwrap();
                    let pk =
                        secp256k1::PublicKey::from_secret_key(&secp256k1::Secp256k1::new(), &sk)
                            .serialize_uncompressed();
                    format!("0x{}", hex::encode(&Keccak256::digest(&pk[1..])[12..]))
                } else {
                    let bytes: Vec<u8> = serde_json::from_str(secret).unwrap();
                    let key = ed25519_dalek::SigningKey::from_keypair_bytes(
                        bytes.as_slice().try_into().unwrap(),
                    )
                    .unwrap();
                    bs58::encode(key.verifying_key().to_bytes()).into_string()
                };
                assert_eq!(derived.to_ascii_lowercase(), address.to_ascii_lowercase());
                let matched = address
                    .strip_prefix("0x")
                    .unwrap_or(address)
                    .to_ascii_lowercase();
                assert!(matched.starts_with(prefix) && matched.ends_with(suffix));
                assert!(
                    row["metrics"]["encodes"].as_u64().unwrap()
                        <= row["metrics"]["iterations"].as_u64().unwrap()
                );
            }
        }
    }
}

#[test]
fn validates_lengths_and_folded_base58() {
    for (chain, len) in [("eth", 41), ("sol", 45)] {
        assert!(!command()
            .args([
                "--chain",
                chain,
                "--prefix",
                &"a".repeat(len),
                "--max-runtime",
                "0"
            ])
            .output()
            .unwrap()
            .status
            .success());
    }
    assert!(command()
        .args([
            "--chain",
            "sol",
            "--prefix",
            "l",
            "--ignore-case",
            "--max-runtime",
            "0"
        ])
        .output()
        .unwrap()
        .status
        .success());
}
