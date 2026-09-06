# Vanity

**CPU/GPU Vanity Address Generator for SVM (Solana) and EVM (Ethereum) blockchains**

## Overview

Vanity is a high-performance tool for generating vanity addresses with custom prefixes and/or suffixes for both Ethereum and Solana blockchains. It leverages GPU acceleration via CUDA for maximum throughput.

### Features

- 🚀 **Blazing Fast**: GPU acceleration with CUDA support
- 🔗 **Multi-Chain**: Supports both Ethereum (EVM) and Solana (SVM)
- 🎯 **Flexible Matching**: Prefix, suffix, or both with case-insensitive options
- 📊 **Multiple Output Formats**: Pretty cards, ASCII-only, or JSON
- 💾 **Optional File Saving**: Secure file saving with proper permissions
- 🔐 **Encrypted Result Files**: Encrypt saved matches on cloud machines and decrypt later on a trusted device
- 🔒 **EIP-55 Checksummed**: Ethereum addresses displayed with proper checksums

## Screenshots
<img width="976" height="824" alt="image" src="https://github.com/user-attachments/assets/f4e77357-8f07-4e10-8d11-e91de0593f2c" />


## Installation

### Prerequisites

- Rust toolchain (1.70+)
- For GPU support: CUDA Toolkit 11.0+

### CPU-Only Build

```bash
cargo install --path .
```

### CPU + GPU Build

```bash
cargo install --path . --features gpu
```

For those without a GPU, consider using cloud GPU providers like [vast.ai](https://cloud.vast.ai/?ref_id=126830).

## Usage

### Basic Command Structure

```bash
vanity --chain <eth|sol> [OPTIONS]
```

Decrypt mode:

```bash
vanity --decrypt <FILE.enc> [--decrypt-out <PATH>]
```

### Options

- `--chain <CHAIN>`: Chain type: `eth|ethereum|evm` or `sol|solana|svm` (required for generation, not needed for `--decrypt`)
- `-p, --prefix <PREFIX>`: Target prefix for the address
- `-s, --suffix <SUFFIX>`: Target suffix for the address
- `-i, --ignore-case`: Case-insensitive matching (Solana only)
- `--gpus <N>`: Number of GPUs to use (default: 1 if GPU detected, else 0) [available only when built with `--features gpu`]
- `--gpu-blocks <N>` (GPU builds): Override GPU blocks per launch
- `--gpu-threads <N>` (GPU builds): Override GPU threads per block
- `--gpu-iters <N>` (GPU builds): Override GPU iterations per thread
- `--gpu-target-ms <MS>` (GPU builds): Adaptive batch time target in ms
- `--cpus <N>`: Number of CPU threads (default: 0 = auto)
- `--count <N>`: Number of matches to find (default: 1)
- `--max-runtime <SECONDS>`: Maximum runtime in seconds (default: unlimited)
- `-o, --output <FORMAT>`: Output format: `card|plain|json` (default: card)
- `--save [PATH]`: Save result to file (optional path, defaults to current directory)
- `--encrypt`: Encrypt the saved result file with AES-256-GCM
- `--encrypt-passphrase-env <NAME>`: Environment variable holding the encryption/decryption passphrase (default: `VANITY_ENCRYPTION_PASSWORD`)
- `--decrypt <FILE.enc>`: Decrypt a previously encrypted result file
- `--decrypt-out <PATH>`: Write decrypted content to a file instead of stdout
- `--debug`: Extra diagnostics to stderr

## Examples

### Ethereum (EVM)

#### Generate with prefix (GPU)

```bash
vanity --chain eth -p cafe
```

Output:
```
✨ Vanity Match
🔗 Chain: ETH (EVM)
🏷️ Address: 0xCAFE935A8e7F3Ad03099c811c2299E5643DdEFf8
🔑 Private Key: 0xe...2bc
⚙️ Method: GPU (CUDA) x1
⏱️ Elapsed: 3.484s
🔁 Iterations: 3,274,049
🚀 Rate: 939,852 / sec
🎯 Pattern: prefix="cafe" (matched on lowercase, displayed with EIP-55 checksum)
```

#### Generate with prefix and suffix

```bash
vanity --chain eth -p dead -s beef
```

#### ASCII-only output

```bash
vanity --chain eth -p 00 -o plain
```

Output:
```
* Vanity Match
- Chain: ETH (EVM)
- Address: 0x0062d7fF8315f842ceF94081b67bE3114a788B95
- Private Key: 0xe...4c1
- Method: GPU (CUDA) x1
- Elapsed: 3.443s
- Iterations: 2,716,412
- Rate: 789,065 / sec
- Pattern: prefix="00" (matched on lowercase, displayed with EIP-55 checksum)
```

#### JSON output

```bash
vanity --chain eth -p 00 -o json
```

#### Save to file

```bash
# Save to current directory with default format (card)
vanity --chain eth -p cafe --save

# Save to specific directory as JSON
vanity --chain eth -p cafe -o json --save ./results

# Save to specific file path
vanity --chain eth -p cafe --save ./my-eth-key.txt

# Save encrypted output on a cloud server
export VANITY_ENCRYPTION_PASSWORD='replace-with-a-strong-passphrase'
vanity --chain eth -p cafe -o json --save ./results --encrypt
```

#### Decrypt an encrypted result

```bash
# Print decrypted content to stdout
export VANITY_ENCRYPTION_PASSWORD='replace-with-a-strong-passphrase'
vanity --decrypt ./results/vanity-eth-cafe1234.json.enc

# Restore to a file
vanity --decrypt ./results/vanity-eth-cafe1234.json.enc --decrypt-out ./restored.json
```

### Solana (SVM)

#### Generate with case-insensitive prefix

```bash
vanity --chain sol -p Test -i
```

Output:
```
✨ Vanity Match
🔗 Chain: SOL (SVM)
🏷️ Address: testvUrgUfrojUjEFohbtpL8ce6Y3gP3K6rxntbkvJx
🔑 Private Key: f90...5bf
⚙️ Method: GPU (CUDA) x1
⏱️ Elapsed: 25.053s
🔁 Iterations: 432,083,169
🚀 Rate: 17,246,647 / sec
🎯 Pattern: prefix="test", ignore_case=true
```

#### Multiple GPUs

```bash
vanity --chain sol -p SOL --gpus 2
```

#### CPU-only mode

```bash
vanity --chain sol -p test --cpus 8 --gpus 0
```

**Note on GPU vs CPU mode:**
- When `--gpus` is greater than 0, **only GPU** will be used (CPU threads are ignored)
- To use CPU-only mode, you **must** specify `--gpus 0`
- There is currently no hybrid mode that uses both GPU and CPU simultaneously

Examples:
```bash
# GPU only (ignores --cpus if specified)
vanity --chain eth -p cafe --gpus 1

# CPU only (must set --gpus 0)
vanity --chain eth -p cafe --cpus 16 --gpus 0

# This will use ONLY GPU (--cpus 24 is ignored)
vanity --chain eth -p cafe --cpus 24 --gpus 1
```

### Advanced Usage

#### Notes on Count and Timeout

- When both `--count` and `--max-runtime` are set, the tool stops when either condition is met. You may get fewer than `--count` results if the timeout expires first.
- CPU threads (`--cpus`) are ignored when GPUs are used (`--gpus > 0`).
- The `--gpus` flag is present only in GPU builds (compiled with `--features gpu`).

#### CI-friendly with timeout

```bash
vanity --chain eth -p cafe --max-runtime 30 -o json
```

#### Debug mode (logs to stderr)

```bash
vanity --chain eth -p cafe --debug 2>debug.log
```

## Performance

The CPU EVM path reuses its curve context and walks public keys by adding the
generator. The GPU EVM path uses batched inversion for 32 point additions.
Solana retains seed-based Ed25519 derivation for standard wallet compatibility;
suffix filters reject candidates before full Base58 encoding.

See [PERFORMANCE.md](PERFORMANCE.md) for measured before/after results, exact
workloads, correctness checks, and reproduction commands. Measurements are
hardware-specific; throughput is not a guarantee of time to the next match.

For a build targeted to this machine's NVIDIA architecture (for example RTX 4080):

```powershell
$env:VANITY_CUDA_ARCH = "89"
cargo build --release --features gpu
```

Omit the environment variable for the existing multi-architecture build. Advanced
users can compare `VANITY_EVM_BATCH_SIZE=8`, `16`, and `32` at build time (default
32). Each GPU launch gets a fresh host CSPRNG key, expanded with ChaCha20. Each
reported GPU match is independently verified on the CPU before printing/saving.

## Chain Synonyms

The tool accepts multiple aliases for chains:

- **Ethereum**: `eth`, `ethereum`, `evm`
- **Solana**: `sol`, `solana`, `svm`

## Pattern Validation

Vanity validates your prefix and suffix patterns before starting the search, providing helpful error messages if invalid characters are detected.

### Ethereum Pattern Rules

Ethereum addresses use **hexadecimal** characters only:
- ✅ **Allowed**: `0-9`, `a-f`, `A-F`
- ❌ **Not allowed**: Any other characters (e.g., `g-z`, `G-Z`, special characters)
- 💡 The `0x` prefix is optional and will be automatically stripped

**Example error message:**
```bash
$ vanity --chain eth -p "hello"
Error: Invalid prefix 'hello' for Ethereum:

❌ Invalid character(s): 'h', 'l', 'o'
✅ Allowed characters: 0-9, a-f, A-F (hexadecimal)
💡 Tip: Remove these characters from your prefix: 'h', 'l', 'o'
```

### Solana Pattern Rules

Solana addresses use **Base58** encoding, which excludes characters that look similar to avoid confusion:
- ✅ **Allowed**: `1-9`, `A-Z`, `a-z` (excluding `0`, `O`, `I`, `l`)
- ❌ **Not allowed**: `0` (zero), `O` (letter O), `I` (letter I), `l` (lowercase L)

**Why these characters are excluded:**
- `0` (zero) looks similar to `O` (letter O)
- `I` (letter I) looks similar to `l` (lowercase L)

**Example error message with suggestions:**
```bash
$ vanity --chain sol -p "COOL"
Error: Invalid prefix 'COOL' for Solana:

❌ Invalid character(s): 'O'
✅ Allowed characters: Base58 (1-9, A-Z, a-z, excluding 0, O, I, l)
💡 Suggestions:
   'O' (letter O) → try 'P' or remove it
💡 Tip: Remove these characters from your prefix: 'O'
```

**Common mistakes and suggestions:**
- `0` (zero) → Use `1` instead or remove it
- `O` (letter O) → Use `P` or `N` instead or remove it
- `I` (letter I) → Use `J` or `H` instead or remove it
- `l` (lowercase L) → Use `k` or `m` instead or remove it

**Valid Solana examples:**
- ✅ `test` - all lowercase, no excluded characters
- ✅ `ABC123` - uppercase letters and numbers (no 0, O, I, l)
- ✅ `MyWallet` - mixed case (note: case-sensitive by default)

### Important: Ethereum EIP-55 Checksum

Ethereum addresses are displayed with EIP-55 checksumming, which mixes uppercase and lowercase letters for error detection. **Pattern matching is always done on lowercase hex**, but the final address display will have mixed case.

**Key points:**
- Patterns are case-insensitive for Ethereum (e.g., `cafe`, `CAFE`, `CaFe` all match the same addresses)
- Matching happens on lowercase: `cafe` matches address `0xcafe...`
- Display uses EIP-55 checksum: `0xCaFE...` (capitals determined by checksum algorithm)
- The `--ignore-case` flag has no effect for Ethereum (always case-insensitive)

Example:
- **Pattern**: `cafe` or `CAFE` or `CaFe` (all equivalent)
- **Matched** (lowercase): `0xcafe...`
- **Displayed** (EIP-55): `0xCaFE...` or `0xcAfE...` (checksum varies per address)

This is correct behavior! When you import the private key into a wallet, it will show the same checksummed address.

## Output Formats

The `-o, --output` flag controls both screen output and file format when saving with `-s`.

### Card (default, `-o card`)

Unicode emojis and formatted output for human-readable terminal display.

### Plain (`-o plain`)

ASCII-only output for maximum compatibility (no Unicode characters).

### JSON (`-o json`)

Machine-readable JSON output with all metadata:

```json
{
  "chain": "eth",
  "address": "0xCAFE935A8e7F3Ad03099c811c2299E5643DdEFf8",
  "secret_type": "private_key",
  "secret": "0xe...2bc",
  "pattern": {
    "prefix": "cafe",
    "suffix": "",
    "ignore_case": false
  },
  "metrics": {
    "backend": "GPU (CUDA) x1",
    "gpus": 1,
    "cpus": 0,
    "elapsed_sec": 3.484,
    "iterations": 3274049,
    "rate_per_sec": 939852.0,
    "encodes": 3274049,
    "encode_rate_per_sec": 939852.0
  },
  "created_at": "2025-10-08T10:00:00Z",
  "version": "0.0.1"
}
```

## Security

### File Permissions

When using `-s, --save`, files are created with secure permissions:
- **Linux/macOS**: `0600` (owner read/write only)
- **Windows**: Warning issued to manually verify permissions

### Private Key Handling

- Private keys are displayed to stdout by default
- Files are only written when explicitly requested via `-s, --save`
- Encrypted files are stored as JSON envelopes ending in `.enc`
- `--encrypt` derives a 256-bit key from your passphrase with `PBKDF2-HMAC-SHA256` and a random salt
- Decryption uses the same passphrase from `--encrypt-passphrase-env`
- **Never** commit private keys to version control
- Use a secure method to transfer keys to your wallet

## Contributions

Contributions are welcome! Please open an issue or pull request.

## License

This project is licensed under the **GNU Lesser General Public License v3.0 (LGPLv3)**.

You can use this software in your projects (including commercial and proprietary software) as long as you:
- Link to this library dynamically or statically
- Provide attribution
- Make any modifications to this library available under LGPLv3

See the [LICENSE](LICENSE) file for full details, or visit: https://www.gnu.org/licenses/lgpl-3.0.html

## Acknowledgements

This project builds upon excellent work from the open-source community:

- **SHA-256/SHA-512 CUDA implementations** from [cuda-hashing-algos](https://github.com/mochimodev/cuda-hashing-algos) (Public Domain) - Used for cryptographic hashing operations
- **Keccak256 and secp256k1 implementations** based on [MrSpike63's vanity-eth-address](https://github.com/MrSpike63/vanity-eth-address) - Core Ethereum address generation algorithms
- **Base58 encoding** from Firedancer (Apache-2.0) - Solana address encoding
- **Ed25519 implementation** - Solana keypair generation and signing

Thank you to all the developers who made their code available for others to build upon!

## Version

Current version: **0.0.1**
#### File Saving Details

- Saved filenames include the chain and the first 8 characters of the address (for Ethereum, `0x` is stripped).
- Encrypted saves append `.enc` to the normal filename, for example `vanity-eth-cafe1234.json.enc`.
- On Windows, file permissions cannot be restricted programmatically; verify permissions manually if saving secrets.
#### ETH GPU Tuning

- Start with autotune: `--gpu-target-ms 1200` (Windows) or `1500` (Linux)
- Manually push launch size if desired: `--gpu-blocks`, `--gpu-threads`, `--gpu-iters`
- Example:
  ```bash
  vanity --chain eth -s 1010 --gpus 1 \
         --gpu-blocks 512 --gpu-threads 512 --gpu-target-ms 1200
  ```

#### Solana Notes

- Solana GPU starts with a small batch and accepts all four GPU tuning flags.
- Suffix patterns use a fast check to avoid most Base58 encodes.
- Output remains wallet-compatible: seed (32) + expanded private (64) + pubkey (32).

#### Metrics

- Iterations/sec: total candidates evaluated per second
- Encodes/sec (Solana): actual full Base58 encodes per second. Rejected suffix candidates avoid encoding; candidates/sec measures search throughput.
