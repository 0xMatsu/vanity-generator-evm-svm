# Performance and verification

Measured on 2026-09-06: Intel Core i5-14600KF, NVIDIA RTX 4080 SUPER,
Windows/WDDM, driver 591.86, CUDA 12.4.99, Rust 1.94.1, release builds.
The baseline is commit `19bf2bb`. These are measurements on one machine,
not a claim of a universal or mathematical performance limit.

## Results

| Workload | Baseline candidates/s | Optimized candidates/s | Change |
| --- | ---: | ---: | ---: |
| CPU EVM, one thread, fixed work | 52,042 | 456,693 | 8.78x |
| CPU Solana, one thread, suffix | 78,178 | 81,261 | +3.9% |
| GPU EVM, suffix | 150,828,372 | 572,556,675 | 3.80x |
| GPU Solana, suffix | 13,421,622 | 14,081,927 | +4.9% |
| GPU Solana, prefix | 13,388,317 | 14,215,244 | +6.2% |

GPU comparisons use the same 160 blocks and 256 threads per block. EVM uses
2,048 iterations/thread; Solana uses 512. The harness discards one warm-up
and reports the median of three complete batches, including key setup,
transfers, kernel execution and result retrieval, but excluding CUDA startup.
GPU measurements run sequentially, with no concurrent benchmark kernels.
The EVM optimized median ranged from 572M to 597M in separate repetitions;
the table uses the paired baseline/optimized run. Clock, display activity,
temperature and batch size affect results.

The fixed-work CPU benchmark runs 100,000 candidates three times using a
deterministic CSPRNG and compares the original hot-loop operations with the new
ones. It excludes Rayon scheduling, output, file encryption and match handling.
Solana gains are modest because SHA-512 and Ed25519 scalar multiplication still
dominate; encoding optimizations cannot eliminate those operations.

## Changes that matter

- CPU EVM: reuse secp256k1 context; derive one random starting point per worker
  and advance by adding G. Match raw hash nibbles; format only hits.
- CPU workers: keep RNG state locally, publish candidate counts every 256
  iterations, check deadlines at batch boundaries, and select exactly one
  winner before serial output. Each search round starts independent walks;
  multiple exported keys never come from the same walk.
- GPU EVM: Montgomery batch inversion shares one modular inverse among 32
  additions. Comparing 8/16/32-point batches gave approximately 476M/562M/597M
  candidates/s in exploratory runs at 160 x 256 x 2048. The default is 32;
  build-time `VANITY_EVM_BATCH_SIZE=8|16|32` permits hardware-specific comparison.
- Solana: constant `58^4` suffix rejection, stack encoding on CPU, and GPU
  Base58 conversion in groups of five digits. GPU encode counters accumulate
  locally and reset on every launch. Prefix/suffix length checks prevent overreads.
- Correctness: fix long-suffix modulus overflow, folded-case false negatives,
  all-zero Base58 encoding, duplicate CPU outputs and cumulative GPU metrics.
- GPU key generation: replace XorShift with ChaCha20 keyed by fresh host CSPRNG
  output. EVM starting keys reject zero and the top 1/256 of scalar space,
  retaining almost 256 bits of entropy while leaving room for any u64-length
  walk without reaching scalar zero/the point at infinity.
- Verify every GPU hit against Rust secp256k1/ed25519-dalek and address hashing
  before printing or saving. This verification is outside the candidate hot loop.
- GPU batches start small and tune upward. The old Solana default processed
  1,310,720,000 candidates in its first launch (~90 seconds here), even with
  `--max-runtime 5`. The new default starts with 1,048,576 candidates. A running
  CUDA launch remains non-preemptible by the application's deadline check.

## Reproduce

CPU checks and fixed-work comparison:

```sh
cargo test --release
cargo test --release cpu_fixed_work -- --ignored --nocapture
```

GPU build and hardware checks (PowerShell):

```powershell
$env:VANITY_CUDA_ARCH = "89"
cargo test --release --features gpu
cargo test --release --features gpu gpu_results_match -- --ignored --nocapture
cargo test --release --features gpu gpu_fixed_work -- --ignored --nocapture
nvcc -arch=sm_89 -o target/test-primitives.exe kernels/test_primitives.cu
nvcc -arch=sm_89 -o target/test-batch.exe kernels/test_batch.cu
target/test-primitives.exe
target/test-batch.exe
```

Run `nvcc` in a Visual Studio developer shell on Windows, or supply `-ccbin`
pointing to the host C++ compiler. Use the architecture appropriate to the GPU.
Omit `VANITY_CUDA_ARCH` to build all of the repository's original architectures.
Native primitive checks cover 1,024 Base58 inputs (including zero/leading-zero
boundaries), all suffix lengths, case folding, the
[RFC 8439 ChaCha20 block vector](https://www.rfc-editor.org/rfc/rfc8439#section-2.3.2),
and 1,024 batched curve points against independent scalar multiplication.
Rust tests also cover scalar-order wraparound, GPU/CPU address reconstruction,
multiple results, prefix+suffix combinations and wallet-compatible key export.

For an identical harness against either revision's compiled CUDA archive, use
`benchmarks/gpu.rs`. After building that revision, locate its native archive:

```powershell
$native = (Get-ChildItem target -Recurse -Filter vanity.lib |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1).DirectoryName
rustc --edition=2021 -O benchmarks/gpu.rs -o target/bench-gpu.exe `
    -L "native=$native" -L "native=$env:CUDA_PATH/lib/x64" `
    -l static=vanity -l cudart_static -l cuda -l ole32 -l oleaut32
target/bench-gpu.exe eth suffix 160 256 2048
target/bench-gpu.exe sol suffix 160 256 512
target/bench-gpu.exe sol prefix 160 256 512
```

Keep baseline and optimized archives/executables separate. The harness reports
median/min/max rates, asserts the exact candidate count, and prints no keys.
The full-length rare/impossible patterns intentionally prevent early termination.

## Remaining limits

The multi-GPU scheduler now runs one independent host thread per CUDA device.
Bounded result delivery and per-worker acknowledgements coordinate global count,
CPU verification, encryption and shutdown without serializing CUDA batches across
devices. Eight simulated workers test overlapping batches, exact global counts,
deadline handling, errors, panics and resource cleanup. Only one physical NVIDIA
GPU was available for regression testing; actual eight-card throughput and scaling
remain unmeasured. `--gpus N` does not promise N-fold throughput.

Run the opt-in scheduler integration test on a CUDA server (default two GPUs):
```bash
VANITY_TEST_GPUS=8 cargo test --release --features gpu --test cli gpu_scheduler_hardware -- --ignored
```
Other GPU architectures and operating systems have not been benchmarked here.
Solana retains standard seed-based wallet keys: simply incrementing an Ed25519
public point would not preserve the seed-to-key relationship. Further substantial
Solana gains require optimizing its SHA-512/curve arithmetic or a different
implementation with equivalent wallet compatibility, not just tuning matching.
