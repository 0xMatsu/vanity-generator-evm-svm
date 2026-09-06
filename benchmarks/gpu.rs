//! Standalone harness that can link either the baseline or optimized CUDA archive.
//! See PERFORMANCE.md. No private keys are printed or saved.
use std::time::Instant;

extern "C" {
    fn cuda_init() -> i32;
    fn eth_vanity_round_ultra(
        id: i32,
        seed: *const u8,
        prefix: *const u8,
        suffix: *const u8,
        prefix_len: u64,
        suffix_len: u64,
        out: *mut u8,
        ignore_case: bool,
        blocks: i32,
        threads: i32,
        iters: u64,
    ) -> i32;
    fn sol_vanity_round_optimized(
        id: i32,
        seed: *const u8,
        prefix: *const u8,
        suffix: *const u8,
        prefix_len: u64,
        suffix_len: u64,
        out: *mut u8,
        ignore_case: bool,
        blocks: i32,
        threads: i32,
        iters: u64,
    ) -> i32;
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() == 6,
        "usage: bench-gpu eth|sol prefix|suffix BLOCKS THREADS ITERS"
    );
    let eth = match args[1].as_str() {
        "eth" => true,
        "sol" => false,
        _ => panic!("invalid chain"),
    };
    let prefix_mode = match args[2].as_str() {
        "prefix" => true,
        "suffix" => false,
        _ => panic!("invalid mode"),
    };
    let blocks: i32 = args[3].parse().unwrap();
    let threads: i32 = args[4].parse().unwrap();
    let iters: u64 = args[5].parse().unwrap();
    assert!(blocks > 0 && threads > 0 && threads <= 1024 && iters > 0);
    let pattern = if eth {
        "ffffffffffffffffffffffffffffffffffffffff"
    } else {
        "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"
    };
    let (prefix, suffix) = if prefix_mode {
        (pattern, "")
    } else {
        ("", pattern)
    };
    assert_eq!(unsafe { cuda_init() }, 0);
    let mut rates = Vec::new();
    for run in 0..4 {
        let seed = [42 + run; 32];
        let mut out = [0u8; 192];
        let start = Instant::now();
        let status = unsafe {
            let round = if eth {
                eth_vanity_round_ultra
            } else {
                sol_vanity_round_optimized
            };
            round(
                0,
                seed.as_ptr(),
                prefix.as_ptr(),
                suffix.as_ptr(),
                prefix.len() as u64,
                suffix.len() as u64,
                out.as_mut_ptr(),
                false,
                blocks,
                threads,
                iters,
            )
        };
        let elapsed = start.elapsed().as_secs_f64();
        assert_eq!(status, 0);
        let offset = if eth { 137 } else { 172 };
        let count = u64::from_le_bytes(out[offset..offset + 8].try_into().unwrap());
        assert_eq!(count, blocks as u64 * threads as u64 * iters);
        if run > 0 {
            rates.push(count as f64 / elapsed);
        }
    }
    rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "{} {} blocks={} threads={} iters={} median={:.0} min={:.0} max={:.0} candidates/sec",
        args[1], args[2], blocks, threads, iters, rates[1], rates[0], rates[2]
    );
}
