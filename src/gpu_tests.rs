//! Explicit opt-in hardware tests: cargo test --release --features gpu -- --ignored --nocapture
use super::*;
static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
#[ignore = "requires a CUDA GPU"]
fn gpu_results_match_independent_cpu_derivation() {
    let _guard = GPU_LOCK.lock().unwrap();
    assert_eq!(unsafe { cuda_init() }, 0);
    let secp = Secp256k1::new();
    for i in 0..8 {
        // Include the Linux default and oversized blocks; the native wrapper
        // must fit the compiled kernel's resources and still return valid keys.
        let threads = [64, 128, 256, 512, 1024, 512, 128, 64][i as usize];
        let seed = [i + 1; 32];
        let mut eth = [0u8; 149];
        assert_eq!(
            unsafe {
                eth_vanity_round_ultra(
                    0,
                    seed.as_ptr(),
                    b"".as_ptr(),
                    b"a".as_ptr(),
                    0,
                    1,
                    eth.as_mut_ptr(),
                    false,
                    4,
                    threads,
                    128,
                )
            },
            0
        );
        assert_eq!(i32::from_le_bytes(eth[145..149].try_into().unwrap()), 1);
        let sk = SecretKey::from_slice(&eth[..32]).unwrap();
        let pk = PublicKey::from_secret_key(&secp, &sk).serialize_uncompressed();
        assert_eq!(&eth[32..97], &pk);
        let address = hex::encode(&Keccak256::digest(&pk[1..])[12..]);
        assert_eq!(&eth[97..137], address.as_bytes());
        assert!(address.ends_with('a'));
        let mut sol = [0u8; 192];
        assert_eq!(
            unsafe {
                sol_vanity_round_optimized(
                    0,
                    seed.as_ptr(),
                    b"".as_ptr(),
                    b"a".as_ptr(),
                    0,
                    1,
                    sol.as_mut_ptr(),
                    true,
                    4,
                    threads,
                    128,
                )
            },
            0
        );
        assert_eq!(i32::from_le_bytes(sol[188..192].try_into().unwrap()), 1);
        let key = SigningKey::from_bytes(sol[..32].try_into().unwrap());
        let pk = key.verifying_key().to_bytes();
        assert_eq!(&sol[96..128], &pk);
        let address = bs58::encode(pk).into_string();
        assert_eq!(&sol[128..128 + address.len()], address.as_bytes());
        assert!(address.to_ascii_lowercase().ends_with('a'));
    }
    unsafe {
        eth_vanity_cleanup_ultra(0);
        sol_vanity_cleanup(0);
    }
}

#[test]
#[ignore = "requires a CUDA GPU; fixed work throughput benchmark"]
fn gpu_fixed_work_benchmark() {
    let _guard = GPU_LOCK.lock().unwrap();
    assert_eq!(unsafe { cuda_init() }, 0);
    for chain in [ChainType::Ethereum, ChainType::Solana] {
        let mut out = [0u8; 192];
        let pattern = if chain == ChainType::Ethereum {
            "ffffffffffffffffffffffffffffffffffffffff"
        } else {
            "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"
        };
        for run in 0..4 {
            let seed = [run + 42; 32];
            let start = Instant::now();
            let status = unsafe {
                if chain == ChainType::Ethereum {
                    eth_vanity_round_ultra(
                        0,
                        seed.as_ptr(),
                        b"".as_ptr(),
                        pattern.as_ptr(),
                        0,
                        pattern.len() as u64,
                        out.as_mut_ptr(),
                        false,
                        160,
                        256,
                        2048,
                    )
                } else {
                    sol_vanity_round_optimized(
                        0,
                        seed.as_ptr(),
                        b"".as_ptr(),
                        pattern.as_ptr(),
                        0,
                        pattern.len() as u64,
                        out.as_mut_ptr(),
                        false,
                        160,
                        256,
                        2048,
                    )
                }
            };
            assert_eq!(status, 0);
            let offset = if chain == ChainType::Ethereum {
                137
            } else {
                172
            };
            let count = u64::from_le_bytes(out[offset..offset + 8].try_into().unwrap());
            assert_eq!(count, 160 * 256 * 2048);
            eprintln!(
                "BENCH {chain} run={run} candidates/sec={:.0}",
                count as f64 / start.elapsed().as_secs_f64()
            );
        }
    }
    unsafe {
        eth_vanity_cleanup_ultra(0);
        sol_vanity_cleanup(0);
    }
}
