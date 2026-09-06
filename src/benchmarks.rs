//! Fixed-work, single-thread reference benchmarks; no keys are printed or saved.
use super::*;
use rand::SeedableRng;
use std::hint::black_box;

#[test]
#[ignore = "performance benchmark: cargo test --release cpu_fixed_work -- --ignored --nocapture"]
fn cpu_fixed_work() {
    const COUNT: u64 = 100_000;
    let secp = Secp256k1::new();
    for chain in [ChainType::Ethereum, ChainType::Solana] {
        for optimized in [false, true] {
            let mut rates = Vec::new();
            for _ in 0..3 {
                let mut rng = rand::rngs::StdRng::seed_from_u64(123);
                let mut walk = eth_walk::EthWalk::new(&secp, SecretKey::new(&mut rng));
                let eth_match = matcher::EthMatcher::new("", "deadbeef");
                let sol_match = matcher::SolMatcher::new("", "deadbeef", false);
                let mut encodes = 0;
                let start = Instant::now();
                for _ in 0..COUNT {
                    let mut bytes = [0; 32];
                    match (chain, optimized) {
                        (ChainType::Ethereum, false) => {
                            rng.fill_bytes(&mut bytes);
                            let context = Secp256k1::new();
                            let key = SecretKey::from_slice(&bytes).unwrap();
                            let public = secp256k1::PublicKey::from_secret_key(&context, &key)
                                .serialize_uncompressed();
                            let hash = Keccak256::digest(&public[1..]);
                            black_box(hex::encode(&hash[12..]).ends_with("deadbeef"));
                        }
                        (ChainType::Ethereum, true) => {
                            let (_, public) = walk.current();
                            black_box(eth_match.matches(&Keccak256::digest(&public[1..])[12..]));
                            walk.advance();
                        }
                        (ChainType::Solana, _) => {
                            rng.fill_bytes(&mut bytes);
                            let public = SigningKey::from_bytes(&bytes).verifying_key().to_bytes();
                            if optimized {
                                black_box(sol_match.matches_counted(
                                    &public,
                                    &mut [0; 44],
                                    &mut encodes,
                                ));
                            } else {
                                let address = bs58::encode(public).into_string();
                                let check_address = address.clone();
                                black_box(check_address.ends_with("deadbeef"));
                            }
                        }
                    }
                }
                rates.push(COUNT as f64 / start.elapsed().as_secs_f64());
            }
            rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
            eprintln!(
                "BENCH CPU {chain} optimized={optimized} median={:.0} candidates/sec",
                rates[1]
            );
        }
    }
}
