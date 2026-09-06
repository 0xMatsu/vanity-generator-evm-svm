use crate::{
    gpu_scheduler::{self, Batch, Event, Worker},
    *,
};

struct CudaWorker<'a> {
    id: u32,
    args: &'a Args,
    prefix: &'a str,
    suffix: &'a str,
    blocks: i32,
    threads: i32,
    iters: u64,
    target_ms: u64,
    round: u64,
}

impl<'a> CudaWorker<'a> {
    fn new(id: u32, args: &'a Args, prefix: &'a str, suffix: &'a str) -> Result<Self, String> {
        // get_gpu_config calls cudaSetDevice on this worker thread.
        let mut config = GpuConfig::default();
        if unsafe { get_gpu_config(id as i32, &mut config) } != 0 {
            return Err("could not initialize device configuration".into());
        }
        let eth = args.generation_chain() == ChainType::Ethereum;
        let default_threads = if eth && !cfg!(target_os = "windows") {
            512
        } else if eth {
            256
        } else {
            128
        };
        let threads = args.gpu_threads.unwrap_or(default_threads) as i32;
        if threads > config.max_threads_per_block {
            return Err(format!(
                "requested {threads} threads, device supports at most {}",
                config.max_threads_per_block
            ));
        }
        let factor = if eth && !cfg!(target_os = "windows") {
            8
        } else if eth {
            4
        } else {
            2
        };
        Ok(Self {
            id,
            args,
            prefix,
            suffix,
            blocks: args
                .gpu_blocks
                .map(|n| n as i32)
                .unwrap_or((config.sm_count * factor).max(1)),
            threads,
            iters: args.gpu_iters.unwrap_or(if eth { 256 } else { 64 }),
            target_ms: args.gpu_target_ms.unwrap_or(if eth { 900 } else { 500 }),
            round: 0,
        })
    }
}

impl Worker for CudaWorker<'_> {
    type Hit = [u8; 192];
    fn batch(&mut self) -> Result<Batch<Self::Hit>, String> {
        let seed = new_gpu_seed(self.id, self.round);
        self.round += 1;
        let mut out = [0u8; 192];
        let eth = self.args.generation_chain() == ChainType::Ethereum;
        let start = Instant::now();
        let status = unsafe {
            let launch = if eth {
                eth_vanity_round_ultra
            } else {
                sol_vanity_round_optimized
            };
            launch(
                self.id as i32,
                seed.as_ptr(),
                self.prefix.as_ptr(),
                self.suffix.as_ptr(),
                self.prefix.len() as u64,
                self.suffix.len() as u64,
                out.as_mut_ptr(),
                self.args.ignore_case,
                self.blocks,
                self.threads,
                self.iters,
            )
        };
        let elapsed = start.elapsed();
        if status != 0 {
            return Err(format!(
                "CUDA batch failed (status={status}, blocks={}, threads={}, iters={})",
                self.blocks, self.threads, self.iters
            ));
        }
        let (count_offset, done_offset) = if eth { (137, 145) } else { (172, 188) };
        let attempts = u64::from_le_bytes(out[count_offset..count_offset + 8].try_into().unwrap());
        let encodes = if eth {
            attempts
        } else {
            u64::from_le_bytes(out[180..188].try_into().unwrap())
        };
        let done = i32::from_le_bytes(out[done_offset..done_offset + 4].try_into().unwrap());
        if done != 0 && done != 1 {
            return Err("invalid CUDA completion flag".into());
        }
        if self.args.gpu_iters.is_none() && done == 0 {
            let ms = elapsed.as_millis().max(1);
            self.iters = (self.iters as u128 * self.target_ms as u128 / ms).clamp(
                (self.iters / 2).max(1) as u128,
                self.iters.saturating_mul(2).min(200_000) as u128,
            ) as u64;
        }
        Ok(Batch {
            attempts,
            encodes,
            elapsed,
            hit: (done == 1).then_some(out),
        })
    }
}

impl Drop for CudaWorker<'_> {
    fn drop(&mut self) {
        unsafe {
            match self.args.generation_chain() {
                ChainType::Ethereum => eth_vanity_cleanup_ultra(self.id as i32),
                ChainType::Solana => sol_vanity_cleanup(self.id as i32),
            }
        }
    }
}

// Only the coordinator validates, encrypts, saves and prints selected hits.
fn verify_hit(
    chain: ChainType,
    prefix: &str,
    suffix: &str,
    ignore: bool,
    out: &[u8; 192],
) -> Result<(String, String, String), String> {
    let invalid = || "GPU result failed independent CPU verification".to_owned();
    match chain {
        ChainType::Ethereum => {
            let key = SecretKey::from_slice(&out[..32]).map_err(|_| invalid())?;
            let public =
                PublicKey::from_secret_key(&Secp256k1::new(), &key).serialize_uncompressed();
            let address = hex::encode(&Keccak256::digest(&public[1..])[12..]);
            if out[32..97] != public
                || out[97..137] != *address.as_bytes()
                || !address.starts_with(prefix)
                || !address.ends_with(suffix)
            {
                return Err(invalid());
            }
            Ok((
                eip55_checksum(&address),
                format!("0x{}", hex::encode(public)),
                format!("0x{}", hex::encode(key.secret_bytes())),
            ))
        }
        ChainType::Solana => {
            let key = SigningKey::from_bytes(out[..32].try_into().unwrap());
            let public = key.verifying_key().to_bytes();
            let address = bs58::encode(public).into_string();
            let len = out[128..172].iter().position(|&b| b == 0).unwrap_or(44);
            let check = if ignore {
                address.to_ascii_lowercase()
            } else {
                address.clone()
            };
            if out[96..128] != public
                || out[128..128 + len] != *address.as_bytes()
                || !check.starts_with(prefix)
                || !check.ends_with(suffix)
            {
                return Err(invalid());
            }
            Ok((
                address.clone(),
                address,
                format_seed_as_array(&key.to_keypair_bytes()),
            ))
        }
    }
}

pub fn generate(
    args: &Args,
    prefix: &str,
    suffix: &str,
    devices: u32,
    cpus: u32,
    start: Instant,
    deadline: Option<Instant>,
    passphrase: Option<&str>,
    progress: &progress::Progress,
) -> Result<(), String> {
    let mut encodes = 0u64;
    let mut accepted = 0u32;
    gpu_scheduler::run(
        devices,
        args.count,
        deadline,
        |id| CudaWorker::new(id, args, prefix, suffix),
        |event| {
            match event {
                Event::Stopped(id) => progress.gpu_stopped(id),
                Event::Batch(id, batch) => {
                    progress.record_batch(id, batch.attempts, batch.elapsed);
                    encodes = encodes.saturating_add(batch.encodes);
                    if args.debug {
                        progress.output(|| {
                            eprintln!(
                                "[debug] GPU {id}: {} candidates in {:.3}s",
                                batch.attempts,
                                batch.elapsed.as_secs_f64()
                            )
                        });
                    }
                    if let Some(raw) = batch.hit {
                        let chain = args.generation_chain();
                        let (address, public_key, secret) =
                            verify_hit(chain, prefix, suffix, args.ignore_case, &raw)?;
                        let elapsed = start.elapsed().as_secs_f64();
                        let iterations = progress.attempts().load(Ordering::Relaxed);
                        let result = VanityResult {
                            chain,
                            address,
                            public_key,
                            secret_type: "private_key",
                            secret,
                            pattern_prefix: prefix.into(),
                            pattern_suffix: suffix.into(),
                            ignore_case: args.ignore_case,
                            backend: format!("GPU (CUDA) x{devices}"),
                            gpus: devices,
                            cpus,
                            elapsed_sec: elapsed,
                            iterations,
                            rate_per_sec: iterations as f64 / elapsed,
                            encodes,
                            encode_rate_per_sec: encodes as f64 / elapsed,
                        };
                        if let Some(path) = &args.save {
                            let mut path = path.clone().unwrap_or_else(|| PathBuf::from("."));
                            if args.count > 1 {
                                path = resolve_save_path(&result, &path, args.output, false);
                                let mut name = path.file_stem().unwrap_or_default().to_os_string();
                                name.push(format!("-{:06}", accepted + 1));
                                if let Some(extension) = path.extension() {
                                    name.push(".");
                                    name.push(extension);
                                }
                                path.set_file_name(name);
                            }
                            save_result(&result, &path, args.output, passphrase)?;
                        }
                        accepted += 1;
                        progress.found();
                        progress.output(|| print_result(&result, args.output));
                    }
                }
            }
            Ok(())
        },
    )?;
    Ok(())
}
