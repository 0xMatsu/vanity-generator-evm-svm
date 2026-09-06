use crate::ChainType;
use clap::ValueEnum;
use std::{
    io::{self, IsTerminal, Write},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc, Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ProgressMode {
    Auto,
    Always,
    Never,
}

struct State {
    attempts: AtomicU64,
    device_rates: [AtomicU64; 16],
    gpu_mode: AtomicBool,
    found: AtomicU64,
    round_start: AtomicU64,
    display: Mutex<()>,
}

pub struct Progress {
    state: Arc<State>,
    stop: Option<mpsc::Sender<()>>,
    worker: Option<JoinHandle<()>>,
    terminal: bool,
}

impl Progress {
    pub fn new(
        mode: ProgressMode,
        json: bool,
        chain: ChainType,
        prefix: &str,
        suffix: &str,
        ignore_case: bool,
        target: u32,
    ) -> Self {
        let terminal = io::stderr().is_terminal();
        let enabled = match mode {
            ProgressMode::Auto => terminal && !json,
            ProgressMode::Always => true,
            ProgressMode::Never => false,
        };
        let state = Arc::new(State {
            attempts: AtomicU64::new(0),
            device_rates: std::array::from_fn(|_| AtomicU64::new(0)),
            gpu_mode: AtomicBool::new(false),
            found: AtomicU64::new(0),
            round_start: AtomicU64::new(0),
            display: Mutex::new(()),
        });
        let mut result = Self {
            state,
            stop: None,
            worker: None,
            terminal,
        };
        if enabled {
            let state = result.state.clone();
            let expected = expected_attempts(chain, prefix, suffix, ignore_case);
            let (tx, rx) = mpsc::channel();
            result.stop = Some(tx);
            let start = Instant::now();
            result.worker = Some(thread::spawn(move || {
                let mut sample = start;
                let mut previous = 0;
                let mut rate = 0.0;
                let interval = Duration::from_millis(if terminal { 250 } else { 1000 });
                loop {
                    let finished =
                        rx.recv_timeout(interval) != Err(mpsc::RecvTimeoutError::Timeout);
                    let now = Instant::now();
                    let attempts = state.attempts.load(Ordering::Relaxed);
                    let gpu_rate: f64 = state
                        .device_rates
                        .iter()
                        .map(|v| f64::from_bits(v.load(Ordering::Relaxed)))
                        .sum();
                    if state.gpu_mode.load(Ordering::Relaxed) {
                        rate = if finished {
                            attempts as f64 / start.elapsed().as_secs_f64().max(1e-9)
                        } else {
                            gpu_rate
                        };
                    } else if attempts > previous {
                        let observed = (attempts - previous) as f64
                            / now.duration_since(sample).as_secs_f64().max(1e-9);
                        rate = if rate == 0.0 {
                            observed
                        } else {
                            0.35 * observed + 0.65 * rate
                        };
                        previous = attempts;
                        sample = now;
                    }
                    let found = state.found.load(Ordering::Relaxed);
                    let round = attempts.saturating_sub(state.round_start.load(Ordering::Relaxed));
                    let line = render(
                        round,
                        attempts,
                        found,
                        target,
                        rate,
                        expected,
                        start.elapsed(),
                        finished,
                    );
                    let _guard = state.display.lock().unwrap();
                    let mut stderr = io::stderr().lock();
                    if terminal {
                        let _ = write!(
                            stderr,
                            "\r\x1b[2K{line}{}",
                            if finished { "\n" } else { "" }
                        );
                    } else {
                        let _ = writeln!(stderr, "{line}");
                    }
                    let _ = stderr.flush();
                    if finished {
                        break;
                    }
                }
            }));
        }
        result
    }

    pub fn attempts(&self) -> &AtomicU64 {
        &self.state.attempts
    }

    /// GPU counters arrive in bursts, so use measured kernel-batch duration,
    /// not the timing of the display thread's polling interval.
    #[cfg(any(test, feature = "gpu"))]
    pub fn record_batch(&self, device: u32, count: u64, elapsed: Duration) {
        let rate = count as f64 / elapsed.as_secs_f64().max(1e-9);
        self.state.gpu_mode.store(true, Ordering::Relaxed);
        self.state.device_rates[device as usize].store(rate.to_bits(), Ordering::Relaxed);
        self.state.attempts.fetch_add(count, Ordering::Relaxed);
    }

    #[cfg(any(test, feature = "gpu"))]
    pub fn gpu_stopped(&self, device: u32) {
        self.state.device_rates[device as usize].store(0, Ordering::Relaxed);
    }

    pub fn found(&self) {
        self.state.round_start.store(
            self.state.attempts.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        self.state.found.fetch_add(1, Ordering::Relaxed);
    }

    pub fn output(&self, output: impl FnOnce()) {
        let _guard = self.state.display.lock().unwrap();
        if self.worker.is_some() && self.terminal {
            let _ = write!(io::stderr().lock(), "\r\x1b[2K");
        }
        output();
        let _ = io::stdout().flush();
    }
}

impl Drop for Progress {
    fn drop(&mut self) {
        if let Some(tx) = self.stop.take() {
            let _ = tx.send(());
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

// EVM: exact constraint probability, including overlapping prefix/suffix.
// SOL: explicitly approximate uniform Base58 model; leading digits are biased.
fn expected_attempts(chain: ChainType, prefix: &str, suffix: &str, ignore: bool) -> f64 {
    let length = if chain == ChainType::Ethereum { 40 } else { 44 };
    let mut masks = vec![u64::MAX; length];
    let alphabet = if chain == ChainType::Ethereum {
        b"0123456789abcdef".as_slice()
    } else {
        b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz".as_slice()
    };
    for (offset, text) in [(0, prefix), (length - suffix.len(), suffix)] {
        for (i, ch) in text.bytes().enumerate() {
            let mask = alphabet
                .iter()
                .enumerate()
                .filter(|(_, c)| {
                    if ignore || chain == ChainType::Ethereum {
                        c.eq_ignore_ascii_case(&ch)
                    } else {
                        **c == ch
                    }
                })
                .fold(0, |mask, (j, _)| mask | (1u64 << j));
            masks[offset + i] &= mask;
        }
    }
    masks
        .into_iter()
        .filter(|&mask| mask != u64::MAX)
        .fold(1.0, |n, mask| {
            if mask == 0 {
                f64::INFINITY
            } else {
                n * alphabet.len() as f64 / mask.count_ones() as f64
            }
        })
}

fn compact(n: f64) -> String {
    for (unit, scale) in [("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)] {
        if n >= scale {
            return format!("{:.1}{unit}", n / scale);
        }
    }
    format!("{n:.0}")
}

fn wait_time(seconds: f64) -> String {
    if !seconds.is_finite() {
        return "--".into();
    }
    if seconds < 60.0 {
        format!("{seconds:.1}s")
    } else if seconds < 3600.0 {
        format!("{:.1}m", seconds / 60.0)
    } else if seconds < 86400.0 {
        format!("{:.1}h", seconds / 3600.0)
    } else {
        format!("{}d", compact(seconds / 86400.0))
    }
}

fn render(
    round: u64,
    attempts: u64,
    found: u64,
    target: u32,
    rate: f64,
    expected: f64,
    elapsed: Duration,
    finished: bool,
) -> String {
    let complete = finished && found >= target as u64;
    let probability = if complete {
        1.0
    } else if expected <= 1.0 {
        if round > 0 {
            1.0
        } else {
            0.0
        }
    } else {
        -((round as f64) * (-1.0 / expected).ln_1p()).exp_m1()
    };
    let filled = (probability * 12.0).floor().clamp(0.0, 12.0) as usize;
    let bar = format!("{}{}", "#".repeat(filled), "-".repeat(12 - filled));
    let state = if finished {
        if found >= target as u64 {
            "done"
        } else {
            "stopped"
        }
    } else {
        "search"
    };
    format!(
        "[{bar}] chance~{:.1}% | {}/s | {} tried | {found}/{target} | mean/hit~{} | {} {state}",
        if complete {
            100.0
        } else {
            (probability * 100.0).min(99.9)
        },
        compact(rate),
        compact(attempts as f64),
        wait_time(if rate > 0.0 {
            expected / rate
        } else {
            f64::INFINITY
        }),
        wait_time(elapsed.as_secs_f64())
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn aggregates_device_rates_and_removes_stopped_devices() {
        let progress = Progress::new(
            ProgressMode::Never,
            false,
            ChainType::Ethereum,
            "a",
            "",
            false,
            1,
        );
        progress.record_batch(0, 100, Duration::from_secs(1));
        progress.record_batch(1, 300, Duration::from_secs(2));
        progress.record_batch(0, 200, Duration::from_secs(1));
        assert_eq!(progress.attempts().load(Ordering::Relaxed), 600);
        let sum = || {
            progress
                .state
                .device_rates
                .iter()
                .map(|r| f64::from_bits(r.load(Ordering::Relaxed)))
                .sum::<f64>()
        };
        assert_eq!(sum(), 350.0);
        progress.gpu_stopped(0);
        assert_eq!(sum(), 150.0);
        progress.gpu_stopped(1);
        assert_eq!(sum(), 0.0);
    }

    #[test]
    fn probability_accounts_for_overlap_and_case() {
        assert_eq!(
            expected_attempts(ChainType::Ethereum, "a", "b", false),
            256.0
        );
        assert_eq!(
            expected_attempts(ChainType::Ethereum, &"a".repeat(40), "a", false),
            16f64.powi(40)
        );
        assert!(expected_attempts(ChainType::Ethereum, &"a".repeat(40), "b", false).is_infinite());
        assert_eq!(expected_attempts(ChainType::Solana, "", "a", true), 29.0);
        assert_eq!(expected_attempts(ChainType::Solana, "", "1", true), 58.0);
    }
    #[test]
    fn progress_is_probability_not_a_countdown() {
        let line = render(256, 256, 0, 1, 128.0, 256.0, Duration::from_secs(2), false);
        assert!(line.contains("chance~63.3%"));
        assert!(line.contains("mean/hit~2.0s"));
        assert!(render(0, 0, 0, 1, 0.0, f64::INFINITY, Duration::ZERO, true).contains("stopped"));
    }
}
