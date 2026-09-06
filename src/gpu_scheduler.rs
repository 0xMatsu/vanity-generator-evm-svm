//! One long-lived host thread per device. The coordinator alone accepts hits.
//! Acknowledgements bound each device to one unconsumed batch, even during saves.
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

pub struct Batch<T> {
    pub attempts: u64,
    pub encodes: u64,
    pub elapsed: Duration,
    pub hit: Option<T>,
}

pub trait Worker: Send {
    type Hit: Send;
    fn batch(&mut self) -> Result<Batch<Self::Hit>, String>;
}

pub enum Event<T> {
    Batch(u32, Batch<T>),
    Stopped(u32),
}

enum Message<T> {
    Batch(u32, Batch<T>),
    Failed(u32, String),
    Stopped(u32),
}

pub fn run<W: Worker, F, H>(
    devices: u32,
    target: u32,
    deadline: Option<Instant>,
    make_worker: F,
    mut handle: H,
) -> Result<u32, String>
where
    F: Fn(u32) -> Result<W, String> + Sync,
    H: FnMut(Event<W::Hit>) -> Result<(), String>,
{
    if target == 0 || deadline.is_some_and(|d| Instant::now() >= d) {
        return Ok(0);
    }
    if devices == 0 {
        return Err("No GPU workers requested".into());
    }
    let stopped = AtomicBool::new(false);
    let expired = || deadline.is_some_and(|d| Instant::now() >= d);
    let mut found = 0;
    let mut error = None;
    thread::scope(|scope| {
        let (tx, rx) = mpsc::sync_channel(devices as usize * 2);
        let mut acknowledgements = Vec::new();
        for id in 0..devices {
            let tx = tx.clone();
            let (ack_tx, ack_rx) = mpsc::channel();
            acknowledgements.push(ack_tx);
            let stopped = &stopped;
            let make_worker = &make_worker;
            let expired = &expired;
            scope.spawn(move || {
                let outcome = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
                    // Construct and drop native resources on their owning thread.
                    let mut worker = make_worker(id)?;
                    while !stopped.load(Ordering::Acquire) && !expired() {
                        let batch = worker.batch()?;
                        if tx.send(Message::Batch(id, batch)).is_err() {
                            break;
                        }
                        if ack_rx.recv().is_err() {
                            break;
                        }
                    }
                    Ok(())
                }));
                match outcome {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        let _ = tx.send(Message::Failed(id, e));
                    }
                    Err(_) => {
                        let _ = tx.send(Message::Failed(id, "worker panicked".into()));
                    }
                }
                let _ = tx.send(Message::Stopped(id));
            });
        }
        drop(tx);
        let mut alive = devices;
        while alive > 0 {
            if expired() {
                stopped.store(true, Ordering::Release);
            }
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(Message::Batch(id, mut batch)) => {
                    if expired() {
                        stopped.store(true, Ordering::Release);
                    }
                    if stopped.load(Ordering::Acquire) {
                        batch.hit = None;
                    }
                    let has_hit = batch.hit.is_some();
                    match handle(Event::Batch(id, batch)) {
                        Ok(()) if has_hit => {
                            found += 1;
                            if found == target {
                                stopped.store(true, Ordering::Release);
                            }
                        }
                        Ok(()) => {}
                        Err(e) => {
                            error.get_or_insert(e);
                            stopped.store(true, Ordering::Release);
                        }
                    }
                    let _ = acknowledgements[id as usize].send(());
                }
                Ok(Message::Failed(id, e)) => {
                    error.get_or_insert(format!("GPU {id}: {e}"));
                    stopped.store(true, Ordering::Release);
                }
                Ok(Message::Stopped(id)) => {
                    alive -= 1;
                    if let Err(e) = handle(Event::Stopped(id)) {
                        error.get_or_insert(e);
                        stopped.store(true, Ordering::Release);
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
    });
    match error {
        Some(e) => Err(e),
        None => Ok(found),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    struct Fake {
        arrived: Arc<AtomicUsize>,
        dropped: Arc<AtomicUsize>,
        gate: usize,
        first: bool,
        mode: u8,
    }
    impl Drop for Fake {
        fn drop(&mut self) {
            self.dropped.fetch_add(1, Ordering::SeqCst);
        }
    }
    impl Worker for Fake {
        type Hit = ();
        fn batch(&mut self) -> Result<Batch<()>, String> {
            if self.first {
                self.first = false;
                self.arrived.fetch_add(1, Ordering::SeqCst);
                let limit = Instant::now() + Duration::from_secs(2);
                while self.arrived.load(Ordering::SeqCst) < self.gate {
                    if Instant::now() >= limit {
                        return Err("workers did not run concurrently".into());
                    }
                    thread::sleep(Duration::from_millis(1));
                }
            }
            if self.mode == 1 {
                return Err("injected device error".into());
            }
            if self.mode == 2 {
                panic!("injected worker panic");
            }
            thread::sleep(Duration::from_millis(2));
            Ok(Batch {
                attempts: 1000,
                encodes: 10,
                elapsed: Duration::from_millis(2),
                hit: Some(()),
            })
        }
    }

    #[test]
    fn eight_devices_overlap_and_count_is_global() {
        for target in [1, 3, 17] {
            let arrived = Arc::new(AtomicUsize::new(0));
            let dropped = Arc::new(AtomicUsize::new(0));
            let (mut hits, mut attempts, mut exits) = (0, 0, 0);
            let found = run(
                8,
                target,
                None,
                |_| {
                    Ok(Fake {
                        arrived: arrived.clone(),
                        dropped: dropped.clone(),
                        gate: 8,
                        first: true,
                        mode: 0,
                    })
                },
                |event| {
                    match event {
                        Event::Batch(id, batch) => {
                            assert!(id < 8);
                            assert_eq!(batch.encodes, 10);
                            assert_eq!(batch.elapsed, Duration::from_millis(2));
                            attempts += batch.attempts;
                            if batch.hit.is_some() {
                                hits += 1;
                            }
                        }
                        Event::Stopped(id) => {
                            assert!(id < 8);
                            exits += 1;
                        }
                    }
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(found, target);
            assert_eq!(hits, target);
            assert!(attempts >= 8000);
            assert_eq!(exits, 8);
            assert_eq!(dropped.load(Ordering::SeqCst), 8);
        }
    }

    #[test]
    fn errors_panics_and_save_failures_stop_and_clean_up_every_worker() {
        for mode in [1, 2, 3] {
            let arrived = Arc::new(AtomicUsize::new(0));
            let dropped = Arc::new(AtomicUsize::new(0));
            let result = run(
                8,
                100,
                None,
                |id| {
                    Ok(Fake {
                        arrived: arrived.clone(),
                        dropped: dropped.clone(),
                        gate: 8,
                        first: true,
                        mode: if id == 0 { mode } else { 0 },
                    })
                },
                |_| {
                    if mode == 3 {
                        Err("save failed".into())
                    } else {
                        Ok(())
                    }
                },
            );
            assert!(result.is_err());
            assert_eq!(dropped.load(Ordering::SeqCst), 8);
        }
    }

    #[test]
    fn deadline_and_zero_target() {
        let arrived = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicUsize::new(0));
        let factory = |_| {
            Ok(Fake {
                arrived: arrived.clone(),
                dropped: dropped.clone(),
                gate: 1,
                first: true,
                mode: 0,
            })
        };
        assert_eq!(run(8, 0, None, &factory, |_| Ok(())).unwrap(), 0);
        assert_eq!(arrived.load(Ordering::SeqCst), 0);
        let found = run(
            8,
            u32::MAX,
            Some(Instant::now() + Duration::from_millis(30)),
            factory,
            |_| Ok(()),
        )
        .unwrap();
        assert!(found < u32::MAX);
        assert_eq!(dropped.load(Ordering::SeqCst), 8);
    }
}
