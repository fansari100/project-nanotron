//! Shared memory interface for reading signals from Mojo/JAX engine.
//!
//! The producer (Mojo strategy core) writes 32-byte `TradingSignal` records
//! into a `/dev/shm` ring buffer with a `RingBufferHeader` prefix. We mmap
//! that file read-only and walk the cursors. If the producer hasn't started
//! yet (file missing), we degrade to a no-op reader so the server still
//! boots and serves `/health` — the readiness probe will report `not_ready`
//! until a producer attaches.

use std::fs::OpenOptions;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};

use memmap2::MmapOptions;

use crate::{NanotronError, Result, TradingSignal};

/// Ring buffer header layout shared with the C++/Mojo producer.
#[repr(C)]
struct RingBufferHeader {
    magic: u64,
    write_pos: u64,
    read_pos: u64,
    record_size: u64,
    max_records: u64,
}

impl RingBufferHeader {
    const MAGIC: u64 = 0x4E414E4F54524F4E; // "NANOTRON"
    const SIZE: usize = std::mem::size_of::<Self>();
}

/// Signal reader from shared memory.
///
/// `mmap` is `None` when the producer file did not exist at startup. In
/// that case `read()` always returns `None` and `is_attached()` returns
/// `false` so the readiness probe can surface the state to the operator.
pub struct SignalReader {
    mmap: Option<memmap2::Mmap>,
    cursor: AtomicU64,
}

impl SignalReader {
    /// Open the shared-memory ring buffer at `/dev/shm{shm_path}`. If the
    /// file does not exist, returns a detached reader that is safe to call
    /// and reports `is_attached() == false`. Any other I/O error is
    /// propagated.
    pub fn new(shm_path: &str) -> Result<Self> {
        let path = format!("/dev/shm{}", shm_path);
        match OpenOptions::new().read(true).open(&path) {
            Ok(file) => {
                let mmap = unsafe { MmapOptions::new().map(&file).map_err(NanotronError::Io)? };
                Ok(Self {
                    mmap: Some(mmap),
                    cursor: AtomicU64::new(0),
                })
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Self::detached()),
            Err(e) => Err(NanotronError::Io(e)),
        }
    }

    /// A reader with no backing mmap. All reads return `None`.
    pub fn detached() -> Self {
        Self {
            mmap: None,
            cursor: AtomicU64::new(0),
        }
    }

    /// True if the producer file was attached at startup.
    pub fn is_attached(&self) -> bool {
        self.mmap.is_some()
    }

    /// Read the next signal if one is available, advancing the cursor.
    pub fn read(&self) -> Option<TradingSignal> {
        let mmap = self.mmap.as_ref()?;
        if mmap.len() < RingBufferHeader::SIZE {
            return None;
        }

        let header = &mmap[..RingBufferHeader::SIZE];
        let magic = u64::from_le_bytes(header[0..8].try_into().ok()?);
        if magic != RingBufferHeader::MAGIC {
            return None;
        }

        let write_pos = u64::from_le_bytes(header[8..16].try_into().ok()?);
        let record_size = u64::from_le_bytes(header[24..32].try_into().ok()?);
        let max_records = u64::from_le_bytes(header[32..40].try_into().ok()?);
        if record_size == 0 || max_records == 0 {
            return None;
        }

        let cur = self.cursor.load(Ordering::Acquire);
        if cur >= write_pos {
            return None;
        }

        let offset = RingBufferHeader::SIZE + (cur % max_records) as usize * record_size as usize;
        let end = offset + TradingSignal::SIZE;
        if end > mmap.len() {
            return None;
        }

        let signal = TradingSignal::from_bytes(&mmap[offset..end])?;
        self.cursor.fetch_add(1, Ordering::Release);
        Some(signal)
    }

    /// Cheap check used by the busy loop to decide whether to back off.
    pub fn has_data(&self) -> bool {
        let Some(mmap) = self.mmap.as_ref() else {
            return false;
        };
        if mmap.len() < RingBufferHeader::SIZE {
            return false;
        }
        let header = &mmap[..RingBufferHeader::SIZE];
        let write_pos = u64::from_le_bytes(header[8..16].try_into().unwrap_or([0; 8]));
        let cur = self.cursor.load(Ordering::Acquire);
        write_pos > cur
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detached_reader_is_safe() {
        let r = SignalReader::detached();
        assert!(!r.is_attached());
        assert!(!r.has_data());
        assert!(r.read().is_none());
    }

    #[test]
    fn missing_file_yields_detached_not_error() {
        // Use a path component that cannot exist under /dev/shm.
        let r = SignalReader::new("/__nanotron_unit_test_missing__").unwrap();
        assert!(!r.is_attached());
    }

    #[test]
    fn signal_serde_roundtrip() {
        let signal = TradingSignal {
            ticker_id: 42,
            direction: 1,
            confidence: 0.85,
            size: 1000.0,
            reasoning_depth: 8,
            latency_us: 50,
            timestamp_ns: 0,
        };
        let bytes = signal.to_bytes();
        let s2 = TradingSignal::from_bytes(&bytes).unwrap();
        assert_eq!(signal.ticker_id, s2.ticker_id);
        assert_eq!(signal.direction, s2.direction);
        assert_eq!(signal.reasoning_depth, s2.reasoning_depth);
        assert_eq!(signal.latency_us, s2.latency_us);
        assert!((signal.confidence - s2.confidence).abs() < 1e-6);
        assert!((signal.size - s2.size).abs() < 1e-6);
    }
}
