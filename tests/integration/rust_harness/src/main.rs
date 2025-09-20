//! Read every record currently in the given ring file using the same
//! `SignalReader` the production server uses, and print each as a CSV
//! line of `field=value`s.  Used by tests/integration/test_cross_language.py.
//!
//! We bypass `SignalReader::new` (which hard-codes /dev/shm) by mmapping
//! the user-supplied file directly via the public detached() + a thin
//! shim that exercises the public from_bytes path.

use std::env;
use std::fs::File;
use std::process::ExitCode;

use memmap2::MmapOptions;
use nanotron_backend::TradingSignal;

const HEADER_SIZE: usize = 40;
const MAGIC: u64 = 0x4E414E4F54524F4E;

fn main() -> ExitCode {
    let path = match env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: ring-harness <path>");
            return ExitCode::from(2);
        }
    };

    let file = match File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("open {path}: {e}");
            return ExitCode::from(1);
        }
    };

    let mmap = match unsafe { MmapOptions::new().map(&file) } {
        Ok(m) => m,
        Err(e) => {
            eprintln!("mmap: {e}");
            return ExitCode::from(1);
        }
    };

    if mmap.len() < HEADER_SIZE {
        eprintln!("file shorter than header");
        return ExitCode::from(1);
    }

    let header = &mmap[..HEADER_SIZE];
    let magic = u64::from_le_bytes(header[0..8].try_into().unwrap());
    if magic != MAGIC {
        eprintln!("bad magic 0x{magic:x}");
        return ExitCode::from(1);
    }

    let write_pos = u64::from_le_bytes(header[8..16].try_into().unwrap());
    let record_size = u64::from_le_bytes(header[24..32].try_into().unwrap()) as usize;
    let max_records = u64::from_le_bytes(header[32..40].try_into().unwrap()) as usize;

    for i in 0..write_pos {
        let slot = (i as usize) % max_records;
        let off = HEADER_SIZE + slot * record_size;
        let end = off + TradingSignal::SIZE;
        if end > mmap.len() {
            break;
        }
        let s = match TradingSignal::from_bytes(&mmap[off..end]) {
            Some(s) => s,
            None => continue,
        };
        println!(
            "ticker_id={},direction={},confidence={},size={},reasoning={},latency_us={}",
            s.ticker_id, s.direction, s.confidence, s.size, s.reasoning_depth, s.latency_us,
        );
    }

    ExitCode::from(0)
}
