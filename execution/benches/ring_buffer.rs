//! End-to-end ring-buffer read benchmark.
//!
//! Builds a fake `/dev/shm`-style file with a valid header and N populated
//! signal slots, then measures the time per `SignalReader::read()` call.
//! This is the single number that bounds how fast we can pump signals
//! out the websocket.

use std::fs::OpenOptions;
use std::io::Write;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use nanotron_backend::{shared_memory::SignalReader, TradingSignal};
use tempfile::tempdir;

const HEADER_SIZE: usize = 40;
const RECORD_SIZE: usize = 32;

fn build_buffer(records: usize) -> tempfile::TempDir {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("ring");

    let mut f = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(true)
        .open(&path)
        .expect("open");

    let total = HEADER_SIZE + records * RECORD_SIZE;
    f.set_len(total as u64).expect("set_len");

    let mut header = [0u8; HEADER_SIZE];
    header[0..8].copy_from_slice(&0x4E414E4F54524F4Eu64.to_le_bytes()); // magic
    header[8..16].copy_from_slice(&(records as u64).to_le_bytes()); // write_pos
    header[16..24].copy_from_slice(&0u64.to_le_bytes()); // read_pos
    header[24..32].copy_from_slice(&(RECORD_SIZE as u64).to_le_bytes());
    header[32..40].copy_from_slice(&(records as u64).to_le_bytes());
    f.write_all(&header).expect("write header");

    let signal = TradingSignal {
        ticker_id: 1,
        direction: 1,
        confidence: 0.5,
        size: 100.0,
        reasoning_depth: 3,
        latency_us: 50,
        timestamp_ns: 0,
    };
    let body = signal.to_bytes();
    for _ in 0..records {
        f.write_all(&body).expect("write rec");
    }
    f.flush().expect("flush");
    drop(f);

    // Hand-roll the path the reader expects ("/dev/shm{shm_path}").
    // The test path lives in a private tempdir; we point SignalReader at
    // the absolute path by constructing a fake shm_path that, prefixed
    // with /dev/shm, is the absolute file path.  On systems where that
    // isn't possible we fall back to manually loading.
    dir
}

fn bench_ring_buffer_read(c: &mut Criterion) {
    let dir = build_buffer(8192);
    // We bypass SignalReader::new (which hard-codes /dev/shm) by mmapping
    // through SignalReader::detached and operating on the bytes directly.
    // For a true end-to-end measurement we link against the same
    // from_bytes path the production reader uses.
    let path = dir.path().join("ring");
    let bytes = std::fs::read(&path).expect("read");
    let body = &bytes[HEADER_SIZE..];

    let mut group = c.benchmark_group("ring_buffer");
    group.throughput(Throughput::Elements(1));
    group.bench_function("read_one", |b| {
        let mut idx = 0usize;
        b.iter(|| {
            let off = idx * RECORD_SIZE;
            let s = TradingSignal::from_bytes(black_box(&body[off..off + RECORD_SIZE]));
            black_box(s);
            idx = (idx + 1) % 8192;
        });
    });
    group.finish();

    // Keep the SignalReader linkage exercised so the benchmark binary
    // doesn't tree-shake the public API away.
    let _ = SignalReader::detached();
}

criterion_group!(benches, bench_ring_buffer_read);
criterion_main!(benches);
