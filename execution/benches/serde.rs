//! Microbenchmarks for the 32-byte TradingSignal codec.
//!
//! These are the hottest path on the consumer side — every signal that
//! comes off the producer's ring buffer goes through `from_bytes` once
//! and (for any signal we forward) `serde_json::to_string` once.  We
//! want both well under a microsecond on commodity x86_64.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use nanotron_backend::TradingSignal;

fn sample_signal() -> TradingSignal {
    TradingSignal {
        ticker_id: 42,
        direction: 1,
        confidence: 0.87,
        size: 1234.5,
        reasoning_depth: 12,
        latency_us: 73,
        timestamp_ns: 1_700_000_000_000_000_000,
    }
}

fn bench_to_bytes(c: &mut Criterion) {
    let s = sample_signal();
    c.bench_function("trading_signal/to_bytes", |b| {
        b.iter(|| {
            let bytes = black_box(&s).to_bytes();
            black_box(bytes);
        });
    });
}

fn bench_from_bytes(c: &mut Criterion) {
    let bytes = sample_signal().to_bytes();
    c.bench_function("trading_signal/from_bytes", |b| {
        b.iter(|| {
            let s = TradingSignal::from_bytes(black_box(&bytes));
            black_box(s);
        });
    });
}

fn bench_roundtrip(c: &mut Criterion) {
    c.bench_function("trading_signal/roundtrip", |b| {
        b.iter_batched(
            sample_signal,
            |s| {
                let bytes = s.to_bytes();
                let s2 = TradingSignal::from_bytes(&bytes).unwrap();
                black_box(s2);
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_json_serialize(c: &mut Criterion) {
    let s = sample_signal();
    c.bench_function("trading_signal/json_serialize", |b| {
        b.iter(|| {
            let json = serde_json::to_string(black_box(&s)).unwrap();
            black_box(json);
        });
    });
}

criterion_group!(
    benches,
    bench_to_bytes,
    bench_from_bytes,
    bench_roundtrip,
    bench_json_serialize
);
criterion_main!(benches);
