//! Property tests for the on-wire TradingSignal encoding.
//!
//! The producer is in another language (C++/Mojo) so the byte layout is
//! load-bearing.  Round-trip and length invariants here are the contract
//! we hold with that producer.

use nanotron_backend::TradingSignal;
use proptest::prelude::*;

fn arb_signal() -> impl Strategy<Value = TradingSignal> {
    (
        any::<u32>(),
        prop_oneof![Just(-1i8), Just(0i8), Just(1i8)],
        any::<f32>().prop_filter("finite", |x| x.is_finite()),
        any::<f32>().prop_filter("finite", |x| x.is_finite()),
        any::<i32>(),
        any::<i64>(),
    )
        .prop_map(
            |(ticker_id, direction, confidence, size, reasoning_depth, latency_us)| TradingSignal {
                ticker_id,
                direction,
                confidence,
                size,
                reasoning_depth,
                latency_us,
                timestamp_ns: 0,
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    #[test]
    fn roundtrip_through_bytes(s in arb_signal()) {
        let bytes = s.to_bytes();
        prop_assert_eq!(bytes.len(), TradingSignal::SIZE);
        let s2 = TradingSignal::from_bytes(&bytes).expect("decode");
        prop_assert_eq!(s.ticker_id, s2.ticker_id);
        prop_assert_eq!(s.direction, s2.direction);
        prop_assert_eq!(s.reasoning_depth, s2.reasoning_depth);
        prop_assert_eq!(s.latency_us, s2.latency_us);
        // f32s must compare bit-for-bit since to/from_le_bytes is exact.
        prop_assert_eq!(s.confidence.to_bits(), s2.confidence.to_bits());
        prop_assert_eq!(s.size.to_bits(), s2.size.to_bits());
    }

    #[test]
    fn from_bytes_rejects_short_buffer(len in 0usize..TradingSignal::SIZE) {
        let buf = vec![0u8; len];
        prop_assert!(TradingSignal::from_bytes(&buf).is_none());
    }

    #[test]
    fn from_bytes_accepts_oversized_buffer(extra in 0usize..256) {
        let s = TradingSignal {
            ticker_id: 1, direction: 1, confidence: 0.5, size: 100.0,
            reasoning_depth: 1, latency_us: 1, timestamp_ns: 0,
        };
        let mut buf = s.to_bytes().to_vec();
        buf.extend(std::iter::repeat_n(0xAB, extra));
        prop_assert!(TradingSignal::from_bytes(&buf).is_some());
    }

    #[test]
    fn direction_str_total(d in any::<i8>()) {
        let s = TradingSignal {
            ticker_id: 0, direction: d, confidence: 0.0, size: 0.0,
            reasoning_depth: 0, latency_us: 0, timestamp_ns: 0,
        };
        let _ = s.direction_str();
    }
}

#[test]
fn size_constant_matches_layout() {
    // We document SIZE = 32 to the C++/Mojo producer; this guards against
    // someone "innocently" adding a field and breaking the on-wire format.
    assert_eq!(TradingSignal::SIZE, 32);
}
