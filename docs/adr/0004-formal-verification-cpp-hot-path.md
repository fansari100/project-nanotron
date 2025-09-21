# ADR-0004: Frama-C/ACSL on the C++ order gateway

- Status: accepted
- Date: 2025-09-18

## Context

`core/cpp/order_gateway.cpp` is the last code on the path before bytes
go on the wire. Every order it emits has a hard, non-negotiable upper
bound on size, notional, and price deviation from mid. A bug here
costs real money in a way that no other component does.

Runtime checks (`if (notional > MAX) return error`) are the standard
defense, but they push the failure to detection time on the hot path.
We want the failure to be impossible to compile, not impossible to
execute.

## Decision

Annotate the gateway with ACSL contracts:

```c
/*@
  predicate valid_notional(double n) = 0.0 <= n <= 1000000.0;
  predicate valid_size(int32_t s) = 0 < s <= 100000;
  predicate valid_price(double p) = 0.0 < p <= 1000000.0;
*/
```

…and run Frama-C/WP in CI:

```
frama-c -wp -wp-rte order_gateway.cpp
```

- Function pre/post conditions encode caller responsibilities at the
  type level. WP verifies them statically.
- `-wp-rte` adds runtime-error checks (overflow, division by zero,
  invalid index) and proves they cannot trip.
- Anything Frama-C cannot prove is treated as a build failure.

Detail in `docs/FORMAL_VERIFICATION.md`.

## Consequences

- New code in `core/cpp/` must come with contracts. Reviewers reject
  PRs that add hot-path C++ without them. The friction is the point.
- ACSL has a learning curve. We mitigate by keeping the hot path small
  (~400 lines of C++) and pushing the bulk of the strategy logic into
  Mojo/JAX where the failure modes are different.
- Compile-time and CI runtime are unaffected; Frama-C runs as a
  separate, advisory job during the soak phase before release tags.
