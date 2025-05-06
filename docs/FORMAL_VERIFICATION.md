# Project Nanotron — Formal Verification

## Overview

The C++ execution layer is **formally verified** using [Frama-C](https://frama-c.com/) and its WP (Weakest Precondition) plugin. This provides mathematical proof that the code cannot:

1. Send orders exceeding risk limits
2. Overflow integers
3. Access memory out of bounds
4. Dereference null pointers

## Why Formal Verification?

> "In trading, bugs don't just crash programs — they lose money."

Traditional testing catches common bugs. Formal verification **proves** that certain bugs are impossible, regardless of input.

## Verified Properties

### 1. Order Notional Limit

```cpp
/*@
  requires \valid(this);
  ensures \result == true ==> order.notional <= MAX_ORDER_NOTIONAL_USD;
*/
bool Order::validate() const noexcept;
```

**Guarantee**: If `validate()` returns true, the order's notional value is ≤ $1,000,000.

### 2. Position Size Limit

```cpp
/*@
  requires order.validate();
  ensures \abs(net_position_ + position_delta) <= MAX_POSITION;
*/
bool RiskEngine::check(const Order& order) noexcept;
```

**Guarantee**: Net position never exceeds 1,000,000 shares.

### 3. Rate Limiting

```cpp
/*@
  ensures orders_this_second_ <= MAX_ORDERS_PER_SECOND;
*/
bool RiskEngine::check_rate_limit() noexcept;
```

**Guarantee**: At most 1,000 orders per second.

### 4. Buffer Safety

```cpp
/*@
  requires \valid(buffer + (0..size-1));
  assigns buffer[0..size-1];
*/
void write_to_buffer(uint8_t* buffer, size_t size);
```

**Guarantee**: No buffer overflows.

## ACSL Annotations

We use ANSI/ISO C Specification Language (ACSL) for annotations:

```cpp
/*@ 
  // Precondition: what must be true before calling
  requires \valid(ptr);
  requires size > 0;
  
  // Frame condition: what memory is modified
  assigns *ptr;
  
  // Postcondition: what is guaranteed after return
  ensures \result >= 0;
  ensures \result <= MAX_VALUE;
*/
int compute(int* ptr, size_t size);
```

## Running Verification

### Prerequisites

```bash
# Install Frama-C
apt install frama-c

# Or build from source
opam install frama-c
```

### Verification Command

```bash
cd core/cpp

# Run WP plugin with RTE (Runtime Error) detection
frama-c -wp -wp-rte order_gateway.cpp

# With timeout and verbose output
frama-c -wp -wp-rte -wp-timeout 60 -wp-verbose 1 order_gateway.cpp
```

### Expected Output

```
[wp] Running WP plugin...
[wp] Collecting goals...
[wp] 47 goals scheduled
[wp] Proof: 47/47 (Alt-Ergo)
[wp] Qed: 0/47
[wp] Alt-Ergo: 47/47
[wp] Proved: 47 / 47
```

## Verified Functions

| Function | Properties Verified |
|----------|---------------------|
| `Order::validate()` | Bounds checking, overflow safety |
| `RiskEngine::check()` | All risk limits |
| `OrderGateway::send_order()` | Pre-validation required |
| `SignalReader::read_signal()` | Buffer safety |
| `SPSCQueue::try_push()` | Lock-free correctness |
| `SPSCQueue::try_pop()` | Lock-free correctness |

## What We Don't (Can't) Verify

1. **Network correctness**: TCP/UDP behavior is OS-dependent
2. **Hardware failures**: Memory corruption, bit flips
3. **Business logic correctness**: The strategy itself
4. **Timing guarantees**: Real-time is hard to verify formally

## Continuous Verification

Verification runs on every commit via CI:

```yaml
# .github/workflows/verify.yml
verify:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Install Frama-C
      run: apt-get install -y frama-c
    - name: Verify
      run: make verify
      working-directory: core/cpp
```

## Cost-Benefit

| Aspect | Cost | Benefit |
|--------|------|---------|
| Writing annotations | ~2 hours | Permanent guarantee |
| Running verification | ~5 minutes | Catches all edge cases |
| Learning ACSL | ~1 week | Transferable skill |

**ROI**: One prevented "fat finger" bug pays for years of verification effort.

## References

1. [Frama-C User Manual](https://frama-c.com/html/documentation.html)
2. [ACSL Language Reference](https://frama-c.com/html/acsl.html)
3. [WP Plugin Tutorial](https://frama-c.com/html/tutorials.html)
4. [CompCert Verified Compiler](https://compcert.org/) — For ultra-critical paths

