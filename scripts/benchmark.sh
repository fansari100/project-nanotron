#!/bin/bash
# Project Nanotron — Performance Benchmarks
# Measures latency and throughput of each component

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           PROJECT NANOTRON — BENCHMARK SUITE                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# 1. GPU MEMORY BANDWIDTH
# ============================================================================

echo "[1/6] GPU Memory Bandwidth Test..."

if command -v nvidia-smi &> /dev/null; then
    # Get peak bandwidth
    BANDWIDTH=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n1)
    echo "  Memory: $BANDWIDTH"
    
    # Run bandwidthTest if available
    if [ -f "/usr/local/cuda/samples/bin/x86_64/linux/release/bandwidthTest" ]; then
        /usr/local/cuda/samples/bin/x86_64/linux/release/bandwidthTest
    fi
fi

echo ""

# ============================================================================
# 2. JAX XLA COMPILATION TEST
# ============================================================================

echo "[2/6] JAX XLA Compilation Test..."

python3 << 'EOF'
import jax
import jax.numpy as jnp
from jax import jit
import time

# Warm up
x = jnp.ones((1000, 1000))

@jit
def matmul(a, b):
    return jnp.dot(a, b)

# First call (compilation)
start = time.perf_counter()
_ = matmul(x, x).block_until_ready()
compile_time = (time.perf_counter() - start) * 1000

# Subsequent calls (execution)
times = []
for _ in range(100):
    start = time.perf_counter()
    _ = matmul(x, x).block_until_ready()
    times.append((time.perf_counter() - start) * 1000)

avg_time = sum(times) / len(times)
print(f"  XLA Compilation: {compile_time:.2f} ms")
print(f"  Execution (avg): {avg_time:.3f} ms")
print(f"  GFLOPS: {(2 * 1000**3 / (avg_time / 1000)) / 1e9:.1f}")
EOF

echo ""

# ============================================================================
# 3. MCTS SEARCH SPEED
# ============================================================================

echo "[3/6] MCTS Search Speed Test..."

python3 << 'EOF'
import sys
sys.path.insert(0, '.')

try:
    from core.jax.mcts import MCTSEngine, create_dummy_state
    import time
    
    engine = MCTSEngine()
    state = create_dummy_state()
    
    # Warm up
    _ = engine.fast_inference(state)
    
    # Benchmark fast inference
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        _ = engine.fast_inference(state)
        times.append((time.perf_counter() - start) * 1e6)  # microseconds
    
    avg_us = sum(times) / len(times)
    print(f"  Fast Inference: {avg_us:.1f} μs")
    
    # Benchmark search
    times = []
    for _ in range(10):
        start = time.perf_counter()
        _ = engine.search(state, max_simulations=100, max_depth=8)
        times.append((time.perf_counter() - start) * 1000)
    
    avg_ms = sum(times) / len(times)
    print(f"  Search (100 sims): {avg_ms:.2f} ms")
    print(f"  Simulations/sec: {100 / (avg_ms / 1000):.0f}")

except Exception as e:
    print(f"  Skipped: {e}")
EOF

echo ""

# ============================================================================
# 4. SHARED MEMORY THROUGHPUT
# ============================================================================

echo "[4/6] Shared Memory Throughput..."

python3 << 'EOF'
import mmap
import time
import os

# Create test file
path = "/dev/shm/nanotron_bench"
size = 1024 * 1024 * 100  # 100MB

with open(path, 'w+b') as f:
    f.truncate(size)

# Benchmark write
with open(path, 'r+b') as f:
    mm = mmap.mmap(f.fileno(), size)
    
    data = b'\x00' * (1024 * 1024)  # 1MB chunks
    
    start = time.perf_counter()
    for i in range(100):
        mm[i * len(data):(i + 1) * len(data)] = data
    elapsed = time.perf_counter() - start
    
    write_gbps = (size / (1024**3)) / elapsed
    print(f"  Write: {write_gbps:.2f} GB/s")
    
    # Benchmark read
    start = time.perf_counter()
    for i in range(100):
        _ = mm[i * len(data):(i + 1) * len(data)]
    elapsed = time.perf_counter() - start
    
    read_gbps = (size / (1024**3)) / elapsed
    print(f"  Read: {read_gbps:.2f} GB/s")
    
    mm.close()

os.remove(path)
EOF

echo ""

# ============================================================================
# 5. ARROW IPC THROUGHPUT
# ============================================================================

echo "[5/6] Arrow IPC Throughput..."

python3 << 'EOF'
import pyarrow as pa
import numpy as np
import time
import io

# Create test data
n_rows = 1_000_000
table = pa.table({
    'timestamp': pa.array(np.arange(n_rows, dtype=np.int64)),
    'price': pa.array(np.random.randn(n_rows).astype(np.float64)),
    'size': pa.array(np.random.randint(0, 10000, n_rows, dtype=np.int32)),
})

# Benchmark serialization
times = []
for _ in range(10):
    sink = pa.BufferOutputStream()
    start = time.perf_counter()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    times.append(time.perf_counter() - start)

avg_time = sum(times) / len(times)
size_mb = sink.getvalue().size / (1024 * 1024)
throughput = size_mb / avg_time

print(f"  Serialize {n_rows:,} rows: {avg_time*1000:.1f} ms")
print(f"  Size: {size_mb:.1f} MB")
print(f"  Throughput: {throughput:.0f} MB/s")
EOF

echo ""

# ============================================================================
# 6. END-TO-END LATENCY
# ============================================================================

echo "[6/6] End-to-End Latency Estimate..."

echo "  Signal Generation (MCTS):  ~100 μs (easy) to ~1000 μs (hard)"
echo "  Shared Memory IPC:         ~1 μs"
echo "  Risk Check (C++):          ~0.5 μs"
echo "  Network Send (TCP):        ~10 μs"
echo "  ────────────────────────────────────"
echo "  Total (Easy):              ~112 μs"
echo "  Total (Hard):              ~1012 μs"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                  BENCHMARK COMPLETE                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"

