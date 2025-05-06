# 🚀 Project Nanotron
## Single-Node B200 Quantitative Engine

**The "God Mode" Stack for Ultra-Low Latency Quantitative Trading**

This architecture is optimized for **Latency (Microseconds)** and **Throughput (Petabytes)** on a single machine, bypassing the network stack entirely to keep data on the PCIe bus.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PROJECT NANOTRON                                       │
│                    Single-Node B200 Quantitative Engine                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                        FRONTEND/RESEARCH LAYER                              ││
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ ││
│  │  │   WebGPU        │  │   React 19      │  │   Rust Axum Backend         │ ││
│  │  │   + deck.gl     │  │   Dashboard     │  │   (WebSocket Streaming)     │ ││
│  │  │   (L3 LOB @120Hz)│  │                 │  │                             │ ││
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                      ▲                                          │
│                                      │ Shared Memory (Read-Only)                │
│                                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                        EXECUTION LAYER (C++23)                              ││
│  │  ┌─────────────────────────────────────────────────────────────────────────┐││
│  │  │  Order Gateway  │  Risk Checks  │  Formal Verification (Frama-C)       │││
│  │  │  (TCP/UDP)      │  (Notional)   │  (Mathematically Proven Safe)        │││
│  │  └─────────────────────────────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                      ▲                                          │
│                                      │ Signal (via Shared Memory)               │
│                                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                     COMPUTE/STRATEGY LAYER                                  ││
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ ││
│  │  │   Mojo 1.0+     │  │   JAX/XLA       │  │   MCTS Search               │ ││
│  │  │   (Controller)  │  │   (Math Engine) │  │   (Prior Network + Tree)   │ ││
│  │  │   DynScaling    │  │   Fused Kernels │  │   Adaptive Depth            │ ││
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ ││
│  │                              │                                              ││
│  │                              ▼                                              ││
│  │  ┌─────────────────────────────────────────────────────────────────────────┐││
│  │  │              NVIDIA B200 GPU (192GB HBM3e, 8 TB/s)                      │││
│  │  │              Full Chip (No MIG) — Single Optimization Tree             │││
│  │  └─────────────────────────────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                      ▲                                          │
│                                      │ GPUDirect Storage (NVMe → GPU VRAM)      │
│                                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                          DATA LAYER                                         ││
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ ││
│  │  │   kdb+/q 4.1    │  │   Apache Arrow  │  │   4x 4TB NVMe PCIe 5.0     │ ││
│  │  │   (GPU Edition) │  │   (Zero-Copy)   │  │   RAID 0 (~50 GB/s)        │ ││
│  │  │   Ticker Plant  │  │   IPC Format    │  │   GPUDirect Storage        │ ││
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure

```
project-nanotron/
├── README.md                    # This file
├── config/                      # Configuration files
│   ├── hardware.toml            # Hardware configuration
│   ├── strategy.toml            # Strategy parameters
│   └── risk.toml                # Risk limits
├── core/                        # Core engine components
│   ├── mojo/                    # Mojo controller (DynScaling logic)
│   │   ├── nanotron.🔥          # Main Mojo entry point
│   │   ├── dynscaling.🔥        # Dynamic compute allocation
│   │   └── difficulty.🔥        # Difficulty estimation
│   ├── jax/                     # JAX math engine
│   │   ├── mcts.py              # Monte Carlo Tree Search
│   │   ├── prior_network.py     # Prior network for tree pruning
│   │   └── kernels.py           # Fused XLA kernels
│   └── cpp/                     # C++23 execution layer
│       ├── order_gateway.cpp    # Order gateway
│       ├── risk_engine.cpp      # Risk checks
│       ├── shared_memory.hpp    # Shared memory interface
│       └── Makefile             # Build configuration
├── data/                        # Data layer
│   ├── kdb/                     # kdb+/q scripts
│   │   ├── schema.q             # Table schemas
│   │   ├── ticker_plant.q       # Ticker plant
│   │   └── gpu_loader.q         # GPUDirect loader
│   └── arrow/                   # Arrow IPC utilities
│       └── zero_copy.py         # Zero-copy interface
├── execution/                   # Data plane (Rust + Axum)
│   ├── Cargo.toml               # Rust dependencies
│   ├── Dockerfile               # Multi-stage build
│   ├── src/
│   │   ├── lib.rs
│   │   ├── main.rs              # Axum server + signal pump
│   │   ├── shared_memory.rs     # /dev/shm ring-buffer reader
│   │   ├── metrics.rs           # /metrics + p-square latency quantiles
│   │   └── websocket.rs         # broadcast fan-out to clients
│   ├── benches/                 # criterion benches (serde, ring buffer)
│   └── tests/                   # proptest signal codec
├── control_plane/               # Control plane (FastAPI + Pydantic v2)
│   ├── pyproject.toml
│   ├── Dockerfile
│   ├── README.md
│   ├── src/nanotron_control/    # FastAPI app + routers + store
│   └── tests/                   # pytest (state machine, risk, backtests)
├── frontend/                    # WebGPU frontend
│   ├── package.json             # NPM dependencies
│   ├── Dockerfile               # nginx runtime, SPA fallback, /ws proxy
│   ├── nginx.conf
│   ├── src/
│   │   ├── App.tsx              # React 19 app
│   │   ├── OrderBook.tsx        # L3 order book (WebGPU)
│   │   └── webgpu/
│   │       └── renderer.ts      # WebGPU renderer
│   └── public/
│       └── index.html           # HTML entry point
├── deploy/helm/project-nanotron/  # Helm chart (Deployments, HPA, PDB, Ingress)
├── docker-compose.yml             # Local-dev stack
├── .github/workflows/ci.yml       # rust · python · ts · c++ · docker · helm
├── research/                    # Research notebooks
│   └── mcts_analysis.ipynb      # MCTS analysis
├── tests/                       # Test suite
│   ├── test_mcts.py             # MCTS tests
│   └── test_execution.cpp       # Execution tests
├── scripts/                     # Utility scripts
│   ├── setup.sh                 # One-click setup
│   └── benchmark.sh             # Performance benchmarks
└── docs/                        # Documentation
    ├── ARCHITECTURE.md          # Detailed architecture
    └── FORMAL_VERIFICATION.md   # Frama-C annotations
```

---

## 🔧 Hardware Requirements

| Component | Specification | Purpose |
|-----------|---------------|---------|
| **GPU** | NVIDIA B200 (Blackwell) | 192GB HBM3e, 8 TB/s bandwidth |
| **Configuration** | MIG Disabled | Full chip for single optimization tree |
| **Storage** | 4x 4TB NVMe PCIe 5.0 (RAID 0) | ~50 GB/s sequential read |
| **Storage Optimization** | GPUDirect Storage (GDS) | NVMe → GPU VRAM (bypass CPU) |
| **Interconnect** | PCIe 6.0 / 5.0 x16 | Maximum bandwidth |
| **Memory** | 256GB+ DDR5 | For CPU-side processing |

---

## 🚀 Quick Start

The whole stack — Rust data plane, FastAPI control plane, React frontend — boots
together with Docker Compose:

```bash
# Bring up data-plane (:8080), control-plane (:8090), frontend (:3000)
docker compose up --build

# Operator dashboard
open http://localhost:3000

# OpenAPI docs for the control plane
open http://localhost:8090/docs

# Rust data-plane health / metrics
curl http://localhost:8080/health
curl http://localhost:8080/metrics
```

For local hacking on a single component:

```bash
# Rust data plane
cd execution && cargo run --release

# FastAPI control plane
cd control_plane && pip install -e .[dev] && python -m nanotron_control

# Frontend
cd frontend && npm install && npm run dev

# Strategy core (real producer)
mojo core/mojo/nanotron.🔥
```

## 🧪 Tests

```bash
# Rust: unit + proptest + criterion benches
cd execution && cargo test --all-targets && cargo bench --bench serde -- --quick

# Python control plane (18 tests, ~1s)
cd control_plane && pytest -q

# Cross-language ABI test (python writer → rust reader)
pytest tests/integration -v
```

## ☸️ Kubernetes

```bash
helm install nanotron deploy/helm/project-nanotron
helm install nanotron deploy/helm/project-nanotron -f deploy/helm/project-nanotron/values.prod.yaml
```

The chart ships Deployments + Services + HPAs (CPU-targeted) + a PDB on the
data plane, plus an optional path-routed Ingress.

---

## 📊 Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Signal Latency** | < 10 μs | From data to signal |
| **Order Latency** | < 1 μs | From signal to wire |
| **Throughput** | 50 GB/s | Data ingestion |
| **MCTS Depth** | 1000+ nodes/ms | Search throughput |
| **Frontend FPS** | 120 Hz | L3 LOB rendering |

---

## 🔬 Key Technologies

### 1. Mojo (Controller Layer)
- Python-like syntax, MLIR compilation
- No GIL overhead
- Direct GPU kernel launches

### 2. JAX/XLA (Math Engine)
- Superior to PyTorch for single-chip optimization
- Fuses entire MCTS into single GPU kernel
- Automatic differentiation for prior network

### 3. kdb+/q 4.1 (Data Layer)
- Industry standard for tick data
- Native GPUDirect support
- Sub-microsecond queries

### 4. C++23 (Execution Layer)
- Minimal attack surface
- Formal verification with Frama-C
- Mathematically proven safety

### 5. WebGPU (Frontend)
- Direct GPU access from browser
- Millions of points at 120Hz
- deck.gl for geospatial-style rendering

---

## 📜 License

Proprietary — For authorized use only.

---

*Project Nanotron*
*Optimized for NVIDIA B200 Blackwell Architecture*

