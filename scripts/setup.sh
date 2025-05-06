#!/bin/bash
# Project Nanotron — One-Click Setup Script
# Configures the entire stack for B200 GPU

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           PROJECT NANOTRON — SETUP SCRIPT                      ║"
echo "║       Single-Node B200 Quantitative Engine                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${YELLOW}Warning: This script is optimized for Linux. Some features may not work on $OSTYPE${NC}"
fi

# ============================================================================
# 1. VERIFY GPU
# ============================================================================

echo -e "${CYAN}[1/8] Checking GPU...${NC}"

if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n1)
    echo -e "${GREEN}  ✓ Found GPU: $GPU_NAME ($GPU_MEMORY)${NC}"
    
    if [[ "$GPU_NAME" == *"B200"* ]]; then
        echo -e "${GREEN}  ✓ B200 Blackwell detected!${NC}"
    elif [[ "$GPU_NAME" == *"H200"* ]] || [[ "$GPU_NAME" == *"H100"* ]]; then
        echo -e "${YELLOW}  ⚠ Hopper GPU detected. Some B200 features won't be available.${NC}"
    fi
else
    echo -e "${YELLOW}  ⚠ nvidia-smi not found. GPU features may not work.${NC}"
fi

# ============================================================================
# 2. PYTHON ENVIRONMENT
# ============================================================================

echo -e "${CYAN}[2/8] Setting up Python environment...${NC}"

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo -e "${GREEN}  ✓ Created virtual environment${NC}"
fi

source .venv/bin/activate

# Install Python dependencies
pip install --upgrade pip > /dev/null 2>&1
pip install \
    jax[cuda12] \
    flax \
    optax \
    chex \
    pyarrow \
    numpy \
    > /dev/null 2>&1

echo -e "${GREEN}  ✓ Python dependencies installed${NC}"

# ============================================================================
# 3. MOJO INSTALLATION
# ============================================================================

echo -e "${CYAN}[3/8] Checking Mojo installation...${NC}"

if command -v mojo &> /dev/null; then
    MOJO_VERSION=$(mojo --version 2>/dev/null || echo "unknown")
    echo -e "${GREEN}  ✓ Mojo installed: $MOJO_VERSION${NC}"
else
    echo -e "${YELLOW}  ⚠ Mojo not found. Install from https://modular.com/mojo${NC}"
    echo "     curl -s https://get.modular.com | sh -"
fi

# ============================================================================
# 4. RUST BUILD
# ============================================================================

echo -e "${CYAN}[4/8] Building Rust backend...${NC}"

if command -v cargo &> /dev/null; then
    cd execution
    cargo build --release > /dev/null 2>&1
    cd ..
    echo -e "${GREEN}  ✓ Rust backend built${NC}"
else
    echo -e "${YELLOW}  ⚠ Cargo not found. Install Rust from https://rustup.rs${NC}"
fi

# ============================================================================
# 5. C++ BUILD
# ============================================================================

echo -e "${CYAN}[5/8] Building C++ execution layer...${NC}"

if command -v g++ &> /dev/null; then
    cd core/cpp
    make clean > /dev/null 2>&1 || true
    make > /dev/null 2>&1
    cd ../..
    echo -e "${GREEN}  ✓ C++ execution layer built${NC}"
else
    echo -e "${YELLOW}  ⚠ g++ not found. Install with: apt install g++${NC}"
fi

# ============================================================================
# 6. FRONTEND SETUP
# ============================================================================

echo -e "${CYAN}[6/8] Setting up frontend...${NC}"

if command -v npm &> /dev/null; then
    cd frontend
    npm install --silent > /dev/null 2>&1
    cd ..
    echo -e "${GREEN}  ✓ Frontend dependencies installed${NC}"
else
    echo -e "${YELLOW}  ⚠ npm not found. Install Node.js from https://nodejs.org${NC}"
fi

# ============================================================================
# 7. SHARED MEMORY SETUP
# ============================================================================

echo -e "${CYAN}[7/8] Setting up shared memory...${NC}"

# Create shared memory directory
sudo mkdir -p /dev/shm/nanotron
sudo chmod 777 /dev/shm/nanotron

# Increase shared memory limits
echo -e "${GREEN}  ✓ Shared memory configured${NC}"

# ============================================================================
# 8. GPUDIRECT STORAGE CHECK
# ============================================================================

echo -e "${CYAN}[8/8] Checking GPUDirect Storage...${NC}"

if [ -f "/usr/local/cuda/gds/tools/gdscheck" ]; then
    /usr/local/cuda/gds/tools/gdscheck -p > /dev/null 2>&1 && \
        echo -e "${GREEN}  ✓ GPUDirect Storage available${NC}" || \
        echo -e "${YELLOW}  ⚠ GPUDirect Storage not configured${NC}"
else
    echo -e "${YELLOW}  ⚠ GDS tools not found. Install CUDA GDS for optimal performance.${NC}"
fi

# ============================================================================
# DONE
# ============================================================================

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    SETUP COMPLETE!                             ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "To start the engine:"
echo ""
echo "  1. Start KDB+ (if available):"
echo "     q data/kdb/schema.q -p 5001"
echo ""
echo "  2. Start the Mojo engine:"
echo "     mojo core/mojo/nanotron.mojo"
echo ""
echo "  3. Start the Rust backend:"
echo "     ./execution/target/release/nanotron-server"
echo ""
echo "  4. Start the frontend:"
echo "     cd frontend && npm run dev"
echo ""
echo "Dashboard will be available at http://localhost:3000"
echo ""

