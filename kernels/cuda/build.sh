#!/bin/bash

# Build script for all CUDA kernels in DiffusionX
# Compiles .cu files to PTX for use with cudarc

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Building DiffusionX CUDA Kernels${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: nvcc not found${NC}"
    echo "Please install CUDA toolkit and add nvcc to PATH"
    echo "Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo -e "${BLUE}Found nvcc version: ${NVCC_VERSION}${NC}"

# Set GPU architectures (adjust based on your GPU)
# Common compute capabilities:
# sm_50,sm_52: Maxwell (GTX 9xx, Quadro Mxxx)
# sm_60,sm_61: Pascal (GTX 10xx, Quadro Pxxx, Tesla P)
# sm_70,sm_75: Volta/Turing (V100, RTX 20xx, GTX 16xx, Quadro RTX)
# sm_80,sm_86: Ampere (A100, RTX 30xx)
# sm_89,sm_90: Ada/Hopper (RTX 40xx, H100)

# Default to widely compatible architectures
ARCH="--gpu-architecture=compute_75 --gpu-code=sm_75,sm_80,sm_86"

# Allow override via environment variable
if [ -n "$CUDA_ARCH" ]; then
    ARCH=$CUDA_ARCH
    echo -e "${YELLOW}Using custom architecture: ${ARCH}${NC}"
fi

# Compilation flags
FLAGS="-O3 -use_fast_math --ptx"
INCLUDE_FLAGS="-I${CUDA_HOME:-/usr/local/cuda}/include"

# Create output directory
mkdir -p ptx
echo -e "${BLUE}Output directory: ptx/${NC}\n"

# Counter for successful compilations
SUCCESS=0
TOTAL=0

# Function to compile a kernel
compile_kernel() {
    local cu_file=$1
    local ptx_file="ptx/$(basename ${cu_file%.cu}.ptx)"

    TOTAL=$((TOTAL + 1))
    echo -e "${YELLOW}[${TOTAL}] Compiling: ${cu_file}${NC}"

    if nvcc ${FLAGS} ${ARCH} ${INCLUDE_FLAGS} ${cu_file} -o ${ptx_file} 2>&1 | grep -v "warning"; then
        echo -e "${GREEN}    ✓ Success: ${ptx_file}${NC}"
        SUCCESS=$((SUCCESS + 1))
        return 0
    else
        echo -e "${RED}    ✗ Failed: ${cu_file}${NC}"
        return 1
    fi
}

# Compile all .cu files
echo -e "${BLUE}Compiling CUDA kernels...${NC}\n"

for cu_file in *.cu; do
    if [ -f "$cu_file" ]; then
        compile_kernel "$cu_file" || true
        echo ""
    fi
done

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Compilation Summary${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Total kernels: ${TOTAL}"
echo -e "Successful:    ${GREEN}${SUCCESS}${NC}"
echo -e "Failed:        $((TOTAL - SUCCESS))"

if [ ${SUCCESS} -eq ${TOTAL} ]; then
    echo -e "\n${GREEN}✓ All kernels compiled successfully!${NC}"
else
    echo -e "\n${YELLOW}⚠ Some kernels failed to compile${NC}"
fi

# List generated PTX files
if [ -d "ptx" ] && [ "$(ls -A ptx/*.ptx 2>/dev/null)" ]; then
    echo -e "\n${BLUE}Generated PTX files:${NC}"
    ls -lh ptx/*.ptx | awk '{printf "  %-40s %10s\n", $9, $5}'

    # Display PTX info
    echo -e "\n${BLUE}PTX Information:${NC}"
    for ptx in ptx/*.ptx; do
        if [ -f "$ptx" ]; then
            echo -e "\n  ${GREEN}$(basename $ptx):${NC}"
            head -10 "$ptx" | grep -E "\.version|\.target|\.address_size" | sed 's/^/    /'
        fi
    done
fi

# Check total size
TOTAL_SIZE=$(du -sh ptx/ 2>/dev/null | awk '{print $1}')
echo -e "\n${BLUE}Total size: ${TOTAL_SIZE}${NC}"

# Create a manifest file
echo -e "\n${YELLOW}Creating kernel manifest...${NC}"
cat > ptx/manifest.txt << EOF
DiffusionX CUDA Kernels Manifest
Generated: $(date)
CUDA Version: ${NVCC_VERSION}
Architecture: ${ARCH}

Compiled Kernels:
EOF

for ptx in ptx/*.ptx; do
    if [ -f "$ptx" ]; then
        SIZE=$(stat -f%z "$ptx" 2>/dev/null || stat -c%s "$ptx" 2>/dev/null || echo "unknown")
        echo "  - $(basename $ptx) (${SIZE} bytes)" >> ptx/manifest.txt
    fi
done

echo -e "${GREEN}✓ Manifest created: ptx/manifest.txt${NC}"

# Exit status
if [ ${SUCCESS} -eq ${TOTAL} ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Build completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    exit 0
else
    echo -e "\n${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Build completed with warnings${NC}"
    echo -e "${YELLOW}========================================${NC}"
    exit 1
fi
