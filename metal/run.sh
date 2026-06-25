#!/bin/bash
# Build & run the Apple Silicon (Metal) fp32-vs-q4 demo.
# Requires: macOS + Xcode command line tools (xcode-select --install).
set -e
cd "$(dirname "$0")"

echo "Compiling..."
swiftc -O main.swift -o matmul_demo -framework Metal -framework Foundation

echo "Running (shaders.metal is compiled at runtime)..."
./matmul_demo .
