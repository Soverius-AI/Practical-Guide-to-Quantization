#!/bin/bash
# Double-clickable launcher for the Metal fp32-vs-q4 benchmark.
cd "$(dirname "$0")"
echo "=== Local-LLM Metal benchmark ==="
echo "Compiling main.swift ..."
swiftc -O main.swift -o matmul_demo -framework Metal -framework Foundation && \
  ./matmul_demo . || echo "BUILD OR RUN FAILED (see errors above)"
echo ""
echo "=== done - leave this window open ==="
