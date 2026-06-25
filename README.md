# Practical Guide to Quantization: Demos

Small, self-contained programs for measuring quantization on your own hardware. An LLM is mostly one matrix multiply repeated billions of times, so local inference speed comes down to how many bytes the weights cost (quantization) and how fast the hardware multiplies them. Each demo isolates one of these.

```text
.
├── metal/          Apple Silicon (Metal): FP32 vs 4-bit weights, same matmul
├── cuda/           NVIDIA (CUDA): plain CUDA cores vs Tensor Cores, same matmul
└── llama-server/   End-to-end: a GGUF quant ladder benchmarked on llama.cpp
```

## What each demo shows

### `metal/`: quantization is a memory win

Runs the same matrix multiply twice on the Apple GPU: once with FP32 weights, once with 4-bit K-quant-style weights dequantized on the fly. The kernel is identical; only the weight format changes. It prints time, throughput, weight memory, and the error introduced by quantization.

The op is memory-bound, so moving ~6x fewer weight bytes is faster even though the kernel reconstructs approximate weights each step. Buffers use `storageModeShared` (unified memory): CPU and GPU read the same bytes with no copy.

Measured on an Apple M5 Max:

```text
fp32   19.603 ms    876.4 GFLOP/s   weights 67.1 MB
q4      9.464 ms   1815.3 GFLOP/s   weights 10.5 MB
weights 6.4x smaller, kernel 2.07x faster, mean rel error ~6% (worst-case random data)
```

Run it (macOS + Xcode command-line tools, `xcode-select --install`):

```bash
cd metal
./run.sh          # or double-click benchmark.command
```

### `cuda/`: Tensor Cores are the matmul win

Isolates the compute hardware. It runs the same 4096³ matmul two ways: a naive FP32 kernel on the ordinary CUDA cores, and a WMMA kernel that issues Tensor Core tile instructions (fp16 in, fp32 accumulate).

Two ties back to quantization: Tensor Core inputs are fp16, not fp32, because lower precision moves more numbers per second (the same logic behind Blackwell's FP8 and FP4), and the `cudaMemcpy` staging is the discrete-GPU memory model that unified-memory parts avoid.

Measured on an NVIDIA GB10 (DGX Spark):

```text
CUDA cores (naive fp32)   114.660 ms     1198.7 GFLOP/s
Tensor Cores (WMMA fp16)   10.479 ms    13115.6 GFLOP/s
Tensor Core speedup: 10.9x
```

Run it (needs an NVIDIA GPU + CUDA toolkit):

```bash
cd cuda
make              # default -arch=sm_121 (GB10 / DGX Spark); override e.g. make ARCH=sm_90
./matmul_demo
```

### `llama-server/`: the trade end to end

The first two demos isolate single kernels; this one runs a full model. It starts llama.cpp's `llama-server` for each GGUF in a quant ladder, sends the same prompts through the OpenAI-compatible endpoint, and prints load time, time to first token, and tokens per second.

Run one model at several quant levels and you can watch file size drop and decode throughput rise, then see where it stops paying off (below ~Q4 you pay for decode work instead of saving bandwidth). On an M5 Max with a Gemma ladder, Q4_K_M was ~40% faster than Q8 at ~35% smaller, and a QAT Q4_0 was fastest.

Run it (needs Node and a llama.cpp `llama-server` build):

```bash
node llama-server/llama-server-quant-demo.mjs \
  --server /path/to/llama-server \
  --q8 /models/model-Q8_0.gguf \
  --q4 /models/model-Q4_K_M.gguf
# add more with --model "LABEL=/path/to.gguf", tune with --ctx, --ngl, --max-tokens
```

Pass `--help` for all options.

## How the three fit together

| Demo | Variable it changes | Lesson |
|---|---|---|
| `metal/` | weight format (FP32 vs 4-bit) | quantization cuts memory traffic, which is what decode waits on |
| `cuda/` | compute path (CUDA cores vs Tensor Cores) | lower precision plus dedicated matmul hardware is the throughput win |
| `llama-server/` | quant level on a real model | the same trade, measured end to end, including where it stops helping |

## Requirements at a glance

- `metal/`: any Apple Silicon Mac, Xcode command-line tools. The `.metal` shader compiles at runtime, so there is no Xcode project to set up.
- `cuda/`: an NVIDIA GPU and the CUDA toolkit (`nvcc`). Does not build on a Mac.
- `llama-server/`: Node 18+ and a built `llama-server` binary, plus one or more GGUF model files.
