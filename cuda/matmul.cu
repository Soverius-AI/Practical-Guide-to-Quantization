// ============================================================================
//  Local-LLM webinar demo  -  NVIDIA Blackwell (DGX Spark / GB10)
//
//  Shows the SAME matrix multiply two ways and times both:
//    1) matmul_naive   - one CUDA thread per output element, plain FP32 math
//                        on the regular CUDA cores.
//    2) matmul_wmma    - uses the 5th-gen TENSOR CORES via the WMMA API.
//                        Tensor Cores do a whole 16x16x16 tile multiply-
//                        accumulate as ONE hardware instruction. That is why
//                        GPUs are fast for LLMs: matmul is ~95% of transformer
//                        compute, and Tensor Cores execute it directly.
//
//  C[MxN] = A[MxK] * B[KxN]
//
//  Build (DGX Spark = GB10 Blackwell):
//     nvcc -O3 -arch=sm_121 matmul.cu -o matmul_demo
//     ./matmul_demo
//  (use -arch=sm_120 for consumer Blackwell, sm_90 for Hopper, sm_80 Ampere)
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define M 4096
#define N 4096
#define K 4096

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
    exit(1);} } while(0)

// ---------------------------------------------------------------------------
//  1) Naive FP32 matmul on the regular CUDA cores. One thread = one C element.
//
//     CUDA execution model, top down (used by BOTH kernels below):
//       - GRID      : the whole kernel launch. One launch spawns a grid of
//                     many equally-sized thread BLOCKS, collectively sized to
//                     cover the entire output. gridDim = the grid's size in
//                     blocks; the grid is what makes the launch "data-parallel".
//       - BLOCK     : a group of threads scheduled together on one streaming
//                     multiprocessor (SM); they can cooperate via fast shared
//                     memory. blockDim = the block's size in threads (`tpb`).
//       - blockIdx  : this block's (x,y) coordinate inside the grid.
//       - threadIdx : this thread's (x,y) coordinate inside its block.
//     The launch syntax  kernel<<<grid, block>>>(...)  sets exactly those two
//     sizes: how many blocks (grid), and how many threads per block.
//     Combining the indices gives every thread a unique global coordinate:
//         global = blockIdx * blockDim + threadIdx
//     which we use directly as the (row, col) of the output element to write.
//
//     How the grid hits the silicon: a CUDA GPU is an array of Streaming
//     Multiprocessors (SMs). The hardware scheduler assigns each BLOCK to one
//     SM, where it stays put for its lifetime; an SM runs several blocks at
//     once if registers and shared memory allow (its "occupancy"). Inside an
//     SM the block's threads run as WARPS of 32 in lockstep (SIMT), and the
//     warp schedulers feed each warp's instruction to the execution units --
//     CUDA cores for FP32, Tensor Cores for mma_sync, load/store units for
//     memory. A grid almost always has far more blocks than the GPU has SMs,
//     so blocks queue and stream through as SMs free up; keeping many warps
//     resident is exactly how an SM hides memory latency -- when one warp
//     stalls on a global-memory read, the scheduler instantly runs another.
//     For this demo's 4096^3 problem:
//         naive : (4096/16)^2 = 65,536 blocks x 256 threads (8 warps each)
//         wmma  : (4096/64)^2 =  4,096 blocks x 512 threads (16 warps each)
//
//     Correct and easy to read, but slow: each thread streams a whole row of A
//     and a whole column of B from global memory, so neighbouring threads
//     re-read the same values over and over. No reuse, no Tensor Cores.
// ---------------------------------------------------------------------------
__global__ void matmul_naive(const float* A, const float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // which output row
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // which output col
    if (row >= M || col >= N) return;                  // guard the ragged edge
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)                        // dot product of row . col
        acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
//  2) Tensor Core matmul via WMMA (Warp Matrix Multiply Accumulate).
//
//     Key shift from kernel 1: the unit of work is now a WARP (32 threads
//     acting in lockstep), not a single thread. Each warp computes one 16x16
//     output tile of C, with fp16 inputs and fp32 accumulation. A single
//     wmma::mma_sync issues a full 16x16x16 tile multiply-accumulate -- 4096
//     MACs -- as ONE hardware instruction on the Tensor Core. That density
//     (thousands of MACs per instruction) is the entire reason it is faster.
// ---------------------------------------------------------------------------
const int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;   // the Tensor Core tile shape

__global__ void matmul_wmma(const half* A, const half* B, float* C) {
    // Which 16x16 output tile does THIS warp own?
    //   blockIdx * blockDim + threadIdx is the usual global thread index
    //   (see the primer on kernel 1). Dividing the x-index by warpSize (32)
    //   collapses each group of 32 threads into one warp, so adjacent warps
    //   land on adjacent tile columns; threadIdx.y already indexes tile rows.
    int warpM =  blockIdx.y * blockDim.y + threadIdx.y;            // tile row
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // tile column

    // Fragments are the per-warp register tiles the Tensor Core reads and
    // writes. A is row-major (M x K); B is declared col-major so each load
    // pulls a contiguous (N x K) row of weights as a tile column.
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> bFrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;
    wmma::fill_fragment(cFrag, 0.0f);                  // accumulator starts at 0

    // Walk the shared K dimension 16 columns at a time, folding each
    // 16x16x16 tile product into the running accumulator.
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;     // top-left corner of this tile in A...
        int bCol = warpN * WMMA_N;     // ...and the matching tile in B
        if (aRow < M && bCol < N) {
            wmma::load_matrix_sync(aFrag, A + aRow * K + k, K);  // 16x16 slab of A
            wmma::load_matrix_sync(bFrag, B + bCol * K + k, K);  // 16x16 slab of B (col-major, N x K)
            wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);          // <-- Tensor Core MAC
        }
    }

    // Store the finished 16x16 tile back to C in row-major order.
    int cRow = warpM * WMMA_M, cCol = warpN * WMMA_N;
    if (cRow < M && cCol < N)
        wmma::store_matrix_sync(C + cRow * N + cCol, cFrag, N, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
float timeKernel(void (*launch)(), int iters) {
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    launch();                       // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) launch();
    cudaEventRecord(stop);
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);
    return ms / iters;
}

float *dA32, *dB32, *dC32; half *dA16, *dB16;

void launchNaive() {
    // 16x16 = 256 threads per block, one thread per output element; the grid
    // tiles the whole MxN output, rounding up so partial edge tiles are covered.
    dim3 tpb(16, 16), grid((N + 15) / 16, (M + 15) / 16);
    matmul_naive<<<grid, tpb>>>(dA32, dB32, dC32);
}
void launchWmma() {
    // 128x4 = 512 threads = 16 warps per block. The x-dim holds 128 threads =
    // 4 warps -> 4 tile columns; the y-dim holds 4 -> 4 tile rows. So each block
    // computes a 4x4 patch of 16x16 tiles = a 64x64 output region, which is why
    // the grid divides N and M by (WMMA_N * 4) = 64 (rounding up for the edges).
    dim3 tpb(128, 4);
    dim3 grid((N + (WMMA_N * 4) - 1) / (WMMA_N * 4),
              (M + (WMMA_M * 4) - 1) / (WMMA_M * 4));
    matmul_wmma<<<grid, tpb>>>(dA16, dB16, dC32);
}

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s   (compute capability %d.%d, %d SMs)\n",
           p.name, p.major, p.minor, p.multiProcessorCount);
    printf("C[%dx%d] = A[%dx%d] * B[%dx%d]\n\n", M, N, M, K, K, N);

    size_t szF = (size_t)M * K * sizeof(float);
    float *hA = (float*)malloc(szF), *hB = (float*)malloc(szF);
    for (size_t i = 0; i < (size_t)M * K; ++i) { hA[i] = (rand()%100)/100.0f; hB[i] = (rand()%100)/100.0f; }
    half *hA16 = (half*)malloc((size_t)M*K*sizeof(half));
    half *hB16 = (half*)malloc((size_t)K*N*sizeof(half));
    for (size_t i = 0; i < (size_t)M * K; ++i) { hA16[i] = __float2half(hA[i]); hB16[i] = __float2half(hB[i]); }

    // Allocate device buffers, then stage the inputs from host (CPU) to device
    // (GPU). On a discrete GPU this copy crosses PCIe and the data must fit in
    // VRAM; on a unified-memory part like GB10 the same call is coherent over
    // NVLink-C2C rather than a real copy. This staging is what Apple Silicon's
    // shared storage mode removes entirely.
    CUDA_CHECK(cudaMalloc(&dA32, szF)); CUDA_CHECK(cudaMalloc(&dB32, szF)); CUDA_CHECK(cudaMalloc(&dC32, (size_t)M*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dA16, (size_t)M*K*sizeof(half))); CUDA_CHECK(cudaMalloc(&dB16, (size_t)K*N*sizeof(half)));
    CUDA_CHECK(cudaMemcpy(dA32, hA, szF, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB32, hB, szF, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA16, hA16, (size_t)M*K*sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB16, hB16, (size_t)K*N*sizeof(half), cudaMemcpyHostToDevice));

    // 2 FLOP per multiply-accumulate (one multiply + one add), M*N outputs,
    // K MACs each. Same work for both kernels, so GFLOP/s is a fair compare.
    double flops = 2.0 * M * N * K;
    float tN = timeKernel(launchNaive, 20);   // fewer iters: the naive path is slow
    float tW = timeKernel(launchWmma, 50);

    printf("---- results -----------------------------------------------\n");
    printf("CUDA cores (naive fp32)  %8.3f ms   %8.1f GFLOP/s\n", tN, flops/(tN/1e3)/1e9);
    printf("Tensor Cores (WMMA fp16) %8.3f ms   %8.1f GFLOP/s\n", tW, flops/(tW/1e3)/1e9);
    printf("------------------------------------------------------------\n");
    printf("Tensor Core speedup: %.1fx\n", tN / tW);
    return 0;
}
