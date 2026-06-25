#include <metal_stdlib>
using namespace metal;

// ============================================================================
//  Webinar demo: transform a batch of tokens through one big weight matrix.
//
//  Math:  Y = X * W^T
//    X : [M tokens, K hidden]   (the activations / token embeddings)
//    W : [N out,    K hidden]   (the weight matrix - this is what we quantize)
//    Y : [M tokens, N out]
//
//  Two versions of the same math:
//    1) matmul_fp32  - weights stored as full 32-bit floats (4 bytes each)
//    2) matmul_q4    - weights stored as 4-bit "K-quant style" blocks
//                      (block of 32 weights -> 16 bytes of nibbles
//                       + 1 fp16 scale + 1 fp16 min = 20 bytes for 32 weights
//                       => 0.625 bytes/weight  vs  4 bytes/weight = 6.4x smaller)
//
//  The point of the demo: the q4 weights are ~6x smaller, so the GPU moves far
//  less data from memory. On a bandwidth-bound op like this, smaller = faster,
//  with only a tiny accuracy cost. That is why local LLMs ship quantized.
// ============================================================================

constant uint QK = 32;   // weights per quant block (matches llama.cpp K-quants)

// ---- One block of 32 four-bit weights, K-quant style (scale + min) ----------
//  Reconstruction:  w_i = scale * nibble_i + min
//  nibble_i is an unsigned 4-bit value in [0, 15].
struct BlockQ4 {
    half  d;          // scale   (fp16)
    half  m;          // min     (fp16)
    uchar qs[QK / 2]; // 32 nibbles packed two-per-byte = 16 bytes
};

// ---------------------------------------------------------------------------
//  Baseline: full fp32 weights. One thread computes one output element Y[row,col]
// ---------------------------------------------------------------------------
kernel void matmul_fp32(
    device const float* X      [[buffer(0)]],   // [M, K]
    device const float* W      [[buffer(1)]],   // [N, K]
    device       float* Y      [[buffer(2)]],   // [M, N]
    constant uint&      M       [[buffer(3)]],
    constant uint&      N       [[buffer(4)]],
    constant uint&      K       [[buffer(5)]],
    uint2 gid                  [[thread_position_in_grid]])
{
    uint row = gid.y;   // token
    uint col = gid.x;   // output neuron
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    device const float* xrow = X + row * K;
    device const float* wrow = W + col * K;   // weights for this output neuron
    for (uint k = 0; k < K; ++k) {
        acc += xrow[k] * wrow[k];
    }
    Y[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
//  Quantized: 4-bit weights, dequantized on the fly inside the inner loop.
//  Same output shape, same math - only the weight storage changed.
// ---------------------------------------------------------------------------
kernel void matmul_q4(
    device const float*   X     [[buffer(0)]],  // [M, K]
    device const BlockQ4* W     [[buffer(1)]],  // [N * K/QK] blocks
    device       float*   Y     [[buffer(2)]],  // [M, N]
    constant uint&        M      [[buffer(3)]],
    constant uint&        N      [[buffer(4)]],
    constant uint&        K      [[buffer(5)]],
    uint2 gid                   [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= M || col >= N) return;

    uint blocksPerRow = K / QK;
    device const float*   xrow = X + row * K;
    device const BlockQ4* wrow = W + col * blocksPerRow;  // blocks for this neuron

    float acc = 0.0f;
    for (uint b = 0; b < blocksPerRow; ++b) {
        BlockQ4 blk = wrow[b];
        float d = float(blk.d);
        float m = float(blk.m);
        device const float* xblk = xrow + b * QK;
        // unpack 32 nibbles -> 32 weights
        for (uint i = 0; i < QK / 2; ++i) {
            uchar byte = blk.qs[i];
            float w0 = d * float(byte & 0x0F) + m;   // low  nibble
            float w1 = d * float(byte >> 4)   + m;   // high nibble
            acc += xblk[2 * i]     * w0;
            acc += xblk[2 * i + 1] * w1;
        }
    }
    Y[row * N + col] = acc;
}
