// ============================================================================
//  Local-LLM webinar demo  -  Apple Silicon GPU (Metal)
//
//  Transforms a batch of tokens through one large weight matrix on the GPU,
//  comparing FULL fp32 weights vs 4-bit K-quant weights for:
//     - GPU time
//     - effective throughput (GFLOP/s)
//     - weight memory footprint
//     - numerical accuracy (how much the q4 answer drifts from fp32)
//
//  Key Apple-Silicon talking point: storageModeShared = UNIFIED MEMORY.
//  The CPU fills these buffers and the GPU reads the SAME physical bytes -
//  no cudaMemcpy, no PCIe copy. That is the whole trick of Apple Silicon.
//
//  Build & run:   ./run.sh        (or see commands in run.sh)
// ============================================================================

import Foundation
import Metal

// ---- Problem size (tweak these live during the webinar) --------------------
let M = 512     // tokens in the batch
let N = 4096    // output neurons  (rows of W)
let K = 4096    // hidden dim      (cols of W)  -> must be a multiple of 32
let ITERS = 50  // timed iterations (we take the best, to ignore warm-up noise)
let QK = 32     // weights per quant block

precondition(K % QK == 0, "K must be a multiple of 32")

// ---- Metal setup -----------------------------------------------------------
guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("No Metal device found.")
}
print("GPU:            \(device.name)")
print("Unified memory: \(device.hasUnifiedMemory)")
print("Max threads/tg: \(device.maxThreadsPerThreadgroup.width)")
print(String(format: "Problem:        Y[%d x %d] = X[%d x %d] * W[%d x %d]^T",
             M, N, M, K, N, K))
print("")

let queue = device.makeCommandQueue()!

// Compile the .metal shader source at runtime (no Xcode project needed).
let shaderURL = URL(fileURLWithPath: "shaders.metal",
                    relativeTo: URL(fileURLWithPath: CommandLine.arguments.count > 1
                                    ? CommandLine.arguments[1] : "."))
let src = try String(contentsOf: shaderURL, encoding: .utf8)
let library = try device.makeLibrary(source: src, options: nil)
let fnFP32 = library.makeFunction(name: "matmul_fp32")!
let fnQ4   = library.makeFunction(name: "matmul_q4")!
let pipeFP32 = try device.makeComputePipelineState(function: fnFP32)
let pipeQ4   = try device.makeComputePipelineState(function: fnQ4)

// ---- Host data -------------------------------------------------------------
var rng = SystemRandomNumberGenerator()
func frand() -> Float { Float.random(in: -1...1, using: &rng) }

var X = [Float](repeating: 0, count: M * K)
var W = [Float](repeating: 0, count: N * K)
for i in 0..<X.count { X[i] = frand() }
for i in 0..<W.count { W[i] = frand() * 0.1 }   // weights are usually small

// ---- Quantize W to 4-bit K-quant style blocks (scale + min) ----------------
// Layout per 32-weight block: [fp16 d][fp16 m][16 bytes of nibbles] = 20 bytes
let blocksPerRow = K / QK
let bytesPerBlock = 2 + 2 + QK / 2          // 20
let qBytes = N * blocksPerRow * bytesPerBlock
var Wq = [UInt8](repeating: 0, count: qBytes)

func f32ToF16Bits(_ f: Float) -> UInt16 {    // minimal fp32 -> fp16 converter
    let bits = f.bitPattern
    let sign = UInt16((bits >> 16) & 0x8000)
    let exp = Int((bits >> 23) & 0xFF) - 127 + 15
    let mant = bits & 0x7FFFFF
    if exp <= 0 { return sign }                       // flush tiny to 0
    if exp >= 31 { return sign | 0x7C00 }             // overflow -> inf
    return sign | UInt16(exp << 10) | UInt16(mant >> 13)
}

var blockIdx = 0
for n in 0..<N {
    for b in 0..<blocksPerRow {
        let base = (n * blocksPerRow + b) * QK
        var lo = Float.greatestFiniteMagnitude
        var hi = -Float.greatestFiniteMagnitude
        for i in 0..<QK { let v = W[base + i]; lo = min(lo, v); hi = max(hi, v) }
        let d = (hi - lo) / 15.0                       // scale over 4-bit range
        let m = lo                                     // min
        let outOff = blockIdx * bytesPerBlock
        let dh = f32ToF16Bits(d); let mh = f32ToF16Bits(m)
        Wq[outOff + 0] = UInt8(dh & 0xFF); Wq[outOff + 1] = UInt8(dh >> 8)
        Wq[outOff + 2] = UInt8(mh & 0xFF); Wq[outOff + 3] = UInt8(mh >> 8)
        for i in stride(from: 0, to: QK, by: 2) {
            let q0 = d > 0 ? UInt8(max(0, min(15, Int(((W[base+i]   - m) / d).rounded())))) : 0
            let q1 = d > 0 ? UInt8(max(0, min(15, Int(((W[base+i+1] - m) / d).rounded())))) : 0
            Wq[outOff + 4 + i/2] = q0 | (q1 << 4)
        }
        blockIdx += 1
    }
}

// ---- Metal buffers (SHARED = unified memory, zero copy) --------------------
func buf<T>(_ a: [T]) -> MTLBuffer {
    a.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!,
                                           length: $0.count,
                                           options: .storageModeShared)! }
}
let bX  = buf(X)
let bW  = buf(W)
let bWq = buf(Wq)
let bY  = device.makeBuffer(length: M * N * MemoryLayout<Float>.stride,
                            options: .storageModeShared)!
var mM = UInt32(M), nN = UInt32(N), kK = UInt32(K)

// ---- Dispatch helper -------------------------------------------------------
func run(_ pipe: MTLComputePipelineState, weights: MTLBuffer) -> Double {
    var best = Double.greatestFiniteMagnitude
    for _ in 0..<ITERS {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipe)
        enc.setBuffer(bX,  offset: 0, index: 0)
        enc.setBuffer(weights, offset: 0, index: 1)
        enc.setBuffer(bY,  offset: 0, index: 2)
        enc.setBytes(&mM, length: 4, index: 3)
        enc.setBytes(&nN, length: 4, index: 4)
        enc.setBytes(&kK, length: 4, index: 5)
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        let grid = MTLSize(width: N, height: M, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        best = min(best, cb.gpuEndTime - cb.gpuStartTime)
    }
    return best
}

// ---- Run fp32 --------------------------------------------------------------
let tF = run(pipeFP32, weights: bW)
let yF = bY.contents().bindMemory(to: Float.self, capacity: M * N)
var ref = [Float](repeating: 0, count: M * N)
for i in 0..<M*N { ref[i] = yF[i] }

// ---- Run q4 ----------------------------------------------------------------
let tQ = run(pipeQ4, weights: bWq)
let yQ = bY.contents().bindMemory(to: Float.self, capacity: M * N)

// ---- Accuracy (q4 vs fp32) -------------------------------------------------
var maxAbs: Float = 0, sumAbs: Float = 0, sumRef: Float = 0
for i in 0..<M*N {
    let e = abs(yQ[i] - ref[i])
    maxAbs = max(maxAbs, e); sumAbs += e; sumRef += abs(ref[i])
}
let meanRelPct = (sumAbs / sumRef) * 100

// ---- Report ----------------------------------------------------------------
let flops = 2.0 * Double(M) * Double(N) * Double(K)   // multiply-add = 2 FLOP
let wFP32MB = Double(N * K * 4) / 1e6
let wQ4MB   = Double(qBytes)    / 1e6

func line(_ label: String, _ ms: Double, _ wMB: Double) {
    let gflops = flops / (ms / 1000.0) / 1e9
    print(String(format: "%-10@  time %7.3f ms   %8.1f GFLOP/s   weights %7.1f MB",
                 label as NSString, ms, gflops, wMB))
}
print("---- results (best of \(ITERS)) --------------------------------------")
line("fp32", tF * 1000, wFP32MB)
line("q4",   tQ * 1000, wQ4MB)
print("------------------------------------------------------------------")
print(String(format: "weights shrank %.2fx   (%.1f MB -> %.1f MB)",
             wFP32MB / wQ4MB, wFP32MB, wQ4MB))
print(String(format: "q4 speedup     %.2fx", tF / tQ))
print(String(format: "q4 accuracy    max abs err %.4f   mean rel err %.3f%%",
             maxAbs, meanRelPct))
