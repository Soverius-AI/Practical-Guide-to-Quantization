# Build the Blackwell / Tensor Core demo.
# DGX Spark (GB10 Grace-Blackwell) = sm_121.
# Override on the command line for other GPUs, e.g.:  make ARCH=sm_90  (Hopper)
ARCH ?= sm_121

matmul_demo: matmul.cu
	nvcc -O3 -arch=$(ARCH) matmul.cu -o matmul_demo

run: matmul_demo
	./matmul_demo

clean:
	rm -f matmul_demo
