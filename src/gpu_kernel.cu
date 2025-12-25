#include <iostream>
#include <cuda_runtime.h>
#include "gpu_kernel.h"

// GPUで実行されるカーネル関数 (A + B = C)
__global__ void vectorAddKernel(const float* A, const float* B, float* C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

// C++から呼び出されるラッパー関数 (ホスト側の処理)
void runVectorAdd(const std::vector<float>& h_A, const std::vector<float>& h_B, std::vector<float>& h_C) {
    int numElements = h_A.size();
    size_t size = numElements * sizeof(float);

    // 1. GPUメモリの確保
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 2. CPU -> GPU へデータをコピー
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    // 3. GPUカーネルの実行
    // (256スレッド/ブロックで計算)
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // エラーチェック
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    // 4. GPU -> CPU へ結果をコピー
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    // 5. メモリ解放
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}