#include "gpu_kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

__global__ void reconstructKernel(
    int width, int height,
    AppConfig config,
    uchar4* outputBuffer // ← 変更: シンプルなポインタ
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // 仮実装: グラデーション
    uchar4 color;
    color.x = (unsigned char)((float)x / width * 255.0f);
    color.y = (unsigned char)((float)y / height * 255.0f);
    color.z = 128;
    color.w = 255;

    // 配列への書き込み (1次元インデックス)
    outputBuffer[y * width + x] = color;
}

// ラッパー関数
void runReconstructionKernel(PointCloudData& data, const AppConfig& config, uchar4* d_output, int width, int height) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    reconstructKernel<<<gridSize, blockSize>>>(width, height, config, d_output);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}