#include "Reconstructor.h"
#include <iostream>
#include <vector>
#include <glad/glad.h>
#include "gpu_kernel.h"

// 出力画像バッファ
static uchar4* d_outputBuffer = nullptr;
static std::vector<uchar4> h_outputBuffer;

// 点群バッファ
static float* d_xyz = nullptr;
static unsigned char* d_rgb = nullptr;
static int currentCapacity = 0;

// ★追加: ボクセルグリッドバッファ (Accumulation Buffer)
// サイズ: 600 * 480 * 60
static unsigned int* d_gridR = nullptr;
static unsigned int* d_gridG = nullptr;
static unsigned int* d_gridB = nullptr;
static unsigned int* d_gridCnt = nullptr;

// 定数 (gpu_kernel.cu と合わせる)
const int VOXEL_W = 600;
const int VOXEL_H = 480;
const int VOXEL_D = 60;

CudaReconstructor::CudaReconstructor() {}
CudaReconstructor::~CudaReconstructor() { cleanup(); }

void CudaReconstructor::initialize(int w, int h) {
    width = w;
    height = h;

    // 出力バッファ
    size_t size = width * height * sizeof(uchar4);
    cudaMalloc(&d_outputBuffer, size);
    h_outputBuffer.resize(width * height);

    // ボクセルバッファ確保
    size_t voxelSize = VOXEL_W * VOXEL_H * VOXEL_D * sizeof(unsigned int);
    cudaMalloc(&d_gridR, voxelSize);
    cudaMalloc(&d_gridG, voxelSize);
    cudaMalloc(&d_gridB, voxelSize);
    cudaMalloc(&d_gridCnt, voxelSize);

    std::cout << "[Reconstructor] Voxel Grid Allocated: " << VOXEL_W << "x" << VOXEL_H << "x" << VOXEL_D << std::endl;
}

void CudaReconstructor::cleanup() {
    if (d_outputBuffer) { cudaFree(d_outputBuffer); d_outputBuffer = nullptr; }
    if (d_xyz) { cudaFree(d_xyz); d_xyz = nullptr; }
    if (d_rgb) { cudaFree(d_rgb); d_rgb = nullptr; }
    
    if (d_gridR) { cudaFree(d_gridR); d_gridR = nullptr; }
    if (d_gridG) { cudaFree(d_gridG); d_gridG = nullptr; }
    if (d_gridB) { cudaFree(d_gridB); d_gridB = nullptr; }
    if (d_gridCnt) { cudaFree(d_gridCnt); d_gridCnt = nullptr; }
}

void CudaReconstructor::process(PointCloudData& input, const AppConfig& config, unsigned int glTextureID, bool updateTexture, ProcessTimings& timings) {
    if (!d_outputBuffer) return;
    if (input.numPoints <= 0) return;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    // --- 1. 点群転送 (H2D) ---
    cudaEventRecord(start);
    
    if (input.numPoints > currentCapacity) {
        if (d_xyz) cudaFree(d_xyz);
        if (d_rgb) cudaFree(d_rgb);
        currentCapacity = input.numPoints * 1.2;
        cudaMalloc(&d_xyz, currentCapacity * 3 * sizeof(float));
        cudaMalloc(&d_rgb, currentCapacity * 3 * sizeof(unsigned char));
    }
    cudaMemcpy(d_xyz, input.h_xyz.data(), input.numPoints * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rgb, input.h_rgb.data(), input.numPoints * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    // ポインタセット
    input.d_xyz = d_xyz;
    input.d_rgb = d_rgb;

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    timings.dataTransferH2D = ms;

    // --- 2. カーネル実行 (GPU側で計測) ---
    // ここで gpu_kernel.cu の関数を呼ぶ
    runReconstructionKernel(
        input, config, 
        d_gridR, d_gridG, d_gridB, d_gridCnt,
        d_outputBuffer, width, height,
        timings // 参照渡し
    );

    // --- 3. 結果書き戻し (D2H) ---
    timings.dataTransferD2H = 0.0;
    if (updateTexture) {
        cudaEventRecord(start);
        
        // GPU -> CPU
        cudaMemcpy(h_outputBuffer.data(), d_outputBuffer, 
                   width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);
        
        // CPU -> OpenGL Texture
        glBindTexture(GL_TEXTURE_2D, glTextureID);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_outputBuffer.data());
        glBindTexture(GL_TEXTURE_2D, 0);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        timings.dataTransferD2H = ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}