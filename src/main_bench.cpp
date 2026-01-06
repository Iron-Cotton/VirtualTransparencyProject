#include "HeadlessContext.h" // Viewer.hの代わり
#include "Reconstructor.h"
#include "CpuOpenGlReconstructor.h"
#include "BenchmarkSource.h"
#include "Common.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <thread>
#include <cuda_runtime.h>
#include <glad/glad.h> // glGenTextures用に必要

int main() {
    AppConfig config;
    CudaReconstructor cudaRecon;
    CpuOpenGlReconstructor cpuGlRecon;
    BenchmarkSource input;

    // 解像度
    int width = config.numLensX * config.elemImgPxX;
    int height = config.numLensY * config.elemImgPxY;

    // パラメータ
    float displayPitch = 13.4f * 0.0254f / std::sqrt(3840.f * 3840.f + 2400.f * 2400.f);
    config.lensPitchX = config.elemImgPxX * displayPitch;
    config.lensPitchY = config.elemImgPxY * displayPitch;
    config.focalLength = 0.0068f; 
    config.numZPlane = 60;

    // ★ Headless Context 初期化
    HeadlessContext headless;
    if (!headless.init(width, height)) {
        std::cerr << "[Error] Failed to initialize Headless Context." << std::endl;
        return -1;
    }

    // ★ ダミーテクスチャの作成
    // (cudaRecon.processがエラーにならないように、書き込み先のテクスチャIDを用意する)
    unsigned int dummyTexID;
    glGenTextures(1, &dummyTexID);
    glBindTexture(GL_TEXTURE_2D, dummyTexID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    cudaRecon.initialize(width, height);
    cpuGlRecon.initialize(width, height);

    if (!input.initialize()) {
        std::cerr << "[Error] Failed to initialize Benchmark Source." << std::endl;
        return -1;
    }

    std::cout << "[System] Headless Benchmark Mode Started." << std::endl;

    int frameCount = 0;
    double sumReconTime = 0.0;
    double sumFrameTime = 0.0;
    bool useCuda = true; 

    // ループ回数制限 (無限ループだと終われないため)
    const int MAX_FRAMES = 3600; 

    while (frameCount < MAX_FRAMES) {
        auto frameStart = std::chrono::high_resolution_clock::now();

        PointCloudData pcData;
        input.update(pcData);

        if (frameCount % 120 == 0) {
            useCuda = !useCuda;
            std::cout << "[Mode Switch] Now running: " << (useCuda ? "All-CUDA" : "CPU+OpenGL") << std::endl;
        }

        auto reconStart = std::chrono::high_resolution_clock::now();

        if (useCuda) {
            // ★ 修正: viewer.getTextureID() -> dummyTexID
            // 60フレームに1回だけテクスチャ更新(転送)を有効にする
            bool doDraw = (frameCount % 60 == 0);
            cudaRecon.process(pcData, config, dummyTexID, doDraw);
            cudaDeviceSynchronize();
        } else {
            // CPU+OpenGL
            cpuGlRecon.process(pcData, config, dummyTexID);
            glFinish();
        }

        auto reconEnd = std::chrono::high_resolution_clock::now();

        // viewer.draw() は削除 (Headlessなので描画しない)

        auto frameEnd = std::chrono::high_resolution_clock::now();

        sumReconTime += std::chrono::duration<double, std::milli>(reconEnd - reconStart).count();
        sumFrameTime += std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();
        frameCount++;

        if (frameCount % 60 == 0) {
            double avgRecon = sumReconTime / 60.0;
            double avgFrame = sumFrameTime / 60.0;
            double fps = 1000.0 / avgFrame;
            std::cout << "FPS: " << fps 
                      << " | " << (useCuda ? "[CUDA]" : "[CPU+GL]")
                      << " Algo: " << avgRecon << "ms" << std::endl;
            sumReconTime = 0.0;
            sumFrameTime = 0.0;
        }
    }
    
    headless.cleanup();
    return 0;
}