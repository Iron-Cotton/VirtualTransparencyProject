#include "Viewer.h"
#include "Reconstructor.h"
#include "CpuOpenGlReconstructor.h"
#include "BenchmarkSource.h"
#include "Common.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <thread>
#include <cuda_runtime.h>

int main() {
    AppConfig config;
    GLViewer viewer;
    CudaReconstructor cudaRecon;
    CpuOpenGlReconstructor cpuGlRecon;
    BenchmarkSource input;

    // ★解像度: 600x600 (レンズ10x10個分)
    int width = config.numLensX * config.elemImgPxX;
    int height = config.numLensY * config.elemImgPxY;

    // パラメータ
    float displayPitch = 13.4f * 0.0254f / std::sqrt(3840.f * 3840.f + 2400.f * 2400.f);
    config.lensPitchX = config.elemImgPxX * displayPitch;
    config.lensPitchY = config.elemImgPxY * displayPitch;
    config.focalLength = 0.0068f; 
    config.numZPlane = 60;


    // 初期化 (600x600)
    if (!viewer.init(width, height, "Virtual Transparency (Benchmark)")) {
        std::cerr << "[Error] Failed to initialize Viewer." << std::endl;
        return -1;
    }

    cudaRecon.initialize(width, height);
    cpuGlRecon.initialize(width, height);

    if (!input.initialize()) {
        std::cerr << "[Error] Failed to initialize Benchmark Source." << std::endl;
        return -1;
    }

    std::cout << "[System] Benchmark Mode Started (600x600, 10x10)." << std::endl;

    int frameCount = 0;
    double sumReconTime = 0.0;
    double sumFrameTime = 0.0;
    bool useCuda = true; 

    while (!viewer.shouldClose()) {
        auto frameStart = std::chrono::high_resolution_clock::now();

        PointCloudData pcData;
        input.update(pcData);

        // 120フレームごとに切り替え
        if (frameCount % 120 == 0) {
            useCuda = !useCuda;
            std::cout << "[Mode Switch] Now running: " << (useCuda ? "All-CUDA" : "CPU+OpenGL") << std::endl;
        }

        auto reconStart = std::chrono::high_resolution_clock::now();

        if (useCuda) {
            cudaRecon.process(pcData, config, viewer.getTextureID());
            cudaDeviceSynchronize();
        } else {
            cpuGlRecon.process(pcData, config, viewer.getTextureID());
            glFinish();
        }

        auto reconEnd = std::chrono::high_resolution_clock::now();

        viewer.draw();

        if (glfwGetKey(viewer.getWindow(), GLFW_KEY_S) == GLFW_PRESS) {
            if (useCuda)
                viewer.saveTexture("../images/screenshot_cuda.bmp");
            else
                viewer.saveTexture("../images/screenshot_cpu_gl.bmp");
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        if (glfwGetKey(viewer.getWindow(), GLFW_KEY_Q) == GLFW_PRESS) {
            break;
        }

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
    return 0;
}