#include "HeadlessContext.h"
#include "Reconstructor.h"
#include "CpuOpenGlReconstructor.h"
#include "BenchmarkSource.h"
#include "Common.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <thread>
#include <cuda_runtime.h>
#include <glad/glad.h>

int main() {
    AppConfig config;
    CudaReconstructor cudaRecon;
    CpuOpenGlReconstructor cpuGlRecon;
    BenchmarkSource input;

    // 解像度設定 (60x20 = 1200)
    int width = config.numLensX * config.elemImgPxX;
    int height = config.numLensY * config.elemImgPxY;

    // パラメータ設定
    float displayPitch = 13.4f * 0.0254f / std::sqrt(3840.f * 3840.f + 2400.f * 2400.f);
    config.lensPitchX = config.elemImgPxX * displayPitch;
    config.lensPitchY = config.elemImgPxY * displayPitch;
    config.focalLength = 0.0068f; 
    config.numZPlane = 60;

    // Headless Context 初期化
    HeadlessContext headless;
    if (!headless.init(width, height)) {
        std::cerr << "[Error] Failed to initialize Headless Context." << std::endl;
        return -1;
    }

    // ダミーテクスチャ作成
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
    bool useCuda = false; // 最初はCPUからスタート
    const int MAX_FRAMES = 1000;
    const int SWITCH_INTERVAL = 200; // モード切替間隔
    const int LOG_INTERVAL = 100;     // ログ出力間隔

    // 計測用変数
    double sumFrameTime = 0.0; // 実時間の合計
    double sumPartsTime = 0.0; // 内部処理時間の合計
    ProcessTimings timings;

    while (frameCount < MAX_FRAMES) {
        // 動作中であることを示すインジケータ ( \r で上書き)
        if (frameCount % 10 == 0) {
            std::cout << "\r[Running] Frame: " << frameCount << "   " << std::flush;
        }

        auto frameStart = std::chrono::high_resolution_clock::now();

        PointCloudData pcData;
        input.update(pcData);

        // モード切替
        if (frameCount % SWITCH_INTERVAL == 0 && frameCount > 0) {
            useCuda = !useCuda;
            std::cout << "\n[Mode Switch] Now running: " << (useCuda ? "All-CUDA" : "CPU+OpenGL") << std::endl;
            // 切り替え直後は計測値をリセット
            sumFrameTime = 0.0;
            sumPartsTime = 0.0;
        }

        // 再構成処理
        auto reconStart = std::chrono::high_resolution_clock::now();

        if (useCuda) {
            bool doDraw = (frameCount % 60 == 0); // テクスチャ書き戻し頻度
            cudaRecon.process(pcData, config, dummyTexID, doDraw, timings);
        } else {
            cpuGlRecon.process(pcData, config, dummyTexID, timings);
        }

        auto reconEnd = std::chrono::high_resolution_clock::now();

        // 1フレームの実測時間 (ms)
        double currentFrameTime = std::chrono::duration<double, std::milli>(reconEnd - reconStart).count();
        
        // 内部処理時間の合計 (ms)
        double currentPartsSum = timings.dataTransferH2D + timings.voxelization + timings.dataTransferInter + timings.rendering + timings.dataTransferD2H;

        sumFrameTime += currentFrameTime;
        sumPartsTime += currentPartsSum;

        frameCount++;

        // ログ出力
        if (frameCount % LOG_INTERVAL == 0) {
            double avgFrameTime = sumFrameTime / (double)LOG_INTERVAL;
            double avgPartsTime = sumPartsTime / (double)LOG_INTERVAL;
            double fps = 1000.0 / avgFrameTime; // 1秒(1000ms) / 平均フレーム時間

            std::cout << "\n--- " << (useCuda ? "[CUDA]" : "[CPU+GL]") << " Frame: " << frameCount << " ---" << std::endl;
            std::cout << "  FPS:           " << fps << " fps" << std::endl;
            std::cout << "  Frame Time:    " << avgFrameTime << " ms (Actual)" << std::endl;
            std::cout << "  Process Sum:   " << avgPartsTime << " ms (Internal Sum)" << std::endl;
            std::cout << "  -----------------------------" << std::endl;
            std::cout << "  [Breakdown]" << std::endl;
            std::cout << "    Input(H2D):  " << timings.dataTransferH2D << " ms" << std::endl;
            std::cout << "    Voxel:       " << timings.voxelization << " ms" << std::endl;
            if (!useCuda) 
                std::cout << "    UpLoad:      " << timings.dataTransferInter << " ms" << std::endl;
            std::cout << "    Render:      " << timings.rendering << " ms" << std::endl;
            if (useCuda && timings.dataTransferD2H > 0)
                std::cout << "    Result(D2H): " << timings.dataTransferD2H << " ms" << std::endl;
            std::cout << "--------------------------------\n" << std::endl;

            // リセット
            sumFrameTime = 0.0;
            sumPartsTime = 0.0;
        }
    }
    
    headless.cleanup();
    return 0;
}