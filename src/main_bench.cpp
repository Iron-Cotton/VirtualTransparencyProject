#include "Viewer.h"
#include "Reconstructor.h"
#include "BenchmarkSource.h"
#include "Common.h"
#include <iostream>
#include <chrono> // 時間計測用
#include <cmath>
#include <thread> // ★これを追加してください
#include <cuda_runtime.h> // cudaDeviceSynchronize用

int main() {
    // 1. 設定
    AppConfig config;
    GLViewer viewer;
    CudaReconstructor reconstructor;
    BenchmarkSource input;

    int width = 4800;
    int height = 3600;

    // --- パラメータ設定 (前回と同じ) ---
    float displayPitch = 13.4f * 0.0254f / std::sqrt(3840.f * 3840.f + 2400.f * 2400.f);
    float elemPx = 60.0f;
    config.lensPitchX = elemPx * displayPitch;
    config.lensPitchY = elemPx * displayPitch;
    config.focalLength = 0.0068f; 
    config.numZPlane = 60;
    // ---------------------------------

    // 2. 初期化
    if (!viewer.init(width, height, "Virtual Transparency (Benchmark)")) {
        std::cerr << "[Error] Failed to initialize Viewer." << std::endl;
        return -1;
    }

    reconstructor.initialize(width, height);

    if (!input.initialize()) {
        std::cerr << "[Error] Failed to initialize Benchmark Source." << std::endl;
        return -1;
    }

    std::cout << "[System] Benchmark Mode Started." << std::endl;

    // ★計測用変数の準備
    int frameCount = 0;
    double sumReconTime = 0.0;
    double sumFrameTime = 0.0;
    auto lastFrameTime = std::chrono::high_resolution_clock::now();

    // 3. メインループ
    while (!viewer.shouldClose()) {
        // ループ開始時刻
        auto frameStart = std::chrono::high_resolution_clock::now();

        PointCloudData pcData;
        
        // データ更新 (CPU)
        input.update(pcData);

        // --- ★ここから再構成時間の計測開始 ---
        auto reconStart = std::chrono::high_resolution_clock::now();

        // GPU処理の実行
        reconstructor.process(pcData, config, viewer.getTextureID());

        // 【重要】GPUの計算完了を待機 (これを入れないと正しい計算時間が測れません)
        cudaDeviceSynchronize();

        auto reconEnd = std::chrono::high_resolution_clock::now();
        // --- ★ここまで ---

        // 描画 (ここでV-Sync待ちが発生する場合がある)
        viewer.draw();

        // ループ終了時刻
        auto frameEnd = std::chrono::high_resolution_clock::now();

        // 時間計算 (ミリ秒)
        double reconMs = std::chrono::duration<double, std::milli>(reconEnd - reconStart).count();
        double frameMs = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();

        // 平均を出すための積算
        sumReconTime += reconMs;
        sumFrameTime += frameMs;
        frameCount++;

        // ★追加: Sキーで保存
        if (glfwGetKey(viewer.getWindow(), GLFW_KEY_S) == GLFW_PRESS) {
            viewer.saveTexture("../images/screenshot.bmp");
            // 連続保存を防ぐための簡易ウェイト（本来はフラグ管理が良いですが簡易的に）
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        // ★追加: Qキーで終了
        if (glfwGetKey(viewer.getWindow(), GLFW_KEY_Q) == GLFW_PRESS) {
            break;
        }


        // 60フレームごとに平均時間を表示
        if (frameCount >= 60) {
            double avgRecon = sumReconTime / frameCount;
            double avgFrame = sumFrameTime / frameCount;
            double fps = 1000.0 / avgFrame;

            std::cout << "FPS: " << fps 
                      << " | Total Frame: " << avgFrame << "ms"
                      << " | Algo(CUDA): " << avgRecon << "ms" << std::endl;

            // カウンタ・積算リセット
            frameCount = 0;
            sumReconTime = 0.0;
            sumFrameTime = 0.0;
        }
    }

    return 0;
}