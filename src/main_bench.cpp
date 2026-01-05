#include "Viewer.h"
#include "Reconstructor.h"
#include "BenchmarkSource.h"
#include "Common.h"
#include <iostream>
#include <chrono>
#include <cmath>

int main() {
    // 1. 設定
    AppConfig config;
    GLViewer viewer;
    CudaReconstructor reconstructor;
    BenchmarkSource input;

    int width = 1280;
    int height = 720;

    // ==========================================
    // ★修正: ベンチマーク用の正しい物理パラメータを設定
    // ==========================================
    
    // 1. ディスプレイの画素ピッチ (m)
    // 13.4インチ, 3840x2400 (WQUXGA) から計算される値
    // 13.4 * 0.0254 / sqrt(3840^2 + 2400^2)
    float displayPitch = 13.4f * 0.0254f / std::sqrt(3840.f * 3840.f + 2400.f * 2400.f);

    // 2. レンズピッチの計算
    // ベンチマークでは「要素画像1つあたり 60px」と定義されている
    float elemPx = 60.0f;
    config.lensPitchX = elemPx * displayPitch; // 約 2.7mm
    config.lensPitchY = elemPx * displayPitch;

    // 3. 焦点距離 (Gap) の設定
    // ベンチマーク定義の FOCAL_LENGTH (6.8mm) に合わせる
    config.centerDistance = 0.0068f; 

    // 4. その他
    config.numZPlane = 60;

    std::cout << "[Config] Lens Pitch: " << (config.lensPitchX * 1000.0f) << " mm" << std::endl;
    std::cout << "[Config] Gap (Focal): " << (config.centerDistance * 1000.0f) << " mm" << std::endl;

    // ==========================================

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

    // 3. メインループ
    while (!viewer.shouldClose()) {
        PointCloudData pcData;
        
        input.update(pcData);

        // 再構成処理
        reconstructor.process(pcData, config, viewer.getTextureID());

        viewer.draw();
    }

    return 0;
}