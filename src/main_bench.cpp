#include "Viewer.h"
#include "Reconstructor.h"
#include "BenchmarkSource.h"
#include "Common.h"
#include <iostream>
#include <chrono>

int main() {
    // 1. 設定とインスタンス作成
    AppConfig config;
    GLViewer viewer;
    CudaReconstructor reconstructor;
    BenchmarkSource input;

    int width = 1280;
    int height = 720;

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
        
        // 静的画像から生成された点群を取得
        input.update(pcData);

        // 計測開始
        auto start = std::chrono::high_resolution_clock::now();

        // 再構成処理
        reconstructor.process(pcData, config, viewer.getTextureID());

        // 計測終了
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

        // ログ出力 (毎フレーム出すと多すぎるので間引いても良い)
        // std::cout << "Reconstruction Time: " << elapsed << " ms" << std::endl;

        // 描画
        viewer.draw();
    }

    return 0;
}