#include "Viewer.h"
#include "Reconstructor.h"
#include "RealSenseSource.h"
#include "Common.h"
#include <iostream>

int main() {
    // 1. 設定とインスタンス作成
    AppConfig config; // 共通設定 (Common.h で定義)
    GLViewer viewer;
    CudaReconstructor reconstructor;
    RealSenseSource input;

    // リモートデスクトップ用に解像度を少し下げて初期化 (本番では必要な解像度に変更してください)
    int width = 1280;
    int height = 720;

    // 2. 初期化
    if (!viewer.init(width, height, "Virtual Transparency (Production)")) {
        std::cerr << "[Error] Failed to initialize Viewer." << std::endl;
        return -1;
    }

    reconstructor.initialize(width, height);

    if (!input.initialize()) {
        std::cerr << "[Error] Failed to initialize RealSense." << std::endl;
        return -1;
    }

    std::cout << "[System] Production Mode Started." << std::endl;

    // 時間計測用構造体 (追加)
    ProcessTimings timings;

    // 3. メインループ
    while (!viewer.shouldClose()) {
        PointCloudData pcData;
        
        // RealSenseから最新フレームを取得
        if (input.update(pcData)) {
            // 再構成処理 (点群 -> 光線再生 -> OpenGLテクスチャ)
            // 引数: 点群データ, 設定, 書き込み先テクスチャID, 描画更新フラグ(true), 計測結果格納先
            reconstructor.process(pcData, config, viewer.getTextureID(), true, timings);
        }

        // 描画
        viewer.draw();
    }

    return 0;
}