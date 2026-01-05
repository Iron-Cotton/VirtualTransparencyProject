#include <iostream>
#include <thread>
#include <chrono>

// 自作ヘッダー
#include "Viewer.h"
#include "InputSource.h"
#include "Reconstructor.h"
#include "Alignment.h"
#include "Common.h"

int main() {
    // 1. 設定データの準備
    AppConfig config;
    PointCloudData pcData; // 点群データの器（今は空っぽ）

    // 2. 各モジュールのインスタンス化
    GLViewer viewer;
    CudaReconstructor reconstructor;
    
    // とりあえずダミーの入力ソースを使う
    FileSource input; 

    // 3. 初期化処理
    // ※ リモートデスクトップで8Kは重すぎるので、一旦HD画質でテストします
    int width = 1280;
    int height = 720;

    if (!viewer.init(width, height, "Virtual Transparency")) {
        std::cerr << "[Error] Failed to initialize Viewer." << std::endl;
        return -1;
    }

    // CUDA再構成モジュールの初期化（メモリ確保など）
    reconstructor.initialize(width, height);
    
    // 入力ソースの初期化
    input.initialize();

    std::cout << "Initialization Complete. Starting Loop..." << std::endl;

    // 4. メインループ
    while (!viewer.shouldClose()) {
        // Step A: 入力データの更新 (今は空のデータが返るだけ)
        input.update(pcData);

        // Step B: 再構成処理 (CUDAで計算 -> CPUへ転送 -> OpenGLテクスチャ更新)
        // ここで「グラデーション」が作られます
        reconstructor.process(pcData, config, viewer.getTextureID());

        // Step C: 描画
        viewer.draw();
        
        // GPU負荷軽減のため少し待機 (約60FPS)
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    std::cout << "App Finished." << std::endl;
    return 0;
}