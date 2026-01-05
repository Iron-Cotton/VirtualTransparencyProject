#include "BenchmarkSource.h"
#include <opencv2/opencv.hpp>
#include <iostream>

bool BenchmarkSource::initialize() {
    // 1. 画像読み込み
    cv::Mat image = cv::imread("./images/standard/parrots.bmp");
    if (image.empty()) {
        std::cerr << "[Benchmark] Failed to load image." << std::endl;
        return false;
    }
    
    // 2. リサイズ
    const int NUM_X = 554; // 提供コードの定数
    const int NUM_Y = 554;
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(NUM_X, NUM_Y), 0, 0, cv::INTER_NEAREST);

    // 3. 点群データ生成 (CPUメモリ上に作成)
    // 提供コードの "点群座標" "点群色情報" 生成ループをここに移植
    int numPoints = NUM_X * NUM_Y;
    
    // PointCloudData への格納準備
    cacheData.h_xyz.resize(numPoints * 3);
    cacheData.h_rgb.resize(numPoints * 3);
    cacheData.numPoints = numPoints;

    float SUBJECT_Z = 1.0f; // 1.0m離れた位置

    // ループ処理 (提供コードより抜粋・簡略化)
    int idx = 0;
    for (int r = 0; r < NUM_Y; ++r) {
        for (int c = 0; c < NUM_X; ++c) {
            // 座標計算 (平面状に配置)
            // ... (提供コードの pointsPos 生成ロジック) ...
            cacheData.h_xyz[idx*3 + 0] = /* x */;
            cacheData.h_xyz[idx*3 + 1] = /* y */;
            cacheData.h_xyz[idx*3 + 2] = SUBJECT_Z;

            // 色コピー (BGR -> RGB注意)
            cv::Vec3b col = resized.at<cv::Vec3b>(NUM_Y - 1 - r, c);
            cacheData.h_rgb[idx*3 + 0] = col[2]; // R
            cacheData.h_rgb[idx*3 + 1] = col[1]; // G
            cacheData.h_rgb[idx*3 + 2] = col[0]; // B
            
            idx++;
        }
    }

    std::cout << "[Benchmark] Initialized with " << numPoints << " points." << std::endl;
    return true;
}

bool BenchmarkSource::update(PointCloudData& outData) {
    // 静止画なので、キャッシュしておいたデータをそのまま渡すだけ
    // (ベンチマークなので、毎回コピー発生させてもよいし、ポインタだけ渡してもよい)
    outData = cacheData; 
    return true;
}