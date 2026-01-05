#include "BenchmarkSource.h"
#include <iostream>
#include <cmath>

// 定数定義 (PCSJ2025-OpenGL-v2-3-2.cpp より移植)
// 計算に必要な最小限のパラメータをここで再定義します
namespace {
    const float MIN_OBSERVE_Z = 1.0f;
    // ディスプレイパラメータ (本来はCommon.hやConfigから取るべきですが、ベンチマーク再現のため固定値を計算)
    const float DISPLAY_PX_PITCH = 13.4f * 0.0254f / std::sqrt(3840.f * 3840.f + 2400.f * 2400.f);
    const int NUM_LENS_X = 20;
    const int NUM_LENS_Y = 20;
    const int NUM_ELEM_IMG_PX_X = 60;
    const int NUM_ELEM_IMG_PX_Y = 60;
    
    const unsigned int NUM_DISPLAY_IMG_PX_Y = NUM_ELEM_IMG_PX_Y * NUM_LENS_Y;
    const float DISPLAY_IMG_SIZE_Y = NUM_DISPLAY_IMG_PX_Y * DISPLAY_PX_PITCH;

    // 被写体サイズ計算
    const float SUBJECT_SIZE_X = DISPLAY_IMG_SIZE_Y * (1.0f + MIN_OBSERVE_Z) / MIN_OBSERVE_Z;
    const float SUBJECT_SIZE_Y = DISPLAY_IMG_SIZE_Y * (1.0f + MIN_OBSERVE_Z) / MIN_OBSERVE_Z;
}

BenchmarkSource::BenchmarkSource() {}
BenchmarkSource::~BenchmarkSource() {}

bool BenchmarkSource::initialize() {
    std::cout << "[Benchmark] Initializing..." << std::endl;

    // 1. 画像読み込み
    // ※ 実行ディレクトリからの相対パスに注意してください
    cv::Mat image_input = cv::imread(IMAGE_PATH);
    if (image_input.empty()) {
        std::cerr << "[Benchmark] Error: 画像を開くことができませんでした: " << IMAGE_PATH << std::endl;
        return false;
    }

    // 2. リサイズ
    cv::Mat resized_image;
    cv::resize(image_input, resized_image, cv::Size(NUM_SUBJECT_POINTS_X, NUM_SUBJECT_POINTS_Y), 0, 0, cv::INTER_NEAREST);

    // 3. 点群データ生成
    int totalPoints = NUM_SUBJECT_POINTS_X * NUM_SUBJECT_POINTS_Y;
    
    // メモリ確保
    cacheData.numPoints = totalPoints;
    cacheData.h_xyz.resize(totalPoints * 3);
    cacheData.h_rgb.resize(totalPoints * 3);

    // ピッチ計算
    const float SUBJECT_POINTS_PITCH_X = SUBJECT_SIZE_X / static_cast<float>(NUM_SUBJECT_POINTS_X);
    const float SUBJECT_POINTS_PITCH_Y = SUBJECT_SIZE_Y / static_cast<float>(NUM_SUBJECT_POINTS_Y);
    const float HALF_SUBJECT_POINTS_PITCH_X = SUBJECT_POINTS_PITCH_X * 0.5f;
    const float HALF_SUBJECT_POINTS_PITCH_Y = SUBJECT_POINTS_PITCH_Y * 0.5f;
    const int HALF_NUM_SUBJECT_POINTS_X = NUM_SUBJECT_POINTS_X / 2;
    const int HALF_NUM_SUBJECT_POINTS_Y = NUM_SUBJECT_POINTS_Y / 2;

    int idx = 0;
    for (int r = -HALF_NUM_SUBJECT_POINTS_Y; r < HALF_NUM_SUBJECT_POINTS_Y; ++r) {
        int row = r + HALF_NUM_SUBJECT_POINTS_Y;
        int reverseRow = NUM_SUBJECT_POINTS_Y - 1 - row; // 画像の上下反転対応

        for (int c = -HALF_NUM_SUBJECT_POINTS_X; c < HALF_NUM_SUBJECT_POINTS_X; ++c) {
            int col = c + HALF_NUM_SUBJECT_POINTS_X;

            // --- 座標計算 (XYZ) ---
            // 提供コード: pointsPos[idx] = ...
            cacheData.h_xyz[idx * 3 + 0] = (2.0f * (float)c + 1.0f) * HALF_SUBJECT_POINTS_PITCH_X; // X
            cacheData.h_xyz[idx * 3 + 1] = (2.0f * (float)r + 1.0f) * HALF_SUBJECT_POINTS_PITCH_Y; // Y
            cacheData.h_xyz[idx * 3 + 2] = SUBJECT_Z;                                              // Z

            // --- 色情報 (RGB) ---
            // OpenCVはBGR順なので、RGB順に入れ替えて格納
            cv::Vec3b pixel = resized_image.at<cv::Vec3b>(reverseRow, col);
            cacheData.h_rgb[idx * 3 + 0] = pixel[2]; // R
            cacheData.h_rgb[idx * 3 + 1] = pixel[1]; // G
            cacheData.h_rgb[idx * 3 + 2] = pixel[0]; // B

            idx++;
        }
    }

    std::cout << "[Benchmark] Data generated. Points: " << totalPoints << std::endl;
    return true;
}

bool BenchmarkSource::update(PointCloudData& outData) {
    // 静的データをコピーして返す
    // (ベンチマーク計測のため、あえてコピーコストを含めるか、参照だけ渡すかは設計次第ですが
    //  ここでは安全性重視でコピーします)
    outData = cacheData;
    return true;
}