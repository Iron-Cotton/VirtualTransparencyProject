#pragma once
#include "InputSource.h"
#include <opencv2/opencv.hpp>
#include <string>

class BenchmarkSource : public InputSource {
public:
    BenchmarkSource();
    ~BenchmarkSource();

    // 画像を読み込み、点群データを生成してメモリにキャッシュする
    bool initialize() override;

    // キャッシュしたデータをoutDataにコピーする（静止画なので毎回同じ）
    bool update(PointCloudData& outData) override;

private:
    // 読み込んだ点群データを保持しておく変数
    PointCloudData cacheData;

    // 定数（提供されたコードに基づく）
    const std::string IMAGE_PATH = "../images/standard/Parrots.bmp";
    const int NUM_SUBJECT_POINTS_X = 554;
    const int NUM_SUBJECT_POINTS_Y = 554;
    const float SUBJECT_Z = 1.0f;
};