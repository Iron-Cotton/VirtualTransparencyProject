#pragma once
#include "Common.h"

// インターフェース
class IInputSource {
public:
    virtual ~IInputSource() = default;
    virtual bool initialize() = 0;
    // データを取得し、Host側の配列を埋める関数
    virtual bool update(PointCloudData& outData) = 0;
};

// リアルタイム版
class RealSenseSource : public IInputSource {
    // rs2::pipeline などを保持
public:
    bool initialize() override;
    bool update(PointCloudData& outData) override;
};

// 録画ファイル版
class FileSource : public IInputSource {
    // cv::VideoCapture などを保持
public:
    bool initialize() override;
    bool update(PointCloudData& outData) override;
};