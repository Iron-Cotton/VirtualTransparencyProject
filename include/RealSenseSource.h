#pragma once
#include "InputSource.h"
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp> // カラー画像の形式変換にOpenCVを使う場合

class RealSenseSource : public InputSource {
public:
    RealSenseSource();
    ~RealSenseSource();

    // RealSenseのパイプラインを開始する
    bool initialize() override;

    // 最新のフレームを待ち受け、点群に変換してoutDataに詰める
    bool update(PointCloudData& outData) override;

private:
    // RealSense制御用オブジェクト
    rs2::pipeline pipe;
    rs2::config cfg;
    rs2::pointcloud pc;
    rs2::points points;
    rs2::align align_to_color; // 深度とカラーの位置合わせ用

    // 定数（提供されたコードに基づく）
    const int WIDTH = 640;
    const int HEIGHT = 480;
    const int FPS = 30;

    // 本番用のキャリブレーションパラメータ（必要に応じて）
    float diffX = 0.0f;
    float diffHeight = -0.06f; // DIFF_HEIGHT_CAMERA_DISPLAY
    float distCameraDisplay = 0.157f; // DISTANCE_CAMERA_DISPLAY
};