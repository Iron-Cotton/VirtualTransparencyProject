#include "RealSenseSource.h"
#include <librealsense2/rs.hpp>

// メンバ変数として pipe, config 等を持つ
bool RealSenseSource::initialize() {
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    pipe.start(cfg);
    // ... センサ設定など ...
    return true;
}

bool RealSenseSource::update(PointCloudData& outData) {
    rs2::frameset frames = pipe.wait_for_frames();
    // ... align処理 ...
    // ... pointcloud計算 ...
    
    // PointCloudData への変換
    // RealSenseの頂点配列(rs2::vertex*)から h_xyz, h_rgb へコピー
    // ...
    return true;
}