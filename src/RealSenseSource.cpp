#include "../include/RealSenseSource.h"
#include <iostream>
#include <algorithm>

RealSenseSource::RealSenseSource() : align_to_color(RS2_STREAM_COLOR) {
    // コンストラクタ
}

RealSenseSource::~RealSenseSource() {
    // パイプラインはデストラクタで自動的に停止しますが、明示的に止めても良い
    try {
        pipe.stop();
    } catch (...) {}
}

bool RealSenseSource::initialize() {
    std::cout << "[RealSense] Initializing..." << std::endl;

    try {
        // 設定 (PCSJ2025-OpenGL-v2-3-2-trial.cpp より)
        cfg.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_BGR8, FPS);
        cfg.enable_stream(RS2_STREAM_DEPTH, WIDTH, HEIGHT, RS2_FORMAT_Z16, FPS);

        // パイプライン開始
        rs2::pipeline_profile profile = pipe.start(cfg);

        // センサ設定 (露出やレーザーパワーなど)
        auto sensors = profile.get_device().query_sensors();
        if (sensors.size() >= 2) {
            auto depthSensor = sensors[0];
            auto colorSensor = sensors[1];

            // Color: 自動露出OFF, 固定値設定
            colorSensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0);
            colorSensor.set_option(RS2_OPTION_EXPOSURE, 150);
            colorSensor.set_option(RS2_OPTION_GAIN, 64);

            // Depth: 自動露出ON, レーザーパワー設定
            depthSensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1);
            depthSensor.set_option(RS2_OPTION_LASER_POWER, 360);
            
            std::cout << "[RealSense] Sensor options configured." << std::endl;
        }

        return true;

    } catch (const rs2::error& e) {
        std::cerr << "[RealSense] Error calling " << e.get_failed_function() 
                  << "(" << e.get_failed_args() << "): " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[RealSense] Exception: " << e.what() << std::endl;
        return false;
    }
}

bool RealSenseSource::update(PointCloudData& outData) {
    try {
        // 1. フレーム待ち受け
        rs2::frameset frames = pipe.wait_for_frames();

        // 2. 位置合わせ (DepthをColorに合わせる)
        frames = align_to_color.process(frames);

        rs2::video_frame color_frame = frames.get_color_frame();
        rs2::depth_frame depth_frame = frames.get_depth_frame();

        if (!color_frame || !depth_frame) return false;

        // 3. 点群生成
        // Depthフレームから頂点座標を計算
        points = pc.calculate(depth_frame);
        
        // テクスチャ座標のマッピング（今回は直接カラー画像を参照するため必須ではないが、SDKの仕様上呼んでおく）
        pc.map_to(color_frame);

        // 4. データ変換 (SDK形式 -> アプリ用構造体)
        auto vertices = points.get_vertices();
        size_t num_vertices = points.size(); // 通常は WIDTH * HEIGHT

        // カラー画像データへのポインタ (BGR8形式)
        const uint8_t* color_data = reinterpret_cast<const uint8_t*>(color_frame.get_data());

        // 出力バッファの準備
        // 無効な点はスキップするため、最大サイズで確保して後で縮小するか、
        // 単純に全点確保して無効フラグで管理するかですが、
        // ここでは trial.cpp のロジック「z <= 0.0f continue」に従い、有効な点だけを詰めます。
        
        outData.h_xyz.clear();
        outData.h_rgb.clear();
        outData.h_xyz.reserve(num_vertices * 3);
        outData.h_rgb.reserve(num_vertices * 3);

        int validCount = 0;

        for (size_t i = 0; i < num_vertices; i++) {
            // 座標変換 (Trialコードのロジックを適用)
            // float tmp_pcd_x = v.x + diffX;         <-- diffXはReconstructor側でやるかここでやるか要検討
            // float tmp_pcd_y = -v.y + diffHeight;   <-- 上下反転と高さ調整
            // float tmp_pcd_z = v.z + distCameraDisplay;
            
            float x = vertices[i].x;
            float y = vertices[i].y;
            float z = vertices[i].z;

            // 座標変換（カメラ位置 -> ディスプレイ基準位置）
            // ※ Trialコードでは「Left/Right」で異なる diffX を足していますが、
            //    Sourceとしては「カメラ基準」または「ディスプレイ中央基準」で渡すのが自然です。
            //    ここでは「ディスプレイ中央基準(Commonパラメータ適用)」に変換します。
            
            float pcd_x = x + diffX;
            float pcd_y = -y + diffHeight; // Y軸反転 + 高さ調整
            float pcd_z = z + distCameraDisplay;

            // 無効な深度、あるいは近すぎる/裏側の点をスキップ
            if (pcd_z <= 0.0f) continue;

            // XYZ格納
            outData.h_xyz.push_back(pcd_x);
            outData.h_xyz.push_back(pcd_y);
            outData.h_xyz.push_back(pcd_z);

            // RGB格納 (BGR -> RGB)
            // i番目のピクセル: color_data[i*3 + 0/1/2]
            int idx = i * 3;
            outData.h_rgb.push_back(color_data[idx + 2]); // R
            outData.h_rgb.push_back(color_data[idx + 1]); // G
            outData.h_rgb.push_back(color_data[idx + 0]); // B

            validCount++;
        }

        outData.numPoints = validCount;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[RealSense] Update failed: " << e.what() << std::endl;
        return false;
    }
}