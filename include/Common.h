#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <cuda_runtime.h> // CUDAの型(uchar4など)を使うため

// --- 定数定義 (CPU/GPU共通) ---
#define Z_PLANE_IMG_PX_X 600
#define Z_PLANE_IMG_PX_Y 480
#define BOX_MIN_Z 0.2f
#define BOX_DETAIL_N 3

// ★追加: 処理時間の内訳 (単位: ms)
struct ProcessTimings {
    double totalTime = 0.0;      // 全体
    double dataTransferH2D = 0.0;// Host -> Device (点群転送)
    double voxelization = 0.0;   // ボクセル化 (Voting/Binning)
    double dataTransferInter = 0.0; // 中間転送 (CPU計算結果のTextureアップロードなど)
    double rendering = 0.0;      // レンダリング (Ray Casting)
    double dataTransferD2H = 0.0;// Device -> Host (描画結果の書き戻し)
};

// アプリケーションの設定・状態管理（Kさんが操作し、Wさんが読む）
struct AppConfig {
    // モード設定
    bool isLiveMode = true;       // true: RealSense, false: Video
    bool isAlignmentMode = false; // true: 位置合わせ操作有効

    // ディスプレイ/レンズパラメータ
    float focalLength = 0.0068f; // ギャップ(m)

    // レンズ数
    int numLensX = 128;
    int numLensY = 80;

    // 要素画像の解像度
    int elemImgPxX = 60;
    int elemImgPxY = 60;
    
    // レンズピッチ
    float lensPitchX = 0.00454f;
    float lensPitchY = 0.00454f;
    
    // UVスケール補正
    float uvScaleX = 1.0f;
    float uvScaleY = 1.0f;
    float frustumUVShiftX = 0.0f;
    float frustumUVShiftY = 0.0f;

    // ボクセル空間パラメータ
    int numZPlane = 60;
    
    // 点群補正用オフセット
    glm::vec3 pointCloudOffset = {0.0f, 0.0f, 0.0f};
};

// 点群データのコンテナ（CPU/GPU転送用）
struct PointCloudData {
    int numPoints; // 有効な点群数
    
    // Host側データ (CPUメモリ)
    // std::vectorよりもポインタ管理の方がCUDAとの相性が良いが、
    // ここでは簡便のためvectorを使用し、転送時にdata()を取得する
    std::vector<float> h_xyz; // x, y, z が一直線に並んだ配列
    std::vector<unsigned char> h_rgb; // r, g, b が一直線に並んだ配列

    // Device側データ (GPUメモリ - Wさんが管理)
    float* d_xyz = nullptr;
    unsigned char* d_rgb = nullptr;
};