#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <cuda_runtime.h> // CUDAの型(uchar4など)を使うため

// アプリケーションの設定・状態管理（Kさんが操作し、Wさんが読む）
struct AppConfig {
    // モード設定
    bool isLiveMode = true;       // true: RealSense, false: Video
    bool isAlignmentMode = false; // true: 位置合わせ操作有効

    // ディスプレイ/レンズパラメータ
    float focalLength = 0.0068f;
    float centerDistance = 0.016f; // ギャップ(m)
    
    // レンズピッチ
    float lensPitchX = 0.001f;
    float lensPitchY = 0.001f;
    
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