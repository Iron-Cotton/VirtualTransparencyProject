#pragma once
#include "Common.h"
#include <cuda_runtime.h>

class CudaReconstructor {
public:
    CudaReconstructor();
    ~CudaReconstructor();

    // 初期化：解像度の設定とメモリ確保
    void initialize(int width, int height);

    // 終了処理
    void cleanup();

    // メイン処理：点群 -> 画像生成
    // input: 点群データ
    // config: パラメータ
    // glTextureID: 書き込み先のOpenGLテクスチャID
    void process(PointCloudData& input, const AppConfig& config, unsigned int glTextureID);

private:
    int width = 0;
    int height = 0;

    // OpenGLとの連携用リソースハンドル
    cudaGraphicsResource_t cudaTexRes = nullptr;

    // GPU内部で使用するボクセル配列など
    // ここに atomicAdd するためのバッファを持つ
    unsigned int* d_voxelGrid = nullptr; // 仮: 3Dボクセルを1次元配列で管理

    // 内部ヘルパー
    void mapGLTexture(cudaArray_t* outArray);
    void unmapGLTexture();
};