#include "gpu_kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>

// ==========================================
// 定数定義 (ベンチマークコードより移植)
// ==========================================
// ※ GPU内で使うため定数として定義
#define Z_PLANE_IMG_PX_X 600
#define Z_PLANE_IMG_PX_Y 480
#define NUM_Z_PLANE 60

#define BOX_DETAIL_N 3
#define HALF_SEARCH_BOX_SIZE (BOX_DETAIL_N / 2)

// 物理定数
#define DISPLAY_PX_PITCH (13.4f * 0.0254f / sqrtf(3840.f * 3840.f + 2400.f * 2400.f))
#define Z_PLANE_IMG_PITCH (DISPLAY_PX_PITCH / (float)BOX_DETAIL_N)
#define FOCAL_LENGTH 0.0068f // 仮の焦点距離 (v2-3-2-trialより)
#define BOX_MIN_Z 0.2f
#define COEF_TRANS (((float)NUM_Z_PLANE - 0.0f) * BOX_MIN_Z)

#define F_OVER_Z_PLANE_PITCH (FOCAL_LENGTH / Z_PLANE_IMG_PITCH)
#define HALF_Z_PLANE_IMG_PX_X (Z_PLANE_IMG_PX_X / 2)
#define HALF_Z_PLANE_IMG_PX_Y (Z_PLANE_IMG_PX_Y / 2)

// ==========================================
// カーネル関数
// ==========================================

// 1. ボクセルグリッドのクリア
__global__ void clearVoxelKernel(
    unsigned int* r, unsigned int* g, unsigned int* b, unsigned int* cnt,
    int totalVoxels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalVoxels) {
        r[idx] = 0;
        g[idx] = 0;
        b[idx] = 0;
        cnt[idx] = 0;
    }
}

// 2. ビニング（投票）カーネル
// 各点が「自分がどこのボクセルに属するか」を計算し、投票する
__global__ void votingKernel(
    int numPoints,
    const float* xyz,
    const unsigned char* rgb,
    unsigned int* gridR,
    unsigned int* gridG,
    unsigned int* gridB,
    unsigned int* gridCnt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    // 座標読み込み
    float px = xyz[idx * 3 + 0];
    float py = xyz[idx * 3 + 1];
    float pz = xyz[idx * 3 + 2];
    
    // 色読み込み
    unsigned char pr = rgb[idx * 3 + 0];
    unsigned char pg = rgb[idx * 3 + 1];
    unsigned char pb = rgb[idx * 3 + 2];

    if (pz <= 0.0f) return; // 無効な点

    // --- 座標変換ロジック (CPU版 binningPointClouds と同一) ---
    float invz = 1.0f / pz;
    float xt = px * invz;
    float yt = py * invz;

    // ボクセルインデックス計算
    int nx = (int)lroundf(F_OVER_Z_PLANE_PITCH * xt) + HALF_Z_PLANE_IMG_PX_X;
    int ny = (int)lroundf(F_OVER_Z_PLANE_PITCH * yt) + HALF_Z_PLANE_IMG_PX_Y;
    
    // Z方向のインデックス計算 (1/z に比例)
    int nz = (int)floorf(COEF_TRANS * invz + 0.5f);

    // 範囲チェック & 投票
    // ※ 周辺ボクセルへの拡散(Anti-aliasing)も含む
    if (nz >= 0 && nz < NUM_Z_PLANE) {
        for (int m = -HALF_SEARCH_BOX_SIZE; m <= HALF_SEARCH_BOX_SIZE; ++m) {
            for (int n = -HALF_SEARCH_BOX_SIZE; n <= HALF_SEARCH_BOX_SIZE; ++n) {
                
                int x = nx + n;
                int y = ny + m;

                if (x >= 0 && x < Z_PLANE_IMG_PX_X && y >= 0 && y < Z_PLANE_IMG_PX_Y) {
                    // 1次元インデックスへ変換
                    int voxelIdx = (nz * Z_PLANE_IMG_PX_Y + y) * Z_PLANE_IMG_PX_X + x;

                    // アトミック加算 (競合を防ぎつつ加算)
                    atomicAdd(&gridR[voxelIdx], (unsigned int)pr);
                    atomicAdd(&gridG[voxelIdx], (unsigned int)pg);
                    atomicAdd(&gridB[voxelIdx], (unsigned int)pb);
                    atomicAdd(&gridCnt[voxelIdx], 1);
                }
            }
        }
    }
}

// 3. スライス可視化カーネル
// 3次元ボクセルの中から、指定したZ平面(sliceIndex)を切り出して画像にする
__global__ void sliceVisualizationKernel(
    unsigned int* gridR,
    unsigned int* gridG,
    unsigned int* gridB,
    unsigned int* gridCnt,
    int sliceIndex,
    int width, int height, // 出力画像のサイズ
    uchar4* outputBuffer
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // 出力画像のX
    int y = blockIdx.y * blockDim.y + threadIdx.y; // 出力画像のY

    if (x >= width || y >= height) return;

    // 出力画像の中央に、ボクセル画像(600x480)を表示するためのオフセット計算
    int voxW = Z_PLANE_IMG_PX_X;
    int voxH = Z_PLANE_IMG_PX_Y;
    
    int offX = (width - voxW) / 2;
    int offY = (height - voxH) / 2;

    int vx = x - offX;
    int vy = y - offY;

    // ボクセル範囲外なら黒
    if (vx < 0 || vx >= voxW || vy < 0 || vy >= voxH) {
        outputBuffer[y * width + x] = make_uchar4(0, 0, 0, 255);
        return;
    }

    // ボクセル取得
    int voxelIdx = (sliceIndex * voxH + vy) * voxW + vx;
    unsigned int count = gridCnt[voxelIdx];

    if (count > 0) {
        // 平均化 (Accum / Count)
        unsigned char r = (unsigned char)(gridR[voxelIdx] / count);
        unsigned char g = (unsigned char)(gridG[voxelIdx] / count);
        unsigned char b = (unsigned char)(gridB[voxelIdx] / count);
        outputBuffer[y * width + x] = make_uchar4(r, g, b, 255);
    } else {
        // データがないボクセルは黒
        outputBuffer[y * width + x] = make_uchar4(0, 0, 0, 255);
    }
}

// ==========================================
// 4. IPレンダリングカーネル (修正版)
// OpenGLシェーダーのロジックを忠実に再現する
// ==========================================
__global__ void renderIPKernel(
    unsigned int* gridR, unsigned int* gridG, unsigned int* gridB, unsigned int* gridCnt,
    int width, int height,
    float lensPitchX, float lensPitchY,
    float focalLen,
    float dispPitch,
    uchar4* outputBuffer,
    float numLensX, float numLensY,  // 追加: レンズの総数
    float elemPxX, float elemPxY,    // 追加: レンズ1つあたりのピクセル数
    float invFocalLen
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= width || v >= height) return;

    // --- OpenGL Shader Logicの移植 ---

    // 1. レンズインデックスの特定 (Pixel Grid Base)
    // シェーダ: lIdx = floor(fragCoord / uElemPx);
    float lx = floorf((float)u / elemPxX);
    float ly = floorf((float)v / elemPxY);

    // 範囲外チェック
    if (lx < 0.0f || lx >= numLensX || ly < 0.0f || ly >= numLensY) {
        outputBuffer[v * width + u] = make_uchar4(0, 0, 0, 255);
        return;
    }

    // 2. レンズ中心(World)の計算
    // シェーダ: lensCenterWorld = (lIdx - (uNumLens - 1.0) * 0.5) * uLensPitchPhy;
    // これにより、レンズアレイ全体が原点を中心に配置されます
    float lensCenterWorldX = (lx - (numLensX - 1.0f) * 0.5f) * lensPitchX;
    float lensCenterWorldY = (ly - (numLensY - 1.0f) * 0.5f) * lensPitchY;

    // 3. ピクセルオフセット(UV)の計算
    // シェーダ: lensCenterPx = lIdx * uElemPx + uElemPx * 0.5;
    // シェーダ: deltaPx = fragCoord - lensCenterPx;
    // ※ fragCoord は画素中心(0.5, 1.5...) を指すが、CUDAのuは整数(0, 1...)
    //    厳密に合わせるなら u + 0.5f だが、ここでは相対位置が重要。
    //    CPU版と完全に合わせるため、画素中心として +0.5f を考慮する
    float currentPxX = (float)u + 0.5f;
    float currentPxY = (float)v + 0.5f;

    float lensCenterPxX = lx * elemPxX + elemPxX * 0.5f;
    float lensCenterPxY = ly * elemPxY + elemPxY * 0.5f;

    float deltaPxX = currentPxX - lensCenterPxX;
    float deltaPxY = currentPxY - lensCenterPxY;

    // シェーダ: vec2 uv = deltaPx * uDispPitch;
    float uvX = deltaPxX * dispPitch;
    float uvY = deltaPxY * dispPitch;

    // 定数準備
    float f_over_p = F_OVER_Z_PLANE_PITCH; 

    bool hit = false;
    uchar4 finalColor = make_uchar4(0, 0, 0, 255);

    // 4. ボクセル空間への逆投影 (Ray Casting)
    for (int nz = NUM_Z_PLANE - 1; nz > 0; --nz) {
        
        // 深度 Z の計算
        float z = COEF_TRANS / (float)nz;

        // 幾何学計算 (相似比)
        float ratio = z * invFocalLen;
        
        // シェーダ: vec2 posAtZ = lensCenterWorld - uv * ratio;
        // ※ピンホールカメラモデルのため、変位(uv)を反転させて投影面にマッピングする
        float x_obj = lensCenterWorldX - uvX * ratio;
        float y_obj = lensCenterWorldY - uvY * ratio;

        // ボクセルインデックス計算
        float xt = x_obj / z;
        float yt = y_obj / z;

        int voxX = (int)lroundf(f_over_p * xt) + HALF_Z_PLANE_IMG_PX_X;
        int voxY = (int)lroundf(f_over_p * yt) + HALF_Z_PLANE_IMG_PX_Y;

        // 範囲チェック
        if ((unsigned int)voxX < Z_PLANE_IMG_PX_X && (unsigned int)voxY < Z_PLANE_IMG_PX_Y) {
            
            int vIdx = (nz * Z_PLANE_IMG_PX_Y + voxY) * Z_PLANE_IMG_PX_X + voxX;
            unsigned int cnt = gridCnt[vIdx];
            
            if (cnt > 0) {
                unsigned char r = (unsigned char)(gridR[vIdx] / cnt);
                unsigned char g = (unsigned char)(gridG[vIdx] / cnt);
                unsigned char b = (unsigned char)(gridB[vIdx] / cnt);
                finalColor = make_uchar4(r, g, b, 255);
                hit = true;
                break;
            }
        }
    }
    outputBuffer[v * width + u] = finalColor;
}

// ラッパー関数の実装
void runReconstructionKernel(
    PointCloudData& data, 
    const AppConfig& config, 
    unsigned int* d_r, unsigned int* d_g, unsigned int* d_b, unsigned int* d_cnt,
    uchar4* d_output, 
    int width, int height,
    ProcessTimings& timings // ★追加
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    // --- 1. Voxelization (Clear + Voting) ---
    cudaEventRecord(start);

    int totalVoxels = Z_PLANE_IMG_PX_X * Z_PLANE_IMG_PX_Y * config.numZPlane;
    int blockSize = 256;
    int gridSize = (totalVoxels + blockSize - 1) / blockSize;
    
    // ボクセルクリア
    clearVoxelKernel<<<gridSize, blockSize>>>(d_r, d_g, d_b, d_cnt, totalVoxels);
    
    // 投票 (点群がある場合)
    if (data.numPoints > 0) {
        int ptBlock = 256;
        int ptGrid = (data.numPoints + ptBlock - 1) / ptBlock;
        votingKernel<<<ptGrid, ptBlock>>>(data.numPoints, data.d_xyz, data.d_rgb, d_r, d_g, d_b, d_cnt);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    timings.voxelization = ms; // 計測結果を格納

    // --- 2. Rendering ---
    cudaEventRecord(start);

    dim3 renderBlock(16, 16);
    dim3 renderGrid((width + renderBlock.x - 1) / renderBlock.x, (height + renderBlock.y - 1) / renderBlock.y);

    float dispPitch = 13.4f * 0.0254f / sqrtf(3840.f * 3840.f + 2400.f * 2400.f); // 簡易計算
    float elemPxX = (config.elemImgPxX > 0) ? (float)config.elemImgPxX : (config.lensPitchX / dispPitch);
    float elemPxY = (config.elemImgPxY > 0) ? (float)config.elemImgPxY : (config.lensPitchY / dispPitch);
    float numLensX = (float)width / elemPxX;
    float numLensY = (float)height / elemPxY;
    float invFocalLen = 1.0f / config.focalLength;

    renderIPKernel<<<renderGrid, renderBlock>>>(
        d_r, d_g, d_b, d_cnt,
        width, height,
        config.lensPitchX, config.lensPitchY,
        config.focalLength,
        dispPitch,
        d_output,
        numLensX, numLensY,
        elemPxX, elemPxY,
        invFocalLen
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    timings.rendering = ms; // 計測結果を格納

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // エラーチェック
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA Error: %s\n", cudaGetErrorString(err));
}