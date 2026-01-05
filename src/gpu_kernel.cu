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
// 4. IPレンダリングカーネル (Light Field Rendering)
// ==========================================
// ディスプレイ上の画素(u,v)に対応する光線を追跡し、ボクセルを積分する
__global__ void renderIPKernel(
    unsigned int* gridR, unsigned int* gridG, unsigned int* gridB, unsigned int* gridCnt,
    int width, int height,       // 出力画像サイズ (例: 1280x720)
    float lensPitchX, float lensPitchY,
    float focalLen,              // レンズとディスプレイの距離 (gap)
    float dispPitch,             // ディスプレイの画素ピッチ
    uchar4* outputBuffer
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= width || v >= height) return;

    // 1. 物理座標の計算 (メートル単位)
    // 座標系を合わせるため、XYともに反転させる
    float metricX = -(u - width / 2.0f) * dispPitch;
    float metricY = -(v - height / 2.0f) * dispPitch;

    // 2. 所属するレンズの特定
    // レンズアレイも中央基準で配置されていると仮定
    // nx, ny はレンズのインデックス
    int lx = (int)floorf((metricX + lensPitchX/2.0f) / lensPitchX);
    int ly = (int)floorf((metricY + lensPitchY/2.0f) / lensPitchY);

    // レンズ中心の座標
    float lensCenterX = lx * lensPitchX;
    float lensCenterY = ly * lensPitchY;

    // 3. ボクセル空間への逆投影 (Ray Casting / Back-projection)
    // ディスプレイ画素(metricX)からレンズ中心(lensCenterX)を通る光線を考える
    // 光線の方程式: X(z) = metricX + (lensCenterX - metricX) * (z / focalLen)
    // ここで z はディスプレイ面からの距離。ボクセル空間の座標系に合わせる必要がある。

    float accumR = 0.0f;
    float accumG = 0.0f;
    float accumB = 0.0f;
    float accumA = 0.0f;

    // 定数再定義 (カーネル内計算用)
    float f_over_p = F_OVER_Z_PLANE_PITCH; 
    
    // 全てのスライス(Z平面)を走査して積分
    for (int nz = 0; nz < NUM_Z_PLANE; ++nz) {
        // このスライスに対応する物理深度 Z を逆算
        // nz = COEF_TRANS * (1/z)  =>  z = COEF_TRANS / nz
        // ゼロ除算回避
        if (nz == 0) continue; 
        float z = COEF_TRANS / (float)nz;

        // 光線とスライス(深度z)の交点 X_obj を計算
        // X_obj = LensCenter + (LensCenter - Pixel) * (z / gap)
        // ※ 符号は座標系によるが、一般的に「画素位置と逆方向」に投影される
        float ratio = z / focalLen;
        float x_obj = lensCenterX + (lensCenterX - metricX) * ratio;
        float y_obj = lensCenterY + (lensCenterY - metricY) * ratio;

        // ボクセルインデックス(nx, ny)に変換
        // VotingKernelの逆: nx = (x/z) * F_OVER_P + HALF_W
        // ここでは x_obj が既に (x) なので、Votingのロジックに合わせるなら:
        // Voting: xt = x/z.  Here: xt = x_obj / z.
        
        float xt = x_obj / z;
        float yt = y_obj / z;

        int voxX = (int)lroundf(f_over_p * xt) + HALF_Z_PLANE_IMG_PX_X;
        int voxY = (int)lroundf(f_over_p * yt) + HALF_Z_PLANE_IMG_PX_Y;

        // 範囲内なら色を加算
        if (voxX >= 0 && voxX < Z_PLANE_IMG_PX_X && voxY >= 0 && voxY < Z_PLANE_IMG_PX_Y) {
            int vIdx = (nz * Z_PLANE_IMG_PX_Y + voxY) * Z_PLANE_IMG_PX_X + voxX;
            unsigned int cnt = gridCnt[vIdx];
            
            if (cnt > 0) {
                // ボクセルの平均色を取得
                // 単純加算(Additive)で合成
                accumR += (float)gridR[vIdx] / cnt;
                accumG += (float)gridG[vIdx] / cnt;
                accumB += (float)gridB[vIdx] / cnt;
                accumA += 1.0f;
            }
        }
    }

    // 4. 色の書き込み
    // 蓄積した値を適当にスケールして表示（正規化しないと真っ白になるかも）
    // 今回は単純に「重なった数」で割るか、あるいは最大値でクリップするか
    if (accumA > 0.0f) {
        // 少し明るめに調整
        float scale = 1.0f; 
        uchar4 c;
        c.x = (unsigned char)fminf(accumR * scale, 255.0f);
        c.y = (unsigned char)fminf(accumG * scale, 255.0f);
        c.z = (unsigned char)fminf(accumB * scale, 255.0f);
        c.w = 255;
        outputBuffer[v * width + u] = c;
    } else {
        outputBuffer[v * width + u] = make_uchar4(0, 0, 0, 255);
    }
}

// ==========================================
// ラッパー関数の更新
// ==========================================
void runReconstructionKernel(
    PointCloudData& data, 
    const AppConfig& config, // ここからパラメータをもらう
    unsigned int* d_r, unsigned int* d_g, unsigned int* d_b, unsigned int* d_cnt,
    uchar4* d_output, 
    int width, int height
) {
    // 1-3. クリア & Voting (変更なし)
    int totalVoxels = Z_PLANE_IMG_PX_X * Z_PLANE_IMG_PX_Y * NUM_Z_PLANE;
    int blockSize = 256;
    int gridSize = (totalVoxels + blockSize - 1) / blockSize;
    clearVoxelKernel<<<gridSize, blockSize>>>(d_r, d_g, d_b, d_cnt, totalVoxels);
    cudaDeviceSynchronize();

    if (data.numPoints > 0) {
        int ptBlock = 256;
        int ptGrid = (data.numPoints + ptBlock - 1) / ptBlock;
        votingKernel<<<ptGrid, ptBlock>>>(data.numPoints, data.d_xyz, data.d_rgb, d_r, d_g, d_b, d_cnt);
        cudaDeviceSynchronize();
    }

    // 4. IPレンダリング (可視化カーネルから差し替え)
    dim3 renderBlock(16, 16);
    dim3 renderGrid((width + renderBlock.x - 1) / renderBlock.x, (height + renderBlock.y - 1) / renderBlock.y);

    // パラメータ設定 (config から取得、無ければデフォルト値)
    // ベンチマーク設定に合わせる
    float lensPitchX = config.lensPitchX > 0 ? config.lensPitchX : 0.001f; // 仮
    float lensPitchY = config.lensPitchY > 0 ? config.lensPitchY : 0.001f; // 仮
    float focalLen = config.centerDistance > 0 ? config.centerDistance : 0.016f; // gap
    
    // 定数定義マクロを利用
    float dispPitch = DISPLAY_PX_PITCH;

    renderIPKernel<<<renderGrid, renderBlock>>>(
        d_r, d_g, d_b, d_cnt,
        width, height,
        lensPitchX, lensPitchY,
        focalLen,
        dispPitch,
        d_output
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA Error: %s\n", cudaGetErrorString(err));
}