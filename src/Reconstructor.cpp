#include "Reconstructor.h"
#include <iostream>
#include <vector>
#include <glad/glad.h>
#include "gpu_kernel.h"

// グローバル変数
static uchar4* d_outputBuffer = nullptr;        // GPU側
static std::vector<uchar4> h_outputBuffer;      // CPU側

CudaReconstructor::CudaReconstructor() {}
CudaReconstructor::~CudaReconstructor() { cleanup(); }

void CudaReconstructor::initialize(int w, int h) {
    width = w;
    height = h;

    size_t size = width * height * sizeof(uchar4);
    cudaMalloc(&d_outputBuffer, size);
    h_outputBuffer.resize(width * height);
    
    // 初期値
    std::fill(h_outputBuffer.begin(), h_outputBuffer.end(), uchar4{255, 0, 0, 255}); // 赤
}

void CudaReconstructor::cleanup() {
    if (d_outputBuffer != nullptr) {
        cudaFree(d_outputBuffer);
        d_outputBuffer = nullptr;
    }
}

void CudaReconstructor::process(PointCloudData& input, const AppConfig& config, unsigned int glTextureID) {
    if (!d_outputBuffer) return;

    // 1. カーネル実行
    runReconstructionKernel(input, config, d_outputBuffer, width, height);
    
    // 2. GPU -> CPU 転送
    cudaMemcpy(h_outputBuffer.data(), d_outputBuffer, 
               width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);

    // 3. OpenGLテクスチャ更新（ここを変更！）
    // SubImage (部分更新) ではなく、glTexImage2D (全確保＆転送) を使う
    // これなら「初期化されてない」というエラーは絶対に起きない
    glBindTexture(GL_TEXTURE_2D, glTextureID);
    
    // おまじない：ピクセルの整列設定を1バイト単位にする（ズレ防止）
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_outputBuffer.data());
    
    // エラーチェック
    GLenum glErr = glGetError();
    if (glErr != GL_NO_ERROR) {
        // もしこれでもエラーならログを出す（が、恐らく消えるはず）
        static bool printed = false;
        if (!printed) {
            std::cerr << "[GL Critical] Update failed: " << glErr << std::endl;
            printed = true;
        }
    }
    
    glBindTexture(GL_TEXTURE_2D, 0);
}