#include "CpuOpenGlReconstructor.h"
#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>
#include <chrono> // 時間計測用
#include <glad/glad.h> // OpenGL関数用

// シェーダコード (変更なし)
const char* IP_VERT_SHADER = R"(
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
void main(){
    vUV = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

const char* IP_FRAG_SHADER = R"(
#version 330 core
out vec4 FragColor;
in vec2 vUV;

uniform sampler2DArray uSlices;
uniform int uNZ;
uniform float uElemPx;
uniform vec2  uNumLens;
uniform float uLensPitchPhy;
uniform float uDispPitch;
uniform float uFocalLen;
uniform float uCoefTrans;
uniform float uFOverP;
uniform vec2  uSliceSize;

void main() {
    vec2 fragCoord = gl_FragCoord.xy;
    vec2 lIdx = floor(fragCoord / uElemPx);

    if (lIdx.x < 0.0 || lIdx.x >= uNumLens.x || lIdx.y < 0.0 || lIdx.y >= uNumLens.y) {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec2 lensCenterWorld = (lIdx - (uNumLens - 1.0) * 0.5) * uLensPitchPhy;
    vec2 lensCenterPx = lIdx * uElemPx + uElemPx * 0.5;
    vec2 deltaPx = fragCoord - lensCenterPx;
    vec2 uv = deltaPx * uDispPitch;

    for(int nz = uNZ - 1; nz > 0; --nz) {
        float z = uCoefTrans / float(nz);
        float ratio = z / uFocalLen;
        vec2 posAtZ = lensCenterWorld - uv * ratio;
        vec2 proj = posAtZ / z;
        
        vec2 projPx = proj * uFOverP;
        ivec2 voxIdx = ivec2(floor(projPx + 0.5)) + ivec2(uSliceSize / 2.0);

        if(voxIdx.x >= 0 && voxIdx.x < int(uSliceSize.x) && voxIdx.y >= 0 && voxIdx.y < int(uSliceSize.y)) {
            vec4 c = texelFetch(uSlices, ivec3(voxIdx, nz), 0);
            if(c.a > 0.0) {
                FragColor = c;
                return;
            }
        }
    }
    FragColor = vec4(0.0, 0.0, 0.0, 1.0);
}
)";

// ★修正: 定数定義(#define)を削除
// Common.h で定義された定数 (Z_PLANE_IMG_PX_X 等) を使用します。
// ただし、計算が必要な定数(DISPLAY_PX_PITCHなど)はローカルで計算するかCommon.hの設計に合わせます。
// ここでは、計算が必要なものは process/render 内で都度計算またはConfigから取得する形とします。

static unsigned int g_ipShaderProgram = 0;

void checkShaderError(unsigned int shader, const char* type) {
    int success;
    char infoLog[1024];
    if (std::string(type) == "PROGRAM") {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "[Shader Link Error] " << infoLog << std::endl;
        }
    } else {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "[Shader Compile Error (" << type << ")] " << infoLog << std::endl;
        }
    }
}

CpuOpenGlReconstructor::CpuOpenGlReconstructor() {}
CpuOpenGlReconstructor::~CpuOpenGlReconstructor() {
    if (quadVAO) glDeleteVertexArrays(1, &quadVAO);
    if (quadVBO) glDeleteBuffers(1, &quadVBO);
    if (sliceTexture) glDeleteTextures(1, &sliceTexture);
    if (fbo) glDeleteFramebuffers(1, &fbo);
}

void CpuOpenGlReconstructor::initialize(int w, int h) {
    width = w;
    height = h;
    initGL();
}

void CpuOpenGlReconstructor::initGL() {
    unsigned int vertex, fragment;
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &IP_VERT_SHADER, NULL);
    glCompileShader(vertex);
    checkShaderError(vertex, "VERTEX");

    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &IP_FRAG_SHADER, NULL);
    glCompileShader(fragment);
    checkShaderError(fragment, "FRAGMENT");

    unsigned int prog = glCreateProgram();
    glAttachShader(prog, vertex);
    glAttachShader(prog, fragment);
    glLinkProgram(prog);
    checkShaderError(prog, "PROGRAM");
    
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    
    g_ipShaderProgram = prog;
}

void CpuOpenGlReconstructor::initQuad() {
    float quadVertices[] = {
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
}

// ★修正: timingsを受け取り、計測を行う
void CpuOpenGlReconstructor::process(const PointCloudData& input, const AppConfig& config, unsigned int targetTexture, ProcessTimings& timings) {
    
    // std::cout << "[Debug] Process start" << std::endl; // ★追加

    if (quadVAO == 0) initQuad();
    
    // Texture初期化 (初回のみ)
    if (sliceTexture == 0) {
        // std::cout << "[Debug] Init Texture..." << std::endl; // ★追加
        glGenTextures(1, &sliceTexture);
        glBindTexture(GL_TEXTURE_2D_ARRAY, sliceTexture);
        // Common.h の定数を使用
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, Z_PLANE_IMG_PX_X, Z_PLANE_IMG_PX_Y, config.numZPlane, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        // std::cout << "[Debug] Init Texture Done" << std::endl; // ★追加
    }

    // 1. ボクセル化 & 転送 (updateSlices内で計測)
    // std::cout << "[Debug] Calling updateSlices..." << std::endl; // ★追加
    updateSlices(input, config, timings);
    // std::cout << "[Debug] updateSlices returned" << std::endl; // ★追加
    // 2. レンダリング計測
    auto startRender = std::chrono::high_resolution_clock::now();
    
    // std::cout << "[Debug] Rendering..." << std::endl; // ★追加
    render(config, targetTexture);
    glFinish();
    // std::cout << "[Debug] Process end" << std::endl; // ★追加    

    auto endRender = std::chrono::high_resolution_clock::now();
    timings.rendering = std::chrono::duration<double, std::milli>(endRender - startRender).count();

    // CPU版は入力データが既にメモリにあるため転送時間は0とみなす
    timings.dataTransferH2D = 0.0;
    timings.dataTransferD2H = 0.0;
}

// ★修正: 内部で計算と転送を分けて計測
void CpuOpenGlReconstructor::updateSlices(const PointCloudData& input, const AppConfig& config, ProcessTimings& timings) {
    // --- 準備 ---
    int W = Z_PLANE_IMG_PX_X;
    int H = Z_PLANE_IMG_PX_Y;
    int D = config.numZPlane;
    int total = W * H * D;
    // std::cout << "[Debug] Voxel Total: " << total << ", Points: " << input.numPoints << std::endl; // ★追加

    // パラメータ計算 (以前のマクロ相当の値を計算)
    float displayPitch = 13.4f * 0.0254f / std::sqrt(3840.f * 3840.f + 2400.f * 2400.f); // 仮計算
    float zPlanePitch = displayPitch / (float)BOX_DETAIL_N;
    float f_over_p = config.focalLength / zPlanePitch;
    float coef_trans = (float)D * BOX_MIN_Z;
    
    int half_w = W / 2;
    int half_h = H / 2;

    // メモリ確保等は計測対象外とするか、Voxelizationに含める
    if (h_gridR.size() != total) {
        h_gridR.assign(total, 0);
        h_gridG.assign(total, 0);
        h_gridB.assign(total, 0);
        h_gridCnt.assign(total, 0);
        uploadBuffer.resize(total * 4);
    } else {
        // クリア処理も計算の一部として計測
    }

    // ==========================================
    // 計測区間1: ボクセル化 (CPU計算)
    // ==========================================
    // std::cout << "[Debug] Voting start..." << std::endl; // ★追加
    auto startVoxel = std::chrono::high_resolution_clock::now();

    // 1. ゼロクリア
    std::fill(h_gridR.begin(), h_gridR.end(), 0);
    std::fill(h_gridG.begin(), h_gridG.end(), 0);
    std::fill(h_gridB.begin(), h_gridB.end(), 0);
    std::fill(h_gridCnt.begin(), h_gridCnt.end(), 0);

    // 2. Voting
    #pragma omp parallel for
    for (int i = 0; i < input.numPoints; ++i) {
        float px = input.h_xyz[i * 3 + 0];
        float py = input.h_xyz[i * 3 + 1];
        float pz = input.h_xyz[i * 3 + 2];
        
        if (pz <= 0.0f) continue;

        float invz = 1.0f / pz;
        float xt = px * invz;
        float yt = py * invz;

        int nx = (int)floorf(f_over_p * xt + 0.5f) + half_w;
        int ny = (int)floorf(f_over_p * yt + 0.5f) + half_h;
        int nz = (int)floorf(coef_trans * invz + 0.5f);

        if (nz >= 0 && nz < D && nx >= 0 && nx < W && ny >= 0 && ny < H) {
            int idx = (nz * H + ny) * W + nx;
            #pragma omp atomic
            h_gridR[idx] += input.h_rgb[i * 3 + 0];
            #pragma omp atomic
            h_gridG[idx] += input.h_rgb[i * 3 + 1];
            #pragma omp atomic
            h_gridB[idx] += input.h_rgb[i * 3 + 2];
            #pragma omp atomic
            h_gridCnt[idx] += 1;
        }
    }

    // 3. Averaging & Buffer Packing
    // std::cout << "[Debug] Averaging start..." << std::endl; // ★追加
    #pragma omp parallel for
    for (int i = 0; i < total; ++i) {
        unsigned int cnt = h_gridCnt[i];
        if (cnt > 0) {
            uploadBuffer[i * 4 + 0] = (unsigned char)(h_gridR[i] / cnt);
            uploadBuffer[i * 4 + 1] = (unsigned char)(h_gridG[i] / cnt);
            uploadBuffer[i * 4 + 2] = (unsigned char)(h_gridB[i] / cnt);
            uploadBuffer[i * 4 + 3] = 255; 
        } else {
            *(unsigned int*)&uploadBuffer[i * 4] = 0;
        }
    }

    auto endVoxel = std::chrono::high_resolution_clock::now();
    timings.voxelization = std::chrono::duration<double, std::milli>(endVoxel - startVoxel).count();

    // ==========================================
    // 計測区間2: データ転送 (CPU -> GPU)
    // ==========================================
    // std::cout << "[Debug] Uploading texture..." << std::endl; // ★追加
    auto startUpload = std::chrono::high_resolution_clock::now();

    // std::cout << "[Debug] Uploading texture..." << std::endl; // ★追加
    glBindTexture(GL_TEXTURE_2D_ARRAY, sliceTexture);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, W, H, D, GL_RGBA, GL_UNSIGNED_BYTE, uploadBuffer.data());
    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    
    // 転送完了を確実に待つ
    glFinish();
    // std::cout << "[Debug] Upload done" << std::endl; // ★追加

    auto endUpload = std::chrono::high_resolution_clock::now();
    timings.dataTransferInter = std::chrono::duration<double, std::milli>(endUpload - startUpload).count();
}

void CpuOpenGlReconstructor::render(const AppConfig& config, unsigned int targetTexture) {
    if (g_ipShaderProgram == 0) return;

    if (fbo == 0) glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, targetTexture, 0);

    glViewport(0, 0, width, height);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(g_ipShaderProgram);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, sliceTexture);
    glUniform1i(glGetUniformLocation(g_ipShaderProgram, "uSlices"), 0);

    float displayPitch = 13.4f * 0.0254f / std::sqrt(3840.f * 3840.f + 2400.f * 2400.f); // 仮計算
    // パラメータ計算 (CUDA版と同様のロジックで設定値を優先)
    float elemPxX = (config.elemImgPxX > 0) ? (float)config.elemImgPxX : (config.lensPitchX / displayPitch);
    // float elemPxY = ... (Yも同様だがシェーダはuElemPxひとつで受けているのでXを使う想定)
    float numLensX = (float)width / elemPxX;
    float numLensY = (float)height / elemPxX;
    
    // Common.h の定数を使って計算
    float zPlanePitch = displayPitch / (float)BOX_DETAIL_N;

    glUniform1i(glGetUniformLocation(g_ipShaderProgram, "uNZ"), config.numZPlane);
    glUniform1f(glGetUniformLocation(g_ipShaderProgram, "uLensPitchPhy"), config.lensPitchX);
    glUniform1f(glGetUniformLocation(g_ipShaderProgram, "uElemPx"), elemPxX);
    glUniform2f(glGetUniformLocation(g_ipShaderProgram, "uNumLens"), numLensX, numLensY);
    glUniform1f(glGetUniformLocation(g_ipShaderProgram, "uDispPitch"), displayPitch);
    glUniform1f(glGetUniformLocation(g_ipShaderProgram, "uFocalLen"), config.focalLength);
    glUniform1f(glGetUniformLocation(g_ipShaderProgram, "uCoefTrans"), (float)config.numZPlane * BOX_MIN_Z);
    glUniform1f(glGetUniformLocation(g_ipShaderProgram, "uFOverP"), config.focalLength / zPlanePitch);
    glUniform2f(glGetUniformLocation(g_ipShaderProgram, "uSliceSize"), (float)Z_PLANE_IMG_PX_X, (float)Z_PLANE_IMG_PX_Y);

    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}