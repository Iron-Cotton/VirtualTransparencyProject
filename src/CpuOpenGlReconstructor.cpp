#include "CpuOpenGlReconstructor.h"
#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>

// シェーダコード
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
        
        // ★修正: texture() をやめ、texelFetch() で整数座標を直接指定する
        // これにより CUDA の (int)floorf(val + 0.5f) と完全に一致させる
        vec2 projPx = proj * uFOverP;
        
        // CUDAと同じ計算式: floor(val + 0.5) + half_size
        ivec2 voxIdx = ivec2(floor(projPx + 0.5)) + ivec2(uSliceSize / 2.0);

        if(voxIdx.x >= 0 && voxIdx.x < int(uSliceSize.x) && voxIdx.y >= 0 && voxIdx.y < int(uSliceSize.y)) {
            // texelFetch は整数の絶対座標で画素を取得する
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

// 定数定義
#define Z_PLANE_IMG_PX_X 600
#define Z_PLANE_IMG_PX_Y 600
#define BOX_MIN_Z 0.2f
#define BOX_DETAIL_N 3
#define DISPLAY_PX_PITCH (13.4f * 0.0254f / sqrtf(3840.f * 3840.f + 2400.f * 2400.f))
#define Z_PLANE_IMG_PITCH (DISPLAY_PX_PITCH / (float)BOX_DETAIL_N)
#define FOCAL_LENGTH 0.0068f 

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

void CpuOpenGlReconstructor::process(const PointCloudData& input, const AppConfig& config, unsigned int targetTexture) {
    if (quadVAO == 0) initQuad();
    if (sliceTexture == 0) {
        glGenTextures(1, &sliceTexture);
        glBindTexture(GL_TEXTURE_2D_ARRAY, sliceTexture);
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, Z_PLANE_IMG_PX_X, Z_PLANE_IMG_PX_Y, config.numZPlane, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    updateSlices(input, config);
    render(config, targetTexture);
}

void CpuOpenGlReconstructor::updateSlices(const PointCloudData& input, const AppConfig& config) {
    int W = Z_PLANE_IMG_PX_X;
    int H = Z_PLANE_IMG_PX_Y;
    int D = config.numZPlane;
    int total = W * H * D;

    if (h_gridR.size() != total) {
        h_gridR.assign(total, 0);
        h_gridG.assign(total, 0);
        h_gridB.assign(total, 0);
        h_gridCnt.assign(total, 0);
        uploadBuffer.resize(total * 4);
    } else {
        std::fill(h_gridR.begin(), h_gridR.end(), 0);
        std::fill(h_gridG.begin(), h_gridG.end(), 0);
        std::fill(h_gridB.begin(), h_gridB.end(), 0);
        std::fill(h_gridCnt.begin(), h_gridCnt.end(), 0);
    }

    float f_over_p = FOCAL_LENGTH / Z_PLANE_IMG_PITCH;
    float coef_trans = (float)D * BOX_MIN_Z;
    int half_w = W / 2;
    int half_h = H / 2;

    #pragma omp parallel for
    for (int i = 0; i < input.numPoints; ++i) {
        float px = input.h_xyz[i * 3 + 0];
        float py = input.h_xyz[i * 3 + 1];
        float pz = input.h_xyz[i * 3 + 2];
        
        if (pz <= 0.0f) continue;

        float invz = 1.0f / pz;
        float xt = px * invz;
        float yt = py * invz;

        // ★統一: floorf(x + 0.5f) で四捨五入 (CPU Voting)
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

    glBindTexture(GL_TEXTURE_2D_ARRAY, sliceTexture);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, W, H, D, GL_RGBA, GL_UNSIGNED_BYTE, uploadBuffer.data());
    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
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

    float elemPx = 60.0f; 
    float numLensX = (float)width / elemPx;
    float numLensY = (float)height / elemPx;

    glUniform1i(glGetUniformLocation(g_ipShaderProgram, "uNZ"), config.numZPlane);
    glUniform1f(glGetUniformLocation(g_ipShaderProgram, "uLensPitchPhy"), config.lensPitchX);
    glUniform1f(glGetUniformLocation(g_ipShaderProgram, "uElemPx"), elemPx);
    glUniform2f(glGetUniformLocation(g_ipShaderProgram, "uNumLens"), numLensX, numLensY);
    glUniform1f(glGetUniformLocation(g_ipShaderProgram, "uDispPitch"), DISPLAY_PX_PITCH);
    glUniform1f(glGetUniformLocation(g_ipShaderProgram, "uFocalLen"), config.focalLength);
    glUniform1f(glGetUniformLocation(g_ipShaderProgram, "uCoefTrans"), (float)config.numZPlane * BOX_MIN_Z);
    glUniform1f(glGetUniformLocation(g_ipShaderProgram, "uFOverP"), FOCAL_LENGTH / Z_PLANE_IMG_PITCH);
    glUniform2f(glGetUniformLocation(g_ipShaderProgram, "uSliceSize"), (float)Z_PLANE_IMG_PX_X, (float)Z_PLANE_IMG_PX_Y);

    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}