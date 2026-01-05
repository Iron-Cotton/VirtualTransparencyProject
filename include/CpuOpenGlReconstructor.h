#pragma once
#include "Common.h"
#include <glad/glad.h>
#include <vector>
#include <string>
#include "shader_m.h"

class CpuOpenGlReconstructor {
public:
    CpuOpenGlReconstructor();
    ~CpuOpenGlReconstructor();

    void initialize(int width, int height);
    
    // 入力点群からボクセルを生成(CPU)し、IP像を描画(GL)して targetTexture に書き込む
    void process(const PointCloudData& input, const AppConfig& config, unsigned int targetTexture);

private:
    int width, height;
    
    // シェーダ
    Shader* shader = nullptr;
    
    // ボクセルデータ用 3Dテクスチャ (または 2D Array)
    unsigned int sliceTexture = 0;
    
    // 描画用
    unsigned int fbo = 0;
    unsigned int quadVAO = 0, quadVBO = 0;

    // CPU側のボクセルバッファ (Voting用)
    // 3次元配列を1次元に展開して保持: [nz * H * W + y * W + x]
    std::vector<unsigned int> h_gridR;
    std::vector<unsigned int> h_gridG;
    std::vector<unsigned int> h_gridB;
    std::vector<unsigned int> h_gridCnt;
    
    // テクスチャアップロード用バッファ (RGBA8)
    std::vector<unsigned char> uploadBuffer;

    void initGL();
    void initQuad();
    
    // CPUでVotingを行いテクスチャを更新する
    void updateSlices(const PointCloudData& input, const AppConfig& config);
    
    // シェーダでレンダリングする
    void render(const AppConfig& config, unsigned int targetTexture);
};