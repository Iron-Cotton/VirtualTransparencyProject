#pragma once
#include <vector>
#include <string>
#include "InputSource.h"
#include "Common.h" // AppConfig, PointCloudData, ProcessTimings の定義が必要

class CpuOpenGlReconstructor {
public:
    CpuOpenGlReconstructor();
    ~CpuOpenGlReconstructor();

    void initialize(int width, int height);
    
    // ★修正: timingsを追加
    void process(const PointCloudData& input, const AppConfig& config, unsigned int targetTexture, ProcessTimings& timings);

private:
    void initGL();
    void initQuad();
    
    // ★修正: timingsを追加 (内部で計算と転送を分けて計測するため)
    void updateSlices(const PointCloudData& input, const AppConfig& config, ProcessTimings& timings);
    
    void render(const AppConfig& config, unsigned int targetTexture);

    int width, height;
    unsigned int sliceTexture = 0;
    unsigned int fbo = 0;
    unsigned int quadVAO = 0, quadVBO = 0;

    // ボクセルデータ (Host側)
    std::vector<float> h_gridR;
    std::vector<float> h_gridG;
    std::vector<float> h_gridB;
    std::vector<unsigned int> h_gridCnt;
    std::vector<unsigned char> uploadBuffer;
};