#pragma once
#include "Common.h"
#include <cuda_runtime.h>

// 引数にボクセルバッファ(r, g, b, cnt)を追加
void runReconstructionKernel(
    PointCloudData& data, 
    const AppConfig& config, 
    unsigned int* d_r, unsigned int* d_g, unsigned int* d_b, unsigned int* d_cnt,
    uchar4* d_output, 
    int width, int height
);