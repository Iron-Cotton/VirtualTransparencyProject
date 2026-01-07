#pragma once
#include "Common.h" // ProcessTimingsのために必要
#include <cuda_runtime.h>

// カーネルラッパー関数
void runReconstructionKernel(
    PointCloudData& data, 
    const AppConfig& config, 
    unsigned int* d_r, unsigned int* d_g, unsigned int* d_b, unsigned int* d_cnt,
    uchar4* d_output, 
    int width, int height,
    ProcessTimings& timings // ★追加
);