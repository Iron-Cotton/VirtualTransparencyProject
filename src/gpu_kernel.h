#pragma once
#include "Common.h"
#include <cuda_runtime.h>

// 引数の型を cudaArray_t から uchar4* に変更
void runReconstructionKernel(PointCloudData& data, const AppConfig& config, uchar4* d_output, int width, int height);