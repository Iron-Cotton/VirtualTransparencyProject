#include <iostream>
#include <vector>
#include "gpu_kernel.h" // GPU関数のヘッダを読み込む

int main() {
    int N = 10000; // 要素数
    std::cout << "Running CUDA Vector Addition with " << N << " elements..." << std::endl;

    // データの準備
    std::vector<float> h_A(N, 1.0f); // 全て 1.0
    std::vector<float> h_B(N, 2.0f); // 全て 2.0
    std::vector<float> h_C(N);       // 結果を入れる箱

    // GPU処理を実行
    runVectorAdd(h_A, h_B, h_C);

    // 結果の検証 (最初の5つだけ表示)
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != 3.0f) { // 1.0 + 2.0 = 3.0 になるはず
            std::cerr << "Error at index " << i << ": " << h_C[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Success! (First 5 results: ";
        for(int i=0; i<5; i++) std::cout << h_C[i] << " ";
        std::cout << "...)" << std::endl;
    }

    return 0;
}