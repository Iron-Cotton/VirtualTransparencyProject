cmake_minimum_required(VERSION 3.10)

# プロジェクト名と使用言語 (CXX = C++, CUDA = GPU)
project(MyResearchApp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ソースコードを指定 (.cpp と .cu を混ぜてOK)
add_executable(ResearchApp 
    src/main.cpp 
    src/gpu_kernel.cu
)

# インクルードパスの設定 (srcフォルダ内のヘッダを見れるようにする)
target_include_directories(ResearchApp PRIVATE src)

# 必要に応じてアーキテクチャ指定 (エラーが出る場合のみコメントアウトを外す)
# set_target_properties(ResearchApp PROPERTIES CUDA_ARCHITECTURES "native")