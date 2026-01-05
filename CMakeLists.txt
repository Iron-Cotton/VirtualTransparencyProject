cmake_minimum_required(VERSION 3.10)
project(VirtualTransparency LANGUAGES CXX CUDA C) # C言語(glad用)も有効化

set(CMAKE_CXX_STANDARD 17)

# 1. パッケージの検索
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(glm REQUIRED)

# OpenCVとRealSense (前回のおさらい)
find_package(OpenCV REQUIRED)
# find_package(realsense2 REQUIRED)

# 2. GLADの場所を指定 (includeフォルダにヘッダがある場合)
include_directories(include)

# 共通のソースファイル群
set(COMMON_SOURCES
    src/Viewer.cpp
    src/Reconstructor.cpp
    src/gpu_kernel.cu
    src/glad.c
    src/Alignment.cpp
    src/InputSource.cpp
)

# -------------------------------------------------------
# 1. 本番用アプリ (Production)
# -------------------------------------------------------
add_executable(AppProd
    src/main_prod.cpp
    src/RealSenseSource.cpp
    ${COMMON_SOURCES}
)
target_include_directories(AppProd PRIVATE include)
target_link_libraries(AppProd PRIVATE
    OpenGL::GL
    realsense2           # RealSenseは本番のみ必須
    ${OpenCV_LIBS}       # OpenCVも必要
    glfw
    glm::glm
)

# -------------------------------------------------------
# 2. ベンチマーク用アプリ (Benchmark)
# -------------------------------------------------------
add_executable(AppBench
    src/main_bench.cpp
    src/BenchmarkSource.cpp
    src/CpuOpenGlReconstructor.cpp
    ${COMMON_SOURCES}
)
target_include_directories(AppBench PRIVATE include)
target_link_libraries(AppBench PRIVATE
    OpenGL::GL
    ${OpenCV_LIBS}       # 画像読み込みに必要
    # RealSenseはリンクしない（依存関係を減らせる）
    glfw
    glm::glm
)

# シェーダフォルダをビルドディレクトリにコピーするコマンド
file(COPY ${CMAKE_SOURCE_DIR}/shaders DESTINATION ${CMAKE_BINARY_DIR})