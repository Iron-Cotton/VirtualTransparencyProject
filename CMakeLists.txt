cmake_minimum_required(VERSION 3.10)
project(VirtualTransparency LANGUAGES CXX CUDA C) # C言語(glad用)も有効化

set(CMAKE_CXX_STANDARD 17)

# 1. パッケージの検索
find_package(OpenMP REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(glm REQUIRED)

# OpenCVとRealSense (前回のおさらい)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
# find_package(realsense2 REQUIRED)

# ★追加: EGLを探す (OpenGLのコンポーネントとして探すのが現代的ですが、念のためPkgConfigも用意)
find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)
# もし上記でEGLが見つからない場合(古いCMakeなど)のための保険
if(NOT TARGET OpenGL::EGL)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(EGL REQUIRED egl)
endif()

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

if(OpenMP_CXX_FOUND)
    target_link_libraries(AppBench PRIVATE OpenMP::OpenMP_CXX)
endif()

target_include_directories(AppBench PRIVATE include)
target_link_libraries(AppBench PRIVATE
    OpenGL::GL
    OpenGL::EGL  # <--- これを追加！
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}       # 画像読み込みに必要
    # RealSenseはリンクしない（依存関係を減らせる）
    glfw
    glm::glm
    dl
)

# シェーダフォルダをビルドディレクトリにコピーするコマンド
file(COPY ${CMAKE_SOURCE_DIR}/shaders DESTINATION ${CMAKE_BINARY_DIR})