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

# 3. 実行ファイルの作成
add_executable(VirTraApp 
    src/main.cpp 
    src/Viewer.cpp
    src/Reconstructor.cpp    # 追加
    src/InputSource.cpp      # (まだ中身空でもファイル作っておいてください)
    src/Alignment.cpp        # (同上)
    src/gpu_kernel.cu
    src/glad.c               # gl.c から変更
)

# ヘッダーファイルの場所を指定
target_include_directories(VirTraApp PRIVATE include)

# 4. ライブラリのリンク
target_link_libraries(VirTraApp PRIVATE
    OpenGL::GL
    glfw
    glm::glm
    ${OpenCV_LIBS}
    # realsense2::realsense2
)

# シェーダフォルダをビルドディレクトリにコピーするコマンド
file(COPY ${CMAKE_SOURCE_DIR}/shaders DESTINATION ${CMAKE_BINARY_DIR})