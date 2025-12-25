cmake_minimum_required(VERSION 3.10)

# プロジェクト名
project(MyResearchApp)

# C++のバージョン指定 (C++17推奨)
set(CMAKE_CXX_STANDARD 17)

# 実行ファイルの作成
# (src/main.cpp をコンパイルして ResearchApp というアプリを作る)
add_executable(ResearchApp src/main.cpp)