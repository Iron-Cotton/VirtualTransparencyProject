#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>
#include "shader_m.h" // 既存のシェーダクラスを利用

class GLViewer {
public:
    GLViewer();
    ~GLViewer();

    // 初期化 (ウィンドウ作成、OpenGL初期化、テクスチャ準備)
    bool init(int width, int height, const std::string& title);

    // 描画ループ用
    bool shouldClose();
    void draw();
    void cleanup();

    // ゲッター
    GLFWwindow* getWindow() const { return window; }
    unsigned int getTextureID() const { return displayTexture; }

private:
    GLFWwindow* window = nullptr;
    int winWidth = 0;
    int winHeight = 0;

    // 表示用シェーダ
    Shader* displayShader = nullptr;

    // CUDAが書き込む対象のテクスチャ
    unsigned int displayTexture = 0;

    // 全画面矩形描画用
    unsigned int quadVAO = 0;
    unsigned int quadVBO = 0;

    // 初期化ヘルパー
    void initGL();
    void initTexture();
    void initQuad();
};