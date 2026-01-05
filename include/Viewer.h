#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>
#include <vector> // 追加
#include "shader_m.h"

class GLViewer {
public:
    GLViewer();
    ~GLViewer();

    bool init(int width, int height, const std::string& title);
    bool shouldClose();
    void draw();
    void cleanup();

    // ★追加: 現在のテクスチャをファイルに保存する関数
    void saveTexture(const std::string& filename);

    GLFWwindow* getWindow() const { return window; }
    unsigned int getTextureID() const { return displayTexture; }

private:
    GLFWwindow* window = nullptr;
    int winWidth = 0;
    int winHeight = 0;

    Shader* displayShader = nullptr;
    unsigned int displayTexture = 0;
    unsigned int quadVAO = 0;
    unsigned int quadVBO = 0;

    void initGL();
    void initTexture();
    void initQuad();
};