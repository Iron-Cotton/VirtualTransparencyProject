#include "Viewer.h"
#include <iostream>

// コールバック関数プロトタイプ
void framebuffer_size_callback(GLFWwindow* window, int width, int height);

GLViewer::GLViewer() {}

GLViewer::~GLViewer() {
    cleanup();
}

bool GLViewer::init(int width, int height, const std::string& title) {
    winWidth = width;
    winHeight = height;

    // GLFW初期化
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    // バージョン指定 (OpenGL 4.6 Core)
// バージョン指定 (OpenGL 3.3 Core に下げる)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // 解像度固定

    // ウィンドウ作成
    window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

// GLADロード (Version 2対応)
    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return false;
    }

    // vsyncオフ (ベンチマーク用)
    glfwSwapInterval(0);

    // 各種初期化
    initGL();
    initTexture();
    initQuad();

    return true;
}

void GLViewer::initGL() {
    // 表示用シェーダの読み込み (後述の display.vert/frag を使用)
    // ファイルパスは環境に合わせて調整してください
    displayShader = new Shader("shaders/display.vert", "shaders/display.frag");
}

void GLViewer::initTexture() {
    // 古いテクスチャがあれば消す（安全対策）
    if (displayTexture) glDeleteTextures(1, &displayTexture);

    glGenTextures(1, &displayTexture);
    glBindTexture(GL_TEXTURE_2D, displayTexture);

    // パラメータ設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // 【修正点】 GL_RGBA8 ではなく GL_RGBA を使う（互換性重視）
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, winWidth, winHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    // エラーチェック
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "[Viewer Error] Failed to create texture: " << err << std::endl;
    } else {
        std::cout << "[Viewer] Texture created successfully. ID: " << displayTexture << std::endl;
    }

    glBindTexture(GL_TEXTURE_2D, 0);
}

void GLViewer::initQuad() {
    // 画面いっぱいの四角形 (X: -1~1, Y: -1~1, U: 0~1, V: 0~1)
    float quadVertices[] = {
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };

    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);

    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);

    // Position (Location 0)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    
    // TexCoord (Location 1)
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    glBindVertexArray(0);
}

bool GLViewer::shouldClose() {
    return glfwWindowShouldClose(window);
}

void GLViewer::draw() {
    // クリア
    glClear(GL_COLOR_BUFFER_BIT);

    // シェーダ使用
    displayShader->use();
    
    // テクスチャバインド (Unit 0)
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, displayTexture);
    displayShader->setInt("screenTexture", 0);

    // 描画
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);

    // バッファスワップとイベント処理
    glfwSwapBuffers(window);
    glfwPollEvents();
}

void GLViewer::cleanup() {
    if (displayShader) delete displayShader;
    if (quadVAO) glDeleteVertexArrays(1, &quadVAO);
    if (quadVBO) glDeleteBuffers(1, &quadVBO);
    if (displayTexture) glDeleteTextures(1, &displayTexture);
    if (window) glfwDestroyWindow(window);
    glfwTerminate();
}

// グローバルコールバック
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}