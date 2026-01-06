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

    // Viewer.cpp の window生成後、glew/glad初期化後に追加
    const GLubyte* renderer = glGetString(GL_RENDERER);
    const GLubyte* version = glGetString(GL_VERSION);
    std::cout << "[OpenGL Info] Renderer: " << renderer << std::endl;
    std::cout << "[OpenGL Info] Version:  " << version << std::endl;

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

// ★追加: BMP書き出しの実装
void GLViewer::saveTexture(const std::string& filename) {
    // 1. テクスチャのサイズを取得
    glBindTexture(GL_TEXTURE_2D, displayTexture);
    int w, h;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &h);

    // 2. ピクセルデータをGPUから読み出す
    // RGBで取得 (BMPはパディングが必要だが、w=2400なら4の倍数なのでパディング不要)
    std::vector<unsigned char> pixels(w * h * 3);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    // 3. BMPヘッダの作成
    unsigned char fileHeader[14] = {
        'B','M',      // magic
        0,0,0,0,      // size in bytes (後で埋める)
        0,0,          // app data
        0,0,          // app data
        54,0,0,0      // start of data offset
    };
    unsigned char infoHeader[40] = {
        40,0,0,0,     // info hd size
        0,0,0,0,      // width (後で埋める)
        0,0,0,0,      // height (後で埋める)
        1,0,          // number color planes
        24,0,         // bits per pixel
        0,0,0,0,      // compression is none
        0,0,0,0,      // image bits size
        0,0,0,0,      // horz resolu
        0,0,0,0,      // vert resolu
        0,0,0,0,      // # colors in plt
        0,0,0,0,      // # important colors
    };

    int fileSize = 54 + pixels.size();
    fileHeader[ 2] = (unsigned char)(fileSize      );
    fileHeader[ 3] = (unsigned char)(fileSize >>  8);
    fileHeader[ 4] = (unsigned char)(fileSize >> 16);
    fileHeader[ 5] = (unsigned char)(fileSize >> 24);

    infoHeader[ 4] = (unsigned char)(w      );
    infoHeader[ 5] = (unsigned char)(w >>  8);
    infoHeader[ 6] = (unsigned char)(w >> 16);
    infoHeader[ 7] = (unsigned char)(w >> 24);

    infoHeader[ 8] = (unsigned char)(h      );
    infoHeader[ 9] = (unsigned char)(h >>  8);
    infoHeader[10] = (unsigned char)(h >> 16);
    infoHeader[11] = (unsigned char)(h >> 24);

    // 4. ファイル書き込み
    std::ofstream f(filename, std::ios::out | std::ios::binary);
    if (!f) {
        std::cerr << "[Error] Failed to open file for saving: " << filename << std::endl;
        return;
    }

    f.write(reinterpret_cast<char*>(fileHeader), 14);
    f.write(reinterpret_cast<char*>(infoHeader), 40);

    // BMPはBGR順、かつ上下反転している場合があるが、
    // OpenGLのテクスチャ座標(左下原点)とBMP(左下原点)は相性が良いので
    // 上下はそのままでOK。ただし色は RGB -> BGR 変換が必要。
    for (int i = 0; i < pixels.size(); i += 3) {
        unsigned char r = pixels[i];
        unsigned char g = pixels[i+1];
        unsigned char b = pixels[i+2];
        unsigned char bgr[] = { b, g, r };
        f.write(reinterpret_cast<char*>(bgr), 3);
    }
    
    f.close();
    std::cout << "[System] Screenshot saved to " << filename << " (" << w << "x" << h << ")" << std::endl;
}