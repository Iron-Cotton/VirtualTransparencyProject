#pragma once
#include "Common.h"
#include <string>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

class AlignmentController {
public:
    // 入力（キーボード/ゲームパッド）を監視し、configの値を書き換える
    void update(GLFWwindow* window, AppConfig& config);
    
    // 設定の保存/ロード (CSV)
    void saveSettings(const std::string& path, const AppConfig& config);
    void loadSettings(const std::string& path, AppConfig& config);
};