#include "Alignment.h"
#include <iostream>

void AlignmentController::update(GLFWwindow* window, AppConfig& config) {
    // キー入力の監視
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
        // 例: 上キーで焦点距離を調整
        // config.focalLength += 0.0001f;
    }
    
    // TODO: ここにKさんの担当する「キーボード操作で config を書き換える処理」を移植する
}

void AlignmentController::saveSettings(const std::string& path, const AppConfig& config) {
    // TODO: CSV保存
}

void AlignmentController::loadSettings(const std::string& path, AppConfig& config) {
    // TODO: CSV読み込み
}