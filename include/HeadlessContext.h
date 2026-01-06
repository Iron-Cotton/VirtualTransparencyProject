#pragma once
#include <EGL/egl.h>
#include <EGL/eglext.h> // 拡張機能用
#include <iostream>
#include <vector>
#include <cstring>
#include <glad/glad.h>

class HeadlessContext {
public:
    EGLDisplay eglDpy = EGL_NO_DISPLAY;
    EGLContext eglCtx = EGL_NO_CONTEXT;
    EGLSurface eglSurf = EGL_NO_SURFACE;

    bool init(int width, int height) {
        // ---------------------------------------------------------
        // 1. EGL拡張関数のロード (Device Enumeration用)
        // ---------------------------------------------------------
        PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT = 
            (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
        PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT = 
            (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");

        if (!eglQueryDevicesEXT || !eglGetPlatformDisplayEXT) {
            std::cerr << "[Headless] EGL extension for device query not found." << std::endl;
            std::cerr << "[Headless] Fallback to default display..." << std::endl;
            eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        } else {
            // ---------------------------------------------------------
            // 2. デバイス（GPU）の列挙と選択
            // ---------------------------------------------------------
            const int MAX_DEVICES = 16;
            EGLDeviceEXT devices[MAX_DEVICES];
            EGLint numDevices = 0;

            if (!eglQueryDevicesEXT(MAX_DEVICES, devices, &numDevices) || numDevices == 0) {
                std::cerr << "[Headless] No EGL devices found." << std::endl;
                return false;
            }

            std::cout << "[Headless] Found " << numDevices << " EGL devices." << std::endl;

            // 最初に見つかった初期化可能なデバイス（通常はNVIDIA）を使用
            for (int i = 0; i < numDevices; ++i) {
                // デバイスからディスプレイを取得 (EGL_PLATFORM_DEVICE_EXT = 0x313F)
                EGLDisplay dpy = eglGetPlatformDisplayEXT(0x313F, devices[i], 0);
                
                if (dpy != EGL_NO_DISPLAY) {
                    EGLint major, minor;
                    if (eglInitialize(dpy, &major, &minor)) {
                        eglDpy = dpy;
                        std::cout << "[Headless] Successfully initialized device index: " << i << std::endl;
                        break; // 成功したらループを抜ける
                    }
                }
            }
        }

        if (eglDpy == EGL_NO_DISPLAY) {
            std::cerr << "[Headless] Failed to get EGL display." << std::endl;
            return false;
        }

        // ---------------------------------------------------------
        // 3. コンフィグ設定 (ここからは以前と同じ)
        // ---------------------------------------------------------
        const EGLint configAttribs[] = {
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_BLUE_SIZE, 8,
            EGL_GREEN_SIZE, 8,
            EGL_RED_SIZE, 8,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
            EGL_NONE
        };

        EGLConfig eglCfg;
        EGLint numConfigs;
        if (!eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs)) {
            std::cerr << "[Headless] Failed to choose config." << std::endl;
            return false;
        }

        // APIのバインド
        eglBindAPI(EGL_OPENGL_API);

        // コンテキスト作成
        const EGLint contextAttribs[] = {
            EGL_CONTEXT_MAJOR_VERSION, 4,
            EGL_CONTEXT_MINOR_VERSION, 5,
            EGL_NONE
        };
        eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, contextAttribs);
        if (eglCtx == EGL_NO_CONTEXT) {
            std::cerr << "[Headless] Failed to create EGL context." << std::endl;
            return false;
        }

        // サーフェス作成
        const EGLint pbufferAttribs[] = {
            EGL_WIDTH, width,
            EGL_HEIGHT, height,
            EGL_NONE,
        };
        eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg, pbufferAttribs);
        if (eglSurf == EGL_NO_SURFACE) {
            std::cerr << "[Headless] Failed to create Pbuffer Surface." << std::endl;
            return false;
        }

        // カレントにする
        eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx);

        // GLADロード
        if (!gladLoadGL((GLADloadfunc)eglGetProcAddress)) {
            std::cerr << "[Headless] Failed to initialize GLAD." << std::endl;
            return false;
        }

        // 確認用ログ出力
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "[Headless] GL_VENDOR:   " << glGetString(GL_VENDOR) << std::endl;
        std::cout << "[Headless] GL_RENDERER: " << glGetString(GL_RENDERER) << std::endl;
        std::cout << "[Headless] GL_VERSION:  " << glGetString(GL_VERSION) << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        
        return true;
    }

    void cleanup() {
        if (eglDpy != EGL_NO_DISPLAY) {
            eglMakeCurrent(eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            if (eglSurf != EGL_NO_SURFACE) eglDestroySurface(eglDpy, eglSurf);
            if (eglCtx != EGL_NO_CONTEXT) eglDestroyContext(eglDpy, eglCtx);
            eglTerminate(eglDpy);
        }
    }
};