#pragma once
#include "Common.h"

// 入力ソースの基底クラス（インターフェース）
class InputSource {
public:
    virtual ~InputSource() {}

    // 初期化（デバイスの開始やファイルの読み込み）
    virtual bool initialize() = 0;

    // 更新（新しいフレームを取得してデータを詰める）
    // 戻り値: 成功ならtrue
    virtual bool update(PointCloudData& outData) = 0;
};