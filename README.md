# 🏓 桌球比賽預測模型

基於深度學習的桌球比賽下一拍預測系統，使用 LSTM + Multi-Head Attention 架構來預測比賽中的動作 (Action)、落點 (Point) 及勝負結果。

## 📋 專案簡介

本專案針對桌球比賽數據，建立序列預測模型，主要預測：
- **動作類型 (ActionId)**：下一拍的擊球動作
- **落點位置 (PointId)**：球的落點區域
- **得分者 (ServerGetPoint)**：該回合的得分方

## 🏗️ 模型架構

### 核心組件

1. **增強數據處理器 (EnhancedProcessor)**
   - 建立統計先驗知識
   - Action → Point 條件概率
   - Position 轉移矩陣
   - Point 序列模式分析

2. **序列編碼器**
   - 雙層雙向 LSTM
   - Multi-Head Self-Attention (4 heads)
   - Attention Pooling

3. **預測頭**
   - Action Head：動作分類
   - Point Head：落點分類（含 BatchNorm）
   - Winner Head：勝負預測

### 特徵工程

模型使用 37 維統計特徵，包括：
- 序列長度與位置特徵
- 最近擊球的動作/落點/位置
- 序列動態變化特徵
- N-gram 特徵
- Action-Point 條件概率
- Point 轉移概率
- Delta 特徵

## 📦 安裝需求

```bash
pip install pandas numpy scikit-learn torch
```

### 環境需求
- Python 3.8+
- PyTorch 1.9+
- CUDA (建議使用 GPU 訓練)

## 🚀 使用方法

### 1. 準備數據

將訓練和測試數據放置於專案目錄：
```
table-tennis-prediction/
├── train.csv
├── test.csv
└── train.py
```

### 2. 訓練模型

```bash
python train.py
```

### 3. 輸出結果

訓練完成後會產生：
- `best_model_fold_0.pth` ~ `best_model_fold_4.pth`：各 fold 的最佳模型
- `submission_optimized.csv`：預測結果

## ⚙️ 超參數設定

| 參數 | 預設值 | 說明 |
|------|--------|------|
| MAX_SEQ_LEN | 25 | 最大序列長度 |
| BATCH_SIZE | 256 | 批次大小 |
| EPOCHS | 30 | 訓練輪數 |
| N_SPLITS | 5 | K-Fold 折數 |
| Learning Rate | 0.0015 | 學習率 |
| Dropout | 0.3 | Dropout 比例 |

## 🔧 訓練策略

### 損失函數
- **Action Loss**：CrossEntropy (權重 0.3)
- **Point Loss**：Adaptive Focal Loss (動態權重 3.0→6.0)
- **Winner Loss**：CrossEntropy (權重 0.5)

### 優化技術
- AdamW 優化器 (weight_decay=5e-5)
- OneCycleLR 學習率調度
- 梯度裁剪 (max_norm=1.0)
- Early Stopping (patience=8)

### 集成策略
- 5-Fold GroupKFold 交叉驗證
- Test-Time Augmentation (TTA, n=3)
- 智慧加權集成：
  - 基於驗證損失的 Fold 權重
  - 基於預測信心的權重
  - 基於一致性的權重

## 📊 模型表現

- 平均驗證損失：4.45 ± 0.02
- Point 預測準確率：~28%
- 預期分數：0.255-0.265

## 📁 專案結構

```
table-tennis-prediction/
├── README.md           # 專案說明文件
├── train.py            # 主要訓練程式
├── requirements.txt    # 依賴套件
└── .gitignore         # Git 忽略檔案
```

## 📝 資料格式

### 輸入資料欄位
- `rally_uid`：回合唯一識別碼
- `strickNumber`：擊球序號
- `gamePlayerId`：選手 ID
- `actionId`：動作類型
- `pointId`：落點位置
- `positionId`：選手位置
- `serveId`：發球方
- `serverGetPoint`：發球方是否得分

### 輸出資料欄位
- `rally_uid`：回合唯一識別碼
- `serverGetPoint`：預測發球方得分
- `pointId`：預測落點
- `actionId`：預測動作


