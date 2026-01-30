# Unit16 卷積神經網路 (Convolutional Neural Networks, CNN)

## 📚 單元簡介

在 Unit15 中，我們學習了深度神經網路 (DNN) 的基礎，掌握了處理結構化表格數據的能力。但當面對**影像數據、空間結構數據、或具有局部相關性的數據**時，傳統 DNN 會遇到嚴重的挑戰：

- **參數爆炸問題**：一張 224×224×3 的 RGB 影像，若使用全連接層處理，假設第一層有 1000 個神經元，參數量約為 **150 百萬 (150,528,000)**
- **空間結構丟失**：DNN 將影像攤平成一維向量，破壞了像素的空間關係（相鄰像素的相關性）
- **平移不變性缺失**：同一物體出現在影像不同位置，DNN 需要重新學習

**卷積神經網路 (Convolutional Neural Networks, CNN)** 透過以下創新設計完美解決這些問題：
- **局部連接 (Local Connectivity)**：每個神經元只連接小區域，大幅減少參數
- **參數共享 (Parameter Sharing)**：同一卷積核在整個影像上使用，進一步壓縮參數
- **空間階層性 (Spatial Hierarchy)**：從邊緣 → 紋理 → 部件 → 物體，逐層抽象特徵
- **平移不變性 (Translation Invariance)**：無論物體在哪裡，都能正確識別

在化工領域，CNN 特別適合：
- **缺陷檢測**：產品表面瑕疵、裂紋、異常斑點的自動識別
- **顯微影像分析**：晶體形態分類、細胞計數、粒徑分布分析
- **製程視覺監控**：火焰狀態監測、液位識別、泡沫特性分析
- **光譜影像處理**：高光譜影像的成分分佈、反應過程視覺化
- **安全監控**：洩漏檢測、人員行為識別、設備異常監測

本單元涵蓋：
- **CNN 核心原理**：卷積運算、池化、感受野、特徵圖
- **經典架構**：LeNet、AlexNet、VGG、ResNet、Inception
- **TensorFlow/Keras 實作**：Conv2D、MaxPooling2D、Data Augmentation
- **四個實際案例**：大腸癌病理影像、中文手寫數字、工業缺陷檢測、遷移學習

本單元是 Part_4 深度學習系列的影像處理專題，建議在完成 Unit15 (DNN) 基礎後學習。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **理解 CNN 的核心概念**：卷積運算、池化、步幅、填充、感受野、特徵圖
2. **掌握 CNN 建模流程**：影像預處理、數據增強、模型設計、訓練策略、遷移學習
3. **熟練使用 TensorFlow/Keras**：Conv2D、MaxPooling2D、Dropout、BatchNormalization、Data Augmentation
4. **解決實際化工問題**：缺陷檢測、顯微影像分析、視覺監控、品質檢測
5. **應用遷移學習**：利用預訓練模型（VGG16、ResNet50、EfficientNet）加速訓練

---

## 📖 單元內容架構

### 1️⃣ 總覽篇：CNN 基礎與原理 ⭐

**檔案**：
- 講義：[Unit16_CNN_Overview.md](Unit16_CNN_Overview.md)
- 範例：[Unit16_CNN_Overview.ipynb](Unit16_CNN_Overview.ipynb)

**內容重點**：
- **為什麼需要 CNN？**：
  - DNN 處理影像的三大問題：參數爆炸、空間結構丟失、平移不變性缺失
  - 224×224×3 影像使用全連接層需要 154M 參數的計算示例
  - CNN 如何透過局部連接與參數共享解決這些問題
  
- **卷積運算 (Convolution) 核心概念**：
  - 卷積核 (Kernel/Filter)：權重矩陣，用於特徵提取
  - 滑動窗口 (Sliding Window)：在影像上移動卷積核
  - 步幅 (Stride)：移動步長，控制輸出尺寸
  - 填充 (Padding)：邊界處理（Valid vs. Same）
  - 特徵圖 (Feature Map)：卷積輸出，表示提取的特徵
  
- **池化 (Pooling) 降採樣**：
  - Max Pooling：取最大值，保留最顯著特徵
  - Average Pooling：取平均值，平滑特徵
  - 作用：降維、增加感受野、提供平移不變性
  
- **CNN 網路架構**：
  - 典型結構：[INPUT] → [CONV → RELU → POOL]×N → [FC → RELU]×M → [OUTPUT]
  - 感受野 (Receptive Field)：每個神經元在原始影像上的視野範圍
  - 特徵階層：淺層（邊緣、紋理）→ 深層（部件、物體）
  
- **經典 CNN 架構演進**：
  - **LeNet-5 (1998)**：手寫數字識別，CNN 的開創性架構
  - **AlexNet (2012)**：ImageNet 冠軍，深度學習崛起的標誌
  - **VGG-16/19 (2014)**：小卷積核堆疊，網路更深
  - **GoogLeNet/Inception (2014)**：多尺度特徵提取
  - **ResNet (2015)**：殘差連接，訓練超深網路（152 層）
  - **EfficientNet (2019)**：複合縮放，效率與精度最佳平衡
  
- **TensorFlow/Keras 實作**：
  - `Conv2D`：卷積層設計（filters、kernel_size、strides、padding、activation）
  - `MaxPooling2D` / `AveragePooling2D`：池化層
  - `Flatten`：特徵攤平，連接全連接層
  - `Dense`：全連接層（分類頭）
  - `Dropout`：防止過擬合
  - `BatchNormalization`：加速訓練穩定性
  
- **數據增強 (Data Augmentation)**：
  - 旋轉 (Rotation)：±15°
  - 平移 (Translation)：水平/垂直移動
  - 縮放 (Zoom)：±20%
  - 翻轉 (Flip)：水平/垂直翻轉
  - 亮度調整 (Brightness)
  - ImageDataGenerator vs. tf.data API
  
- **遷移學習 (Transfer Learning)**：
  - 預訓練模型：VGG16、ResNet50、Inception、EfficientNet
  - 特徵提取 (Feature Extraction)：凍結預訓練權重
  - 微調 (Fine-Tuning)：解凍部分層進行訓練
  - 優勢：小數據集也能達到高精度，訓練時間大幅縮短
  
- **化工領域 CNN 應用**：
  - **顯微影像分析**：晶體形態分類、粒徑分布
  - **缺陷檢測**：產品表面瑕疵、裂紋識別
  - **製程監控**：火焰監測、液位識別、泡沫分析
  - **安全應用**：洩漏檢測、人員行為識別

**適合讀者**：所有學員，**建議最先閱讀**以建立完整的 CNN 理論基礎

---

### 2️⃣ 實際案例 1：大腸癌病理影像分類 (Colorectal Cancer Histology) ⭐

**檔案**：
- 講義：[Unit16_Example_ColorectalCancer.md](Unit16_Example_ColorectalCancer.md)
- 程式範例：[Unit16_Example_ColorectalCancer.ipynb](Unit16_Example_ColorectalCancer.ipynb)

**內容重點**：
- **問題背景**：
  - 病理影像分析是癌症診斷的黃金標準
  - 人工判讀耗時、主觀、受經驗影響
  - 目標：自動分類大腸癌組織切片影像（8 種組織類型）
  
- **數據特性**：
  - Kather 大腸癌病理影像數據集
  - 150×150 像素 RGB 影像
  - 8 類別：腫瘤、基質、免疫細胞、壞死、正常黏膜等
  - 醫學影像特點：顏色、紋理、結構重要
  
- **模型設計**：
  - 自定義 CNN 架構（3-4 層卷積 + 2 層全連接）
  - Conv2D (32 → 64 → 128 filters) + MaxPooling2D
  - BatchNormalization 穩定訓練
  - Dropout (0.5) 防止過擬合
  
- **關鍵技術**：
  - 數據增強：旋轉、翻轉、縮放（模擬切片角度差異）
  - 類別平衡檢查
  - 混淆矩陣分析（哪些類別容易混淆）
  - Grad-CAM 視覺化（模型關注哪些區域）
  
- **工程意義**：
  - 輔助病理醫師診斷
  - 提升診斷效率與一致性
  - 大規模篩檢應用

**適合場景**：醫學影像分析、細胞分類、組織病理學、生物影像處理

---

### 3️⃣ 實際案例 2：中文手寫數字識別 (Chinese MNIST)

**檔案**：
- 講義：[Unit16_Example_ChineseMNIST.md](Unit16_Example_ChineseMNIST.md)
- 程式範例：[Unit16_Example_ChineseMNIST.ipynb](Unit16_Example_ChineseMNIST.ipynb)

**內容重點**：
- **問題背景**：
  - 手寫識別是 CNN 的經典應用（LeNet-5 原始應用）
  - 中文數字（一、二、三...）比阿拉伯數字複雜
  - 目標：識別 15 個中文手寫字元
  
- **數據特性**：
  - Chinese MNIST 數據集
  - 64×64 灰階影像
  - 15 類別（0-9 的中文數字 + 5 個中文數字單位字元）
  - 手寫變異大（筆觸、風格、粗細）
  
- **模型設計**：
  - LeNet-5 架構改良
  - Conv2D (6 → 16 filters) + Subsampling
  - 較淺網路，適合小影像
  
- **關鍵技術**：
  - 灰階影像處理
  - 數據標準化 (0-1)
  - Softmax 輸出層（15 類別）
  - 學習曲線分析（訓練 vs. 驗證）
  
- **工程意義**：
  - 表單自動識別
  - 文檔數位化
  - OCR 系統開發

**適合場景**：文字識別、OCR、手寫輸入系統、文檔處理

---

### 4️⃣ 實際案例 3：工業缺陷檢測 (Defect Detection) ⭐

**檔案**：
- 講義：[Unit16_Example_DefectDetection.md](Unit16_Example_DefectDetection.md)
- 程式範例：[Unit16_Example_DefectDetection.ipynb](Unit16_Example_DefectDetection.ipynb)

**內容重點**：
- **問題背景**：
  - 產品表面缺陷檢測是製造業的關鍵品質控制環節
  - 人工檢測速度慢、易疲勞、主觀判斷不一致
  - 目標：自動檢測產品表面的裂紋、刮痕、斑點等缺陷
  
- **數據特性**：
  - 工業缺陷影像數據集（如 NEU Surface Defect Database）
  - 200×200 像素灰階影像
  - 缺陷類型：裂紋 (Crack)、壓痕 (Scratch)、斑點 (Patch) 等
  - 類別不平衡（正常樣本多、缺陷樣本少）
  
- **模型設計**：
  - 中深度 CNN（4-5 層卷積）
  - 小卷積核 (3×3) 堆疊
  - 全局平均池化 (Global Average Pooling) 減少參數
  
- **關鍵技術**：
  - 類別不平衡處理：Class Weights、SMOTE、Focal Loss
  - 數據增強：針對缺陷特性設計（旋轉、縮放、亮度）
  - 精確率 vs. 召回率權衡（誤判成本考量）
  - 邊緣檢測預處理（增強缺陷特徵）
  
- **工程意義**：
  - 自動化品質檢測
  - 即時生產線監控
  - 降低人工成本
  - 提升產品一致性

**適合場景**：品質檢測、缺陷識別、表面檢查、自動光學檢測 (AOI)

---

### 5️⃣ 實際案例 4：遷移學習應用 (Transfer Learning) ⭐

**檔案**：
- 講義：[Unit16_Example_TransferLearning.md](Unit16_Example_TransferLearning.md)
- 程式範例：[Unit16_Example_TransferLearning.ipynb](Unit16_Example_TransferLearning.ipynb)

**內容重點**：
- **問題背景**：
  - 化工領域影像數據量通常有限（標註成本高）
  - 從頭訓練 CNN 需要大量數據（數萬至數百萬張）
  - 目標：利用預訓練模型（ImageNet）加速訓練，提升小數據集性能
  
- **遷移學習原理**：
  - **預訓練模型**：在 ImageNet (1000 類別、120 萬張影像) 上訓練
  - **特徵提取**：凍結預訓練權重，只訓練新的分類頭
  - **微調**：解凍部分頂層，進行精細調整
  - **為何有效**：淺層特徵（邊緣、紋理）具有通用性
  
- **預訓練模型選擇**：
  - **VGG16**：簡單、穩定，適合入門
  - **ResNet50**：深度殘差網路，性能優異
  - **InceptionV3**：多尺度特徵，適合複雜影像
  - **EfficientNetB0-B7**：效率與精度最佳，推薦使用
  - **MobileNetV2**：輕量化，適合邊緣部署
  
- **應用案例**：
  - 化工晶體形態分類（數據量 < 1000 張）
  - 製程視覺監控（火焰狀態分類）
  - 產品外觀檢測（包裝缺陷識別）
  
- **關鍵技術**：
  - 預訓練權重載入：`weights='imagenet'`
  - 凍結層：`layer.trainable = False`
  - 全局平均池化替代 Flatten
  - 學習率調度：微調時使用更小學習率（1e-5）
  - 數據增強仍然重要
  
- **對比實驗**：
  - 從頭訓練 vs. 特徵提取 vs. 微調
  - 訓練時間、收斂速度、最終精度對比
  - 小數據集（< 1000 張）效果尤其顯著
  
- **工程意義**：
  - 小數據集也能達到高精度
  - 訓練時間從數天縮短至數小時
  - 降低算力需求
  - 快速原型開發

**適合場景**：小數據集影像分類、快速原型開發、資源受限環境

---

### 6️⃣ 實作練習

**檔案**：[Unit16_CNN_Homework.ipynb](Unit16_CNN_Homework.ipynb)

**練習內容**：
- 建立完整的 CNN 影像分類模型
- 比較不同 CNN 架構（淺層 vs. 深層）
- 實驗數據增強的效果
- 應用遷移學習（選擇不同預訓練模型）
- 視覺化卷積核與特徵圖
- Grad-CAM 可解釋性分析
- 模型部署（TensorFlow Lite）

---

## 📊 數據集說明

### 1. 大腸癌病理影像 (`data/colorectal_cancer/`)
- Kather 大腸癌數據集
- 150×150 像素 RGB 影像
- 8 類別組織類型
- 用於醫學影像分類演示

### 2. 中文手寫數字 (`data/chinese_mnist/`)
- Chinese MNIST 數據集
- 64×64 灰階影像
- 15 個中文字元類別
- 用於 OCR 與手寫識別

### 3. 工業缺陷檢測 (`data/defects/`)
- NEU Surface Defect Database
- 200×200 灰階影像
- 6 種缺陷類型（裂紋、刮痕等）
- 用於品質檢測應用

### 4. 遷移學習示例 (`data/custom_images/`)
- 化工晶體形態影像（自製數據集）
- 224×224 RGB 影像
- 3-5 類別（根據應用場景）
- 小數據集（< 1000 張），用於遷移學習演示

---

## 🎓 CNN 建模決策指南

### 網路架構設計原則

| 影像尺寸 | 建議架構 | 卷積層數 | 參數量 | 適用場景 |
|---------|---------|---------|-------|---------|
| **< 32×32** | LeNet | 2-3 層 | < 100K | MNIST、簡單紋理 |
| **64×64** | 自定義淺層 | 3-4 層 | 100K-1M | 手寫、簡單物體 |
| **128×128** | 自定義中層 | 4-5 層 | 1M-5M | 缺陷檢測、細胞分類 |
| **224×224** | VGG/ResNet | 5-10 層 | 5M-25M | 通用影像分類 |
| **> 224×224** | EfficientNet | 7-15 層 | 10M-50M | 高解析度影像 |

### 卷積層設計建議

| 層級 | Filters 數量 | Kernel Size | 特徵類型 | 感受野 |
|------|------------|------------|---------|-------|
| **第 1 層** | 32-64 | 3×3 或 5×5 | 邊緣、線條 | 小 |
| **第 2-3 層** | 64-128 | 3×3 | 紋理、角點 | 中 |
| **第 4-5 層** | 128-256 | 3×3 | 部件、形狀 | 大 |
| **第 6+ 層** | 256-512 | 3×3 | 物體、語義 | 非常大 |

### 池化策略

| 池化類型 | 尺寸 | 作用 | 適用場景 |
|---------|-----|------|---------|
| **Max Pooling** | 2×2 | 保留最顯著特徵 | **最常用**，物體檢測 |
| **Average Pooling** | 2×2 | 平滑特徵 | 紋理分析 |
| **Global Average Pooling** | - | 替代 Flatten，減少參數 | 現代架構（ResNet 後） |

### 數據增強策略

| 增強方法 | 參數建議 | 適用場景 | 不適用場景 |
|---------|---------|---------|-----------|
| **水平翻轉** | 50% 機率 | 通用（物體無方向性） | 文字、定向物體 |
| **垂直翻轉** | 50% 機率 | 顯微影像、衛星影像 | 自然場景 |
| **旋轉** | ±15° | 通用 | 文字（小角度 ±5° 可接受） |
| **縮放** | ±20% | 通用 | 尺寸敏感應用 |
| **平移** | ±10% | 通用 | - |
| **亮度調整** | ±20% | 照明變化大 | 顏色關鍵應用 |
| **對比度調整** | ±20% | 通用 | - |

### 遷移學習模型選擇

| 模型 | 參數量 | 精度 | 速度 | 適用場景 |
|------|-------|------|------|---------|
| **MobileNetV2** | 3.5M | 中 | 快 | 邊緣部署、實時應用 |
| **VGG16** | 138M | 高 | 慢 | 教學、特徵提取 |
| **ResNet50** | 26M | 高 | 中 | **通用首選** |
| **InceptionV3** | 24M | 高 | 中 | 複雜影像 |
| **EfficientNetB0** | 5.3M | 高 | 快 | **最佳平衡** |
| **EfficientNetB7** | 66M | 極高 | 慢 | 精度要求極高 |

---

## 💻 實作環境需求

### 必要套件
```python
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
opencv-python >= 4.5.0           # 影像處理
Pillow >= 8.0.0                  # 影像讀取
tensorflow >= 2.10.0             # 深度學習框架
keras >= 2.10.0                  # 高階 API
scikit-learn >= 1.0.0            # 評估工具
```

### 選用套件
```python
albumentations >= 1.3.0          # 進階數據增強
imgaug >= 0.4.0                  # 影像增強
tensorflow-hub >= 0.12.0         # 預訓練模型中心
tensorboard >= 2.10.0            # 訓練視覺化
keras-tuner >= 1.1.0             # 超參數調整
scikit-image >= 0.19.0           # 影像處理工具
grad-cam >= 1.4.0                # 可解釋性視覺化
```

### 硬體建議
- **CPU**：Intel i5 以上（可訓練小模型，速度較慢）
- **GPU**：NVIDIA GTX 1060 以上（推薦 RTX 3060+），**大幅加速訓練**
- **記憶體**：16GB RAM 以上
- **儲存空間**：10GB+ SSD（數據集 + 模型權重）

---

## 📈 學習路徑建議

### 第一階段：理論基礎建立（必讀）
1. 閱讀 [Unit16_CNN_Overview.md](Unit16_CNN_Overview.md)
2. 執行 [Unit16_CNN_Overview.ipynb](Unit16_CNN_Overview.ipynb)
3. 重點掌握：
   - 為何 DNN 不適合影像？（參數爆炸計算）
   - 卷積運算原理（滑動窗口、步幅、填充）
   - 池化的作用（降採樣、平移不變性）
   - 經典架構演進（LeNet → AlexNet → VGG → ResNet）

### 第二階段：從簡單到複雜的案例學習

**建議學習順序**：

1. **中文手寫數字**（最簡單，灰階小影像）
   - 閱讀 [Unit16_Example_ChineseMNIST.md](Unit16_Example_ChineseMNIST.md)
   - 執行 [Unit16_Example_ChineseMNIST.ipynb](Unit16_Example_ChineseMNIST.ipynb)
   - 重點：LeNet 架構、灰階影像處理、Softmax 輸出
   
2. **工業缺陷檢測**（中等難度，實用性強）
   - 閱讀 [Unit16_Example_DefectDetection.md](Unit16_Example_DefectDetection.md)
   - 執行 [Unit16_Example_DefectDetection.ipynb](Unit16_Example_DefectDetection.ipynb)
   - 重點：類別不平衡、數據增強、精確率 vs. 召回率
   
3. **大腸癌病理影像**（較複雜，RGB 多類別）
   - 閱讀 [Unit16_Example_ColorectalCancer.md](Unit16_Example_ColorectalCancer.md)
   - 執行 [Unit16_Example_ColorectalCancer.ipynb](Unit16_Example_ColorectalCancer.ipynb)
   - 重點：多類別分類、混淆矩陣、Grad-CAM 可解釋性
   
4. **遷移學習**（必學，實務最重要）
   - 閱讀 [Unit16_Example_TransferLearning.md](Unit16_Example_TransferLearning.md)
   - 執行 [Unit16_Example_TransferLearning.ipynb](Unit16_Example_TransferLearning.ipynb)
   - 重點：預訓練模型、特徵提取 vs. 微調、小數據集策略

### 第三階段：綜合練習
1. 完成 [Unit16_CNN_Homework.ipynb](Unit16_CNN_Homework.ipynb)
2. 比較從頭訓練 vs. 遷移學習的效果
3. 實驗不同數據增強組合
4. 嘗試將 CNN 應用於自己的化工影像數據

### 第四階段：進階技術
1. 學習物體檢測（YOLO、Faster R-CNN）
2. 語義分割（U-Net、DeepLab）
3. 實例分割（Mask R-CNN）
4. 模型壓縮與部署（TensorFlow Lite、ONNX）

---

## 🔍 化工領域核心應用

### 1. 缺陷檢測與品質檢測 ⭐
- **應用場景**：
  - 產品表面瑕疵檢測（裂紋、刮痕、斑點）
  - 包裝完整性檢查
  - 焊接品質檢測
  - 印刷品質控制
  
- **CNN 優勢**：
  - 自動特徵提取（無需手工設計特徵）
  - 即時檢測（毫秒級）
  - 精度超越人眼
  - 一致性高
  
- **實作關鍵**：
  - 類別不平衡處理（正常樣本多、缺陷樣本少）
  - 數據增強模擬各種缺陷變異
  - 精確率 vs. 召回率權衡（誤判成本）
  - 邊緣部署（工廠現場無網路）

### 2. 顯微影像分析
- **應用場景**：
  - 晶體形態分類（多晶型、針狀、片狀）
  - 細胞計數與分類
  - 粒徑分布分析
  - 組織病理學
  
- **CNN 優勢**：
  - 捕捉微小形態差異
  - 多尺度特徵提取
  - 定量分析
  
- **實作關鍵**：
  - 高解析度影像處理（1024×1024+）
  - 預處理（去噪、對比度增強）
  - 語義分割（逐像素標註）
  - 實例分割（單個晶體/細胞識別）

### 3. 製程視覺監控
- **應用場景**：
  - 火焰狀態監測（燃燒器）
  - 液位識別
  - 泡沫特性分析
  - 流型識別（兩相流）
  - 反應顏色變化監測
  
- **CNN 優勢**：
  - 即時監控
  - 環境魯棒性（照明變化、角度變化）
  - 無接觸測量
  
- **實作關鍵**：
  - 影像預處理（去背景、增強對比）
  - 時序整合（連續幀分析）
  - 邊緣計算（現場部署）
  - 異常檢測（正常 vs. 異常狀態）

### 4. 高光譜影像分析
- **應用場景**：
  - 成分分佈分析
  - 反應過程視覺化
  - 混合均勻度評估
  
- **CNN 優勢**：
  - 處理高維光譜數據（數百個波長）
  - 自動特徵學習
  - 空間-光譜聯合分析
  
- **實作關鍵**：
  - 3D 卷積（2D 空間 + 1D 光譜）
  - 降維預處理（PCA）
  - 數據標準化

### 5. 安全監控
- **應用場景**：
  - 洩漏檢測（液體、氣體）
  - 人員行為識別（穿戴 PPE 檢查）
  - 設備異常監測（腐蝕、變形）
  - 煙霧火災檢測
  
- **CNN 優勢**：
  - 多目標檢測
  - 即時預警
  - 24/7 監控
  
- **實作關鍵**：
  - 物體檢測（YOLO、SSD）
  - 多攝影機整合
  - 誤報抑制
  - 報警邏輯設計

---

## 📝 評估指標總結

### 分類任務
- **Accuracy**：整體準確率，適合類別平衡數據
- **Precision**：查準率，預測為正類中真正為正類的比例（誤報代價高時重視）
- **Recall**：查全率，真實正類中被正確預測的比例（漏報代價高時重視）
- **F1-Score**：Precision 與 Recall 的調和平均
- **ROC-AUC**：分類能力的綜合評估
- **混淆矩陣 (Confusion Matrix)**：詳細錯誤分析

### 訓練監控
- **Training Loss vs. Validation Loss**：過擬合診斷
- **Accuracy Curves**：訓練/驗證準確率變化
- **Learning Rate Schedule**：學習率調整記錄
- **Gradient Norm**：梯度穩定性

### 模型複雜度
- **參數量 (Parameters)**：模型大小
- **FLOPs**：計算量（浮點運算次數）
- **推論時間 (Inference Time)**：預測速度（ms/image）
- **模型檔案大小**：部署考量

---

## 🚀 進階學習方向

完成本單元後，可以進一步探索：

1. **物體檢測 (Object Detection)**：YOLO、Faster R-CNN、SSD（偵測多個物體及位置）
2. **語義分割 (Semantic Segmentation)**：U-Net、DeepLab（逐像素分類）
3. **實例分割 (Instance Segmentation)**：Mask R-CNN（區分個別物體）
4. **生成對抗網路 (GAN)**：影像生成、數據增強、超解析度
5. **自監督學習 (Self-Supervised Learning)**：無標註數據預訓練
6. **神經架構搜索 (NAS)**：自動設計最佳 CNN 架構
7. **輕量化模型**：MobileNet、ShuffleNet、SqueezeNet（邊緣部署）
8. **可解釋 AI**：Grad-CAM、SHAP、LIME（解釋模型決策）
9. **3D CNN**：影片分析、醫學影像（CT、MRI）

---

## 📚 參考資源

### 教科書
1. *Deep Learning for Computer Vision* by Rajalingappaa Shanmugamani
2. *Dive into Deep Learning* by Aston Zhang et al.（線上免費，含 CNN 詳細推導）
3. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* by Aurélien Géron（第 14 章 CNN）

### 經典論文
- **LeNet-5** (1998): "Gradient-Based Learning Applied to Document Recognition" - Yann LeCun
- **AlexNet** (2012): "ImageNet Classification with Deep Convolutional Neural Networks" - Krizhevsky et al.
- **VGG** (2014): "Very Deep Convolutional Networks for Large-Scale Image Recognition" - Simonyan & Zisserman
- **ResNet** (2015): "Deep Residual Learning for Image Recognition" - He et al.
- **EfficientNet** (2019): "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" - Tan & Le

### 線上資源
- [Stanford CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [TensorFlow CNN 教學](https://www.tensorflow.org/tutorials/images/cnn)
- [Keras 官方文件](https://keras.io/guides/sequential_model/)

### 化工領域應用論文
- "Deep learning for automated visual inspection in manufacturing" (Journal of Manufacturing Systems, 2020)
- "Convolutional neural networks for crystallization image classification" (Chemical Engineering Science, 2019)
- "Flame image recognition for industrial burners using CNN" (Energy, 2021)

---

## ✍️ 課後習題提示

1. **參數計算**：計算 VGG16 第一層的參數量（224×224×3 → Conv(64, 3×3) → ...）
2. **感受野計算**：計算 5 層 3×3 卷積堆疊的感受野大小
3. **架構設計**：為 128×128 影像設計 4 層 CNN（指定 filters、kernel_size、pooling）
4. **數據增強實驗**：比較有無數據增強對模型性能的影響
5. **遷移學習對比**：在小數據集（< 500 張）上比較從頭訓練 vs. VGG16 vs. EfficientNetB0
6. **可解釋性**：使用 Grad-CAM 視覺化模型決策，分析錯誤樣本
7. **化工應用**：將 CNN 應用於自己的化工影像數據（晶體、缺陷、監控影像），撰寫完整報告

---

## 📞 技術支援

如有疑問或需要協助，歡迎透過以下方式聯繫：
- 課程討論區
- 指導教授 Office Hour
- 課程助教

---

**🎉 CNN 是電腦視覺的核心技術！掌握 CNN 讓化工製程的「眼睛」更加智能！**

---

**課程資訊**
- 課程名稱：AI在化工上之應用
- 課程單元：Unit16 卷積神經網路（CNN）與影像分析
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-01-28

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---