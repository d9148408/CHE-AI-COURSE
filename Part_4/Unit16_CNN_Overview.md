# Unit 16: 卷積神經網路 (Convolutional Neural Network, CNN) 概述

## 學習目標
- 理解卷積神經網路 (CNN) 的基本原理與數學基礎
- 認識常見的 CNN 架構及其特點
- 學習使用 TensorFlow/Keras 建立 CNN 模型
- 掌握 CNN 模型的訓練、評估與預測流程
- 了解 CNN 在化工領域的應用案例

---

## 1. CNN 理論背景

### 1.1 什麼是卷積神經網路？

**卷積神經網路 (Convolutional Neural Network, CNN)** 是一種專門用於處理具有網格狀拓撲結構數據的深度學習模型,特別適合處理影像數據。CNN 的設計靈感來自於生物視覺皮層的神經元組織方式,能夠自動學習並提取數據中的空間特徵。

### 1.2 CNN 的核心概念

#### 1.2.1 卷積運算 (Convolution)

卷積是 CNN 的核心操作,透過可學習的濾波器 (kernel 或 filter) 在輸入數據上滑動,提取局部特徵。

**二維卷積的數學定義:**

$$
S(i, j) = (I * K)(i, j) = \sum_{m} \sum_{n} I(i+m, j+n) \cdot K(m, n)
$$

其中:
- $I$ 是輸入影像
- $K$ 是卷積核 (通常 3×3 或 5×5)
- $(i, j)$ 是輸出特徵圖的位置
- $(m, n)$ 是卷積核的索引

**卷積運算的關鍵特性:**

1. **平移不變性 (Translation Invariance)**  
   數字在影像中移動,CNN 仍能正確辨識 (不像全連接層對位置敏感)

2. **參數共享 (Parameter Sharing)**  
   同一個濾波器掃描整張影像,參數量遠小於全連接層  
   例如:28×28 影像用 3×3 濾波器,僅需 9 個參數 (vs 全連接需 784×隱藏層數)

3. **局部連接 (Local Connectivity)**  
   每個神經元只看一小塊影像區域 (感受野 Receptive Field)

**化工類比:濾波器像「化學濾紙」**

| 濾波器類型 | 化學實驗類比 | 檢測特徵 |
|-----------|-------------|---------|
| **邊緣檢測** | 微分器 (檢測濃度梯度) | 裂紋邊界、輪廓線 |
| **平滑濾波** | 低通濾波器 (去除雜訊) | 消除顆粒感、保留主結構 |
| **紋理檢測** | 週期性圖案分析 | 金屬軋製紋路、編織布料 |

**常見濾波器範例:**

Sobel 邊緣檢測濾波器 (手工設計,CNN 會自動學習類似功能):

$$
K_x = \begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{bmatrix}, \quad
K_y = \begin{bmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
1 & 2 & 1
\end{bmatrix}
$$

#### 1.2.2 特徵圖 (Feature Map)

卷積運算的輸出稱為特徵圖,表示輸入數據中特定特徵的激活程度。一個卷積層通常包含多個卷積核,產生多個特徵圖。

**特徵圖尺寸計算:**

$$
O = \frac{W - K + 2P}{S} + 1
$$

其中:
- $O$ 是輸出特徵圖的尺寸
- $W$ 是輸入尺寸
- $K$ 是卷積核尺寸
- $P$ 是填充 (padding) 大小
- $S$ 是步長 (stride)

#### 1.2.3 池化 (Pooling)

池化層用於降低特徵圖的空間維度,減少參數量並提高模型的平移不變性。

**最大池化 (Max Pooling):**

$$
y_{i,j} = \max_{(p,q) \in \mathcal{R}_{i,j}} x_{p,q}
$$

其中 $\mathcal{R}_{i,j}$ 為 2×2 池化視窗覆蓋的區域。

**平均池化 (Average Pooling):**

$$
y_{i,j} = \frac{1}{|\mathcal{R}_{i,j}|} \sum_{(p,q) \in \mathcal{R}_{i,j}} x_{p,q}
$$

其中 $\mathcal{R}_{i,j}$ 是池化窗口的區域。

**生物學啟發:**
- 模擬視覺皮層的「側抑制」機制
- 只保留最強烈的特徵訊號 (類似「贏者全拿」)

**池化層的效果:**
- **降低維度**: 28×28 → 14×14 → 7×7
- **提升平移容忍度**: 數字稍微移動幾個像素仍能辨識
- **防止過擬合**: 減少參數數量

#### 1.2.4 感受野 (Receptive Field)

感受野是指輸出特徵圖中的一個神經元在原始輸入影像上所對應的區域大小。隨著網路層數的增加,感受野會逐漸擴大。

**感受野計算:**

$$
RF_l = RF_{l-1} + (K_l - 1) \times \prod_{i=1}^{l-1} S_i
$$

其中:
- $RF_l$ 是第 $l$ 層的感受野
- $K_l$ 是第 $l$ 層的卷積核尺寸
- $S_i$ 是第 $i$ 層的步長

### 1.3 CNN 的典型架構

一個典型的 CNN 架構通常包含以下幾個部分:

1. **輸入層 (Input Layer)**: 接收原始影像數據
2. **卷積層 (Convolutional Layer)**: 提取局部特徵
3. **激活函數 (Activation Function)**: 引入非線性
4. **池化層 (Pooling Layer)**: 降維與特徵選擇
5. **全連接層 (Fully Connected Layer)**: 整合特徵進行分類或回歸
6. **輸出層 (Output Layer)**: 產生最終預測結果

**典型 CNN 架構流程:**

```
Input → [Conv → Activation → Pooling] × N → Flatten → Dense → Output
```

---

## 2. 常見的 CNN 架構

### 2.1 LeNet-5 (1998)

- **提出者**: Yann LeCun
- **特點**: 最早成功應用於手寫數字識別的 CNN 架構
- **結構**: 2 個卷積層 + 2 個池化層 + 3 個全連接層
- **應用**: MNIST 手寫數字識別

### 2.2 AlexNet (2012)

- **提出者**: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton
- **特點**: 首次在 ImageNet 競賽中展現深度學習的強大能力
- **創新**: 使用 ReLU 激活函數、Dropout、數據增強、GPU 加速
- **結構**: 5 個卷積層 + 3 個全連接層,約 6000 萬參數

### 2.3 VGGNet (2014)

- **提出者**: Visual Geometry Group, Oxford
- **特點**: 使用小尺寸卷積核 (3×3) 堆疊深層網路
- **變體**: VGG-16 (16 層)、VGG-19 (19 層)
- **優點**: 結構簡單、易於理解和實現

### 2.4 GoogLeNet / Inception (2014)

- **提出者**: Google
- **特點**: 引入 Inception 模組,在同一層使用多種尺寸的卷積核
- **創新**: 1×1 卷積降維、全局平均池化
- **優點**: 參數量少、計算效率高

### 2.5 ResNet (2015)

- **提出者**: Kaiming He et al., Microsoft Research
- **特點**: 引入殘差連接 (Residual Connection) 解決深層網路訓練困難問題
- **創新**: 跳躍連接 (Skip Connection) 允許梯度直接傳播
- **變體**: ResNet-50, ResNet-101, ResNet-152
- **影響**: 使訓練超過 100 層的網路成為可能

**殘差塊數學表示:**

$$
y = F(x, \{W_i\}) + x
$$

其中 $F(x, \{W_i\})$ 是殘差映射,$x$ 是跳躍連接。

### 2.6 MobileNet (2017)

- **提出者**: Google
- **特點**: 專為移動設備和嵌入式系統設計的輕量級 CNN
- **創新**: 深度可分離卷積 (Depthwise Separable Convolution)
- **優點**: 參數量少、計算量小、適合資源受限環境

### 2.7 EfficientNet (2019)

- **提出者**: Google
- **特點**: 系統化地平衡網路深度、寬度和解析度
- **創新**: 複合縮放方法 (Compound Scaling)
- **優點**: 在相同計算資源下達到更高的準確率

---

## 3. CNN 的應用場景

### 3.1 電腦視覺領域

1. **影像分類 (Image Classification)**: 識別影像中的物體類別
2. **物體檢測 (Object Detection)**: 定位並識別影像中的多個物體
3. **影像分割 (Image Segmentation)**: 將影像分割成不同的區域
4. **人臉識別 (Face Recognition)**: 識別和驗證人臉身份
5. **姿態估計 (Pose Estimation)**: 估計人體或物體的姿態

### 3.2 化工領域應用案例

#### 3.2.1 產品品質檢測

- **應用**: 使用 CNN 分析產品表面影像,自動檢測缺陷
- **範例**: 塑膠製品表面瑕疵檢測、藥品外觀品質控制
- **優勢**: 自動化、高效率、一致性高

#### 3.2.2 顯微影像分析

- **應用**: 分析顯微鏡影像,識別晶體結構、顆粒形態
- **範例**: 結晶過程監控、催化劑表面分析
- **優勢**: 能夠識別複雜的微觀結構特徵

#### 3.2.3 製程監控與異常檢測

- **應用**: 分析製程中的影像數據,即時檢測異常狀況
- **範例**: 反應器內部狀態監控、管線腐蝕檢測
- **優勢**: 即時性、預防性維護

#### 3.2.4 光譜影像分析

- **應用**: 分析高光譜影像,進行成分識別和濃度預測
- **範例**: 近紅外光譜影像分析、拉曼光譜影像處理
- **優勢**: 結合空間和光譜信息

#### 3.2.5 安全監控

- **應用**: 分析監控影像,識別危險行為或異常事件
- **範例**: 人員穿戴防護裝備檢測、洩漏檢測
- **優勢**: 提高工廠安全性

---

## 4. 使用 TensorFlow/Keras 建立 CNN 模型

### 4.1 導入必要的模組

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
```

### 4.2 CNN 常用的層 (Layers)

#### 4.2.1 卷積層 (Conv2D)

**功能**: 對輸入進行二維卷積運算,提取空間特徵。

**主要參數**:
- `filters`: 卷積核的數量 (輸出特徵圖的數量)
- `kernel_size`: 卷積核的尺寸,例如 `(3, 3)` 或 `3`
- `strides`: 步長,預設為 `(1, 1)`
- `padding`: 填充方式,`'valid'` (不填充) 或 `'same'` (填充使輸出尺寸相同)
- `activation`: 激活函數,例如 `'relu'`
- `input_shape`: 輸入形狀 (僅第一層需要指定)

**使用範例**:
```python
# 第一個卷積層,需要指定 input_shape
layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', 
              input_shape=(28, 28, 1), padding='same')

# 後續卷積層
layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')
```

**輸出形狀計算**:
- `padding='valid'`: 輸出尺寸 = $\lfloor \frac{W - K}{S} \rfloor + 1$
- `padding='same'`: 輸出尺寸 = $\lceil \frac{W}{S} \rceil$

#### 4.2.2 最大池化層 (MaxPooling2D)

**功能**: 對特徵圖進行下採樣,取局部區域的最大值。

**主要參數**:
- `pool_size`: 池化窗口大小,例如 `(2, 2)` 或 `2`
- `strides`: 步長,預設與 `pool_size` 相同
- `padding`: 填充方式,預設為 `'valid'`

**使用範例**:
```python
layers.MaxPooling2D(pool_size=(2, 2))
```

**效果**: 將特徵圖的空間維度減半 (使用 2×2 池化窗口時)

#### 4.2.3 平均池化層 (AveragePooling2D)

**功能**: 對特徵圖進行下採樣,取局部區域的平均值。

**主要參數**: 與 MaxPooling2D 相同

**使用範例**:
```python
layers.AveragePooling2D(pool_size=(2, 2))
```

**比較**: 
- MaxPooling 保留最顯著的特徵,常用於分類任務
- AveragePooling 保留整體信息,有時用於特定架構

#### 4.2.4 展平層 (Flatten)

**功能**: 將多維特徵圖展平為一維向量,用於連接卷積層和全連接層。

**使用範例**:
```python
layers.Flatten()
```

**範例**: 輸入形狀 `(7, 7, 64)` → 輸出形狀 `(3136,)`

#### 4.2.5 全連接層 (Dense)

**功能**: 標準的全連接神經網路層。

**主要參數**:
- `units`: 神經元數量
- `activation`: 激活函數

**使用範例**:
```python
layers.Dense(units=128, activation='relu')  # 隱藏層
layers.Dense(units=10, activation='softmax')  # 輸出層 (10 類分類)
```

#### 4.2.6 Dropout 層

**功能**: 隨機丟棄一定比例的神經元,防止過擬合。

**主要參數**:
- `rate`: 丟棄比例,例如 `0.5` 表示丟棄 50% 的神經元

**使用範例**:
```python
layers.Dropout(rate=0.5)
```

**注意**: Dropout 僅在訓練時啟用,預測時自動關閉。

#### 4.2.7 批次正規化層 (BatchNormalization)

**功能**: 對每個批次的數據進行正規化,加速訓練並提高穩定性。

**使用範例**:
```python
layers.BatchNormalization()
```

**優點**:
- 加速收斂
- 允許使用更高的學習率
- 減少對初始化的敏感性
- 具有一定的正則化效果

**放置位置**: 通常放在卷積層或全連接層之後、激活函數之前

### 4.3 建立 CNN 模型

#### 4.3.1 使用 Sequential API

**Sequential API** 適合建立線性堆疊的模型。

**範例: 簡單的 CNN 模型**
```python
model = models.Sequential([
    # 第一個卷積塊
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # 第二個卷積塊
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # 第三個卷積塊
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # 展平並連接全連接層
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

#### 4.3.2 使用 Functional API

**Functional API** 更靈活,適合建立複雜的模型 (例如多輸入、多輸出、殘差連接)。

**範例: 使用 Functional API 建立相同模型**
```python
inputs = keras.Input(shape=(28, 28, 1))

# 第一個卷積塊
x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = layers.MaxPooling2D((2, 2))(x)

# 第二個卷積塊
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

# 第三個卷積塊
x = layers.Conv2D(64, (3, 3), activation='relu')(x)

# 展平並連接全連接層
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

#### 4.3.3 參數計算詳解

理解模型參數數量對於評估模型複雜度和防止過擬合至關重要。

**卷積層參數公式:**

$$
\text{Params} = (K_h \times K_w \times C_{\text{in}} + 1) \times C_{\text{out}}
$$

其中:
- $K_h, K_w$: 濾波器高度、寬度 (此處為 3×3)
- $C_{\text{in}}$: 輸入通道數
- $C_{\text{out}}$: 輸出通道數 (濾波器個數)
- $+1$: 偏置項 (每個輸出通道一個)

**全連接層參數公式:**

$$
\text{Params} = (N_{\text{in}} + 1) \times N_{\text{out}}
$$

其中:
- $N_{\text{in}}$: 輸入神經元數量
- $N_{\text{out}}$: 輸出神經元數量

**範例模型參數統計:**

以上述 MNIST CNN 模型為例 (使用 `padding='same'` 版本):

| 層別 | 參數計算 | 參數量 | 佔比 |
|------|---------|--------|------|
| Conv2D(32) | $(3 \times 3 \times 1 + 1) \times 32$ | 320 | 0.08% |
| Conv2D(64) | $(3 \times 3 \times 32 + 1) \times 64$ | 18,496 | 4.40% |
| Conv2D(64) | $(3 \times 3 \times 64 + 1) \times 64$ | 36,928 | 8.78% |
| Dense(128) | $(3136 + 1) \times 128$ | 401,536 | 95.45% |
| Dense(10) | $(128 + 1) \times 10$ | 1,290 | 0.31% |
| **總計** | | **421,642** | 100% |

**關鍵觀察:**

1. **全連接層佔主導**: 95% 的參數集中在全連接層  
   → 這是為何現代架構 (如 ResNet) 減少全連接層的原因

2. **卷積層高效**: 僅 13% 參數卻提取關鍵特徵  
   → 參數共享 (Parameter Sharing) 的威力

3. **計算複雜度分析**:  
   雖然全連接層參數多,但卷積層的浮點運算 (FLOPs) 更高:

$$
\text{FLOPs}_{\text{conv}} = 2 \times H_{\text{out}} \times W_{\text{out}} \times K_h \times K_w \times C_{\text{in}} \times C_{\text{out}}
$$

**化工應用啟示:**
- 參數量 ≈ 模型複雜度 → 需要更多數據防止過擬合
- 大型模型訓練需要更多計算資源 (GPU)
- 工業部署時考慮「模型壓縮」(Pruning, Quantization)

### 4.4 常用的激活函數

#### 4.4.1 ReLU (Rectified Linear Unit)

**數學定義**:

$$
\text{ReLU}(x) = \max(0, x)
$$

**特點**:
- 計算簡單、效率高
- 緩解梯度消失問題
- CNN 中最常用的激活函數

**使用**:
```python
layers.Conv2D(32, (3, 3), activation='relu')
```

#### 4.4.2 Softmax

**數學定義**:

$$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$

**特點**:
- 將輸出轉換為機率分布
- 所有輸出值總和為 1
- 用於多類別分類的輸出層

**使用**:
```python
layers.Dense(10, activation='softmax')  # 10 類分類
```

#### 4.4.3 Sigmoid

**數學定義**:

$$
\text{Sigmoid}(x) = \frac{1}{1 + e^{-x}}
$$

**特點**:
- 輸出範圍 (0, 1)
- 用於二元分類的輸出層

**使用**:
```python
layers.Dense(1, activation='sigmoid')  # 二元分類
```

#### 4.4.4 其他激活函數

- **Leaky ReLU**: $\text{LeakyReLU}(x) = \max(\alpha x, x)$,其中 $\alpha$ 是小的正數 (例如 0.01)
- **ELU (Exponential Linear Unit)**: 平滑版本的 ReLU
- **Tanh**: $\text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$,輸出範圍 (-1, 1)

---

## 5. 模型編譯 (Model Compilation)

### 5.1 model.compile() 方法

**功能**: 配置模型的學習過程,指定優化器、損失函數和評估指標。

**基本語法**:
```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### 5.2 優化器 (Optimizer)

優化器決定了模型如何根據損失函數更新權重。

#### 5.2.1 Adam (Adaptive Moment Estimation)

**特點**: 結合動量和自適應學習率,是目前最常用的優化器。

**使用方式**:
```python
# 方式 1: 使用預設參數
model.compile(optimizer='adam', ...)

# 方式 2: 自定義學習率
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.001), ...)
```

**主要參數**:
- `learning_rate`: 學習率,預設為 0.001
- `beta_1`: 一階動量的指數衰減率,預設為 0.9
- `beta_2`: 二階動量的指數衰減率,預設為 0.999

#### 5.2.2 SGD (Stochastic Gradient Descent)

**特點**: 經典的梯度下降法,可加入動量。

**使用方式**:
```python
from tensorflow.keras.optimizers import SGD
model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), ...)
```

**主要參數**:
- `learning_rate`: 學習率
- `momentum`: 動量,預設為 0.0
- `nesterov`: 是否使用 Nesterov 動量,預設為 False

#### 5.2.3 RMSprop

**特點**: 適合處理非平穩目標,常用於 RNN。

**使用方式**:
```python
from tensorflow.keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(learning_rate=0.001), ...)
```

### 5.3 損失函數 (Loss Function)

損失函數衡量模型預測與真實值之間的差異。

#### 5.3.1 分類問題的損失函數

**Sparse Categorical Crossentropy**:
- **使用時機**: 多類別分類,標籤為整數 (例如 0, 1, 2, ...)
- **數學定義**: $L = -\log(p_{y_{true}})$,其中 $p_{y_{true}}$ 是真實類別的預測機率

```python
model.compile(loss='sparse_categorical_crossentropy', ...)
```

**Categorical Crossentropy**:
- **使用時機**: 多類別分類,標籤為 one-hot 編碼
- **數學定義**: $L = -\sum_{i} y_i \log(p_i)$

```python
model.compile(loss='categorical_crossentropy', ...)
```

**Binary Crossentropy**:
- **使用時機**: 二元分類
- **數學定義**: $L = -[y \log(p) + (1-y) \log(1-p)]$

```python
model.compile(loss='binary_crossentropy', ...)
```

#### 5.3.2 回歸問題的損失函數

**Mean Squared Error (MSE)**:
- **數學定義**: $\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$

```python
model.compile(loss='mse', ...)
```

**Mean Absolute Error (MAE)**:
- **數學定義**: $\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$

```python
model.compile(loss='mae', ...)
```

### 5.4 評估指標 (Metrics)

評估指標用於監控模型的訓練和測試性能。

**常用指標**:
- `'accuracy'`: 準確率 (分類問題)
- `'mae'`: 平均絕對誤差 (回歸問題)
- `'mse'`: 均方誤差 (回歸問題)

**使用方式**:
```python
# 單一指標
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 多個指標
model.compile(optimizer='adam', loss='mse', 
              metrics=['mae', 'mse'])
```

### 5.5 查看模型結構

#### 5.5.1 model.summary()

**功能**: 顯示模型的層結構、輸出形狀和參數數量。

**使用方式**:
```python
model.summary()
```

**輸出範例**:
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 1600)              0         
_________________________________________________________________
dense (Dense)                (None, 64)                102464    
_________________________________________________________________
dense_1 (Dense)              (None, 10)                650       
=================================================================
Total params: 121,930
Trainable params: 121,930
Non-trainable params: 0
_________________________________________________________________
```

**解讀**:
- **Output Shape**: 每層的輸出形狀 (None 表示批次大小)
- **Param #**: 每層的參數數量
- **Total params**: 模型總參數數量

---

## 6. 模型訓練 (Model Training)

### 6.1 model.fit() 方法

**功能**: 訓練模型。

**基本語法**:
```python
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
```

### 6.2 主要參數說明

#### 6.2.1 epochs

**定義**: 訓練的輪數,即整個訓練集被遍歷的次數。

**範例**:
```python
model.fit(x_train, y_train, epochs=20)
```

**建議**: 
- 從較小的 epochs 開始 (例如 10-20)
- 觀察訓練曲線,根據收斂情況調整
- 使用 Early Stopping 自動停止訓練

#### 6.2.2 batch_size

**定義**: 每次梯度更新使用的樣本數量。

**範例**:
```python
model.fit(x_train, y_train, batch_size=64)
```

**常用值**: 32, 64, 128, 256

**影響**:
- **較小的 batch_size**: 訓練較慢,但可能有更好的泛化能力
- **較大的 batch_size**: 訓練較快,但需要更多記憶體

#### 6.2.3 validation_split

**定義**: 從訓練集中分割出一部分作為驗證集的比例。

**範例**:
```python
model.fit(x_train, y_train, validation_split=0.2)  # 20% 作為驗證集
```

**注意**: 驗證集從訓練數據的**末尾**分割,不會打亂順序。

#### 6.2.4 validation_data

**定義**: 直接提供驗證集數據。

**範例**:
```python
model.fit(x_train, y_train, validation_data=(x_val, y_val))
```

**比較**:
- `validation_split`: 自動分割,方便但不靈活
- `validation_data`: 手動提供,更靈活

#### 6.2.5 verbose

**定義**: 訓練過程的顯示模式。

**選項**:
- `0`: 不顯示
- `1`: 顯示進度條 (預設)
- `2`: 每個 epoch 顯示一行

#### 6.2.6 callbacks

**定義**: 在訓練過程中執行的回調函數列表。

**範例**:
```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

model.fit(x_train, y_train, callbacks=callbacks)
```

### 6.3 常用的 Callbacks

#### 6.3.1 EarlyStopping

**功能**: 當監控指標不再改善時,提前停止訓練。

**主要參數**:
- `monitor`: 監控的指標,例如 `'val_loss'` 或 `'val_accuracy'`
- `patience`: 容忍的 epochs 數量
- `restore_best_weights`: 是否恢復最佳權重,預設為 False

**範例**:
```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

model.fit(x_train, y_train, callbacks=[early_stop])
```

#### 6.3.2 ModelCheckpoint

**功能**: 在訓練過程中保存模型。

**主要參數**:
- `filepath`: 保存路徑
- `monitor`: 監控的指標
- `save_best_only`: 是否只保存最佳模型
- `save_weights_only`: 是否只保存權重

**範例**:
```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    filepath='best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

model.fit(x_train, y_train, callbacks=[checkpoint])
```

#### 6.3.3 ReduceLROnPlateau

**功能**: 當監控指標停止改善時,降低學習率。

**主要參數**:
- `monitor`: 監控的指標
- `factor`: 學習率降低的倍數
- `patience`: 容忍的 epochs 數量
- `min_lr`: 學習率的最小值

**範例**:
```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

model.fit(x_train, y_train, callbacks=[reduce_lr])
```

#### 6.3.4 TensorBoard

**功能**: 將訓練過程記錄到 TensorBoard,用於視覺化分析。

**範例**:
```python
from tensorflow.keras.callbacks import TensorBoard
import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x_train, y_train, callbacks=[tensorboard_callback])
```

**啟動 TensorBoard**:
```bash
tensorboard --logdir=logs/fit
```

---

## 7. 訓練歷史記錄與視覺化

### 7.1 History 物件

`model.fit()` 返回一個 `History` 物件,記錄了訓練過程中的指標。

**範例**:
```python
history = model.fit(x_train, y_train, epochs=20, validation_split=0.2)
```

**History 物件的屬性**:
- `history.history`: 字典,包含每個 epoch 的指標值

**可用的鍵**:
- `'loss'`: 訓練損失
- `'accuracy'`: 訓練準確率 (如果有)
- `'val_loss'`: 驗證損失 (如果有驗證集)
- `'val_accuracy'`: 驗證準確率 (如果有驗證集)

### 7.2 視覺化訓練過程

**範例: 繪製損失和準確率曲線**
```python
import matplotlib.pyplot as plt

# 提取歷史記錄
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']
epochs_range = range(1, len(loss) + 1)

# 繪製損失曲線
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss, 'bo-', label='Training Loss')
plt.plot(epochs_range, val_loss, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 繪製準確率曲線
plt.subplot(1, 2, 2)
plt.plot(epochs_range, accuracy, 'bo-', label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

**解讀訓練曲線**:
- **訓練損失持續下降,驗證損失也下降**: 模型正常訓練
- **訓練損失下降,驗證損失上升**: 過擬合 (Overfitting)
- **訓練損失和驗證損失都很高**: 欠擬合 (Underfitting)

### 7.3 過擬合診斷與正則化

#### 7.3.1 什麼是過擬合？

**定義**: 模型在訓練集表現很好,但在未見過的數據 (驗證/測試集) 表現差。

**化工類比**:
> 就像只用「標準品」標定分析儀器,卻用來測量複雜的實際樣品  
> 儀器「記住」了標準品特徵,但對真實樣品泛化能力差

**症狀**:
- 訓練 Accuracy → 99%+
- 驗證 Accuracy → 85-90% (差距 > 10%)
- 驗證 Loss 先降後升 (U 型曲線)

#### 7.3.2 過擬合的成因

1. **模型容量過大**  
   例如:用 3 層 CNN + 512 neurons 全連接層訓練 1000 張影像

2. **訓練資料太少**  
   工業缺陷檢測常見:正常品 10,000 張,缺陷品僅 50 張

3. **訓練時間過長**  
   Epoch 過多,模型開始「背答案」

#### 7.3.3 正則化技術

**(1) Dropout**

在訓練時隨機「關閉」部分神經元 (如 50%),強迫網路學習更魯棒的特徵:

```python
Dense(128, activation='relu'),
Dropout(0.5),  # 訓練時隨機丟棄 50% 神經元
Dense(10, activation='softmax')
```

**數學原理**:

$$
\mathbf{h}_{\text{dropout}} = \mathbf{h} \odot \mathbf{m}, \quad m_i \sim \text{Bernoulli}(0.5)
$$

**效果**:
- 類似「集成學習」(Ensemble):每次 batch 訓練不同子網路
- 測試時使用完整網路 (但權重乘以 keep_prob)

**(2) L2 正則化 (Weight Decay)**

在損失函數中加入權重懲罰項:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \sum_{i} w_i^2
$$

```python
from tensorflow.keras import regularizers

Dense(128, activation='relu', 
      kernel_regularizer=regularizers.l2(0.001))
```

**效果**:
- 限制權重大小,避免模型過度依賴少數特徵
- $\lambda$ 越大,正則化越強 (但可能欠擬合)

**(3) 資料增強 (Data Augmentation)**

人工擴充訓練集,模擬真實變化:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,      # 隨機旋轉 ±10°
    width_shift_range=0.1,  # 水平平移 ±10%
    height_shift_range=0.1, # 垂直平移 ±10%
    zoom_range=0.1          # 縮放 ±10%
)

model.fit(datagen.flow(train_images, train_labels, batch_size=128),
          epochs=10)
```

**化工應用**:
- 鋼材缺陷影像:旋轉、翻轉、亮度調整
- 顯微鏡影像:縮放、對比度變化

#### 7.3.4 過擬合 vs 欠擬合判斷

| 指標 | 欠擬合 | 適配良好 | 過擬合 |
|------|--------|---------|--------|
| 訓練 Acc | 低 (< 90%) | 高 (95-98%) | 極高 (> 99%) |
| 驗證 Acc | 低 (< 90%) | 高 (95-98%) | 中 (85-92%) |
| 訓練/驗證差距 | 小 (< 3%) | 小 (< 5%) | 大 (> 10%) |
| 解決方法 | 增加容量、訓練更久 | - | Dropout, L2, 資料增強 |

---

## 8. 進階: TensorBoard 使用教學

### 8.1 TensorBoard 簡介

**TensorBoard** 是 TensorFlow 提供的視覺化工具,可以:
- 視覺化訓練指標 (損失、準確率等)
- 視覺化模型結構
- 顯示權重和梯度的分布
- 顯示影像、文字等數據

### 8.2 基本使用流程

**步驟 1: 設定 TensorBoard Callback**
```python
from tensorflow.keras.callbacks import TensorBoard
import datetime

# 建立日誌目錄
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# 建立 TensorBoard callback
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,  # 每個 epoch 記錄權重分布
    write_graph=True,  # 記錄模型結構
    write_images=True  # 記錄影像
)
```

**步驟 2: 訓練模型**
```python
history = model.fit(
    x_train, y_train,
    epochs=20,
    validation_data=(x_val, y_val),
    callbacks=[tensorboard_callback]
)
```

**步驟 3: 啟動 TensorBoard**

在 Jupyter Notebook 中:
```python
%load_ext tensorboard
%tensorboard --logdir logs/fit
```

在命令列中:
```bash
tensorboard --logdir=logs/fit
```

然後在瀏覽器中開啟 `http://localhost:6006`

### 8.3 TensorBoard 功能介紹

#### 8.3.1 Scalars (標量)

顯示訓練過程中的標量指標,例如損失和準確率。

#### 8.3.2 Graphs (圖)

顯示模型的計算圖結構。

#### 8.3.3 Distributions (分布)

顯示權重和偏差的分布隨時間的變化。

#### 8.3.4 Histograms (直方圖)

顯示權重和偏差的直方圖。

---

## 9. 模型評估 (Model Evaluation)

### 9.1 model.evaluate() 方法

**功能**: 在測試集上評估模型性能。

**基本語法**:
```python
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
```

**返回值**:
- 如果只有一個指標 (例如只有損失),返回單一值
- 如果有多個指標,返回列表

**參數**:
- `x`: 測試數據
- `y`: 測試標籤
- `batch_size`: 批次大小,預設為 32
- `verbose`: 顯示模式

### 9.2 詳細評估

對於分類問題,可以使用混淆矩陣、分類報告等進行詳細評估。

**範例**:
```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 預測
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# 混淆矩陣
cm = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(cm)

# 分類報告
report = classification_report(y_test, y_pred_classes)
print("\nClassification Report:")
print(report)
```

### 9.3 混淆矩陣深入分析

混淆矩陣是評估分類模型性能的強大工具,能夠揭示模型的系統性錯誤。

#### 9.3.1 混淆矩陣定義

對於 $K$ 類分類問題,混淆矩陣是一個 $K \times K$ 方陣,元素 $C_{ij}$ 表示「真實類別為 $i$,預測為 $j$」的樣本數。

**視覺化混淆矩陣**:
```python
import seaborn as sns
import matplotlib.pyplot as plt

# 計算混淆矩陣
cm = confusion_matrix(y_test, y_pred_classes)

# 繪製熱圖
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('預測標籤', fontsize=12)
plt.ylabel('真實標籤', fontsize=12)
plt.title('混淆矩陣', fontsize=14, fontweight='bold')
plt.show()

# 計算每個類別的準確率
for i in range(10):
    class_accuracy = cm[i, i] / cm[i, :].sum()
    print(f'數字 {i}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)')
```

#### 9.3.2 評估指標數學定義

**Precision (精確率)**:

$$
\text{Precision}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FP}_k} = \frac{C_{kk}}{\sum_{i=0}^{K-1} C_{ik}}
$$

「預測為類別 $k$ 的樣本中,實際為 $k$ 的比例」

**Recall (召回率)**:

$$
\text{Recall}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k} = \frac{C_{kk}}{\sum_{j=0}^{K-1} C_{kj}}
$$

「實際為類別 $k$ 的樣本中,被正確預測的比例」

**F1 Score (調和平均)**:

$$
F1_k = 2 \times \frac{\text{Precision}_k \times \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}
$$

#### 9.3.3 常見混淆模式分析

**對角線元素 (正確預測)**:
- 數值越大越好
- 表示該類別的正確預測數量

**非對角線元素 (系統性錯誤)**:
- 揭示哪些類別容易被混淆
- 例如:MNIST 中 '5' 常被誤判為 '3'

**分析範例**:
```python
# 找出最容易混淆的數字對
errors = []
for i in range(10):
    for j in range(10):
        if i != j and cm[i, j] > 0:
            errors.append((i, j, cm[i, j]))

# 排序並顯示前 5 個
errors_sorted = sorted(errors, key=lambda x: x[2], reverse=True)
print("最容易混淆的數字對:")
for true_label, pred_label, count in errors_sorted[:5]:
    print(f"數字 {true_label} 被誤判為 {pred_label}: {count} 次")
```

#### 9.3.4 化工應用:成本敏感決策

在工業應用中,不同類型的錯誤有不同的成本:

**鋼材缺陷檢測範例**:

| 錯誤類型 | 成本 | 應對策略 |
|---------|------|---------|
| 漏檢裂紋 (FN) | 極高 (產品召回、安全事故) | 最大化 Recall |
| 誤報裂紋 (FP) | 中等 (人工複檢成本) | 接受一定 FP |

**調整策略**:
```python
# 調整類別權重
class_weight = {
    0: 1,  # 正常
    1: 5,  # 裂紋 (提高權重)
    2: 3,  # 劃痕
    # ...
}

model.fit(x_train, y_train, class_weight=class_weight)
```

**雙閾值決策系統**:

| 置信度範圍 | 決策 | 化工類比 |
|-----------|------|----------|
| > 95% | 自動分類 | 合格品直接放行 |
| 70-95% | 人工複檢 | 送資深工程師判定 |
| < 70% | 重新拍攝 | 可能影像品質問題 |

---

## 10. 模型預測 (Model Prediction)

### 10.1 model.predict() 方法

**功能**: 對新數據進行預測。

**基本語法**:
```python
predictions = model.predict(x_new)
```

**返回值**:
- 對於分類問題,返回每個類別的機率
- 對於回歸問題,返回預測值

### 10.2 分類問題的預測

**範例: 預測單一樣本**
```python
# 選擇一個測試樣本
sample = x_test[0:1]  # 保持維度 (1, 28, 28, 1)

# 預測
prediction = model.predict(sample)
predicted_class = np.argmax(prediction)
confidence = np.max(prediction)

print(f'Predicted Class: {predicted_class}')
print(f'Confidence: {confidence:.4f}')
```

**範例: 批次預測**
```python
# 預測多個樣本
predictions = model.predict(x_test[:10])
predicted_classes = np.argmax(predictions, axis=1)

print('Predicted Classes:', predicted_classes)
```

### 10.3 視覺化預測結果

**範例: 顯示預測結果與真實標籤**
```python
import matplotlib.pyplot as plt

# 預測
predictions = model.predict(x_test[:25])
predicted_classes = np.argmax(predictions, axis=1)

# 視覺化
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'Pred: {predicted_classes[i]}\nTrue: {y_test[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()
```

---

## 11. 模型保存與載入

### 11.1 保存整個模型

**方法 1: 使用 model.save() (推薦)**

保存模型架構、權重和訓練配置。

**保存**:
```python
# 保存為 .h5 格式
model.save('my_cnn_model.h5')

# 或保存為 SavedModel 格式 (TensorFlow 2.x 預設)
model.save('my_cnn_model')
```

**載入**:
```python
from tensorflow.keras.models import load_model

# 載入模型
loaded_model = load_model('my_cnn_model.h5')

# 或
loaded_model = load_model('my_cnn_model')

# 使用載入的模型進行預測
predictions = loaded_model.predict(x_test)
```

### 11.2 只保存權重

**保存**:
```python
model.save_weights('my_model_weights.h5')
```

**載入**:
```python
# 需要先建立相同架構的模型
model = create_model()  # 自定義函數建立模型
model.load_weights('my_model_weights.h5')
```

### 11.3 保存模型架構

**保存為 JSON**:
```python
# 保存架構
model_json = model.to_json()
with open('model_architecture.json', 'w') as json_file:
    json_file.write(model_json)
```

**載入架構**:
```python
from tensorflow.keras.models import model_from_json

# 載入架構
with open('model_architecture.json', 'r') as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)

# 載入權重
loaded_model.load_weights('my_model_weights.h5')
```

---

## 12. 進階主題

### 12.1 卷積核可視化

理解 CNN 學到了什麼特徵是模型解釋性的重要一環。

#### 12.1.1 第一層卷積核可視化

```python
# 提取第一層卷積核
kernels = model.layers[0].get_weights()[0]  # 形狀: (3, 3, 1, 32)

# 繪製前 16 個濾波器
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    if i < 32:
        ax.imshow(kernels[:, :, 0, i], cmap='gray')
        ax.set_title(f'Filter {i}', fontsize=9)
    ax.axis('off')

plt.suptitle('第一層卷積核 (前 16 個)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

**典型學習結果**:
- **水平邊緣檢測器**: 檢測數字的橫線 (如 '5' 的頂部)
- **垂直邊緣檢測器**: 檢測數字的豎線 (如 '1')
- **45° 斜線檢測器**: 檢測斜筆畫 (如 '7' 的斜槓)
- **圓弧檢測器**: 檢測彎曲部分 (如 '8', '6', '9')

#### 12.1.2 特徵圖可視化

```python
from tensorflow.keras import Model

# 建立中間層輸出模型
layer_outputs = [layer.output for layer in model.layers[:4]]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# 選一張影像
test_img = x_test[0:1]

# 獲取各層特徵圖
activations = activation_model.predict(test_img)

# 繪製第一層卷積後的 32 個特徵圖
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    if i < 32:
        ax.imshow(activations[0][0, :, :, i], cmap='viridis')
        ax.set_title(f'Feature {i}', fontsize=8)
    ax.axis('off')

plt.suptitle('第一層卷積特徵圖', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

**觀察結果**:
- **第一層**: 低階特徵 (邊緣、角點)
- **第二層**: 中階特徵 (筆畫組合、局部形狀)
- **全連接層**: 高階抽象特徵 (數字身份)

### 12.2 從 MNIST 到工業應用

#### 12.2.1 關鍵差異

| 特徵 | MNIST | 工業影像檢測 |
|------|-------|-------------|
| **資料量** | 60,000 訓練樣本 | 常 < 1,000 樣本 |
| **類別平衡** | 均勻分佈 | 極度不平衡 (99:1) |
| **背景複雜度** | 純黑背景 | 雜訊、反光、污漬 |
| **尺度變化** | 居中、等大 | 位置、大小、角度隨機 |
| **即時性要求** | 無 | 毫秒級推論 |

#### 12.2.2 推論速度考量

**批次處理加速原理**:

批次推論將多張影像組成張量同時處理,充分利用 GPU 並行計算能力:

$$
\text{Speedup} = \frac{T_{\text{single}} \times B}{T_{\text{batch}}}
$$

其中:
- $T_{\text{single}}$: 單張推論時間
- $T_{\text{batch}}$: 批次推論總時間
- $B$: 批次大小

**工業場景適配性**:

| 應用場景 | 處理速度需求 | 典型 CNN 表現 | 評估 |
|---------|------------|-------------|------|
| **高速產線** (鋼材連鑄) | < 10 ms/張 | 1-2 ms/張 | ✓✓ 大幅超標 |
| **中速產線** (零件檢測) | < 50 ms/張 | 1-2 ms/張 | ✓ 符合需求 |
| **批次檢測** (終端品檢) | < 200 ms/張 | 1-2 ms/張 | ✓ 綽綽有餘 |

#### 12.2.3 模型優化策略

**1. 量化 (Quantization): FP32 → INT8**

$$
w_{\text{int8}} = \text{round}\left( \frac{w_{\text{fp32}} - \min(w)}{\max(w) - \min(w)} \times 255 \right)
$$

- 速度提升: 2-4 倍
- 模型縮小: 4 倍
- 準確率損失: < 1%

**2. 剪枝 (Pruning): 移除不重要的權重**

$$
w_i^{\text{new}} = \begin{cases} w_i & \text{if } |w_i| > \tau \\ 0 & \text{otherwise} \end{cases}
$$

- 可減少 50-70% 參數
- 需 Fine-tuning 恢復準確率

**3. 知識蒸餾 (Knowledge Distillation)**

訓練小模型模仿大模型:

$$
\mathcal{L}_{\text{KD}} = \alpha \mathcal{L}_{\text{CE}} + (1-\alpha) \text{KL}(p_{\text{teacher}} \| p_{\text{student}})
$$

#### 12.2.4 部署架構建議

```
化工產線影像檢測系統架構:

[相機陣列] → [邊緣裝置] → [本地伺服器] → [雲端監控]
     ↓           ↓              ↓              ↓
  拍攝影像   即時推論(90%)   複雜案例(8%)    數據分析
             1-5 ms         10-20 ms       離線處理
             
             置信度 > 95%: 自動通過/拒絕
             置信度 70-95%: 送本地伺服器
             置信度 < 70%: 人工複檢
```

---

## 13. 總結

本單元介紹了卷積神經網路 (CNN) 的基本原理和使用 TensorFlow/Keras 建立 CNN 模型的完整流程:

### 13.1 核心概念
- CNN 透過卷積運算提取空間特徵,具有局部連接、參數共享和平移不變性
- 典型 CNN 架構包含卷積層、池化層、展平層和全連接層
- 常見的 CNN 架構包括 LeNet、AlexNet、VGG、ResNet、MobileNet、EfficientNet 等
- 理解參數計算對於評估模型複雜度至關重要

### 13.2 實作流程
1. **建立模型**: 使用 Sequential 或 Functional API 堆疊層
2. **編譯模型**: 指定優化器、損失函數和評估指標
3. **訓練模型**: 使用 `model.fit()` 訓練,配合 callbacks 控制訓練過程
4. **評估模型**: 使用 `model.evaluate()` 在測試集上評估性能
5. **預測**: 使用 `model.predict()` 對新數據進行預測
6. **保存/載入**: 使用 `model.save()` 和 `load_model()` 保存和載入模型

### 13.3 重要技巧
- **防止過擬合**: 使用 Dropout、L2 正則化、資料增強
- **訓練控制**: 使用 EarlyStopping、ModelCheckpoint、ReduceLROnPlateau
- **模型診斷**: 視覺化訓練曲線、混淆矩陣分析
- **模型解釋**: 卷積核可視化、特徵圖可視化
- **性能監控**: 使用 TensorBoard 視覺化訓練過程

### 13.4 化工應用
CNN 在化工領域有廣泛應用,包括:
- **產品品質檢測**: 表面缺陷自動檢測
- **顯微影像分析**: 晶體結構、顆粒形態識別
- **製程監控**: 即時異常檢測與預防性維護
- **光譜影像分析**: 成分識別和濃度預測
- **安全監控**: 危險行為識別、洩漏檢測

### 13.5 工業部署考量
- **資料挑戰**: 工業數據常面臨樣本少、類別不平衡問題
- **速度要求**: 批次處理可達 1-2 ms/張,滿足高速產線需求
- **模型優化**: 量化、剪枝、知識蒸餾可大幅減少模型大小和推論時間
- **部署架構**: 邊緣裝置 + 本地伺服器 + 雲端監控的混合架構
- **成本敏感**: 根據錯誤成本調整類別權重和決策閾值

---

## 參考資料

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
2. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.
3. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems*, 25.
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 770-778.
5. Howard, A. G., et al. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications. *arXiv preprint arXiv:1704.04861*.
6. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *International Conference on Machine Learning*, 6105-6114.
7. TensorFlow Documentation: https://www.tensorflow.org/
8. Keras Documentation: https://keras.io/

---

**下一步**: 請參考 `Unit16_CNN_Overview.ipynb` 進行實際的程式演練,學習如何使用 MNIST 手寫數字資料集建立和訓練 CNN 模型。
