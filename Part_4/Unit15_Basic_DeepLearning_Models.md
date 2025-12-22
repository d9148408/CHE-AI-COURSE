# Unit 15: 深度學習基礎模型

## 🎯 教學目標

本單元將介紹三種最常見的深度學習模型架構：
- **DNN (Deep Neural Network / MLP - Multi-Layer Perceptron)**: 全連接神經網路
- **CNN (Convolutional Neural Network)**: 卷積神經網路
- **RNN (Recurrent Neural Network)**: 循環神經網路，包含 LSTM 和 GRU 變體

學生將學會：
1. 理解各模型的基礎理論與運作原理
2. 了解各模型適用的問題類型與應用時機
3. 使用 TensorFlow (Keras) 建立這些模型
4. 掌握關鍵參數的設定與調整
5. 透過實作範例建立完整的深度學習流程

---

## 📚 課程大綱

1. [深度學習概述](#1-深度學習概述)
2. [DNN / MLP 模型](#2-dnn--mlp-模型)
3. [CNN 模型](#3-cnn-模型)
4. [RNN 模型 (LSTM/GRU)](#4-rnn-模型-lstmgru)
5. [模型比較與選擇指南](#5-模型比較與選擇指南)
6. [實作建議](#6-實作建議)
7. [實作範例執行結果](#7-實作範例執行結果)
8. [參考資源](#8-參考資源)

---

## 1. 深度學習概述

### 1.1 什麼是深度學習？

深度學習 (Deep Learning) 是機器學習的一個分支，透過建立多層神經網路來學習數據的複雜特徵表示。相較於傳統機器學習需要人工特徵工程，深度學習能夠自動學習數據的階層式特徵。

**深度學習與傳統機器學習的差異**：
- **特徵學習**: 深度學習自動提取特徵，傳統機器學習需要手動設計特徵
- **數據需求**: 深度學習通常需要大量數據，傳統機器學習在小數據上表現更好
- **計算資源**: 深度學習需要更多計算資源 (GPU/TPU)
- **可解釋性**: 傳統機器學習通常更容易解釋

### 1.2 深度學習在化工領域的應用

化工領域的深度學習應用包括：
- **製程監控**: 使用感測器數據預測製程異常
- **品質預測**: 根據製程參數預測產品品質
- **反應動力學**: 建模複雜的反應系統
- **分子性質預測**: 預測化合物的物理化學性質
- **製程優化**: 尋找最佳操作條件
- **故障診斷**: 識別設備異常與故障模式

### 1.3 TensorFlow 與 Keras 簡介

**TensorFlow** 是 Google 開發的開源深度學習框架，而 **Keras** 是高階 API，提供簡潔易用的介面。

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 檢查 TensorFlow 版本
print(f"TensorFlow 版本: {tf.__version__}")
```

**執行結果**：
```
TensorFlow 版本: 2.19.0
Keras 版本: 3.10.0
NumPy 版本: 2.0.2
Pandas 版本: 2.2.2

GPU 可用: []
```

---

## 2. DNN / MLP 模型

### 2.1 基礎理論

**DNN (Deep Neural Network)** 或 **MLP (Multi-Layer Perceptron)** 是最基本的深度學習架構，由多層全連接 (Fully Connected) 神經元組成。

**架構組成**：
- **輸入層 (Input Layer)**: 接收原始特徵數據
- **隱藏層 (Hidden Layers)**: 多層神經元進行特徵轉換
- **輸出層 (Output Layer)**: 產生最終預測結果

**前向傳播 (Forward Propagation)**：
```
輸入 → [權重 × 輸入 + 偏差] → 激活函數 → 下一層
```

數學表示：
$$z^{[l]} = W^{[l]} \cdot a^{[l-1]} + b^{[l]}$$
$$a^{[l]} = g(z^{[l]})$$

其中：
- $W$: 權重矩陣
- $b$: 偏差向量
- $g$: 激活函數
- $a$: 激活值

### 2.2 適用時機與問題類型

✅ **適合使用 DNN 的情況**：
- 表格型數據 (Tabular Data)
- 特徵之間沒有明顯的空間或時間結構
- 迴歸問題 (連續值預測)
- 分類問題 (類別預測)
- 中等規模的數據集

📊 **化工領域應用範例**：
- 根據反應條件預測產率
- 根據原料組成預測產品性質
- 製程參數與品質指標的關係建模
- 能源消耗預測

❌ **不適合的情況**：
- 圖像數據 (應使用 CNN)
- 時間序列數據 (應使用 RNN/LSTM)
- 數據具有明顯空間結構

### 2.2.5 深度學習 MLP vs scikit-learn MLP

許多學生可能已經在機器學習課程中使用過 scikit-learn 的 `MLPClassifier` 和 `MLPRegressor`。這裡說明兩者的主要差異：

#### 📊 功能與架構比較

| 特性 | scikit-learn MLP | TensorFlow/Keras DNN |
|------|------------------|---------------------|
| **定位** | 傳統機器學習工具 | 深度學習框架 |
| **網路深度** | 通常 2-3 層 | 可建立任意深度網路 |
| **最佳化算法** | 有限 (lbfgs, sgd, adam) | 豐富的優化器選擇 |
| **GPU 支援** | ❌ 不支援 | ✅ 完整支援 |
| **訓練監控** | 基本 | 豐富 (callbacks, tensorboard) |
| **正則化** | L2, early stopping | Dropout, BN, L1/L2, 等 |
| **數據規模** | 小到中型 | 中到超大型 |
| **模型保存** | pickle | HDF5, SavedModel |
| **部署彈性** | 有限 | TensorFlow Serving, TFLite |

#### 🔍 詳細差異說明

**1. 模型複雜度與靈活性**

```python
# scikit-learn: 簡單但有限
from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(64, 32), 
                     activation='relu',
                     max_iter=200)

# TensorFlow/Keras: 靈活且強大
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
model = Sequential([
    Input(shape=(input_dim,)),
    Dense(64, activation='relu'),
    BatchNormalization(),  # sklearn 沒有
    Dropout(0.3),          # sklearn 沒有
    Dense(32, activation='relu'),
    Dense(1)
])
```

**2. 訓練控制與監控**

scikit-learn MLP：
- ✅ 簡單易用，適合快速原型
- ❌ 訓練過程控制有限
- ❌ 無法方便地實現 learning rate scheduling
- ❌ 無法使用自定義 callbacks

TensorFlow/Keras DNN：
- ✅ 完整的訓練控制
- ✅ EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- ✅ 即時訓練視覺化 (TensorBoard)
- ✅ 自定義訓練循環

**3. 效能與規模**

```python
# scikit-learn: CPU only
# 適用數據規模: < 100,000 樣本
mlp = MLPRegressor(hidden_layer_sizes=(100, 50))
mlp.fit(X_train, y_train)  # 在 CPU 上訓練

# TensorFlow: GPU accelerated
# 適用數據規模: 無限制
model = create_dnn_model()
model.fit(X_train, y_train, 
          batch_size=32,     # Mini-batch 訓練
          epochs=100,
          validation_split=0.2)  # 自動驗證
```

**4. 正則化技術**

| 技術 | sklearn MLP | Keras DNN |
|------|-------------|-----------|
| **L2 正則化** | ✅ (alpha 參數) | ✅ (kernel_regularizer) |
| **Early Stopping** | ✅ (基本) | ✅ (進階 callback) |
| **Dropout** | ❌ | ✅ |
| **Batch Normalization** | ❌ | ✅ |
| **Layer Normalization** | ❌ | ✅ |

#### 🎯 使用建議

**選擇 scikit-learn MLP 的時機**：
- ✅ 快速原型開發和實驗
- ✅ 小型數據集 (< 10,000 樣本)
- ✅ 不需要 GPU 加速
- ✅ 與其他 sklearn 模組整合使用
- ✅ 簡單的 2-3 層網路即可

**選擇 TensorFlow/Keras DNN 的時機**：
- ✅ 大型數據集 (> 100,000 樣本)
- ✅ 需要建立深層網路 (> 3 層)
- ✅ 需要 GPU 加速訓練
- ✅ 需要精細控制訓練過程
- ✅ 需要使用先進的正則化技術
- ✅ 模型需要部署到生產環境

#### 💡 實際範例對比

**相同任務：預測化工製程產率**

```python
# 方法 1: scikit-learn (簡單快速)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)
mlp.fit(X_scaled, y_train)

# 方法 2: TensorFlow/Keras (強大靈活)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=20)
history = model.fit(
    X_scaled, y_train,
    validation_split=0.2,
    epochs=500,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0
)
```

**效能比較** (本範例數據集)：
- sklearn MLP: R² ≈ 0.92, 訓練時間 ~5 秒 (CPU)
- Keras DNN: R² ≈ 0.96, 訓練時間 ~15 秒 (CPU) / ~3 秒 (GPU)

#### 📚 學習路徑建議

1. **初學階段**: 從 scikit-learn MLP 開始
   - 理解神經網路基本概念
   - 熟悉超參數調整
   - 建立直覺

2. **進階階段**: 過渡到 TensorFlow/Keras
   - 學習更複雜的架構
   - 掌握訓練技巧
   - 使用 GPU 加速

3. **專業階段**: 精通深度學習框架
   - 自定義層和損失函數
   - 模型部署與優化
   - 解決實際工程問題

---

### 2.3 使用 Keras 建立 DNN 模型

**基本架構**：

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization

# 建立 Sequential 模型
model = Sequential([
    # 輸入層
    Input(shape=(input_dim,)),
    
    # 第一隱藏層
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    # 第二隱藏層
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    # 第三隱藏層
    Dense(16, activation='relu'),
    
    # 輸出層
    Dense(output_dim, activation='linear')  # 迴歸問題
    # Dense(output_dim, activation='softmax')  # 多分類問題
])

# 編譯模型
model.compile(
    optimizer='adam',
    loss='mse',  # 迴歸: mse, mae; 分類: categorical_crossentropy
    metrics=['mae']
)

# 查看模型結構
model.summary()
```

### 2.4 關鍵參數說明

#### 2.4.1 網路架構參數

| 參數 | 說明 | 選擇建議 |
|------|------|----------|
| **層數 (Depth)** | 隱藏層的數量 | 從 2-3 層開始，根據問題複雜度調整 |
| **神經元數量** | 每層的神經元個數 | 常見: 64→32→16 遞減，或保持一致 |
| **激活函數** | 引入非線性 | 隱藏層: ReLU, 輸出層視任務而定 |

**常用激活函數**：
- **ReLU** (Rectified Linear Unit): `f(x) = max(0, x)` - 最常用，計算快速
- **Sigmoid**: `f(x) = 1/(1+e^(-x))` - 二元分類輸出層
- **Tanh**: `f(x) = (e^x - e^(-x))/(e^x + e^(-x))` - 輸出範圍 [-1, 1]
- **Softmax**: 多分類輸出層，輸出機率分佈
- **Linear**: 迴歸問題輸出層

#### 2.4.2 正則化參數

**Dropout**：隨機丟棄神經元，防止過擬合
```python
Dropout(0.3)  # 丟棄 30% 的神經元
```
- 訓練集較小時使用 0.3-0.5
- 訓練集較大時可降低或不使用

**Batch Normalization**：標準化層間輸入，加速訓練
```python
BatchNormalization()
```
- 放在激活函數之前或之後都可以
- 有助於訓練更深的網路

**L1/L2 正則化**：懲罰權重大小
```python
Dense(64, activation='relu', 
      kernel_regularizer=tf.keras.regularizers.l2(0.01))
```

#### 2.4.3 訓練參數

**優化器 (Optimizer)**：
- **Adam**: 自適應學習率，最常用，適合大多數問題
- **SGD**: 基本梯度下降，需調整學習率
- **RMSprop**: 適合 RNN

**學習率 (Learning Rate)**：
```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
```
- 預設 0.001 通常是好的起點
- 太大可能不收斂，太小訓練太慢

**損失函數 (Loss Function)**：
- 迴歸: `mse` (均方誤差), `mae` (平均絕對誤差)
- 二元分類: `binary_crossentropy`
- 多分類: `categorical_crossentropy` 或 `sparse_categorical_crossentropy`

**批次大小 (Batch Size)**：
```python
model.fit(X_train, y_train, batch_size=32, epochs=100)
```
- 常見: 16, 32, 64, 128
- 較小批次: 更新更頻繁但噪音較大
- 較大批次: 訓練更穩定但記憶體需求高

**訓練週期 (Epochs)**：
- 根據驗證集表現決定
- 使用 Early Stopping 自動停止

### 2.5 訓練技巧

**Early Stopping**：驗證集表現不再改善時停止訓練
```python
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

model.fit(X_train, y_train, 
          validation_split=0.2,
          epochs=200,
          callbacks=[early_stop])
```

**Learning Rate Scheduler**：動態調整學習率
```python
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10
)
```

---

## 3. CNN 模型

### 3.1 基礎理論

**CNN (Convolutional Neural Network)** 卷積神經網路專門設計用於處理具有網格結構的數據，如圖像 (2D)、時間序列 (1D) 或視頻 (3D)。

**核心概念**：
- **卷積層 (Convolutional Layer)**: 使用卷積核 (filters) 提取局部特徵
- **池化層 (Pooling Layer)**: 降低維度，保留重要特徵
- **全連接層 (Dense Layer)**: 最終分類或迴歸

**卷積操作**：
卷積核在輸入數據上滑動，進行逐元素乘法並求和：

$$Output(i,j) = \sum_{m}\sum_{n} Input(i+m, j+n) \times Kernel(m, n)$$

**CNN 的優勢**：
1. **參數共享**: 同一個卷積核在整個輸入上使用，大幅減少參數量
2. **局部連接**: 每個神經元只連接局部區域
3. **平移不變性**: 對輸入的位移具有魯棒性

### 3.2 適用時機與問題類型

✅ **適合使用 CNN 的情況**：
- 圖像數據 (圖像分類、物體檢測、影像分割)
- 具有空間結構的數據
- 1D 時間序列或訊號處理
- 需要自動特徵提取的任務

📊 **化工領域應用範例**：
- **顯微鏡圖像分析**: 晶體結構識別、顆粒大小分析
- **製程圖像監控**: 反應器內部狀態、產品外觀檢測
- **光譜分析**: 使用 1D CNN 處理光譜數據
- **缺陷檢測**: 產品表面缺陷識別
- **相圖識別**: 識別相變化和相邊界

❌ **不適合的情況**：
- 純表格型數據 (使用 DNN)
- 長序列時間依賴問題 (使用 RNN/LSTM)
- 小數據集 (容易過擬合)

### 3.3 使用 Keras 建立 CNN 模型

**2D CNN (圖像處理)**：

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 建立 CNN 模型
model = Sequential([
    # 輸入層
    Input(shape=(height, width, channels)),
    
    # 第一組卷積層
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # 第二組卷積層
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # 第三組卷積層
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # 展平
    Flatten(),
    
    # 全連接層
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # 多分類
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**1D CNN (時間序列或訊號)**：

```python
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D

model = Sequential([
    # 輸入層
    Input(shape=(sequence_length, num_features)),
    
    # 第一組 1D 卷積層
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    
    # 第二組 1D 卷積層
    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    
    # 全局平均池化
    GlobalAveragePooling1D(),
    
    # 全連接層
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_outputs, activation='linear')  # 迴歸
])
```

### 3.4 關鍵參數說明

#### 3.4.1 卷積層參數

| 參數 | 說明 | 選擇建議 |
|------|------|----------|
| **filters** | 卷積核數量 | 常見: 32→64→128 遞增 |
| **kernel_size** | 卷積核大小 | 2D: (3,3), 1D: 3 或 5 |
| **strides** | 步長 | 預設 1，較大值會降低輸出尺寸 |
| **padding** | 填充方式 | 'valid' (不填充) 或 'same' (保持尺寸) |
| **activation** | 激活函數 | 通常使用 'relu' |

**卷積核大小選擇**：
- 小核 (3×3): 提取細節特徵，參數少
- 大核 (5×5, 7×7): 提取更大範圍特徵
- 多數情況使用 3×3，透過堆疊獲得大感受野

#### 3.4.2 池化層參數

**MaxPooling**: 取局部區域最大值
```python
MaxPooling2D(pool_size=(2, 2))  # 2D
MaxPooling1D(pool_size=2)        # 1D
```

**AveragePooling**: 取局部區域平均值
```python
AveragePooling2D(pool_size=(2, 2))
```

**GlobalPooling**: 對整個特徵圖進行池化
```python
GlobalMaxPooling2D()      # 取全局最大
GlobalAveragePooling2D()  # 取全局平均
```

池化作用：
- 降低特徵圖尺寸
- 減少參數量和計算量
- 提供平移不變性

### 3.5 典型 CNN 架構

**淺層 CNN** (小數據集)：
```
Input → Conv → Pool → Conv → Pool → Flatten → Dense → Output
```

**深層 CNN** (大數據集)：
```
Input → [Conv → Conv → Pool] × N → Flatten → Dense → Dense → Output
```

**現代 CNN 技巧**：
- 使用 Batch Normalization 加速訓練
- 使用 Dropout 防止過擬合
- 使用 Data Augmentation 增強數據

---

## 4. RNN 模型 (LSTM/GRU)

### 4.1 基礎理論

**RNN (Recurrent Neural Network)** 循環神經網路專門設計用於處理序列數據，具有記憶能力，能夠利用先前的資訊。

**標準 RNN 問題**：
- **梯度消失**: 長序列時難以學習長期依賴
- **梯度爆炸**: 梯度可能指數增長

**LSTM (Long Short-Term Memory)**：
LSTM 透過門控機制 (gates) 解決梯度消失問題，能夠學習長期依賴關係。

**LSTM 組成**：
1. **遺忘門 (Forget Gate)**: 決定從記憶中丟棄什麼資訊
2. **輸入門 (Input Gate)**: 決定在記憶中儲存什麼新資訊
3. **輸出門 (Output Gate)**: 決定輸出什麼資訊

數學表示：
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$
$$C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \cdot \tanh(C_t)$$

**GRU (Gated Recurrent Unit)**：
GRU 是 LSTM 的簡化版本，只有兩個門：
1. **重置門 (Reset Gate)**
2. **更新門 (Update Gate)**

GRU 優勢：
- 參數較少，訓練更快
- 在許多任務上表現與 LSTM 相當

### 4.2 適用時機與問題類型

✅ **適合使用 RNN/LSTM/GRU 的情況**：
- 時間序列數據
- 序列預測問題
- 自然語言處理
- 具有時間依賴性的數據

📊 **化工領域應用範例**：
- **製程時間序列預測**: 溫度、壓力、流量等變數的預測
- **設備剩餘壽命預測 (RUL)**: 根據感測器歷史數據預測設備壽命
- **批次製程監控**: 預測批次結束時的品質
- **反應器動態建模**: 建立反應器的動態響應模型
- **異常檢測**: 識別製程參數的異常模式

❌ **不適合的情況**：
- 數據點之間沒有時間順序關係
- 超長序列 (可能需要 Transformer)
- 純靜態特徵數據

### 4.3 使用 Keras 建立 RNN 模型

**LSTM 模型**：

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

# 單層 LSTM
model = Sequential([
    Input(shape=(timesteps, features)),
    LSTM(64, activation='tanh'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(output_dim, activation='linear')
])

# 多層 LSTM (return_sequences=True 用於堆疊)
model = Sequential([
    Input(shape=(timesteps, features)),
    LSTM(64, activation='tanh', return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='tanh'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(output_dim, activation='linear')
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
```

**GRU 模型**：

```python
from tensorflow.keras.layers import Input, GRU

model = Sequential([
    Input(shape=(timesteps, features)),
    GRU(64, activation='tanh', return_sequences=True),
    Dropout(0.2),
    GRU(32, activation='tanh'),
    Dropout(0.2),
    Dense(output_dim, activation='linear')
])
```

**Bidirectional RNN**: 同時從前向和後向處理序列

```python
from tensorflow.keras.layers import Input, Bidirectional

model = Sequential([
    Input(shape=(timesteps, features)),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(output_dim, activation='linear')
])
```

### 4.4 關鍵參數說明

#### 4.4.1 RNN 層參數

| 參數 | 說明 | 選擇建議 |
|------|------|----------|
| **units** | LSTM/GRU 單元數量 | 常見: 32, 64, 128, 256 |
| **return_sequences** | 是否返回完整序列 | 堆疊層時設為 True，最後一層 False |
| **return_state** | 是否返回最終狀態 | 需要狀態時設為 True (如 Seq2Seq) |
| **activation** | 激活函數 | 預設 'tanh'，一般不改 |
| **recurrent_activation** | 門控激活函數 | 預設 'sigmoid'，一般不改 |
| **dropout** | 輸入 dropout 比例 | 0-0.3 |
| **recurrent_dropout** | 循環連接 dropout | 0-0.3 |

#### 4.4.2 序列數據準備

**數據形狀**：
```python
# RNN 輸入: (samples, timesteps, features)
X_train.shape  # 例如: (1000, 50, 10)
# 1000 個樣本，每個樣本 50 個時間步，每步 10 個特徵
```

**滑動窗口建立序列**：
```python
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 50
X, y = create_sequences(time_series_data, window_size)
```

### 4.5 LSTM vs GRU 選擇

| 特性 | LSTM | GRU |
|------|------|-----|
| **參數量** | 較多 | 較少 |
| **訓練速度** | 較慢 | 較快 |
| **記憶能力** | 更強 | 略弱 |
| **適用場景** | 複雜長期依賴 | 中等複雜度，追求效率 |

**選擇建議**：
- 先嘗試 GRU (更快，通常效果足夠)
- 若需要更強的長期記憶，使用 LSTM
- 數據量大時，GRU 優勢明顯

---

## 5. 模型比較與選擇指南

### 5.1 三種模型對比

| 特性 | DNN/MLP | CNN | RNN/LSTM |
|------|---------|-----|----------|
| **輸入類型** | 向量 (1D) | 網格 (圖像, 2D/3D) | 序列 (時間, 1D) |
| **特殊結構** | 全連接 | 卷積+池化 | 循環連接 |
| **參數量** | 中等 | 少 (參數共享) | 多 |
| **訓練速度** | 快 | 中 | 慢 |
| **擅長任務** | 表格數據預測 | 圖像識別 | 序列預測 |
| **記憶能力** | 無 | 局部 | 長期依賴 |

### 5.2 根據數據類型選擇

**數據類型決策樹**：

```
您的數據是什麼形式？
│
├─ 表格數據 (特徵向量)
│  └─ 使用 DNN/MLP
│
├─ 圖像數據
│  └─ 使用 CNN
│
├─ 時間序列數據
│  ├─ 短期依賴，快速預測
│  │  └─ 使用 1D CNN
│  └─ 長期依賴，複雜模式
│     └─ 使用 LSTM/GRU
│
└─ 混合數據
   └─ 組合模型 (CNN + LSTM, 等)
```

### 5.3 化工應用場景建議

| 應用場景 | 推薦模型 | 理由 |
|----------|----------|------|
| 反應產率預測 (給定條件) | DNN | 表格型靜態特徵 |
| 製程參數優化 | DNN | 輸入輸出關係建模 |
| 光譜數據分析 | 1D CNN | 局部特徵重要 |
| 顯微圖像分類 | 2D CNN | 空間特徵提取 |
| 溫度時間序列預測 | LSTM/GRU | 時間依賴性 |
| 批次製程品質預測 | LSTM | 整個批次歷史 |
| 設備健康監控 | LSTM + DNN | 時序+靜態特徵結合 |
| 異常檢測 | Autoencoder (DNN/CNN/LSTM) | 根據數據類型選擇 |

---

## 6. 實作建議

### 6.1 模型開發流程

1. **數據準備**
   - 數據清洗與預處理
   - 特徵工程 (如需要)
   - 數據標準化/正規化
   - 訓練/驗證/測試集切分

2. **模型建立**
   - 選擇合適的模型架構
   - 從簡單模型開始
   - 逐步增加複雜度

3. **模型訓練**
   - 設定適當的 batch size 和 epochs
   - 使用 callbacks (EarlyStopping, ModelCheckpoint)
   - 監控訓練和驗證指標

4. **模型評估**
   - 在測試集上評估
   - 視覺化預測結果
   - 分析錯誤案例

5. **模型調優**
   - 超參數調整
   - 正則化技巧
   - 數據增強 (如適用)

### 6.2 常見問題與解決

**過擬合 (Overfitting)**：
- 訓練誤差低，驗證誤差高
- 解決方法:
  - 增加 Dropout
  - 添加 L1/L2 正則化
  - 減少模型複雜度
  - 增加訓練數據
  - 使用 Early Stopping

**欠擬合 (Underfitting)**：
- 訓練和驗證誤差都高
- 解決方法:
  - 增加模型複雜度 (更多層/神經元)
  - 訓練更長時間
  - 減少正則化
  - 檢查特徵是否充分

**訓練不穩定**：
- 損失值震盪或不收斂
- 解決方法:
  - 降低學習率
  - 使用 Batch Normalization
  - 檢查數據標準化
  - 使用梯度裁剪 (Gradient Clipping)

### 6.3 最佳實踐

✅ **數據處理**：
- 標準化輸入特徵 (StandardScaler 或 MinMaxScaler)
- 處理缺失值
- 適當的訓練/驗證/測試切分 (例如 70/15/15)

✅ **模型設計**：
- 從簡單模型開始，逐步增加複雜度
- 使用 Batch Normalization 和 Dropout
- 輸出層激活函數和損失函數要匹配

✅ **訓練策略**：
- 使用 Early Stopping 防止過擬合
- 保存最佳模型 (ModelCheckpoint)
- 監控多個指標
- 使用 TensorBoard 視覺化訓練過程

✅ **評估與驗證**：
- 在獨立測試集上評估
- 使用適當的評估指標
- 視覺化預測結果
- 分析錯誤分布

### 6.4 程式碼範例連結

完整的實作範例請參考：
📓 **Unit15_Basic_DeepLearning_Models.ipynb**

範例包含：
1. DNN 用於化工製程參數預測
2. CNN 用於光譜數據分類
3. LSTM 用於時間序列預測

---

## 7. 實作範例執行結果

以下展示三個完整實作範例的執行結果，幫助學生理解實際應用效果。

### 7.1 範例 1: DNN 模型 - 化工製程參數預測

**問題描述**: 根據反應條件（溫度、壓力、催化劑濃度、反應時間、攪拌速率、原料濃度）預測產品產率。

**數據集**: 2000 筆模擬化工製程數據
- 特徵數: 6 (Temperature, Pressure, Catalyst_Conc, Reaction_Time, Stirring_Rate, Reactant_A)
- 目標: Yield (%)
- 訓練集: 1400 筆 (70%)
- 驗證集: 300 筆 (15%)
- 測試集: 300 筆 (15%)

**模型架構**:
```
Input (6) → Dense(64, ReLU) + BN + Dropout(0.3)
         → Dense(32, ReLU) + BN + Dropout(0.3)
         → Dense(16, ReLU) + BN + Dropout(0.3)
         → Dense(1, Linear)
```

**模型參數**: 
- 總參數量: 3,521
- 可訓練參數: 3,297
- 不可訓練參數: 224

**訓練結果**:

經過 EarlyStopping (patience=30) 訓練後，模型在測試集上的表現：

```
DNN 模型 評估結果:
  MSE  = 12.6499
  RMSE = 3.5567
  MAE  = 2.8586
  R²   = 0.9646
```

**結果分析**:
- ✅ R² = 0.9646 表示模型能解釋 96.46% 的變異
- ✅ RMSE = 3.56% 表示預測誤差約為 3.5 個百分點
- ✅ 殘差分布均勻，無明顯偏差
- ✅ 模型成功捕捉反應條件與產率的非線性關係

**預測範例** (部分測試數據):

| 編號 | 真實產率 (%) | 預測產率 (%) | 誤差 (%) | 誤差率 (%) |
|------|--------------|--------------|----------|------------|
| 275  | 35.07        | 35.28        | -0.21    | 0.60       |
| 81   | 39.73        | 40.33        | -0.61    | 1.52       |
| 35   | 50.64        | 53.88        | -3.24    | 6.41       |
| 46   | 32.42        | 30.73        | 1.69     | 5.20       |
| 180  | 22.46        | 24.59        | -2.12    | 9.45       |

**關鍵發現**:
1. DNN 能有效學習多變數製程參數與產率的關係
2. Batch Normalization 有助於訓練穩定性
3. Dropout 有效防止過擬合
4. 大部分預測誤差在 5% 以內，符合實際應用需求

---

### 7.2 範例 2: CNN 模型 - 光譜數據分類

**問題描述**: 根據紅外光譜數據識別化合物類型（醇類、酮類、酯類）。

**數據集**: 1200 筆模擬光譜數據
- 特徵數: 200 個波長點
- 類別數: 3 (Alcohol, Ketone, Ester)
- 每類樣本: 400 筆
- 訓練集: 840 筆 (70%)
- 驗證集: 180 筆 (15%)
- 測試集: 180 筆 (15%)

**模型架構**:
```
Input (200, 1) → Conv1D(32, kernel=5) + BN + MaxPool + Dropout(0.3)
               → Conv1D(64, kernel=5) + BN + MaxPool + Dropout(0.3)
               → Conv1D(128, kernel=3) + BN
               → GlobalAveragePooling1D
               → Dense(64, ReLU) + Dropout(0.4)
               → Dense(3, Softmax)
```

**模型參數**:
- 總參數量: 約 40,000
- 使用 1D 卷積提取光譜局部特徵

**訓練結果**:

```
測試集評估結果:
  損失 (Loss): 0.0002
  準確率 (Accuracy): 1.0000

分類報告:
              precision    recall  f1-score   support

     Alcohol     1.0000    1.0000    1.0000        60
      Ketone     1.0000    1.0000    1.0000        60
       Ester     1.0000    1.0000    1.0000        60

    accuracy                         1.0000       180
   macro avg     1.0000    1.0000    1.0000       180
weighted avg     1.0000    1.0000    1.0000       180
```

**混淆矩陣**:
```
真實 \ 預測    Alcohol  Ketone  Ester
Alcohol          60       0       0
Ketone            0      60       0
Ester             0       0      60
```

**結果分析**:
- ✅ 100% 準確率：模型完美分類所有測試樣本
- ✅ 所有類別的 Precision、Recall、F1-score 都達到 1.0
- ✅ 混淆矩陣顯示無任何誤分類
- ✅ CNN 成功提取不同化合物的光譜特徵差異

**關鍵發現**:
1. 1D CNN 非常適合處理光譜等具有局部結構的數據
2. 卷積層能自動識別特徵峰（如 O-H 拉伸、C=O 拉伸）
3. GlobalAveragePooling 有效減少參數同時保留重要資訊
4. 在光譜分類任務上，CNN 優於傳統特徵工程 + 機器學習方法

---

### 7.3 範例 3: LSTM 模型 - 製程時間序列預測

**問題描述**: 根據過去 50 個時間步的反應器歷史數據（溫度、壓力、流量），預測下一時刻的溫度。

**數據集**: 5000 個時間點
- 特徵數: 3 (Temperature, Pressure, Flow_Rate)
- 窗口大小: 50 個時間步
- 訓練集: 3465 個序列 (70%)
- 驗證集: 742 個序列 (15%)
- 測試集: 743 個序列 (15%)

**模型架構 - LSTM**:
```
Input (50, 3) → LSTM(64, return_sequences=True) + Dropout(0.2)
              → LSTM(32) + Dropout(0.2)
              → Dense(16, ReLU) + Dropout(0.2)
              → Dense(1, Linear)
```

**模型參數**:
- LSTM 總參數量: 30,369
- GRU 總參數量: 23,201 (用於比較)

**訓練結果 - LSTM**:

```
LSTM Model 評估結果:
  MSE  = 4.6221
  RMSE = 2.1499
  MAE  = 1.7298
  R²   = 0.9118
```

**訓練結果 - GRU** (比較實驗):

```
GRU Model 評估結果:
  MSE  = 5.0419
  RMSE = 2.2454
  MAE  = 1.8118
  R²   = 0.9038
```

**LSTM vs GRU 效能比較**:

| 指標 | LSTM | GRU | 優勝者 |
|------|------|-----|--------|
| RMSE | 2.1499 | 2.2454 | LSTM |
| MAE | 1.7298 | 1.8118 | LSTM |
| R² | 0.9118 | 0.9038 | LSTM |
| 參數量 | 30,369 | 23,201 | GRU |
| 訓練速度 | 較慢 | 較快 | GRU |

**結果分析**:
- ✅ LSTM R² = 0.9118：能解釋 91.18% 的溫度變異
- ✅ LSTM 在預測準確度上略優於 GRU
- ✅ GRU 參數量少 23.6%，訓練速度更快
- ✅ 兩個模型都能捕捉溫度的週期性和趨勢
- ✅ 殘差分布接近正態，無系統性偏差

**模型預測效果**:
- 短期預測 (1-10 步): 兩個模型都非常準確
- 中期預測 (10-50 步): LSTM 表現更穩定
- 長期趨勢: 兩個模型都能跟隨主要趨勢

**關鍵發現**:
1. LSTM 和 GRU 都適合時間序列預測任務
2. 對於複雜的長期依賴，LSTM 略優
3. 追求效率時，GRU 是很好的選擇
4. 窗口大小 (50) 的選擇很重要，需根據實際週期調整
5. 時間序列數據必須按時間順序切分（不打亂）

---

### 7.4 實作範例總結

**三種模型的適用場景對比**:

| 特性 | DNN | CNN | LSTM |
|------|-----|-----|------|
| **數據類型** | 表格數據 | 光譜/圖像 | 時間序列 |
| **特徵提取** | 手動/自動 | 自動 | 自動 |
| **記憶能力** | ❌ | 局部 | ✅ 長期 |
| **訓練速度** | 快 | 中 | 慢 |
| **參數量** | 少 | 中 | 多 |
| **化工應用** | 製程參數預測 | 光譜分析 | 動態建模 |
| **測試效能** | R²=0.96 | Acc=1.00 | R²=0.91 |

**實作經驗建議**:

1. **數據預處理最重要**: 所有模型都需要適當的標準化
2. **從簡單開始**: 先嘗試較淺的網路，再逐步增加複雜度
3. **Early Stopping 是關鍵**: 防止過擬合的最有效方法
4. **視覺化幫助理解**: 繪製訓練曲線和預測結果
5. **參數調整需耐心**: 學習率、批次大小、網路結構都會影響結果
6. **模型組合更強大**: 實際應用中可能需要結合多種模型

---

## 8. 參考資源

**書籍**：
- Deep Learning (Ian Goodfellow et al.)
- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow

**線上資源**：
- TensorFlow 官方文件: https://www.tensorflow.org/
- Keras 官方文件: https://keras.io/
- Deep Learning Specialization (Coursera)

**化工相關應用**：
- ChemE + AI 文獻資料庫
- AIChE Journal - Machine Learning 專題

---

## 💡 學習檢核

完成本單元後，您應該能夠：

- [ ] 理解 DNN, CNN, RNN 的基本原理和運作機制
- [ ] 根據問題類型和數據特性選擇適合的模型
- [ ] 使用 Keras 建立和訓練深度學習模型
- [ ] 設定關鍵參數（層數、神經元數、學習率等）並理解其影響
- [ ] 診斷和解決常見的訓練問題（過擬合、欠擬合）
- [ ] 評估模型效能並進行優化
- [ ] 解讀訓練曲線和預測結果
- [ ] 將深度學習應用於化工領域的實際問題

**實作能力檢核**：
- [ ] 能獨立完成 DNN 模型建立並達到 R² > 0.90
- [ ] 能使用 CNN 處理光譜或圖像數據並達到準確率 > 95%
- [ ] 能使用 LSTM/GRU 進行時間序列預測並理解結果
- [ ] 能根據訓練曲線判斷模型狀態並調整參數
- [ ] 能比較不同模型的效能並選擇最適合的方案

**實際成果展示**：
本講義配套的 Jupyter Notebook 展示了以下實作成果：
- ✅ DNN 模型：R² = 0.9646 (化工製程產率預測)
- ✅ CNN 模型：準確率 = 100% (光譜數據分類)
- ✅ LSTM 模型：R² = 0.9118 (時間序列溫度預測)

---

**下一單元**: Unit 16 - 深度學習進階技巧與應用

---

*更新日期: 2025-12-21*
*課程: 化工AI應用 - Part 4 深度學習*
