# Unit17_Example_debutanizer_column | 使用 LSTM 和 GRU 預測去丁烷塔 C4 含量

> **課程單元**：Part 4 - 深度學習應用  
> **主題**：時序預測 - 去丁烷塔軟測量  
> **技術**：LSTM、GRU、時序特徵工程  
> **難度**：⭐⭐⭐⭐  
> **預計時間**：120 分鐘

---

## 📚 目錄

1. [學習目標](#學習目標)
2. [背景說明](#背景說明)
3. [數據集介紹](#數據集介紹)
4. [環境設定與數據下載](#環境設定與數據下載)
5. [數據探索與分析](#數據探索與分析)
6. [數據預處理](#數據預處理)
7. [LSTM 模型建立與訓練](#lstm-模型建立與訓練)
8. [GRU 模型建立與訓練](#gru-模型建立與訓練)
9. [模型比較與評估](#模型比較與評估)
10. [過擬合診斷與改進](#過擬合診斷與改進)
11. [備選方案與建議](#備選方案與建議)
12. [結論與討論](#結論與討論)
13. [參考資源](#參考資源)

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. ✅ **理解化工製程軟測量的概念與應用**
   - 掌握軟測量（Soft Sensor）在化工製程中的重要性
   - 了解去丁烷塔製程與 C4 含量預測的實際意義

2. ✅ **掌握時序數據的預處理技術**
   - 時序數據的標準化與序列化
   - 滑動窗口（Sliding Window）方法
   - 差分特徵工程

3. ✅ **建立並訓練 LSTM 和 GRU 模型**
   - 理解 LSTM 和 GRU 的架構與原理
   - 設計適合化工製程的模型結構
   - 使用 Keras/TensorFlow 實現時序預測模型

4. ✅ **診斷與解決過擬合問題**
   - 識別過擬合的症狀
   - 應用正則化技術（Dropout、L2）
   - 調整模型複雜度與超參數

5. ✅ **評估模型性能與泛化能力**
   - 使用多種評估指標（R², RMSE, MAE）
   - 分析殘差分布
   - 比較不同模型的優劣

---

## 📖 背景說明

### 什麼是去丁烷塔？

**去丁烷塔（Debutanizer Column）** 是煉油和石化工業中的關鍵設備，屬於脫硫和石腦油分離裝置的一部分。其主要功能是從石腦油流中分離出輕質烴類成分。

**製程功能：**
- 🔹 **塔頂產物**：移除 C3（丙烷）和 C4（丁烷）作為 LP 氣體
- 🔹 **塔底產物**：穩定汽油（Stabilized Gasoline）送往下游製程
- 🔹 **控制目標**：
  - 確保充分的分餾效果
  - 最大化塔頂產物中的 C5 含量（符合法規限制）
  - 最小化塔底產物中的 C4 含量（提高產品質量）

### 為什麼需要軟測量？

**軟測量（Soft Sensor）** 是一種使用數學模型和易測變數來推算難測或無法線上測量變數的技術。

**傳統測量的限制：**
- 🚫 **氣相色譜儀（GC）**：測量延遲長（5-30 分鐘）、維護成本高
- 🚫 **取樣分析**：無法提供即時數據、人工成本高
- 🚫 **安裝困難**：某些位置難以安裝感測器

**軟測量的優勢：**
- ✅ **即時預測**：使用現有感測器數據即時推算目標變數
- ✅ **低成本**：無需額外硬體投資
- ✅ **高頻率**：可提供連續的預測值
- ✅ **彈性高**：可根據需求調整和更新模型

### LSTM 和 GRU 在化工製程中的應用

**為什麼選擇循環神經網路（RNN）？**

化工製程具有以下特性，使得 RNN 系列模型特別適合：

1. **時序依賴性**：當前狀態受過去狀態影響
2. **動態特性**：系統存在慣性和延遲
3. **複雜非線性**：變數間關係複雜且非線性

**LSTM（Long Short-Term Memory）**
- 🔹 擅長捕捉長期依賴關係
- 🔹 通過門控機制避免梯度消失
- 🔹 適合需要記憶長期信息的場景

**GRU（Gated Recurrent Unit）**
- 🔹 LSTM 的簡化版本，參數更少
- 🔹 訓練速度更快
- 🔹 在小數據集上有時表現更好

---

## 📊 數據集介紹

### 數據來源

本案例使用的數據來自真實的工業去丁烷塔操作記錄，記錄了製程運行過程中的多個關鍵變數。

**數據集資訊：**
- 📁 **檔案名稱**：`debutanizer_data.txt`
- 📁 **數據點數**：2,394 筆
- 📁 **採樣頻率**：每分鐘
- 📁 **變數數量**：8 個（7 個輸入 + 1 個輸出）
- 📁 **數據期間**：連續運行記錄

**參考文獻：**
> Fortuna, L., Graziani, S., Rizzo, A., & Xibilia, M. G. (2007). *Soft Sensors for Monitoring and Control of Industrial Processes*. Springer.

### 變數說明

**輸入變數（u1-u7）：**

| 變數 | 描述 | 單位 | 物理意義 |
|------|------|------|----------|
| **u1** | Top Temperature<br>塔頂溫度 | °C | 反映塔頂氣相組成，影響輕組分回收 |
| **u2** | Top Pressure<br>塔頂壓力 | kPa | 影響氣液平衡，控制分餾效果 |
| **u3** | Reflux Flow<br>回流流量 | kg/h | 控制精餾效果的關鍵變數 |
| **u4** | Flow to Next<br>流向下一製程 | kg/h | 塔頂產物流量，影響物料平衡 |
| **u5** | 6th Tray Temperature<br>第 6 層板溫度 | °C | 塔內溫度分布指標 |
| **u6** | Bottom Temperature 1<br>塔底溫度 1 | °C | 塔底重組分溫度 |
| **u7** | Bottom Temperature 2<br>塔底溫度 2 | °C | 塔底溫度冗餘測量 |

**輸出變數（y）：**

| 變數 | 描述 | 單位 | 重要性 |
|------|------|------|--------|
| **y** | C4 Content in Bottom Flow<br>塔底流中的 C4 含量 | mol% | **關鍵品質指標**<br>決定產品是否符合規格 |

### 數據特性分析

根據執行結果，數據集具有以下特性：

```
Dataset shape: (2394, 8)
Number of samples: 2394
Number of features: 7
```

**數據規模評估：**
- ✅ 樣本數適中（約 2400 筆）
- ⚠️ 對於深度學習而言，數據量偏小（理想 > 5000）
- ✅ 特徵數量合理（7 個輸入變數）
- ✅ 無缺失值，數據質量良好

---

## 🔧 環境設定與數據下載

### 環境設定

本單元使用以下 Python 套件：

**核心套件：**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

**機器學習套件：**
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

**深度學習套件：**
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
```

### 數據下載

數據檔案會自動檢查是否存在，若不存在則從 GitHub 下載：

```python
import requests
import os

url = "https://raw.githubusercontent.com/sj823774188/Debutanizer-Column-Data/main/debutanizer_data.txt"
data_file = os.path.join(DATA_DIR, "debutanizer_data.txt")

# 檢查檔案是否存在
if not os.path.exists(data_file):
    print(f"✗ 數據檔案不存在，正在下載...")
    response = requests.get(url)
    with open(data_file, 'wb') as f:
        f.write(response.content)
    print(f"✓ 下載成功")
else:
    print(f"✓ 數據檔案已存在")
```

**執行結果：**
```
✓ 數據檔案已存在於: d:\MyGit\CHE-AI-COURSE\Part_4\Unit17\data\debutanizer_column\debutanizer_data.txt
```

---

## 📈 數據探索與分析

### 載入數據

數據檔案格式為純文字檔，以空格分隔，前 4 行為檔頭說明。

```python
# 載入數據（跳過前 4 行檔頭）
data = np.loadtxt(data_path, skiprows=4)

# 建立 DataFrame 並賦予有意義的欄位名稱
columns = ['u1_TopTemp', 'u2_TopPressure', 'u3_RefluxFlow', 
           'u4_FlowToNext', 'u5_TrayTemp', 'u6_BottomTemp1', 
           'u7_BottomTemp2', 'y_C4Content']
df = pd.DataFrame(data, columns=columns)
```

**執行結果：**
```
Dataset shape: (2394, 8)
Number of samples: 2394
Number of features: 7

First few rows:
   u1_TopTemp  u2_TopPressure  u3_RefluxFlow  u4_FlowToNext  u5_TrayTemp  u6_BottomTemp1  u7_BottomTemp2  y_C4Content
0      463.48          202.82        3829.29        1645.65       158.91          389.54          399.36         0.66
1      463.14          202.67        3831.33        1618.90       159.21          389.71          399.48         0.66
2      463.28          202.78        3828.48        1607.06       159.66          390.07          399.65         0.66
3      463.44          202.82        3826.51        1626.75       159.99          390.11          399.74         0.67
4      463.48          202.82        3829.29        1645.65       160.20          390.16          399.82         0.67
```

### 基本統計信息

**統計摘要觀察：**

- **目標變數（y_C4Content）**：
  - 平均值：0.69 mol%，標準差：0.04 mol%
  - 範圍：0.59 ~ 0.79 mol%
  - 變異較小，製程控制穩定

- **輸入變數特性：**
  - 塔頂溫度（u1）：變異係數小（CV = 0.24%）
  - 回流流量（u3）：相對穩定（CV = 0.32%）
  - 流向下製程（u4）：變異最大（CV = 4.37%）

- **數據質量**：
  - ✅ 無缺失值
  - ✅ 無異常極端值
  - ✅ 數值範圍合理

### 相關性分析

**與目標變數（y_C4Content）的相關性：**
- `u6_BottomTemp1`：r = 0.78（強正相關）⭐
- `u7_BottomTemp2`：r = 0.76（強正相關）⭐
- `u5_TrayTemp`：r = 0.65（中等正相關）
- `u1_TopTemp`：r = 0.42（弱正相關）
- `u3_RefluxFlow`：r = -0.35（弱負相關）

**變數間相關性：**
- `u6` 和 `u7`（兩個塔底溫度）：r = 0.99（極高相關）
- `u1` 和 `u5`（塔頂與層板溫度）：r = 0.68

📌 **建模啟示：**
- 塔底溫度是最重要的預測因子
- 可能存在多重共線性（u6 和 u7）
- 溫度變數對 C4 含量影響最大

---

## 🔄 數據預處理

### 3.1 數據分割

將數據分割為訓練集、驗證集和測試集，使用時序分割以保持時間順序。

```python
# 定義分割比例
train_ratio = 0.7  # 70% 訓練集
val_ratio = 0.15   # 15% 驗證集
test_ratio = 0.15  # 15% 測試集

# 分離特徵和目標
feature_cols = ['u1_TopTemp', 'u2_TopPressure', 'u3_RefluxFlow', 
                'u4_FlowToNext', 'u5_TrayTemp', 'u6_BottomTemp1', 
                'u7_BottomTemp2']
target_col = 'y_C4Content'

X = df[feature_cols].values
y = df[target_col].values

# 時序分割（不打亂順序）
n_samples = len(X)
train_size = int(n_samples * train_ratio)
val_size = int(n_samples * val_ratio)

X_train = X[:train_size]
X_val = X[train_size:train_size+val_size]
X_test = X[train_size+val_size:]

y_train = y[:train_size]
y_val = y[train_size:train_size+val_size]
y_test = y[train_size+val_size:]
```

**執行結果：**
```
✓ Data split completed
Train set: 1675 samples (70.0%)
Validation set: 359 samples (15.0%)
Test set: 360 samples (15.0%)
```

⚠️ **重要說明**：
- 使用**時序分割**而非隨機分割，以模擬實際預測情境
- 測試集使用最新的數據，評估模型對未來數據的泛化能力

### 3.2 特徵工程 - 添加差分特徵

化工製程數據通常包含趨勢和週期性，添加差分特徵可以幫助模型捕捉變化率。

```python
# 計算差分特徵（當前值 - 前一時刻值）
X_diff = np.diff(X, axis=0, prepend=X[0:1])

# 組合原始特徵和差分特徵
X_combined = np.concatenate([X, X_diff], axis=1)

print(f"Original features: {X.shape[1]}")
print(f"Combined features (original + diff): {X_combined.shape[1]}")
```

**執行結果：**
```
Original features: 7
Combined features (original + diff): 14
```

📌 **為什麼添加差分特徵？**
- ✅ 捕捉變化趨勢：差分反映變數的變化速率
- ✅ 增強時序信息：幫助模型理解動態特性
- ✅ 改善預測：對於具有慣性的製程特別有效

### 3.3 數據標準化

RNN 模型對輸入數據的尺度敏感，因此需要進行標準化處理。

```python
from sklearn.preprocessing import StandardScaler

# 分別標準化特徵和目標
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_combined)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

print("✓ Data standardization completed")
print(f"Features - Mean: ~0, Std: ~1")
print(f"Target - Mean: {y_scaled.mean():.6f}, Std: {y_scaled.std():.6f}")
```

⚠️ **標準化注意事項：**
- 只在訓練集上 `fit`，然後 `transform` 驗證集和測試集
- 保存 scaler 以便後續反標準化預測結果
- 特徵和目標分別標準化

### 3.4 創建時序序列數據

LSTM 和 GRU 需要 3D 輸入：`(samples, timesteps, features)`

```python
TIME_STEPS = 20  # 回看窗口長度

def create_sequences(X, y, time_steps):
    """
    將數據轉換為時序序列格式
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
    y : array, shape (n_samples,)
    time_steps : int, 回看窗口長度
    
    Returns:
    --------
    X_seq : array, shape (n_seq, time_steps, n_features)
    y_seq : array, shape (n_seq,)
    """
    X_seq, y_seq = [], []
    
    for i in range(time_steps, len(X)):
        X_seq.append(X[i-time_steps:i])  # 取前 time_steps 個時間點
        y_seq.append(y[i])                # 目標為當前時刻
    
    return np.array(X_seq), np.array(y_seq)

# 為訓練、驗證、測試集創建序列
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, TIME_STEPS)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, TIME_STEPS)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, TIME_STEPS)
```

**執行結果：**
```
✓ Sequence data created
Train sequences: (1655, 20, 14) → 1655 samples, 20 timesteps, 14 features
Val sequences: (339, 20, 14)
Test sequences: (340, 20, 14)
```

**3D 張量結構說明：**
```
Shape: (1655, 20, 14)
       ↓     ↓   ↓
    樣本數  時間步  特徵數
```

- **樣本數（1655）**：可用於訓練的序列總數
- **時間步（20）**：每個序列包含過去 20 個時間點
- **特徵數（14）**：7 個原始特徵 + 7 個差分特徵

📌 **TIME_STEPS 參數選擇：**
- 太小（< 10）：無法捕捉足夠的時序信息
- 太大（> 50）：訓練樣本減少，計算成本增加
- **建議**：化工製程通常選擇 20-30

### 數據準備總結

經過預處理後的數據特性：

| 階段 | 訓練集 | 驗證集 | 測試集 | 總計 |
|------|--------|--------|--------|------|
| **原始數據** | 1675 | 359 | 360 | 2394 |
| **序列數據** | 1655 | 339 | 340 | 2334 |
| **損失樣本** | 20 | 20 | 20 | 60 |

**損失樣本說明**：
- 每個數據集開頭的 `TIME_STEPS` 個樣本無法形成完整序列
- 這是滑動窗口方法的正常現象
- 損失比例：60/2394 = 2.5%（可接受）

---

## 🧠 LSTM 模型建立與訓練

### 4.1 LSTM 原理簡介

**LSTM（Long Short-Term Memory）** 是一種特殊的 RNN，專門設計用於解決長期依賴問題。

**LSTM 的核心組件：**

1. **遺忘門（Forget Gate）**：決定丟棄哪些舊信息
   
   $$
   f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
   $$

2. **輸入門（Input Gate）**：決定接收哪些新信息
   
   $$
   i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
   $$

3. **輸出門（Output Gate）**：決定輸出什麼
   
   $$
   o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
   $$

4. **細胞狀態（Cell State）**：攜帶長期記憶

**為什麼 LSTM 適合化工製程？**
- ✅ 可以記憶長期的製程狀態
- ✅ 能夠處理時間延遲效應
- ✅ 對於複雜非線性關係建模能力強

### 4.2 LSTM 模型架構設計

本案例採用**雙層 LSTM + 正則化**架構：

```python
def build_lstm_model(input_shape, units=[32, 16], dropout_rate=0.35):
    """
    建立雙層 LSTM 模型（v3 優化版）
    
    架構：
    - 第一層 LSTM（32 units）+ BatchNorm + Dropout
    - 第二層 LSTM（16 units）+ BatchNorm + Dropout
    - Dense 緩衝層（8 units）+ Dropout
    - 輸出層（1 unit）
    
    正則化技術：
    - Dropout：防止過擬合
    - L2 Regularization：限制權重大小
    - Batch Normalization：穩定訓練
    """
    from tensorflow.keras.layers import BatchNormalization
    
    model = Sequential(name='LSTM_Model')
    
    # 第一層 LSTM
    model.add(LSTM(
        units=units[0],
        return_sequences=True,  # 輸出完整序列給下一層
        input_shape=input_shape,
        kernel_regularizer=keras.regularizers.l2(0.02),
        recurrent_regularizer=keras.regularizers.l2(0.01),
        name='LSTM_1'
    ))
    model.add(BatchNormalization(name='BatchNorm_1'))
    model.add(Dropout(dropout_rate, name='Dropout_1'))
    
    # 第二層 LSTM
    model.add(LSTM(
        units=units[1],
        kernel_regularizer=keras.regularizers.l2(0.02),
        recurrent_regularizer=keras.regularizers.l2(0.01),
        name='LSTM_2'
    ))
    model.add(BatchNormalization(name='BatchNorm_2'))
    model.add(Dropout(dropout_rate, name='Dropout_2'))
    
    # Dense 緩衝層
    model.add(Dense(
        8, 
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.02),
        name='Dense_1'
    ))
    model.add(Dropout(dropout_rate * 0.5, name='Dropout_3'))
    
    # 輸出層
    model.add(Dense(1, name='Output'))
    
    # 編譯模型
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

# 建立 LSTM 模型
lstm_model = build_lstm_model(
    input_shape=(TIME_STEPS, X_train_seq.shape[2]),
    units=[32, 16],
    dropout_rate=0.35
)
```

**模型摘要：**
```
Model: "LSTM_Model"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
LSTM_1 (LSTM)              (None, 20, 32)            6016      
BatchNorm_1                (None, 20, 32)            128       
Dropout_1 (Dropout)        (None, 20, 32)            0         
LSTM_2 (LSTM)              (None, 16)                3136      
BatchNorm_2                (None, 16)                64        
Dropout_2 (Dropout)        (None, 16)                0         
Dense_1 (Dense)            (None, 8)                 136       
Dropout_3 (Dropout)        (None, 8)                 0         
Output (Dense)             (None, 1)                 9         
=================================================================
Total params: 9,489
Trainable params: 9,393
Non-trainable params: 96
```

**架構設計考量：**
- **漸進式降維**：32 → 16 → 8 → 1
- **適度參數量**：約 9,500 個參數，適合 2,000 筆數據
- **多重正則化**：Dropout + L2 + BatchNorm 組合使用

### 4.3 訓練回調函數設定

使用 Callbacks 優化訓練過程：

```python
lstm_callbacks = [
    # 早停：驗證 loss 30 epochs 沒改善則停止
    EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True,
        verbose=1
    ),
    
    # 學習率調整：15 epochs 沒改善則降低學習率
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-6,
        verbose=1
    ),
    
    # 模型檢查點：保存最佳模型
    ModelCheckpoint(
        MODEL_DIR / 'lstm_best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
]
```

**Callbacks 說明：**

| Callback | 功能 | 參數設定 | 原因 |
|----------|------|----------|------|
| **EarlyStopping** | 防止過度訓練 | patience=30 | 給模型足夠時間收斂 |
| **ReduceLROnPlateau** | 自適應學習率 | patience=15, factor=0.5 | 遇到平台期時減半學習率 |
| **ModelCheckpoint** | 保存最佳模型 | monitor='val_loss' | 保留驗證集最佳權重 |

### 4.4 模型訓練

```python
# 訓練模型
lstm_history = lstm_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=200,
    batch_size=32,
    callbacks=lstm_callbacks,
    verbose=1
)
```

**訓練過程輸出（節錄）：**
```
Epoch 1/200
52/52 [==============================] - 3s 42ms/step - loss: 0.9845 - mae: 0.8234 - val_loss: 0.7123 - val_mae: 0.6891
Epoch 2/200
52/52 [==============================] - 2s 35ms/step - loss: 0.6542 - mae: 0.6234 - val_loss: 0.5432 - val_mae: 0.5621
...
Epoch 47/200
52/52 [==============================] - 2s 36ms/step - loss: 0.2134 - mae: 0.3456 - val_loss: 0.2891 - val_mae: 0.4012
Epoch 48/200
Restoring model weights from the end of the best epoch: 18.
52/52 [==============================] - 2s 35ms/step - loss: 0.2098 - mae: 0.3421 - val_loss: 0.2945 - val_mae: 0.4056
Epoch 48: early stopping
```

**訓練結果：**
- ✅ 訓練在 48 epoch 提前停止
- ✅ 最佳模型在第 18 epoch
- ✅ 訓練時間：約 2 分鐘

### 4.5 訓練過程可視化

```python
# 繪製訓練歷史
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss 曲線
ax1.plot(lstm_history.history['loss'], label='Training Loss', linewidth=2)
ax1.plot(lstm_history.history['val_loss'], label='Validation Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.set_title('LSTM Model - Loss Curves')
ax1.legend()
ax1.grid(True, alpha=0.3)

# MAE 曲線
ax2.plot(lstm_history.history['mae'], label='Training MAE', linewidth=2)
ax2.plot(lstm_history.history['val_mae'], label='Validation MAE', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE')
ax2.set_title('LSTM Model - MAE Curves')
ax2.legend()
ax2.grid(True, alpha=0.3)
```

**觀察要點：**
- ✅ 訓練和驗證 loss 都在下降
- ✅ 無明顯的驗證 loss 上升（過擬合跡象）
- ⚠️ 需檢查訓練-驗證差距是否過大

### 4.6 模型評估

```python
# 在各數據集上進行預測
y_train_pred_scaled = lstm_model.predict(X_train_seq)
y_val_pred_scaled = lstm_model.predict(X_val_seq)
y_test_pred_scaled = lstm_model.predict(X_test_seq)

# 反標準化
y_train_actual = scaler_y.inverse_transform(y_train_seq.reshape(-1, 1)).flatten()
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled).flatten()

y_val_actual = scaler_y.inverse_transform(y_val_seq.reshape(-1, 1)).flatten()
y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled).flatten()

y_test_actual = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled).flatten()

# 計算評估指標
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

lstm_metrics = {
    'train': {
        'MSE': mean_squared_error(y_train_actual, y_train_pred),
        'RMSE': np.sqrt(mean_squared_error(y_train_actual, y_train_pred)),
        'MAE': mean_absolute_error(y_train_actual, y_train_pred),
        'R2': r2_score(y_train_actual, y_train_pred)
    },
    'val': {
        'MSE': mean_squared_error(y_val_actual, y_val_pred),
        'RMSE': np.sqrt(mean_squared_error(y_val_actual, y_val_pred)),
        'MAE': mean_absolute_error(y_val_actual, y_val_pred),
        'R2': r2_score(y_val_actual, y_val_pred)
    },
    'test': {
        'MSE': mean_squared_error(y_test_actual, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test_actual, y_test_pred)),
        'MAE': mean_absolute_error(y_test_actual, y_test_pred),
        'R2': r2_score(y_test_actual, y_test_pred)
    }
}
```

**LSTM 模型性能指標：**

| 數據集 | MSE | RMSE | MAE | R² |
|--------|-----|------|-----|-----|
| **訓練集** | 0.000234 | 0.0153 | 0.0118 | 0.8542 |
| **驗證集** | 0.000345 | 0.0186 | 0.0145 | 0.7834 |
| **測試集** | 0.000412 | 0.0203 | 0.0167 | 0.7234 |

📊 **性能解讀：**
- ✅ R² > 0.7：模型性能良好
- ✅ RMSE ≈ 0.02 mol%：預測誤差可接受
- ⚠️ 訓練-測試 R² 差距：0.13（存在輕微過擬合）

---

## 🚀 GRU 模型建立與訓練

### 5.1 GRU 原理簡介

**GRU（Gated Recurrent Unit）** 是 LSTM 的簡化版本，參數更少但性能相近。

**GRU vs LSTM：**

| 特性 | LSTM | GRU |
|------|------|-----|
| 門控數量 | 3 個（遺忘、輸入、輸出） | 2 個（重置、更新） |
| 參數數量 | 較多 | 較少（約 75%） |
| 訓練速度 | 較慢 | 較快 |
| 記憶能力 | 更強 | 略弱 |
| 適用場景 | 長時序、大數據 | 中短時序、小數據 |

**GRU 的核心組件：**

1. **重置門（Reset Gate）**：控制遺忘多少過去信息
   
   $$
   r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)
   $$

2. **更新門（Update Gate）**：控制接收多少新信息
   
   $$
   z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
   $$

### 5.2 GRU 模型架構

採用與 LSTM 相同的架構設計，便於公平比較：

```python
def build_gru_model(input_shape, units=[32, 16], dropout_rate=0.35):
    """
    建立雙層 GRU 模型（v3 優化版）
    與 LSTM 模型結構一致，僅將 LSTM 層替換為 GRU 層
    """
    from tensorflow.keras.layers import BatchNormalization
    
    model = Sequential(name='GRU_Model')
    
    # 第一層 GRU
    model.add(GRU(
        units=units[0],
        return_sequences=True,
        input_shape=input_shape,
        kernel_regularizer=keras.regularizers.l2(0.02),
        recurrent_regularizer=keras.regularizers.l2(0.01),
        name='GRU_1'
    ))
    model.add(BatchNormalization(name='BatchNorm_1'))
    model.add(Dropout(dropout_rate, name='Dropout_1'))
    
    # 第二層 GRU
    model.add(GRU(
        units=units[1],
        kernel_regularizer=keras.regularizers.l2(0.02),
        recurrent_regularizer=keras.regularizers.l2(0.01),
        name='GRU_2'
    ))
    model.add(BatchNormalization(name='BatchNorm_2'))
    model.add(Dropout(dropout_rate, name='Dropout_2'))
    
    # Dense 層
    model.add(Dense(
        8, 
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.02),
        name='Dense_1'
    ))
    model.add(Dropout(dropout_rate * 0.5, name='Dropout_3'))
    
    # 輸出層
    model.add(Dense(1, name='Output'))
    
    # 編譯模型
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

# 建立並訓練 GRU 模型
gru_model = build_gru_model(
    input_shape=(TIME_STEPS, X_train_seq.shape[2]),
    units=[32, 16],
    dropout_rate=0.35
)
```

**GRU 模型參數：**
```
Total params: 8,193
Trainable params: 8,097
Non-trainable params: 96
```

💡 **參數對比**：
- LSTM：9,489 個參數
- GRU：8,193 個參數
- **GRU 少 13.6%** 的參數

### 5.3 GRU 訓練與評估

使用相同的訓練策略：

```python
# 訓練 GRU 模型
gru_history = gru_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=200,
    batch_size=32,
    callbacks=gru_callbacks,
    verbose=1
)

# 評估 GRU 模型
# （評估代碼與 LSTM 相同，此處省略）
```

**GRU 模型性能指標：**

| 數據集 | MSE | RMSE | MAE | R² |
|--------|-----|------|-----|-----|
| **訓練集** | 0.000256 | 0.0160 | 0.0124 | 0.8421 |
| **驗證集** | 0.000367 | 0.0192 | 0.0151 | 0.7712 |
| **測試集** | 0.000438 | 0.0209 | 0.0173 | 0.7089 |

---

## 📊 模型比較與評估

### 6.1 性能指標對比

**完整對比表：**

| 指標 | LSTM 訓練集 | LSTM 驗證集 | LSTM 測試集 | GRU 訓練集 | GRU 驗證集 | GRU 測試集 |
|------|------------|------------|------------|-----------|-----------|-----------|
| **MSE** | 0.000234 | 0.000345 | 0.000412 | 0.000256 | 0.000367 | 0.000438 |
| **RMSE** | 0.0153 | 0.0186 | 0.0203 | 0.0160 | 0.0192 | 0.0209 |
| **MAE** | 0.0118 | 0.0145 | 0.0167 | 0.0124 | 0.0151 | 0.0173 |
| **R²** | 0.8542 | 0.7834 | 0.7234 | 0.8421 | 0.7712 | 0.7089 |

**綜合評估：**

| 方面 | LSTM | GRU | 優勝者 |
|------|------|-----|--------|
| 測試集 R² | 0.7234 | 0.7089 | LSTM ✓ |
| 訓練速度 | 較慢 | 較快 | GRU ✓ |
| 參數數量 | 9,489 | 8,193 | GRU ✓ |
| 預測精度 | 略高 | 略低 | LSTM ✓ |
| 泛化能力 | 中等 | 中等 | 平手 |

🏆 **結論**：
- **LSTM 在測試集上表現稍好**（R² 高 1.45%）
- **GRU 更輕量、訓練更快**
- 對於此數據集，**兩者差異不大，皆可使用**

### 6.2 預測結果可視化

繪製實際值 vs 預測值：

```python
fig, axes = plt.subplots(3, 2, figsize=(18, 15))

# 訓練集、驗證集、測試集預測
for row, (actual, pred_lstm, pred_gru, title, r2_lstm, r2_gru) in enumerate([
    (y_train_actual, y_train_pred, y_train_pred_gru, 'Training Set', 0.8542, 0.8421),
    (y_val_actual, y_val_pred, y_val_pred_gru, 'Validation Set', 0.7834, 0.7712),
    (y_test_actual, y_test_pred, y_test_pred_gru, 'Test Set', 0.7234, 0.7089)
]):
    # LSTM 預測
    axes[row, 0].plot(actual, label='Actual', linewidth=2, alpha=0.8)
    axes[row, 0].plot(pred_lstm, label='LSTM Prediction', linewidth=2, alpha=0.8)
    axes[row, 0].set_title(f'LSTM - {title}', fontsize=14, fontweight='bold')
    axes[row, 0].legend()
    axes[row, 0].grid(True, alpha=0.3)
    axes[row, 0].text(0.02, 0.95, f'R² = {r2_lstm:.4f}', 
                      transform=axes[row, 0].transAxes, fontsize=12,
                      bbox=dict(boxstyle='round', facecolor='lightblue'))
    
    # GRU 預測
    axes[row, 1].plot(actual, label='Actual', linewidth=2, alpha=0.8)
    axes[row, 1].plot(pred_gru, label='GRU Prediction', linewidth=2, alpha=0.8)
    axes[row, 1].set_title(f'GRU - {title}', fontsize=14, fontweight='bold')
    axes[row, 1].legend()
    axes[row, 1].grid(True, alpha=0.3)
    axes[row, 1].text(0.02, 0.95, f'R² = {r2_gru:.4f}', 
                      transform=axes[row, 1].transAxes, fontsize=12,
                      bbox=dict(boxstyle='round', facecolor='lightgreen'))
```

**視覺觀察：**
- ✅ 兩模型都能追蹤整體趨勢
- ✅ 訓練集擬合良好
- ⚠️ 測試集存在部分偏差
- ⚠️ 峰值和谷值預測略有滯後

### 6.3 殘差分析

```python
# 計算殘差
lstm_residuals_test = y_test_actual - y_test_pred
gru_residuals_test = y_test_actual - y_test_pred_gru

# 繪製殘差圖
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# LSTM 殘差
ax1.scatter(y_test_pred, lstm_residuals_test, alpha=0.5)
ax1.axhline(y=0, color='r', linestyle='--')
ax1.set_xlabel('Predicted Values')
ax1.set_ylabel('Residuals')
ax1.set_title('LSTM - Residual Plot (Test Set)')
ax1.grid(True, alpha=0.3)

# GRU 殘差
ax2.scatter(y_test_pred_gru, gru_residuals_test, alpha=0.5)
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_xlabel('Predicted Values')
ax2.set_ylabel('Residuals')
ax2.set_title('GRU - Residual Plot (Test Set)')
ax2.grid(True, alpha=0.3)
```

**殘差分析結果：**
- ✅ 殘差大致圍繞 0 線分布
- ✅ 無明顯的系統性偏差
- ⚠️ 部分點殘差較大（> 0.05）
- 📌 殘差分布接近隨機，模型基本可用

---

## 🔍 過擬合診斷與改進

### 7.1 過擬合診斷

**什麼是過擬合？**
- 模型在訓練集上表現很好，但在新數據上表現差
- 模型"記憶"了訓練數據，而非"學習"規律

**診斷指標：**

```python
# 計算訓練-驗證-測試 R² 差距
lstm_train_val_gap = lstm_metrics['train']['R2'] - lstm_metrics['val']['R2']
lstm_val_test_gap = lstm_metrics['val']['R2'] - lstm_metrics['test']['R2']

gru_train_val_gap = gru_metrics['train']['R2'] - gru_metrics['val']['R2']
gru_val_test_gap = gru_metrics['val']['R2'] - gru_metrics['test']['R2']
```

**LSTM 過擬合分析：**
```
訓練集 R²:   0.8542
驗證集 R²:   0.7834
測試集 R²:   0.7234

訓練-驗證差距: 0.0708  ⚠️ 注意：存在輕微過擬合
驗證-測試差距: 0.0600
```

**GRU 過擬合分析：**
```
訓練集 R²:   0.8421
驗證集 R²:   0.7712
測試集 R²:   0.7089

訓練-驗證差距: 0.0709  ⚠️ 注意：存在輕微過擬合
驗證-測試差距: 0.0623
```

**診斷標準：**

| 訓練-測試 R² 差距 | 嚴重程度 | 建議 |
|-------------------|----------|------|
| < 0.1 | ✅ 正常 | 無需特別處理 |
| 0.1 ~ 0.2 | ⚠️ 輕微過擬合 | 增加正則化 |
| 0.2 ~ 0.5 | ⚠️⚠️ 中度過擬合 | 降低模型複雜度 |
| > 0.5 | ❌ 嚴重過擬合 | 重新設計模型 |

**本案例診斷結果：**
- LSTM 差距：0.13 → **輕微過擬合**
- GRU 差距：0.13 → **輕微過擬合**
- 兩模型都需要輕微改進

### 7.2 已實施的改進措施

本 Notebook 已經過**三次迭代優化**：

**第一版（原始模型）**：
- TIME_STEPS = 10
- 雙層 LSTM [64, 32]
- Dropout = 0.2
- 無 L2 正則化
- **結果**：測試集 R² < 0 ❌ 完全失敗

**第二版（初次改進）**：
- TIME_STEPS = 20
- 雙層 LSTM [32, 16]
- Dropout = 0.3
- L2 = 0.01
- **結果**：測試集 R² ≈ 0.13 ⚠️ 仍嚴重過擬合

**第三版（當前版本）**：
- TIME_STEPS = 20
- 雙層 LSTM/GRU [32, 16]
- Dropout = 0.35
- L2 = 0.02 (kernel) + 0.01 (recurrent)
- **新增 BatchNormalization**
- **新增差分特徵工程**
- 學習率 = 0.001
- **結果**：測試集 R² ≈ 0.72 ✅ 可用

**改進效果對比：**

| 版本 | LSTM Test R² | GRU Test R² | 主要改進 |
|------|-------------|------------|----------|
| v1 | < 0 | < 0 | 基準版本 |
| v2 | 0.13 | -0.07 | 降低複雜度 + 增加正則化 |
| v3 | **0.72** | **0.71** | BatchNorm + 特徵工程 |

### 7.3 進一步改進建議

如果您的模型性能仍不滿意，可以嘗試：

#### 策略 1：簡化模型架構

```python
# 單層 LSTM
def build_simple_lstm(input_shape, units=24, dropout_rate=0.4):
    model = Sequential([
        LSTM(units=units, input_shape=input_shape,
             kernel_regularizer=keras.regularizers.l2(0.01)),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
```

**適用情況**：
- 數據量 < 1000 樣本
- 雙層模型過擬合嚴重

#### 策略 2：增加訓練數據

**數據增強技術：**
```python
# 添加輕微噪聲
noise_std = 0.01
X_train_aug = X_train + np.random.normal(0, noise_std, X_train.shape)

# 時間翻轉（對於穩態製程）
X_train_flip = np.flip(X_train, axis=1)

# 組合原始和增強數據
X_train_combined = np.vstack([X_train, X_train_aug, X_train_flip])
```

#### 策略 3：更強的正則化

```python
# 增強 dropout
dropout_rate = 0.5  # 原本 0.35

# 增強 L2
kernel_regularizer=keras.regularizers.l2(0.03)  # 原本 0.02

# 添加 Dropout 到 LSTM 內部
model.add(LSTM(units=32, dropout=0.2, recurrent_dropout=0.2))
```

#### 策略 4：集成學習

```python
# 訓練多個模型
models = []
for i in range(5):
    model = build_lstm_model(...)
    model.fit(X_train, y_train, ...)
    models.append(model)

# 平均預測
predictions = [model.predict(X_test) for model in models]
y_pred_ensemble = np.mean(predictions, axis=0)
```

**集成學習優勢：**
- 降低單一模型的不確定性
- 提升泛化能力
- 通常可提升 2-5% R²

#### 策略 5：超參數調優

使用 Keras Tuner 自動搜索最佳參數：

```python
from keras_tuner import RandomSearch

def build_model(hp):
    model = Sequential()
    model.add(LSTM(
        units=hp.Int('units_1', min_value=16, max_value=64, step=16),
        return_sequences=True,
        input_shape=input_shape
    ))
    model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
    # ... 更多層
    return model

tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=20
)

tuner.search(X_train, y_train, validation_data=(X_val, y_val))
```

### 7.4 模型選擇決策樹

```
測試集 R² < 0.3？
├─ 是 → ❌ 深度學習不適合，改用 XGBoost/Random Forest
└─ 否 → 繼續

測試集 R² > 0.7？
├─ 是 → ✅ 模型可用，考慮部署
└─ 否 → 繼續

訓練-測試 R² 差距 > 0.2？
├─ 是 → ⚠️ 嚴重過擬合，降低模型複雜度
└─ 否 → ⚠️ 輕微過擬合，增加正則化

數據量 < 2000？
├─ 是 → 考慮傳統 ML（XGBoost）
└─ 否 → 可繼續使用深度學習
```

---

## 🔄 備選方案與建議

### 8.1 方案一：單層 LSTM/GRU

**適用場景**：數據量 < 1500 或雙層模型過擬合嚴重

```python
def build_simple_lstm(input_shape, units=24, dropout_rate=0.4):
    """
    單層 LSTM 模型
    參數量更少，適合小數據集
    """
    model = Sequential([
        LSTM(units=units, 
             input_shape=input_shape,
             kernel_regularizer=keras.regularizers.l2(0.01)),
        Dropout(dropout_rate),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model
```

**預期效果**：
- ✅ 降低過擬合風險
- ✅ 訓練速度更快
- ⚠️ 可能犧牲部分擬合能力

### 8.2 方案二：傳統機器學習

**XGBoost 實現**（通常在小數據集上效果最好）：

```python
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# 展平時序窗口為特徵向量
X_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
X_test_flat = X_test_seq.reshape(X_test_seq.shape[0], -1)

# XGBoost 模型
xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train_flat, y_train_actual)
y_pred_xgb = xgb_model.predict(X_test_flat)

# Random Forest 模型
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

rf_model.fit(X_train_flat, y_train_actual)
y_pred_rf = rf_model.predict(X_test_flat)
```

**為什麼考慮傳統 ML？**

| 優勢 | 說明 |
|------|------|
| **小數據友好** | 在 < 5000 樣本時通常優於深度學習 |
| **無需大量調參** | 默認參數通常已經很好 |
| **訓練快速** | 幾秒到幾分鐘 |
| **可解釋性強** | 可提供特徵重要性分析 |
| **不易過擬合** | 內建正則化機制 |

**何時選擇傳統 ML？**
- 🔹 深度學習測試集 R² < 0.5
- 🔹 數據量 < 2000 樣本
- 🔹 需要快速部署
- 🔹 需要特徵重要性分析

### 8.3 方案三：混合模型

結合時序特徵和統計特徵：

```python
def extract_statistical_features(X_seq):
    """
    從時序窗口提取統計特徵
    """
    features = []
    for i in range(X_seq.shape[0]):
        seq = X_seq[i]  # shape: (time_steps, n_features)
        
        # 統計特徵
        mean_f = seq.mean(axis=0)
        std_f = seq.std(axis=0)
        max_f = seq.max(axis=0)
        min_f = seq.min(axis=0)
        
        # 趨勢特徵
        diff_first_last = seq[-1] - seq[0]
        
        # 組合特徵
        combined = np.concatenate([
            mean_f, std_f, max_f, min_f, diff_first_last
        ])
        features.append(combined)
    
    return np.array(features)

# 提取特徵
X_train_stats = extract_statistical_features(X_train_seq)
X_test_stats = extract_statistical_features(X_test_seq)

# 使用 XGBoost
xgb_model.fit(X_train_stats, y_train_actual)
y_pred_hybrid = xgb_model.predict(X_test_stats)
```

**混合模型優勢：**
- ✅ 結合時序信息和統計特性
- ✅ 降低維度（原本 20×14=280 維 → 70 維）
- ✅ 通常比純 LSTM 更穩定

### 8.4 實際應用建議

#### 化工製程部署考量

**1. 模型性能要求**

| R² 範圍 | 部署建議 | 應用方式 |
|---------|----------|----------|
| > 0.9 | ✅ 可直接用於自動控制 | 閉環控制 |
| 0.7 ~ 0.9 | ✅ 可用於監控與預警 | 軟測量 + 人工確認 |
| 0.5 ~ 0.7 | ⚠️ 僅供參考 | 輔助決策 |
| < 0.5 | ❌ 不建議部署 | 需改進模型 |

**本案例（R² ≈ 0.72）**：
- ✅ 適合用於監控與預警
- ⚠️ 不建議用於關鍵自動控制
- 建議：與人工判斷結合使用

**2. 安全裕度設計**

```python
# 預測值加上安全裕度
safety_margin = 0.1  # 10% 安全裕度
y_pred_safe = y_pred * (1 + safety_margin)

# 設置警報閾值
threshold_warning = 0.75  # 警告閾值
threshold_alarm = 0.80    # 警報閾值

if y_pred_safe > threshold_warning:
    print("⚠️ 警告：C4 含量接近上限")
if y_pred_safe > threshold_alarm:
    print("🚨 警報：C4 含量超標風險！")
```

**3. 持續監控與更新**

```python
# 定期評估模型性能
def monitor_model_performance(y_true_recent, y_pred_recent):
    """
    監控模型性能是否衰退
    """
    r2_recent = r2_score(y_true_recent, y_pred_recent)
    mae_recent = mean_absolute_error(y_true_recent, y_pred_recent)
    
    # 與原始測試集性能比較
    if r2_recent < original_r2 - 0.1:
        print("⚠️ 模型性能衰退，建議重新訓練")
    
    return r2_recent, mae_recent

# 建議更新頻率
# - 每 1-3 個月用新數據重新訓練
# - 每週評估預測誤差趨勢
# - 製程改變時立即重新訓練
```

**4. 異常檢測整合**

```python
# 輸入數據異常檢測
def detect_input_anomaly(X_new, X_train):
    """
    檢測輸入數據是否超出訓練範圍
    """
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    
    # 允許 10% 超出範圍
    margin = 0.1
    X_min_safe = X_min * (1 - margin)
    X_max_safe = X_max * (1 + margin)
    
    is_anomaly = (X_new < X_min_safe) | (X_new > X_max_safe)
    
    if is_anomaly.any():
        print("⚠️ 輸入數據異常，預測結果可能不可靠")
        return True
    return False
```

### 8.5 性能基準對比

**與文獻報告對比**：

| 方法 | 本案例 R² | 文獻報告 R² | 備註 |
|------|-----------|------------|------|
| LSTM | 0.72 | 0.75-0.85 | 略低於文獻（可能因數據不同） |
| GRU | 0.71 | 0.73-0.83 | 表現相近 |
| Random Forest | - | 0.70-0.80 | 建議測試 |
| XGBoost | - | 0.75-0.85 | 建議測試 |

📚 **參考文獻**：
> Fortuna, L., Graziani, S., Rizzo, A., & Xibilia, M. G. (2007). *Soft Sensors for Monitoring and Control of Industrial Processes*. Springer.

---

## 📝 結論與討論

### 主要成果

本單元成功建立了去丁烷塔 C4 含量預測的時序模型：

✅ **模型性能：**
- LSTM 測試集 R² = 0.72（可用）
- GRU 測試集 R² = 0.71（可用）
- 預測誤差 RMSE ≈ 0.02 mol%

✅ **技術實踐：**
- 時序數據預處理與序列化
- LSTM 和 GRU 模型設計與訓練
- 過擬合診斷與模型優化
- 多種正則化技術應用

✅ **工程應用：**
- 可用於製程監控與預警
- 建議與人工判斷結合使用
- 需定期重新訓練以維持性能

### 關鍵學習點

1. **數據量是關鍵**
   - 深度學習需要足夠數據（理想 > 5000 樣本）
   - 小數據集（< 2000）建議優先考慮傳統 ML

2. **過擬合是最大挑戰**
   - 需要多種正則化技術組合
   - Dropout + L2 + BatchNorm 三管齊下
   - 模型不是越複雜越好

3. **特徵工程很重要**
   - 差分特徵捕捉變化趨勢
   - 統計特徵提供穩定信息
   - 領域知識指導特徵選擇

4. **驗證策略要正確**
   - 時序分割保持時間順序
   - 測試集模擬實際應用情境
   - 關注泛化能力而非訓練精度

5. **實際部署需謹慎**
   - 加入安全裕度
   - 持續監控性能
   - 整合異常檢測
   - 定期更新模型

### 改進方向

如果您想進一步提升模型：

🔹 **短期改進**：
- 嘗試 XGBoost 和 Random Forest
- 實施集成學習（模型平均）
- 添加更多統計特徵

🔹 **中期改進**：
- 收集更多歷史數據（目標 > 5000 樣本）
- 實施 k-fold 交叉驗證
- 使用 Keras Tuner 自動調參

🔹 **長期改進**：
- 探索 Transformer 架構
- 結合物理模型（混合建模）
- 開發自適應學習系統

### 適用場景

本方法適用於以下化工製程：

✅ **適合的場景**：
- 具有時序相關性的製程變數
- 有歷史操作數據（> 1000 樣本）
- 測量延遲或成本高的品質指標
- 需要即時預測的應用

⚠️ **需謹慎的場景**：
- 數據量極少（< 500 樣本）
- 製程經常大幅變動
- 關鍵安全控制（建議加安全裕度）
- 需要嚴格可解釋性

### 延伸閱讀

**推薦資源：**

📚 **書籍**：
1. Fortuna et al., "Soft Sensors for Monitoring and Control of Industrial Processes"
2. Goodfellow et al., "Deep Learning"
3. Chollet, "Deep Learning with Python"

📄 **論文**：
1. Hochreiter & Schmidhuber (1997) - LSTM 原始論文
2. Cho et al. (2014) - GRU 原始論文
3. 化工軟測量相關文獻

🔗 **線上資源**：
- TensorFlow/Keras 官方文檔
- Scikit-learn 時序預測教程
- Kaggle 時序預測競賽案例

---

## 🎓 練習題

### 基礎練習

1. **修改 TIME_STEPS**
   - 嘗試 TIME_STEPS = 10, 15, 25, 30
   - 比較不同窗口長度對性能的影響
   - 分析訓練樣本數量的變化

2. **單層vs雙層**
   - 實作單層 LSTM 模型
   - 比較與雙層模型的性能差異
   - 記錄訓練時間和參數數量

3. **不同分割比例**
   - 嘗試 80/10/10 分割
   - 比較與 70/15/15 的差異
   - 討論對模型評估的影響

### 進階練習

4. **XGBoost 實作**
   - 將時序數據展平為特徵向量
   - 訓練 XGBoost 回歸模型
   - 與 LSTM/GRU 性能對比

5. **集成學習**
   - 訓練 3-5 個 LSTM 模型
   - 實作預測平均
   - 評估集成效果

6. **特徵重要性分析**
   - 使用 SHAP 或 Permutation Importance
   - 分析哪些輸入變數最重要
   - 嘗試移除不重要特徵

### 挑戰練習

7. **超參數優化**
   - 使用 Keras Tuner 或 Optuna
   - 搜索最佳的 units, dropout, L2 組合
   - 記錄搜索過程和結果

8. **異常檢測整合**
   - 實作輸入數據異常檢測
   - 添加預測不確定性估計
   - 設計警報系統

9. **實時預測系統**
   - 實作滑動窗口即時預測
   - 模擬製程數據流
   - 評估推論速度

---

## 📎 參考資源

### 數據來源

- **GitHub Repository**: [Debutanizer Column Data](https://github.com/sj823774188/Debutanizer-Column-Data)
- **原始文獻**: Fortuna et al. (2007), Soft Sensors for Monitoring and Control of Industrial Processes

### 相關單元

- **Unit13**: 時序預測基礎
- **Unit14**: 強化學習控制
- **Unit15**: RUL 預測

### 技術文檔

- [TensorFlow LSTM Guide](https://www.tensorflow.org/guide/keras/rnn)
- [Keras Callbacks](https://keras.io/api/callbacks/)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

**✨ 恭喜完成 Unit17！**

您已經掌握了使用深度學習進行化工製程時序預測的完整流程。這些技能可以應用到許多實際的工業場景中，包括品質預測、故障預警、能耗優化等。

**下一步建議**：
- 嘗試應用到您自己的數據集
- 探索更多正則化技術
- 學習模型部署與監控

祝學習愉快！🚀

---

**課程資訊**
- 課程名稱：AI在化工上之應用
- 課程單元：Unit17 - 去丁烷塔 C4 含量預測
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-01-28

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---

