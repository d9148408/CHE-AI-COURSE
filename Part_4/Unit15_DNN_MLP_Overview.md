# Unit 15: 深度神經網路(DNN)與多層感知機(MLP)概述

## 課程目標
- 理解深度神經網路(DNN)與多層感知機(MLP)的基本概念與數學原理
- 學會使用TensorFlow/Keras建立、訓練、評估DNN模型
- 掌握模型優化技巧與超參數調整方法
- 了解DNN在化工領域的應用場景

---

## 1. DNN與MLP基礎理論

### 1.1 什麼是神經網路?

**人工神經網路(Artificial Neural Network, ANN)** 是一種受生物神經系統啟發的機器學習模型。它透過模擬神經元之間的連接與訊號傳遞，來學習輸入與輸出之間的複雜關係。

**多層感知機(Multi-Layer Perceptron, MLP)** 是最基本的前饋式神經網路(Feedforward Neural Network)，由多層神經元組成:
- **輸入層(Input Layer)**: 接收原始特徵數據
- **隱藏層(Hidden Layers)**: 進行特徵轉換與學習
- **輸出層(Output Layer)**: 產生最終預測結果

**深度神經網路(Deep Neural Network, DNN)** 是指具有**多個隱藏層**的神經網路。當隱藏層數量增加時，網路能夠學習更複雜、更抽象的特徵表示。

### 1.2 歷史發展

- **1943**: McCulloch & Pitts 提出第一個神經元數學模型
- **1958**: Rosenblatt 發明感知機(Perceptron)
- **1986**: Rumelhart 等人提出反向傳播演算法(Backpropagation)
- **2006**: Hinton 提出深度學習(Deep Learning)概念
- **2012**: AlexNet 在ImageNet競賽中大放異彩，開啟深度學習時代

### 1.3 神經元數學模型

單一神經元的運算可表示為:

$$
z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^T \mathbf{x} + b
$$

$$
a = f(z)
$$

其中:
- $x_i$ : 輸入特徵
- $w_i$ : 權重(weight)
- $b$ : 偏差(bias)
- $z$ : 加權總和
- $f$ : 激活函數(activation function)
- $a$ : 神經元輸出(activation)

### 1.4 前向傳播(Forward Propagation)

對於一個具有 $L$ 層的神經網路，前向傳播過程為:

**第一層(輸入層到第一個隱藏層)**:
$$
\mathbf{z}^{[1]} = \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]}
$$
$$
\mathbf{a}^{[1]} = f^{[1]}(\mathbf{z}^{[1]})
$$

**第 $l$ 層(一般化)**:
$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}
$$
$$
\mathbf{a}^{[l]} = f^{[l]}(\mathbf{z}^{[l]})
$$

**輸出層**:
$$
\hat{y} = \mathbf{a}^{[L]}
$$

其中:
- $\mathbf{W}^{[l]}$ : 第 $l$ 層的權重矩陣
- $\mathbf{b}^{[l]}$ : 第 $l$ 層的偏差向量
- $f^{[l]}$ : 第 $l$ 層的激活函數

### 1.5 損失函數(Loss Function)

損失函數用於衡量模型預測值與真實值之間的差異:

**回歸問題常用損失函數**:

1. **均方誤差(Mean Squared Error, MSE)**:
$$
L_{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

**適用場景**:
- 預測連續數值的回歸問題
- 對大誤差敏感,因為誤差被平方放大
- 適合目標變數分布較為均勻的情況

**優點**: 數學性質良好,可微分,梯度計算簡單  
**缺點**: 對異常值非常敏感

2. **平均絕對誤差(Mean Absolute Error, MAE)**:
$$
L_{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

**適用場景**:
- 回歸問題,特別是存在異常值的數據
- 對所有誤差一視同仁(不會放大大誤差)
- 適合需要更穩健(robust)模型的場景

**優點**: 對異常值不敏感  
**缺點**: 在0點不可微,優化較困難

3. **Huber Loss**:
$$
L_{\delta}(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
$$

**適用場景**:
- 結合MSE和MAE的優點
- 小誤差時使用MSE(平滑),大誤差時使用MAE(穩健)
- 適合工業數據中含有噪音和異常值的情況

**分類問題常用損失函數**:

1. **二元交叉熵(Binary Crossentropy)**:
$$
L_{BCE} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]
$$

**適用場景**:
- **二元分類問題** (如:是/否、正常/異常)
- 輸出層使用Sigmoid激活函數
- 標籤為0或1

**範例**: 化工設備故障預測、產品合格與否判定

2. **類別交叉熵(Categorical Crossentropy)**:
$$
L_{CCE} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{C} y_{ij} \log(\hat{y}_{ij})
$$

**適用場景**:
- **多類別分類問題** (C > 2)
- 輸出層使用Softmax激活函數
- 標籤為one-hot編碼格式

**範例**: 產品品質等級分類(A/B/C級)、化學反應類型識別

3. **稀疏類別交叉熵(Sparse Categorical Crossentropy)**:
$$
L_{SCCE} = -\frac{1}{n} \sum_{i=1}^{n} \log(\hat{y}_{i,c_i})
$$

其中 $c_i$ 是第 $i$ 個樣本的真實類別索引。

**適用場景**:
- **多類別分類問題**
- 標籤為整數格式(0, 1, 2, ..., C-1),而非one-hot編碼
- 節省記憶體,適合類別數量很多的情況

### 損失函數選擇指南

| 問題類型 | 推薦損失函數 | 輸出層激活函數 |
|---------|------------|---------------|
| 回歸(一般) | MSE | Linear / 不指定 |
| 回歸(有異常值) | MAE 或 Huber | Linear / 不指定 |
| 二元分類 | Binary Crossentropy | Sigmoid |
| 多類別分類(one-hot) | Categorical Crossentropy | Softmax |
| 多類別分類(整數標籤) | Sparse Categorical Crossentropy | Softmax |


### 1.6 反向傳播(Backpropagation)

反向傳播演算法用於計算損失函數對每個參數的梯度，使用**鏈式法則(Chain Rule)**從輸出層往回計算:

$$
\frac{\partial L}{\partial \mathbf{W}^{[l]}} = \frac{\partial L}{\partial \mathbf{z}^{[l]}} \cdot \frac{\partial \mathbf{z}^{[l]}}{\partial \mathbf{W}^{[l]}} = \delta^{[l]} \cdot (\mathbf{a}^{[l-1]})^T
$$

$$
\frac{\partial L}{\partial \mathbf{b}^{[l]}} = \delta^{[l]}
$$

其中 $\delta^{[l]}$ 是第 $l$ 層的誤差項。

### 1.7 梯度下降與參數更新

使用梯度下降法更新參數:

$$
\mathbf{W}^{[l]} := \mathbf{W}^{[l]} - \alpha \frac{\partial L}{\partial \mathbf{W}^{[l]}}
$$

$$
\mathbf{b}^{[l]} := \mathbf{b}^{[l]} - \alpha \frac{\partial L}{\partial \mathbf{b}^{[l]}}
$$

其中 $\alpha$ 是學習率(learning rate)。

---

## 2. 激活函數(Activation Functions)

激活函數為神經網路引入非線性，使其能夠學習複雜的函數關係。

### 2.1 常用激活函數

#### 2.1.1 ReLU (Rectified Linear Unit)
$$
f(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}
$$

**優點**:
- 計算簡單、速度快
- 有效緩解梯度消失問題
- 使網路具有稀疏性

**缺點**:
- 可能出現"神經元死亡"問題(dying ReLU)

**適用場景**: 隱藏層的首選激活函數

#### 2.1.2 Leaky ReLU
$$
f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}
$$

其中 $\alpha$ 通常設為 0.01。

**優點**: 解決ReLU的神經元死亡問題

#### 2.1.3 Sigmoid
$$
f(x) = \frac{1}{1 + e^{-x}}
$$

**特性**:
- 輸出範圍: (0, 1)
- 可解釋為機率

**缺點**:
- 容易出現梯度消失
- 輸出不是以零為中心

**適用場景**: 二元分類的輸出層

#### 2.1.4 Tanh (雙曲正切)
$$
f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

**特性**:
- 輸出範圍: (-1, 1)
- 以零為中心

**缺點**: 仍有梯度消失問題

**適用場景**: 隱藏層(但ReLU通常更好)

#### 2.1.5 Softmax
$$
f(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{C} e^{x_j}}
$$

**特性**:
- 輸出總和為1
- 可解釋為機率分布

**適用場景**: 多類別分類的輸出層

#### 2.1.6 Linear (線性)
$$
f(x) = x
$$

**適用場景**: 回歸問題的輸出層

### 2.2 激活函數選擇指南

| 層類型 | 問題類型 | 推薦激活函數 |
|--------|----------|--------------|
| 隱藏層 | 一般情況 | ReLU |
| 隱藏層 | 避免dying ReLU | Leaky ReLU |
| 輸出層 | 二元分類 | Sigmoid |
| 輸出層 | 多類別分類 | Softmax |
| 輸出層 | 回歸 | Linear (或不指定) |

---

## 3. DNN/MLP應用場景

### 3.1 適合使用DNN/MLP的情境

1. **非線性關係複雜**: 輸入與輸出之間存在高度非線性關係
2. **特徵交互作用**: 特徵之間有複雜的交互作用
3. **大量數據**: 有足夠的訓練數據支持深度模型
4. **特徵工程困難**: 難以手動設計有效特徵時，DNN可自動學習

### 3.2 化工領域應用案例

#### 3.2.1 製程參數優化
- **應用**: 預測反應器溫度、壓力、流量等操作條件對產品品質的影響
- **優勢**: 可處理多變數、非線性的製程關係

#### 3.2.2 產品品質預測
- **應用**: 根據原料成分與製程條件預測最終產品性質
- **範例**: 紅酒品質預測、聚合物性質預測

#### 3.2.3 設備故障診斷
- **應用**: 透過感測器數據預測設備異常或故障
- **優勢**: 可學習複雜的時間序列模式

#### 3.2.4 分離程序模擬
- **應用**: 蒸餾塔、萃取塔等分離設備的快速模擬
- **優勢**: 比傳統數值模擬快速，適合即時控制

#### 3.2.5 環境排放預測
- **應用**: 預測燃燒程序的污染物排放量
- **範例**: NOx、SOx、CO2排放預測

#### 3.2.6 礦業浮選過程
- **應用**: 預測礦石浮選過程的矽石濃度
- **優勢**: 可整合多種感測器數據進行即時預測

### 3.3 DNN的優勢與限制

**優勢**:
- 強大的非線性建模能力
- 自動特徵學習
- 可處理高維度數據
- 擴展性好

**限制**:
- 需要大量訓練數據
- 計算資源需求高
- 模型可解釋性較差(黑盒模型)
- 容易過擬合
- 超參數調整複雜

---

## 4. TensorFlow/Keras框架介紹

### 4.1 TensorFlow與Keras簡介

**TensorFlow** 是Google開發的開源深度學習框架，提供完整的機器學習生態系統。

**Keras** 是高階神經網路API，現已整合進TensorFlow 2.x (tf.keras)，提供:
- 簡潔易用的介面
- 模組化設計
- 易於擴展
- 支援多種後端

### 4.2 環境安裝

```bash
# 安裝TensorFlow (包含Keras)
pip install tensorflow

# 或安裝特定版本
pip install tensorflow==2.15.0

# 驗證安裝
python -c "import tensorflow as tf; print(tf.__version__)"
```

### 4.3 基本導入

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
```

---

## 5. 使用Keras建立DNN模型

### 5.1 模型架構: Sequential vs Functional API

Keras提供兩種建立模型的方式:

#### 5.1.1 Sequential API (序列模型)

適用於**單輸入、單輸出、線性堆疊**的簡單模型:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 建立Sequential模型
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1)
])

# 或使用add方法逐層添加
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(10,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
```

#### 5.1.2 Functional API (函數式API)

適用於**多輸入、多輸出、複雜連接**的模型:

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# 定義輸入
inputs = Input(shape=(10,))

# 定義隱藏層
x = Dense(64, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)

# 定義輸出
outputs = Dense(1)(x)

# 建立模型
model = Model(inputs=inputs, outputs=outputs)
```

### 5.2 常用層(Layers)

#### 5.2.1 Dense Layer (全連接層)

**功能**: 實現全連接的神經網路層

```python
from tensorflow.keras.layers import Dense

layer = Dense(
    units=64,              # 神經元數量
    activation='relu',     # 激活函數
    use_bias=True,         # 是否使用偏差
    kernel_initializer='glorot_uniform',  # 權重初始化方法
    bias_initializer='zeros',             # 偏差初始化方法
    kernel_regularizer=None,              # 權重正則化
    bias_regularizer=None,                # 偏差正則化
    activity_regularizer=None             # 輸出正則化
)
```

**參數說明**:
- `units`: 該層神經元數量
- `activation`: 激活函數 ('relu', 'sigmoid', 'tanh', 'softmax', 'linear', None)
- `kernel_initializer`: 權重初始化策略

#### 5.2.2 Dropout Layer (隨機失活層)

**功能**: 訓練時隨機將部分神經元輸出設為0，防止過擬合

```python
from tensorflow.keras.layers import Dropout

layer = Dropout(rate=0.5)  # 失活比例
```

**使用時機**:
- 模型出現過擬合時
- 通常放在Dense層之後
- 典型dropout rate: 0.2 ~ 0.5

**範例**:
```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(10,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1)
])
```

#### 5.2.3 BatchNormalization Layer (批次正規化層)

**功能**: 對每個batch的輸入進行標準化，加速訓練並提高穩定性

```python
from tensorflow.keras.layers import BatchNormalization

layer = BatchNormalization()
```

**優點**:
- 加快訓練速度
- 允許使用更高的學習率
- 減少對初始化的敏感度
- 具有輕微的正則化效果
- 緩解梯度消失/爆炸問題

**適用時機與問題種類**:

1. **深度網路** (層數 > 10層):
   - BatchNorm幫助梯度在深層網路中順利傳播
   - 特別適合圖像識別、自然語言處理等複雜任務

2. **訓練不穩定**:
   - 損失曲線波動劇烈
   - 梯度爆炸或消失
   - 對學習率過於敏感

3. **需要更快收斂**:
   - 訓練時間受限的場景
   - 大規模數據集

**使用位置**: 通常放在Dense層與激活函數之間

```python
model = Sequential([
    Dense(128, input_shape=(10,)),
    BatchNormalization(),
    Activation('relu'),
    Dense(64),
    BatchNormalization(),
    Activation('relu'),
    Dense(1)
])
```

### ⚠️ 工業數據回歸任務中的BatchNormalization使用建議

**問題**: 如果輸入數據已經使用StandardScaler進行標準化,是否還需要BatchNormalization?

**答案**: **視情況而定**

#### 情況1: 淺層網路 (≤ 3-4層) + 已標準化數據
**建議**: **不需要BatchNormalization**

**理由**:
- 輸入數據已標準化,第一層的輸入分布已經良好
- 淺層網路梯度傳播問題不明顯
- 增加BatchNorm會增加計算成本和模型複雜度
- 在小數據集上可能產生過擬合

```python
# 淺層網路範例 (數據已標準化)
model = Sequential([
    Dense(64, activation='relu', input_shape=(n_features,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])
```

#### 情況2: 深層網路 (> 4層) + 已標準化數據
**建議**: **建議使用BatchNormalization**

**理由**:
- 即使輸入標準化了,深層網路中間層的分布仍可能shift
- BatchNorm在每一層都重新標準化,穩定各層分布
- 幫助梯度傳播,加速訓練

```python
# 深層網路範例 (數據已標準化,仍使用BatchNorm)
model = Sequential([
    Dense(128, input_shape=(n_features,)),
    BatchNormalization(),
    Activation('relu'),
    
    Dense(64),
    BatchNormalization(),
    Activation('relu'),
    
    Dense(32),
    BatchNormalization(),
    Activation('relu'),
    
    Dense(16, activation='relu'),
    Dense(1)
])
```

#### 情況3: 數據未標準化
**建議**: **強烈建議使用StandardScaler + BatchNormalization**

**理由**:
- 先用StandardScaler標準化輸入特徵(必要步驟)
- 再用BatchNorm穩定訓練過程
- 兩者作用不同,可以互補

### 最佳實踐建議

| 網路深度 | 數據是否標準化 | 是否使用BatchNorm | 說明 |
|---------|--------------|-----------------|------|
| 淺層(≤4層) | 是 | 可選 | 通常不需要,除非訓練不穩定 |
| 淺層(≤4層) | 否 | 建議使用 | 先StandardScaler,可選BatchNorm |
| 深層(>4層) | 是 | 建議使用 | 穩定各層分布,加速收斂 |
| 深層(>4層) | 否 | 強烈建議 | StandardScaler + BatchNorm都需要 |

**化工工業應用經驗**:
- 製程數據回歸(如溫度、壓力預測): 淺層網路 + StandardScaler通常已足夠
- 複雜非線性系統(如蒸餾塔多變數控制): 深層網路 + BatchNorm效果更好
- 小數據集(<1000樣本): 謹慎使用BatchNorm,可能導致過擬合


#### 5.2.4 Activation Layer (激活層)

**功能**: 單獨定義激活函數層

```python
from tensorflow.keras.layers import Activation

layer = Activation('relu')
```

**等價寫法**:
```python
# 方法1: 在Dense中指定
Dense(64, activation='relu')

# 方法2: 使用單獨的Activation層
Dense(64)
Activation('relu')
```

### 5.3 權重初始化策略

合適的權重初始化可以加速訓練並避免梯度消失/爆炸問題。

| 初始化方法 | 說明 | 適用激活函數 |
|-----------|------|--------------|
| `glorot_uniform` (Xavier Uniform) | 預設值，均勻分布 | Sigmoid, Tanh, Softmax |
| `glorot_normal` (Xavier Normal) | 常態分布 | Sigmoid, Tanh, Softmax |
| `he_uniform` | 均勻分布 | ReLU, Leaky ReLU |
| `he_normal` | 常態分布 | ReLU, Leaky ReLU |
| `zeros` | 全部初始化為0 | 偏差項 |
| `ones` | 全部初始化為1 | - |

**使用範例**:
```python
from tensorflow.keras.initializers import HeNormal

model = Sequential([
    Dense(64, activation='relu', 
          kernel_initializer=HeNormal(),
          input_shape=(10,)),
    Dense(32, activation='relu',
          kernel_initializer=HeNormal()),
    Dense(1)
])
```

### 5.4 正則化(Regularization)

防止過擬合的技術。

#### 5.4.1 L1/L2正則化

```python
from tensorflow.keras.regularizers import l1, l2, l1_l2

# L2正則化 (Ridge)
Dense(64, activation='relu', kernel_regularizer=l2(0.01))

# L1正則化 (Lasso)
Dense(64, activation='relu', kernel_regularizer=l1(0.01))

# L1+L2正則化 (Elastic Net)
Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))
```

---

## 6. 模型編譯 (Model Compilation)

編譯模型時需要指定**優化器**、**損失函數**和**評估指標**。

### 6.1 model.compile() 方法

```python
model.compile(
    optimizer='adam',           # 優化器
    loss='mse',                 # 損失函數
    metrics=['mae', 'mse']      # 評估指標
)
```

### 6.2 優化器(Optimizers)

優化器決定如何根據梯度更新權重。

#### 6.2.1 Adam (Adaptive Moment Estimation)
**推薦首選**，結合了Momentum和RMSprop的優點。

```python
from tensorflow.keras.optimizers import Adam

optimizer = Adam(
    learning_rate=0.001,    # 學習率
    beta_1=0.9,             # 一階矩估計的指數衰減率
    beta_2=0.999,           # 二階矩估計的指數衰減率
    epsilon=1e-07           # 數值穩定性常數
)

model.compile(optimizer=optimizer, loss='mse')
```

**優點**:
- 自適應學習率
- 對超參數不敏感
- 適用於大多數問題

#### 6.2.2 SGD (Stochastic Gradient Descent)

```python
from tensorflow.keras.optimizers import SGD

optimizer = SGD(
    learning_rate=0.01,
    momentum=0.9,           # 動量
    nesterov=True           # 是否使用Nesterov動量
)
```

#### 6.2.3 RMSprop

```python
from tensorflow.keras.optimizers import RMSprop

optimizer = RMSprop(
    learning_rate=0.001,
    rho=0.9
)
```

#### 6.2.4 AdaGrad, Adadelta, Nadam 等

```python
from tensorflow.keras.optimizers import AdaGrad, Adadelta, Nadam
```

**優化器選擇建議**:
- **首選**: Adam (適用大多數情況)
- **需要更好泛化**: SGD with momentum
- **RNN問題**: RMSprop 或 Adam

### 6.3 損失函數(Loss Functions)

損失函數根據問題類型選擇,以下列出Keras中常用損失函數及其數學公式。

#### 6.3.1 回歸問題

**1. Mean Squared Error (MSE) - 均方誤差**

**數學公式**:
$$
L_{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

**Keras使用**:
```python
# 均方誤差 (Mean Squared Error)
model.compile(optimizer='adam', loss='mse')
model.compile(optimizer='adam', loss='mean_squared_error')
from tensorflow.keras.losses import MeanSquaredError
model.compile(optimizer='adam', loss=MeanSquaredError())
```

**特性**: 對大誤差懲罰重,梯度與誤差成正比

---

**2. Mean Absolute Error (MAE) - 平均絕對誤差**

**數學公式**:
$$
L_{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

**Keras使用**:
```python
# 平均絕對誤差 (Mean Absolute Error)
model.compile(optimizer='adam', loss='mae')
model.compile(optimizer='adam', loss='mean_absolute_error')
from tensorflow.keras.losses import MeanAbsoluteError
model.compile(optimizer='adam', loss=MeanAbsoluteError())
```

**特性**: 對異常值穩健,梯度為常數

---

**3. Huber Loss**

**數學公式**:
$$
L_{\delta}(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta (|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
$$

**Keras使用**:
```python
# Huber Loss (對異常值較不敏感)
from tensorflow.keras.losses import Huber
model.compile(optimizer='adam', loss=Huber(delta=1.0))
```

**特性**: 結合MSE和MAE優點,小誤差用MSE(平滑),大誤差用MAE(穩健)

---

**4. Mean Squared Logarithmic Error (MSLE) - 均方對數誤差**

**數學公式**:
$$
L_{MSLE} = \frac{1}{n} \sum_{i=1}^{n} (\log(y_i + 1) - \log(\hat{y}_i + 1))^2
$$

**Keras使用**:
```python
from tensorflow.keras.losses import MeanSquaredLogarithmicError
model.compile(optimizer='adam', loss='msle')
model.compile(optimizer='adam', loss=MeanSquaredLogarithmicError())
```

**適用場景**: 目標變數範圍很大,關注相對誤差而非絕對誤差

#### 6.3.2 二元分類問題

**Binary Crossentropy - 二元交叉熵**

**數學公式**:
$$
L_{BCE} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]
$$

其中:
- $y_i \in \{0, 1\}$ : 真實標籤
- $\hat{y}_i \in (0, 1)$ : 預測機率(Sigmoid輸出)

**Keras使用**:
```python
# 二元交叉熵
model.compile(optimizer='adam', loss='binary_crossentropy')
from tensorflow.keras.losses import BinaryCrossentropy
model.compile(optimizer='adam', loss=BinaryCrossentropy())
```

**配合使用**: 輸出層使用`Sigmoid`激活函數

#### 6.3.3 多類別分類問題

**1. Categorical Crossentropy - 類別交叉熵**

**數學公式**:
$$
L_{CCE} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{C} y_{ij} \log(\hat{y}_{ij})
$$

其中:
- $C$ : 類別數量
- $y_{ij}$ : one-hot編碼標籤 (第i個樣本屬於第j類時為1,否則為0)
- $\hat{y}_{ij}$ : 預測機率(Softmax輸出)

**Keras使用**:
```python
# 類別交叉熵 (標籤為one-hot編碼)
model.compile(optimizer='adam', loss='categorical_crossentropy')
from tensorflow.keras.losses import CategoricalCrossentropy
model.compile(optimizer='adam', loss=CategoricalCrossentropy())
```

**標籤格式**: `[[0, 0, 1], [1, 0, 0], [0, 1, 0]]` (one-hot)

---

**2. Sparse Categorical Crossentropy - 稀疏類別交叉熵**

**數學公式**:
$$
L_{SCCE} = -\frac{1}{n} \sum_{i=1}^{n} \log(\hat{y}_{i,c_i})
$$

其中:
- $c_i$ : 第i個樣本的真實類別索引
- $\hat{y}_{i,c_i}$ : 該樣本在真實類別上的預測機率

**Keras使用**:
```python
# 稀疏類別交叉熵 (標籤為整數)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
from tensorflow.keras.losses import SparseCategoricalCrossentropy
model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy())
```

**標籤格式**: `[2, 0, 1]` (整數索引)

**差異**: 與Categorical Crossentropy數學上等價,僅標籤格式不同

### 損失函數選擇快速參考

| 問題類型 | 損失函數 | 數學特性 |
|---------|---------|---------|
| 回歸(一般) | MSE | 對大誤差敏感 |
| 回歸(有異常值) | MAE 或 Huber | 穩健性強 |
| 回歸(大範圍目標值) | MSLE | 關注相對誤差 |
| 二元分類 | Binary Crossentropy | 機率解釋清晰 |
| 多類別(one-hot) | Categorical Crossentropy | 標準多類別損失 |
| 多類別(整數標籤) | Sparse Categorical Crossentropy | 節省記憶體 |


### 6.4 評估指標(Metrics)

#### 什麼是評估指標?

評估指標(Metrics)用於**監控和評估模型性能**,但**不會影響模型訓練過程和最終結果**。

> [!IMPORTANT]
> **常見誤解**: 許多初學者誤以為在`metrics`中添加指標會影響模型訓練和參數更新。
> 
> **正確理解**:
> - ✅ **Loss Function**: 決定模型如何學習和更新參數  
> - ✅ **Metrics**: 僅用於評估和監控,不影響訓練

#### Metrics的作用

1. **訓練過程監控**: 在訓練時顯示額外的評估指標
2. **模型比較**: 使用多個指標全面評估模型性能
3. **早停判斷**: 可作為EarlyStopping的監控指標
4. **結果記錄**: 保存在History物件中供後續分析

#### 使用範例

**回歸問題指標**:
```python
# 回歸指標
model.compile(
    optimizer='adam',
    loss='mse',                                  # 訓練優化目標 (影響訓練)
    metrics=['mae', 'mse', 'RootMeanSquaredError']  # 僅監控 (不影響訓練)
)
```

**分類問題指標**:
```python
# 分類指標
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',                  # 訓練優化目標
    metrics=['accuracy', 'Precision', 'Recall', 'AUC']  # 僅監控
)
```

**自訂指標**:
```python
# 自訂指標
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=[MeanAbsoluteError(name='MAE'),
             RootMeanSquaredError(name='RMSE')]
)
```

#### Loss vs Metrics 的關鍵區別

| 項目 | Loss Function | Metrics |
|------|--------------|---------|
| **數量** | 必須指定1個 | 可指定0個或多個 |
| **作用** | 計算梯度,更新權重 | 僅評估性能 |
| **影響訓練** | ✅ 是 | ❌ 否 |
| **顯示** | 訓練和驗證 | 訓練和驗證 |
| **儲存** | History物件 | History物件 |

#### 實際運作範例

```python
# 模型編譯
model.compile(
    optimizer='adam',
    loss='mse',           # 用於計算梯度並更新權重
    metrics=['mae']       # 僅用於顯示,不影響訓練
)

# 訓練
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# history物件中會記錄:
# - history.history['loss']      : 訓練集MSE (影響訓練)
# - history.history['val_loss']  : 驗證集MSE (影響訓練)
# - history.history['mae']       : 訓練集MAE (僅監控)
# - history.history['val_mae']   : 驗證集MAE (僅監控)
```

#### 常用Metrics列表

**回歸Metrics**:
- `mae` / `MeanAbsoluteError`: 平均絕對誤差
- `mse` / `MeanSquaredError`: 均方誤差
- `RootMeanSquaredError`: 均方根誤差
- `MeanAbsolutePercentageError`: 平均絕對百分比誤差

**分類Metrics**:
- `accuracy` / `BinaryAccuracy` / `CategoricalAccuracy`: 準確率
- `Precision`: 精確率
- `Recall`: 召回率
- `AUC`: ROC曲線下面積
- `F1Score`: F1分數


### 6.5 模型摘要與視覺化

#### 6.5.1 model.summary() - 文字摘要

查看模型架構、參數數量:

```python
model.summary()
```

**輸出範例**:
```
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
dense (Dense)               (None, 64)                704       
dense_1 (Dense)             (None, 32)                2080      
dense_2 (Dense)             (None, 1)                 33        
=================================================================
Total params: 2,817
Trainable params: 2,817
Non-trainable params: 0
_________________________________________________________________
```

**參數計算**:
- Dense層參數數量 = (輸入特徵數 + 1) × 神經元數
- 第一層: (10 + 1) × 64 = 704
- 第二層: (64 + 1) × 32 = 2,080
- 第三層: (32 + 1) × 1 = 33

#### 6.5.2 plot_model() - 圖形化視覺化

使用`plot_model()`將模型架構繪製成圖片,更直觀地理解網路結構。

**基本使用**:
```python
from tensorflow.keras.utils import plot_model

# 繪製模型架構並保存為圖片
plot_model(
    model, 
    to_file='model_architecture.png',  # 輸出檔案名稱
    show_shapes=True,                  # 顯示每層的輸出形狀
    show_layer_names=True              # 顯示層名稱
)
```

**完整參數說明**:
```python
plot_model(
    model,
    to_file='model.png',               # 圖片保存路徑
    show_shapes=True,                  # 是否顯示輸出形狀 (建議True)
    show_dtype=False,                  # 是否顯示數據類型
    show_layer_names=True,             # 是否顯示層名稱 (建議True)
    rankdir='TB',                      # 排列方向: 'TB'(上到下)或'LR'(左到右)
    expand_nested=False,               # 是否展開嵌套模型
    dpi=96,                            # 圖片解析度
    show_layer_activations=False       # 是否顯示激活函數(TF 2.9+)
)
```

**關鍵參數詳解**:

1. **show_shapes** (建議開啟):
   - `True`: 顯示每層的輸出形狀,幫助理解數據流
   - `False`: 只顯示層名稱

2. **rankdir** (排列方向):
   - `'TB'` (Top to Bottom): 從上到下,適合深層網路
   - `'LR'` (Left to Right): 從左到右,適合較寬的網路

3. **dpi** (解析度):
   - 預設96,可調高到150-300以獲得更清晰的圖片
   - 數值越高檔案越大

**使用範例**:
```python
# 範例1: 基本視覺化
plot_model(model, to_file='model_basic.png')

# 範例2: 詳細視覺化 (推薦)
plot_model(
    model,
    to_file='model_detailed.png',
    show_shapes=True,
    show_layer_names=True,
    rankdir='TB',
    dpi=150
)

# 範例3: 橫向排列
plot_model(
    model,
    to_file='model_horizontal.png',
    show_shapes=True,
    rankdir='LR'
)
```

**在Jupyter Notebook中直接顯示**:
```python
from IPython.display import Image, display
from tensorflow.keras.utils import plot_model

# 繪製並顯示模型
plot_model(model, to_file='model.png', show_shapes=True)
display(Image('model.png'))
```

**注意事項**:
1. **需要安裝graphviz**:
   ```bash
   # 安裝Python套件
   pip install pydot graphviz
   
   # 安裝系統套件 (Windows)
   # 下載並安裝: https://graphviz.org/download/
   # 並將bin目錄加入系統PATH
   
   # 安裝系統套件 (Linux)
   sudo apt-get install graphviz
   
   # 安裝系統套件 (Mac)
   brew install graphviz
   ```

2. **如果無法安裝graphviz**:
   - 使用`model.summary()`作為替代
   - 或使用TensorBoard的模型圖功能

**比較: summary() vs plot_model()**

| 特性 | model.summary() | plot_model() |
|------|----------------|--------------|
| 輸出形式 | 文字 | 圖片 |
| 視覺化 | 表格形式 | 流程圖 |
| 安裝要求 | 無 | 需要graphviz |
| 適用場景 | 快速查看參數 | 理解架構 |
| 簡報展示 | 較不適合 | 適合 |


---

## 7. 模型訓練 (Model Training)

### 7.1 model.fit() 方法

```python
history = model.fit(
    x=X_train,                          # 訓練特徵
    y=y_train,                          # 訓練標籤
    batch_size=32,                      # 批次大小
    epochs=100,                         # 訓練輪數
    verbose=1,                          # 顯示模式
    validation_split=0.2,               # 驗證集分割比例
    # validation_data=(X_val, y_val),  # 或直接提供驗證集
    callbacks=[callback1, callback2],   # 回調函數列表
    shuffle=True                        # 是否每輪打亂數據
)
```

### 7.2 重要參數說明

#### 7.2.1 batch_size (批次大小)

**定義**: 每次梯度更新使用的樣本數量

**選擇建議**:
- 小batch (8-32): 訓練穩定但較慢，泛化能力可能較好
- 中batch (32-128): **推薦範圍**
- 大batch (128-256): 訓練快但可能泛化較差

**記憶體限制**: 較大batch需要更多GPU記憶體

#### 7.2.2 epochs (訓練輪數)

**發音**: /'epɒks/ (eh-poks),不是"ee-pocks"  
**定義**: 完整遍歷整個訓練集的次數

**選擇建議**:
- 設定較大值(如100-500)
- 搭配EarlyStopping自動停止

### 🔑 Batch Size, Iteration, Epoch 關係詳解

初學者常常混淆這三個概念,以下用實例說明它們的關係。

#### 基本概念

假設我們有:
- **訓練數據總量**: 1000筆
- **batch_size**: 32
- **epochs**: 10

#### 計算關係

**1. Iteration (迭代)**:
- **定義**: 處理一個batch並進行一次參數更新
- **計算**: 每個epoch的iterations = 訓練數據總量 ÷ batch_size
- **本例**: 1000 ÷ 32 = **31.25 → 32 iterations** (向上取整)

**2. Epoch (輪)**:
- **定義**: 完整遍歷整個訓練集一次
- **本例**: 設定10 epochs

**3. 總更新次數**:
- **計算**: Total updates = iterations × epochs
- **本例**: 32 × 10 = **320次參數更新**

#### 完整訓練流程圖

```
訓練數據: 1000筆
batch_size: 32
epochs: 10

Epoch 1:
  ├─ Iteration 1: 處理樣本 1-32    → 更新權重 (第1次)
  ├─ Iteration 2: 處理樣本 33-64   → 更新權重 (第2次)
  ├─ Iteration 3: 處理樣本 65-96   → 更新權重 (第3次)
  │  ...
  └─ Iteration 32: 處理樣本 993-1000 → 更新權重 (第32次)

Epoch 2:
  ├─ Iteration 1: 處理樣本 1-32    → 更新權重 (第33次)
  ├─ Iteration 2: 處理樣本 33-64   → 更新權重 (第34次)
  │  ...
  └─ Iteration 32: 處理樣本 993-1000 → 更新權重 (第64次)

...

Epoch 10:
  └─ Iteration 32: 處理樣本 993-1000 → 更新權重 (第320次)
```

#### 數學公式

$$
\text{Iterations per Epoch} = \left\lceil \frac{\text{Training Samples}}{\text{batch\_size}} \right\rceil
$$

$$
\text{Total Updates} = \text{Iterations per Epoch} \times \text{epochs}
$$

#### 不同batch_size的影響

| batch_size | Iterations/Epoch | 總更新次數 (10 epochs) | 特性 |
|-----------|------------------|---------------------|------|
| 8 | 125 | 1250 | 更新頻繁,梯度噪音大,泛化好 |
| 32 | 32 | 320 | **平衡推薦** |
| 128 | 8 | 80 | 更新少,訓練快,可能欠擬合 |
| 1000 (全batch) | 1 | 10 | 每epoch只更新一次(不推薦) |

#### 實用建議

**選擇batch_size的考量**:
1. **記憶體限制**: GPU記憶體不足時降低batch_size
2. **數據集大小**:
   - 小數據集 (<1000): batch_size=16-32
   - 中數據集 (1000-10000): batch_size=32-64
   - 大數據集 (>10000): batch_size=64-128
3. **訓練穩定性**: batch太小會導致梯度估計不準確

**監控訓練進度**:
```python
# Keras會自動顯示iteration進度
Epoch 1/10
32/32 [==============================] - 2s 50ms/step - loss: 0.5430
```
- `32/32` 表示完成了32個iterations中的32個
- 每個iteration處理32筆數據 (假設batch_size=32)

#### 常見錯誤觀念

❌ **誤解1**: "epochs越多越好"  
✅ **正確**: 過多epochs會過擬合,應搭配EarlyStopping

❌ **誤解2**: "batch_size越大訓練越好"  
✅ **正確**: 適中的batch_size平衡訓練速度與泛化能力

❌ **誤解3**: "iteration和epoch是同一個東西"  
✅ **正確**: 1 epoch = 多個 iterations


#### 7.2.3 validation_split vs validation_data

**validation_split**:
```python
# 從訓練數據末尾分割20%作為驗證集
model.fit(X_train, y_train, validation_split=0.2)
```

**validation_data**:
```python
# 直接提供驗證集
model.fit(X_train, y_train, validation_data=(X_val, y_val))
```

**建議**: 使用`validation_data`可更好控制數據分割

#### 7.2.4 verbose (顯示模式)

- `0`: 不顯示訓練過程
- `1`: 顯示進度條 (預設)
- `2`: 每輪顯示一行

### 7.3 Callbacks (回調函數)

Callbacks在訓練過程中的特定時間點執行特定操作。

#### 7.3.1 EarlyStopping (早停)

**功能**: 當驗證指標不再改善時自動停止訓練

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',      # 監控的指標
    patience=10,             # 容忍多少輪沒有改善
    restore_best_weights=True,  # 恢復最佳權重
    verbose=1,
    mode='min'               # 'min'表示指標越小越好, 'max'表示越大越好
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=500,
    callbacks=[early_stopping]
)
```

**優點**:
- 防止過擬合
- 節省訓練時間
- 自動找到最佳訓練輪數

#### 7.3.2 ModelCheckpoint (模型檢查點)

**功能**: 在訓練過程中自動保存模型

```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    filepath='best_model.keras',  # 保存路徑
    monitor='val_loss',           # 監控指標
    save_best_only=True,          # 只保存最佳模型
    save_weights_only=False,      # False保存完整模型, True只保存權重
    mode='min',
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=[checkpoint]
)
```

#### 7.3.3 ReduceLROnPlateau (動態調整學習率)

**功能**: 當訓練停滯時降低學習率

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,              # 學習率縮減倍數
    patience=5,              # 容忍輪數
    min_lr=1e-7,             # 最小學習率
    verbose=1
)
```

#### 7.3.4 TensorBoard (訓練視覺化)

**功能**: 使用TensorBoard記錄訓練過程

```python
from tensorflow.keras.callbacks import TensorBoard
import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,        # 記錄權重分布的頻率
    write_graph=True,        # 記錄模型圖
    update_freq='epoch'      # 更新頻率
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=[tensorboard_callback]
)
```

**啟動TensorBoard**:
```bash
tensorboard --logdir=logs/fit
```

在瀏覽器開啟 `http://localhost:6006/`

#### 7.3.5 組合使用多個Callbacks

```python
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_loss',
        save_best_only=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7
    ),
    TensorBoard(log_dir='logs')
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=500,
    batch_size=32,
    callbacks=callbacks
)
```

### 7.4 History物件

`model.fit()`返回一個`History`物件，記錄了訓練過程中的指標。

```python
# 查看可用的指標
print(history.history.keys())
# 輸出: dict_keys(['loss', 'mae', 'val_loss', 'val_mae'])

# 存取訓練損失
train_loss = history.history['loss']

# 存取驗證損失
val_loss = history.history['val_loss']
```

---

## 8. 訓練過程視覺化

### 8.1 繪製損失曲線

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# 損失曲線
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid(True)

# 指標曲線 (例如MAE)
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Model MAE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

### 8.2 判斷過擬合與欠擬合

**過擬合(Overfitting)**:
- 訓練損失持續下降
- 驗證損失開始上升
- 訓練與驗證損失差距大

**欠擬合(Underfitting)**:
- 訓練與驗證損失都很高
- 兩者差距小但都無法下降

**良好擬合**:
- 訓練與驗證損失都下降
- 兩者差距小且趨於穩定

### 8.3 使用TensorBoard進階監控

TensorBoard提供豐富的視覺化功能:

**1. 損失與指標曲線**:
- 實時監控訓練與驗證指標
- 支援平滑處理

**2. 模型架構圖**:
- 視覺化網路結構
- 查看張量維度與連接關係

**3. 權重與梯度分布**:
- 監控參數分布變化
- 檢測梯度消失/爆炸

**4. 超參數調整**:
- 比較不同超參數設定的效果

**設定範例**:
```python
# 在不同超參數下訓練
for learning_rate in [1e-2, 1e-3, 1e-4]:
    for batch_size in [16, 32, 64]:
        log_dir = f"logs/lr{learning_rate}_bs{batch_size}"
        
        model = create_model()
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=batch_size,
            callbacks=[TensorBoard(log_dir=log_dir)],
            verbose=0
        )
```

---

## 9. 模型評估 (Model Evaluation)

### 9.1 model.evaluate() 方法

```python
# 在測試集上評估
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

print(f'Test Loss: {test_loss:.4f}')
print(f'Test MAE: {test_mae:.4f}')
```

**返回值**:
- 第一個值: 損失函數值
- 後續值: metrics中指定的指標

### 9.2 詳細評估指標計算

```python
# 進行預測
y_pred = model.predict(X_test)

# 計算各種指標
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'MAE:  {mae:.4f}')
print(f'MSE:  {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R²:   {r2:.4f}')
```

---

## 10. 模型預測 (Model Prediction)

### 10.1 model.predict() 方法

```python
# 對測試集進行預測
predictions = model.predict(X_test)

# 對單一樣本預測
single_sample = X_test[0:1]  # 保持2D形狀
prediction = model.predict(single_sample)
print(f'Prediction: {prediction[0][0]:.4f}')
print(f'Actual: {y_test[0]:.4f}')
```

### 10.2 預測結果視覺化

```python
# 回歸問題: 真實值 vs 預測值散點圖
plt.figure(figsize=(8, 8))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.legend()
plt.grid(True)
plt.show()

# 殘差圖
residuals = y_test - predictions
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(predictions, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=30, edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## 11. 模型保存與載入

### 11.1 保存完整模型

Keras提供兩種主要的模型保存格式:**Keras格式 (.keras)** 和 **HDF5格式 (.h5)**。

#### 11.1.1 Keras格式 (推薦,TensorFlow 2.x預設)

**檔案格式**: `.keras` (單一檔案,實際上是zip壓縮檔)

**保存方式**:
```python
# 保存模型
model.save('my_model.keras')

# 或不指定副檔名(會自動使用.keras)
model.save('my_model')

# 載入模型
from tensorflow.keras.models import load_model
loaded_model = load_model('my_model.keras')

# 驗證載入的模型
predictions = loaded_model.predict(X_test)
```

**保存內容**:
- ✅ 模型架構 (層的配置與連接)
- ✅ 模型權重 (所有層的參數)
- ✅ 優化器狀態 (optimizer的內部變數)
- ✅ 編譯配置 (loss, metrics, optimizer設定)
- ✅ 訓練配置 (如果調用過`model.fit()`)

**優點**:
- **TensorFlow 2.x官方推薦格式**
- 更好的跨平台兼容性
- 支援自訂對象的序列化
- 可直接用於TensorFlow Serving部署
- 檔案結構清晰(zip格式,可解壓查看)
- 支援大型模型(>2GB)

**缺點**:
- 不向後兼容TensorFlow 1.x
- 檔案稍大於HDF5格式

#### 11.1.2 HDF5格式 (舊版,仍支援但不推薦新專案使用)

**檔案格式**: `.h5` 或 `.hdf5` (HDF5 binary格式)

**保存方式**:
```python
# 保存為HDF5格式
model.save('my_model.h5')

# 載入HDF5模型
from tensorflow.keras.models import load_model
loaded_model = load_model('my_model.h5')
```

**保存內容**:
- ✅ 模型架構
- ✅ 模型權重
- ✅ 優化器狀態
- ✅ 編譯配置

**優點**:
- 向後兼容TensorFlow 1.x
- 檔案較小
- 廣泛支援,許多工具可讀取HDF5格式

**缺點**:
- **官方已不推薦用於新專案**
- 對自訂對象支援較差
- 大型模型(>2GB)可能有問題
- Windows上長檔名可能有問題

### h5 vs keras 格式詳細比較

| 特性 | Keras格式 (.keras) | HDF5格式 (.h5) |
|------|-------------------|----------------|
| **官方推薦** | ✅ 是 (TF 2.x) | ❌ 否 (舊格式) |
| **檔案類型** | ZIP壓縮檔 | HDF5 binary |
| **單一檔案** | ✅ 是 | ✅ 是 |
| **保存內容** | 架構+權重+優化器+配置 | 架構+權重+優化器+配置 |
| **TF 1.x兼容** | ❌ 否 | ✅ 是 |
| **TF 2.x兼容** | ✅ 是 | ✅ 是 |
| **自訂層/對象** | ✅ 完整支援 | ⚠️ 有限支援 |
| **大型模型(>2GB)** | ✅ 支援 | ⚠️ 可能有問題 |
| **TF Serving** | ✅ 原生支援 | ⚠️ 需轉換 |
| **檔案大小** | 稍大 | 稍小 |
| **跨平台** | ✅ 優秀 | ⚠️ Windows長檔名問題 |
| **檔案結構** | 可解壓查看 | 需專用工具 |

### 選擇建議

**使用 Keras格式 (.keras) 當**:
- ✅ 新專案 (強烈推薦)
- ✅ 使用TensorFlow 2.x
- ✅ 有自訂層或損失函數
- ✅ 需要部署到TensorFlow Serving
- ✅ 模型大於2GB

**使用 HDF5格式 (.h5) 當**:
- ⚠️ 需要與TensorFlow 1.x兼容
- ⚠️ 維護舊專案
- ⚠️ 檔案大小極度敏感

**最佳實踐**:
```python
# 推薦: 使用.keras格式
model.save('my_best_model.keras')

# 如果需要兼容性,可同時保存兩種格式
model.save('model_keras_format.keras')  # 主要格式
model.save('model_h5_format.h5')        # 備用格式
```


### 11.2 僅保存權重

```python
# 保存權重
model.save_weights('model_weights.h5')

# 載入權重(需先建立相同架構的模型)
new_model = create_model()  # 建立相同架構
new_model.load_weights('model_weights.h5')
```

### 11.3 保存模型架構

```python
# 保存為JSON
json_config = model.to_json()
with open('model_architecture.json', 'w') as json_file:
    json_file.write(json_config)

# 從JSON載入
from tensorflow.keras.models import model_from_json
with open('model_architecture.json', 'r') as json_file:
    json_config = json_file.read()
new_model = model_from_json(json_config)
```

### 11.4 保存訓練歷史

訓練歷史(`history.history`)是一個Python字典,記錄了訓練過程中的所有指標。可以使用**pickle**或**joblib**來保存。

#### 11.4.1 使用Pickle保存 (Python內建)

**Pickle** 是Python標準庫的序列化工具。

```python
import pickle

# 保存history
with open('training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

# 載入history
with open('training_history.pkl', 'rb') as file:
    history_dict = pickle.load(file)

# 查看載入的歷史
print(history_dict.keys())  # dict_keys(['loss', 'val_loss', 'mae', 'val_mae'])
```

**優點**:
- Python內建,無需額外安裝
- 適合小型對象
- 廣泛支援

**缺點**:
- 對大型numpy數組效率較低
- 不支援壓縮
- 跨Python版本可能有兼容性問題

#### 11.4.2 使用Joblib保存 (推薦用於numpy數組)

**Joblib** 是scikit-learn推薦的持久化工具,對numpy數組優化更好。

**安裝**:
```bash
pip install joblib
```

**使用方式**:
```python
import joblib

# 保存history
joblib.dump(history.history, 'training_history.joblib')

# 載入history
history_dict = joblib.load('training_history.joblib')

# 使用壓縮(推薦,可大幅減小檔案大小)
joblib.dump(history.history, 'training_history_compressed.joblib', compress=3)
# compress參數: 0-9, 數字越大壓縮率越高但速度越慢,推薦3-5
```

**優點**:
- **對numpy數組優化,速度快**
- **支援壓縮,節省空間**
- 記憶體效率高
- 更好的跨Python版本兼容性

**缺點**:
- 需要額外安裝
- 對非numpy對象優勢不明顯

### Pickle vs Joblib 詳細比較

| 特性 | Pickle | Joblib |
|------|--------|--------|
| **安裝** | ✅ Python內建 | ⚠️ 需安裝 (`pip install joblib`) |
| **numpy數組速度** | ⚠️ 較慢 | ✅ 快 (優化過) |
| **檔案大小** | ⚠️ 較大 | ✅ 支援壓縮,可大幅縮小 |
| **記憶體效率** | 一般 | ✅ 高 (大型數組) |
| **通用對象** | ✅ 支援所有Python對象 | ✅ 支援所有Python對象 |
| **跨版本兼容** | ⚠️ 可能有問題 | ✅ 較好 |
| **使用場景** | 小型Python對象 | numpy數組、大型數據 |
| **sklearn推薦** | - | ✅ 是 |

### 實際比較範例

```python
import pickle
import joblib
import numpy as np
import time
import os

# 模擬一個較大的history (典型DNN訓練100 epochs)
history_data = {
    'loss': np.random.rand(100).tolist(),
    'val_loss': np.random.rand(100).tolist(),
    'mae': np.random.rand(100).tolist(),
    'val_mae': np.random.rand(100).tolist()
}

# Pickle保存
start = time.time()
with open('history_pickle.pkl', 'wb') as f:
    pickle.dump(history_data, f)
pickle_time = time.time() - start
pickle_size = os.path.getsize('history_pickle.pkl')

# Joblib保存 (無壓縮)
start = time.time()
joblib.dump(history_data, 'history_joblib.joblib')
joblib_time = time.time() - start
joblib_size = os.path.getsize('history_joblib.joblib')

# Joblib保存 (壓縮)
start = time.time()
joblib.dump(history_data, 'history_joblib_compressed.joblib', compress=3)
joblib_compressed_time = time.time() - start
joblib_compressed_size = os.path.getsize('history_joblib_compressed.joblib')

print(f"Pickle      - 時間: {pickle_time:.4f}s, 大小: {pickle_size} bytes")
print(f"Joblib      - 時間: {joblib_time:.4f}s, 大小: {joblib_size} bytes")
print(f"Joblib壓縮  - 時間: {joblib_compressed_time:.4f}s, 大小: {joblib_compressed_size} bytes")
```

### 選擇建議

**使用 Pickle 當**:
- history數據很小 (<100 epochs)
- 不想安裝額外套件
- 簡單快速的原型開發

**使用 Joblib 當**:
- ✅ history數據較大 (>100 epochs)
- ✅ 訓練時間很長,數據很多
- ✅ 需要頻繁保存/載入
- ✅ 磁碟空間有限 (使用壓縮)
- ✅ 已安裝scikit-learn (joblib會自動安裝)

### 最佳實踐

```python
import joblib

# 推薦: 使用joblib with適度壓縮
joblib.dump(
    history.history, 
    'training_history.joblib',
    compress=3  # 平衡壓縮率與速度
)

# 載入
history_dict = joblib.load('training_history.joblib')

# 視覺化載入的歷史
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_dict['loss'], label='Training Loss')
plt.plot(history_dict['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss History')

plt.subplot(1, 2, 2)
plt.plot(history_dict['mae'], label='Training MAE')
plt.plot(history_dict['val_mae'], label='Validation MAE')
plt.legend()
plt.title('MAE History')

plt.show()
```

### 完整範例:同時保存模型和歷史

```python
import joblib
from datetime import datetime

# 訓練後保存
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. 保存模型 (使用.keras格式)
model.save(f'model_{timestamp}.keras')

# 2. 保存訓練歷史 (使用joblib壓縮)
joblib.dump(
    history.history,
    f'history_{timestamp}.joblib',
    compress=3
)

print(f"模型和歷史已保存,時間戳: {timestamp}")

# 載入
model_loaded = load_model(f'model_{timestamp}.keras')
history_loaded = joblib.load(f'history_{timestamp}.joblib')
```


---

## 12. TensorFlow/Keras vs sklearn MLPRegressor/MLPClassifier

### 12.1 主要差異比較

| 特性 | TensorFlow/Keras | sklearn MLP |
|------|------------------|-------------|
| **靈活性** | 極高，可自訂各種層與架構 | 有限，僅基本MLP |
| **模型規模** | 支援大型深度網路 | 適合中小型網路 |
| **GPU支援** | 原生支援 | 不支援 |
| **訓練控制** | 精細控制(callbacks, 自訂訓練循環) | 基本控制 |
| **部署** | 支援多種部署方案 | 有限 |
| **學習曲線** | 較陡峭 | 較平緩 |
| **API風格** | Keras API | Scikit-learn API |
| **適用場景** | 大規模、複雜深度學習 | 快速原型、小型問題 |

### 12.2 sklearn MLP範例

```python
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 回歸問題
mlp_reg = MLPRegressor(
    hidden_layer_sizes=(64, 32),   # 隱藏層結構
    activation='relu',              # 激活函數
    solver='adam',                  # 優化器
    alpha=0.0001,                   # L2正則化參數
    batch_size='auto',
    learning_rate='constant',
    learning_rate_init=0.001,
    max_iter=500,                   # 最大迭代次數
    random_state=42,
    early_stopping=True,            # 早停
    validation_fraction=0.1,
    n_iter_no_change=10
)

# 訓練
mlp_reg.fit(X_train, y_train)

# 預測
y_pred = mlp_reg.predict(X_test)

# 評估
score = mlp_reg.score(X_test, y_test)  # R² score
```

### 12.3 選擇建議

**使用TensorFlow/Keras當**:
- 需要複雜的網路架構
- 數據量大(>10,000樣本)
- 需要GPU加速
- 需要精細控制訓練過程
- 計畫部署到生產環境
- 進行深度學習研究

**使用sklearn MLP當**:
- 快速原型驗證
- 數據量小到中等
- 需要與其他sklearn工具整合
- 不需要GPU
- 偏好簡潔的sklearn API

---

## 13. 完整工作流程範例

### 13.1 標準DNN回歸流程

```python
# 1. 導入必要套件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 2. 準備數據 (此處使用模擬數據)
np.random.seed(42)
X = np.random.randn(1000, 10)
y = X[:, 0]**2 + 2*X[:, 1] - X[:, 2] + np.random.randn(1000)*0.1

# 3. 分割數據
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# 4. 特徵縮放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 5. 建立模型
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(16, activation='relu'),
    Dense(1)
])

# 6. 編譯模型
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# 7. 查看模型結構
model.summary()

# 8. 設定callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# 9. 訓練模型
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# 10. 視覺化訓練過程
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Model MAE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 11. 評估模型
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'\nTest Loss: {test_loss:.4f}')
print(f'Test MAE: {test_mae:.4f}')

# 12. 進行預測
y_pred = model.predict(X_test_scaled)

# 13. 計算詳細指標
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'\n詳細評估指標:')
print(f'MAE:  {mae:.4f}')
print(f'MSE:  {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R²:   {r2:.4f}')

# 14. 視覺化預測結果
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.grid(True)
plt.show()

# 15. 保存模型
model.save('final_model.keras')

# 16. 載入並驗證模型
loaded_model = tf.keras.models.load_model('final_model.keras')
loaded_predictions = loaded_model.predict(X_test_scaled)
print(f'\n模型載入驗證 - 預測一致性: {np.allclose(y_pred, loaded_predictions)}')
```

---

## 14. 最佳實踐與建議

### 14.1 數據準備
1. **特徵縮放**: 使用`StandardScaler`或`MinMaxScaler`標準化輸入特徵
2. **數據分割**: 訓練集(70%) + 驗證集(15%) + 測試集(15%)
3. **數據增強**: 對於小數據集，考慮數據增強技術

### 14.2 模型設計
1. **層數與寬度**: 從小模型開始，逐步增加複雜度
2. **激活函數**: 隱藏層使用ReLU，輸出層根據問題選擇
3. **正則化**: 使用Dropout (0.2-0.5) 和 L2正則化防止過擬合
4. **BatchNormalization**: 加速訓練並提高穩定性

### 14.3 訓練策略
1. **學習率**: Adam優化器從0.001開始
2. **Batch Size**: 通常32-128之間
3. **Early Stopping**: 設定patience=10-20
4. **ModelCheckpoint**: 保存驗證集上最佳模型

### 14.4 調試技巧
1. **過擬合**: 增加Dropout、L2正則化、減少模型複雜度、增加數據
2. **欠擬合**: 增加模型容量、訓練更多輪、降低正則化
3. **訓練不穩定**: 降低學習率、使用BatchNormalization、檢查數據縮放
4. **梯度消失**: 使用ReLU、適當的權重初始化、BatchNormalization

### 14.5 超參數조정
建議調整順序:
1. 學習率 (最重要)
2. 網路架構 (層數、神經元數)
3. Batch size
4. 正則化參數 (dropout rate, L2係數)
5. 優化器選擇

---

## 16. 課堂作業

### 作業目標

透過實作完整的DNN建模流程,深入理解模型參數對訓練效果的影響,並學會如何系統性地調整超參數以獲得最佳模型性能。

---

### 📋 作業一:完整DNN建模流程實作 (60分)

#### 任務描述

使用提供的化工製程數據集,建立一個DNN回歸模型來預測產品品質指標。

#### 數據集說明

**化工反應器溫度控制數據集**
- **特徵** (8個):
  - 進料流量 (L/min)
  - 反應器溫度 (°C)
  - 反應器壓力 (bar)
  - 催化劑濃度 (%)
  - 攪拌速度 (rpm)
  - 冷卻水溫度 (°C)
  - 原料純度 (%)
  - 停留時間 (min)
- **目標變數**: 產品轉化率 (%)
- **樣本數**: 2000筆

#### 數據生成程式碼

```python
import numpy as np
import pandas as pd

# 設定隨機種子
np.random.seed(42)

# 生成特徵數據
n_samples = 2000
data = {
    '進料流量': np.random.uniform(10, 50, n_samples),
    '反應器溫度': np.random.uniform(150, 250, n_samples),
    '反應器壓力': np.random.uniform(5, 15, n_samples),
    '催化劑濃度': np.random.uniform(0.5, 5, n_samples),
    '攪拌速度': np.random.uniform(100, 500, n_samples),
    '冷卻水溫度': np.random.uniform(15, 35, n_samples),
    '原料純度': np.random.uniform(90, 99.5, n_samples),
    '停留時間': np.random.uniform(30, 120, n_samples)
}

df = pd.DataFrame(data)

# 生成目標變數(產品轉化率) - 複雜非線性關係
conversion_rate = (
    0.3 * df['反應器溫度'] +
    0.2 * df['催化劑濃度'] * df['停留時間'] +
    -0.15 * (df['進料流量'] - 30)**2 +
    0.1 * np.log(df['攪拌速度']) * df['反應器壓力'] +
    0.05 * df['原料純度'] * df['反應器溫度'] / 100 +
    np.random.normal(0, 5, n_samples)  # 添加噪音
)/2

# 限制轉化率在合理範圍
conversion_rate = np.clip(conversion_rate, 0, 100)

df['產品轉化率'] = conversion_rate

# 保存數據
df.to_csv('reactor_data.csv', index=False, encoding='utf-8-sig')
print("數據集已生成: reactor_data.csv")
print(f"數據形狀: {df.shape}")
print(f"\n前5筆數據:\n{df.head()}")
```

#### 必須完成的步驟 (每步10分)

**1. 數據準備與探索 (10分)**
- 載入數據並檢查基本統計資訊
- 繪製目標變數分布圖
- 繪製至少3個特徵與目標變數的散點圖
- 數據分割:訓練集70%、驗證集15%、測試集15%
- **對X和Y都進行StandardScaler標準化**

**2. 模型建立 (10分)**
- 使用Sequential API建立DNN模型
- 至少包含3個隱藏層
- 使用ReLU激活函數
- 加入Dropout層(rate=0.3)
- 使用`model.summary()`查看模型結構
- **加分項**: 使用`plot_model()`繪製模型架構圖

**3. 模型編譯 (10分)**
- 使用Adam優化器(learning_rate=0.001)
- 損失函數:MSE
- 評估指標:至少包含MAE和RMSE
- 說明為何選擇這些設定

**4. 模型訓練 (10分)**
- 設定至少2個callbacks:
  - EarlyStopping (patience=20)
  - ModelCheckpoint
- epochs=200, batch_size=32
- 使用validation_data進行驗證
- **加分項**: 加入TensorBoard callback

**5. 訓練過程視覺化 (10分)**
- 繪製訓練與驗證的Loss曲線
- 繪製訓練與驗證的MAE曲線
- 分析是否有過擬合或欠擬合現象

**6. 模型評估與預測 (10分)**
- **重要**: 對預測結果進行反標準化
- 在測試集上計算MAE、RMSE、R²
- 繪製實際值vs預測值散點圖
- 繪製殘差圖
- 分析模型性能

---

### 🔬 作業二:超參數探討與比較 (40分)

#### 任務描述

系統性地探討不同超參數對模型性能的影響,並撰寫分析報告。

#### 必須完成的實驗

**實驗1: 網路深度影響 (10分)**

比較以下3種網路架構:
- **淺層網路**: 2層 [64, 32]
- **中層網路**: 3層 [128, 64, 32]
- **深層網路**: 4層 [256, 128, 64, 32]

**要求**:
- 其他參數保持一致
- 記錄每個模型的:
  - 訓練時間
  - 最佳驗證Loss
  - 測試集MAE、RMSE、R²
  - 總參數量(從model.summary()獲取)
- 製作比較表格
- **分析**: 哪種深度最適合?為什麼?

**實驗2: Dropout Rate影響 (10分)**

比較以下4種dropout rate:
- 0.0 (無dropout)
- 0.2
- 0.3
- 0.5

**要求**:
- 使用相同的網路架構
- 記錄訓練與驗證Loss的差距
- 分析過擬合程度
- **分析**: 最佳dropout rate是多少?

**實驗3: Batch Size影響 (10分)**

比較以下4種batch size:
- 16
- 32
- 64
- 128

**要求**:
- 記錄每個epoch的訓練時間
- 記錄最終測試集性能
- 計算每個epoch的iterations數量
- **分析**: batch size如何影響訓練速度和模型性能?

**實驗4: 學習率影響 (10分)**

比較以下4種學習率:
- 0.0001
- 0.001
- 0.01
- 0.1

**要求**:
- 觀察訓練曲線的收斂速度
- 記錄是否出現訓練不穩定
- **分析**: 最佳學習率是多少?學習率過大或過小會有什麼問題?

---

### 📊 實驗結果整理格式

#### 表格範例

**實驗1: 網路深度比較**

| 網路架構 | 參數量 | 訓練時間(s) | 驗證Loss | 測試MAE | 測試RMSE | 測試R² |
|---------|--------|------------|---------|---------|----------|--------|
| [64, 32] | XXX | XX | X.XX | X.XX | X.XX | X.XX |
| [128, 64, 32] | XXX | XX | X.XX | X.XX | X.XX | X.XX |
| [256, 128, 64, 32] | XXX | XX | X.XX | X.XX | X.XX | X.XX |

**分析**:
- 最佳架構: ___
- 原因: ___
- 觀察到的現象: ___

#### 視覺化要求

每個實驗至少包含:
1. 訓練曲線對比圖
2. 測試集性能柱狀圖
3. 實際值vs預測值散點圖(最佳模型)

---

### 💡 加分項目 (最多+20分)

1. **使用TensorBoard** (+5分)
   - 記錄所有實驗的訓練過程
   - 在報告中展示TensorBoard截圖

2. **BatchNormalization探討** (+5分)
   - 比較有無BatchNormalization的差異
   - 分析在已標準化數據上的效果

3. **不同激活函數比較** (+5分)
   - 比較ReLU、LeakyReLU、tanh
   - 分析各自的優缺點

4. **模型保存與載入** (+5分)
   - 保存最佳模型(.keras格式)
   - 保存scalers(joblib格式)
   - 展示如何載入並使用模型進行新預測

---

### 📝 繳交格式

#### 1. Jupyter Notebook檔案

**檔名**: `學號_姓名_Unit15作業.ipynb`

**內容結構**:
```
# Unit15 DNN課堂作業
## 學生資訊
- 學號: ___
- 姓名: ___
- 繳交日期: ___

## 作業一:完整DNN建模流程
### 1. 數據準備與探索
### 2. 模型建立
### 3. 模型編譯
### 4. 模型訓練
### 5. 訓練過程視覺化
### 6. 模型評估與預測

## 作業二:超參數探討
### 實驗1: 網路深度影響
### 實驗2: Dropout Rate影響
### 實驗3: Batch Size影響
### 實驗4: 學習率影響

## 總結與心得
```

#### 2. 報告PDF檔案

**檔名**: `學號_姓名_Unit15作業報告.pdf`

**內容**:
- 所有實驗結果表格
- 所有視覺化圖表
- 詳細分析與討論
- 個人心得與學習收穫

#### 3. 模型檔案 (加分項)

- `best_model.keras`: 最佳模型
- `X_scaler.joblib`: X特徵scaler
- `y_scaler.joblib`: Y目標scaler

---

### ⏰ 繳交期限與評分標準

**繳交期限**: 課程結束後2週內

**評分標準**:

| 項目 | 配分 | 評分重點 |
|------|------|---------|
| 作業一完成度 | 60分 | 每個步驟的正確性與完整性 |
| 作業二實驗設計 | 30分 | 實驗設計合理性、結果記錄完整性 |
| 分析與討論 | 10分 | 分析深度、邏輯性、洞察力 |
| 程式碼品質 | 加分 | 註解清楚、結構良好 |
| 加分項目 | +20分 | 額外探討與創新 |

**總分**: 100分 + 加分最多20分

---

### 💭 思考題 (不計分,但建議思考)

1. 為什麼在回歸任務中,Y數據也需要標準化?
2. 如果不進行反標準化直接計算評估指標,會有什麼問題?
3. 為什麼深層網路不一定比淺層網路好?
4. Dropout如何防止過擬合?它的工作原理是什麼?
5. 學習率過大和過小分別會導致什麼問題?
6. 在工業應用中,如何選擇合適的batch size?
7. 如果測試集性能遠差於驗證集,可能是什麼原因?

---

### 📚 參考資源

- 課程講義: Unit15_DNN_MLP_Overview.md
- 課程範例: Unit15_DNN_MLP_Overview.ipynb
- TensorFlow官方文檔: https://www.tensorflow.org/
- Keras官方文檔: https://keras.io/

---

### ❓ 常見問題

**Q1: 數據集太大,訓練太慢怎麼辦?**
A: 可以先用較小的子集(如500筆)進行實驗,確認程式碼正確後再用完整數據集。

**Q2: 如何知道模型是否過擬合?**
A: 觀察訓練Loss持續下降但驗證Loss開始上升,或兩者差距過大。

**Q3: 實驗結果不理想怎麼辦?**
A: 重點在於分析過程和理解原因,不要求一定要達到很高的R²。

**Q4: 可以使用其他數據集嗎?**
A: 可以,但必須是回歸問題,並在報告中說明數據來源。

**Q5: 需要使用GPU嗎?**
A: 不需要,這個作業在CPU上即可完成。

---

## 15. 總結

本單元涵蓋了DNN/MLP的完整知識體系:

✅ **理論基礎**: 神經網路數學原理、前向傳播、反向傳播  
✅ **激活函數**: ReLU、Sigmoid、Tanh、Softmax的特性與選擇  
✅ **TensorFlow/Keras**: 完整的模型建立、訓練、評估流程  
✅ **模型優化**: Callbacks、正則化、超參數調整技巧  
✅ **實務應用**: 化工領域應用案例與最佳實踐  

### 下一步學習
- Unit15附錄案例: 實際化工問題應用
  - 燃料氣體排放預測
  - 蒸餾塔操作控制
  - 紅酒品質預測
  - 礦業浮選過程預測

---

## 參考資料

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
2. TensorFlow官方文檔: https://www.tensorflow.org/
3. Keras官方文檔: https://keras.io/
4. Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.
5. Chollet, F. (2021). Deep Learning with Python (2nd Edition).

---

**課程編號**: CHE-AI-114  
**授課教師**: 莊曜禎 助理教授  
**逢甲大學化學工程學系**
