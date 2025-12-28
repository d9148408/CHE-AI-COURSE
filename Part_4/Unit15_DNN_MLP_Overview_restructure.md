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

4. **均方對數誤差(Mean Squared Logarithmic Error, MSLE)**:
$$
L_{MSLE} = \frac{1}{n} \sum_{i=1}^{n} (\log(y_i + 1) - \log(\hat{y}_i + 1))^2
$$

**適用場景**:
- 目標值範圍很大的回歸問題
- 更關注相對誤差而非絕對誤差
- 對小值的預測誤差更敏感

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
| 回歸(大範圍) | MSLE | Linear / 不指定 |
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

> [!TIP]
> 關於 ReLU 為何成為深度學習標準配備的深入理論分析，請參考 **第3.4節**。

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

## 3. DNN架構設計理論

在學習如何使用Keras建立DNN模型之前，了解網路架構設計的理論基礎非常重要。本章將深入探討DNN架構設計的核心原則，幫助您做出明智的設計決策。

### 3.1 漏斗型架構 (Funnel Architecture)

**什麼是漏斗型架構？**

漏斗型架構（也稱為倒金字塔結構）是指神經網路的隱藏層節點數量由寬變窄，逐層遞減的設計模式。例如：

```
Input (6 features) → 256 → 128 → 64 → 32 → 16 → Output (1)
```

雖然沒有嚴格的數學定律規定「必須」每次減半，但這種設計有強而有力的理論支撐。

**理論基礎 1: 資訊瓶頸理論 (Information Bottleneck Theory)**

這是最核心的理論解釋。想像您的輸入數據包含了很多雜訊和冗餘資訊：

- **概念**: 每一層神經網路的任務是將上一層的資訊進行「過濾」與「重組」
- **作用**: 透過逐漸減少節點數量，我們強迫模型**丟棄不重要的細節，只保留對預測結果最關鍵的特徵**
- **比喻**: 就像閱讀一本書並做摘要
  - 第一層：把每一句話都讀進去（256 個節點）
  - 下一層：開始歸納段落大意（128 個節點）
  - 最後：只提煉出最核心的結論（1 個節點）

> [!NOTE]
> 如果中間層數太寬，模型可能會「死記硬背」整本書的廢話，而不是學會「理解」。

**理論基礎 2: 特徵抽象化與表示學習 (Representation Learning)**

深度神經網路的運作邏輯是分層抽象：

- **淺層（節點多）**: 負責捕捉低階特徵
  - 例如在處理房價預測時，淺層可能在看「坪數」、「房間數」、「地點」這些原始數據的交互作用
- **深層（節點少）**: 負責將低階特徵組合成高階概念
  - 隨著層數加深，數據被壓縮成更抽象的概念，例如「生活機能」、「豪華程度」
  - 這些高階概念不需要那麼多維度（節點）來描述，但訊息密度更高

**理論基礎 3: 控制參數量與防止過度擬合**

讓我們看看參數量的計算：

$$\text{參數量} = (\text{輸入節點數} \times \text{輸出節點數}) + \text{偏差項}$$

範例比較：
- `Dense(256 → 128)`: $256 \times 128 = 32,768$ 個權重
- 如果下一層不減半 `Dense(128 → 128)`: $128 \times 128 = 16,384$ 個權重
- 如果下一層變寬 `Dense(128 → 256)`: 參數量會暴增

**理論依據**：
根據**奧卡姆剃刀 (Occam's Razor)** 原則，在效能相似的情況下，越簡單的模型（參數越少）越好。

- 逐漸減少節點數可以有效控制參數量
- 減少模型「記憶」訓練數據的能力
- 強迫模型學習通用的規則，提升泛化能力（Generalization）

**理論基礎 4: 幾何金字塔規則 (Geometric Pyramid Rule)**

早在 1993 年，學者 Masters 就提出了經驗法則：
- 在前饋神經網路中，隱藏層的幾何形狀如果是金字塔型（由輸入層向輸出層逐漸收斂），通常能獲得較好的收斂效果

**為什麼是 2 的次方？(64, 128, 256)**

這主要是電腦科學界的工程習慣：

1. **記憶體配置**: 電腦記憶體管理通常以 2 的次方為單位，這樣配置對硬體運算（GPU/TPU）的矩陣乘法效率稍微友善一些
2. **超參數調整的方便性**: 當工程師在調參時，比起測試 100, 110, 120，直接測試 64, 128, 256 更能快速在大範圍內找到合適的容量

**這種結構是絕對的嗎？**

不是。雖然「漏斗型」是處理結構化數據（Tabular Data）最常見的起手式，但也有例外：

- **長方形結構 (Cylindrical)**: 例如 BERT 或 ResNet 的中間層，往往保持固定的寬度（例如從頭到尾都是 512）。這是為了在深層網路中保持梯度的流動，避免資訊過早遺失。
- **紡錘形結構 (Bottleneck / Autoencoder)**: 先寬、變窄（壓縮）、再變寬（解壓縮）。這通常用於生成模型或降維任務。

**總結**

256 → 128 → 64 是一種**「透過壓縮來提取精華」**的設計哲學：
- ✅ 理論上符合資訊瓶頸原理
- ✅ 實務上能有效節省運算資源
- ✅ 降低過度擬合的風險
- ✅ 促進特徵的階層式抽象

### 3.2 第一層節點數的決定

**核心邏輯：特徵擴張 vs. 特徵壓縮**

第一層隱藏層扮演著**「特徵交互（Feature Interaction）」**的角色。決定第一層大小時，首先要看您的目的：

**情況 1: 輸入特徵很少（< 20 個）**
- 通常需要把第一層設得比輸入層大（例如 2 倍或更高）
- **原因**: 假設您有兩個特徵 $x_1$ 和 $x_2$，模型可能需要學習它們的組合（如 $x_1 \times x_2$, $x_1^2$, $x_1+x_2$）
- 如果第一層太窄，模型就沒有足夠的空間去「展開」這些潛在的組合特徵

**情況 2: 輸入特徵很多（> 1000 個）**
- 通常會把第一層設得比輸入層小
- **原因**: 這時候需要做「特徵壓縮」或「降維」，強迫模型立刻過濾掉不重要的輸入

**常見的經驗公式 (Heuristics)**

假設 $N_i$ 是輸入層節點數（特徵數），$N_o$ 是輸出層節點數，$N_h$ 是第一層隱藏層節點數。

**公式 A: 「2/3 規則」（最經典）**

$$N_h = \frac{2}{3} N_i + N_o$$

- 解讀：隱藏層的大小介於輸入和輸出之間，但更偏向輸入層的大小
- 這是一個非常老派但穩健的建議

**公式 B: 「兩倍上限」規則**

$$N_h < 2 \times N_i$$

- 解讀：為了避免過度擬合，第一層的大小最好不要超過輸入特徵的兩倍
- 這是一個防止參數量暴增的安全界線

**公式 C: 「幾何平均」規則**

$$N_h = \sqrt{N_i \times N_o}$$

- 解讀：取輸入與輸出的幾何平均數
- 這通常會得到一個比較小的數字，適合數據量較少、怕過度擬合的情況

**公式 D: 「平均值」規則**

$$N_h = \frac{N_i + N_o}{2}$$

- 解讀：單純取平均，中規中矩的起點

**現代深度學習的實務做法**

在現代使用 Keras/PyTorch 時，我們更傾向於結合**「2 的次方」與「寬度優先」**策略：

**步驟一：對齊 2 的次方**
- 不管公式算出多少（例如算出 105），我們通常會直接選最接近的 32, 64, 128, 256, 512, 1024
- 這是為了配合 GPU 硬體加速的習慣

**步驟二：根據數據量級決定**

| 數據規模 | 建議第一層節點數 | 說明 |
|----------|------------------|------|
| 小數據集（幾千筆，10-50 個特徵） | 64 或 128 | 通常大於輸入特徵數，捕捉非線性關係 |
| 中數據集（幾萬筆，50-200 個特徵） | 256 或 512 | 平衡表達能力與訓練效率 |
| 大數據集（百萬筆資料） | 1024 或更深 | 充分利用大數據的資訊 |

**步驟三：監控「過度擬合」**

第一層設得越大，模型能學到的東西越複雜，但也越容易把雜訊當成訊號：

- 如果 Training Loss 很低，但 Validation Loss 很高 → 第一層設太大了，請減少節點數
- 如果 Training Loss 和 Validation Loss 都降不下去 → 模型太小（Underfitting），請增加第一層節點數

**實例說明**

假設您有 20 個輸入特徵，第一層設為 256 nodes：
- 比例：$256 / 20 \approx 12.8$ 倍
- 這在深度學習中是很常見的配置
- 雖然遠大於「2倍規則」，但因為後面接了漏斗狀的收縮（256 → 128 → 64），且現代模型通常會搭配 Dropout 或 Batch Normalization 來防止過度擬合
- 因此**「第一層寬一點」**通常是被允許且鼓勵的，因為它能確保不錯過任何微小的特徵關聯

**總結建議**

如果您正在從頭建立一個模型，建議的起手式如下：

1. **起點**: 將第一層設為 128 或 256（除非您的特徵數大於這個數字，那就設為大於特徵數的下一個 2 的次方）
2. **觀察**: 跑個 20 epochs，看 Loss 下降的情況
3. **調整**:
   - 如果學不動（Underfitting）：加倍第一層 (e.g., 128 → 256)
   - 如果學太快但驗證差（Overfitting）：減半第一層，或加入 Dropout 層

### 3.3 層數深度選擇

如果第一層是 256，且採用每次減半（漏斗型）的策略，理論上您可以有兩種極端的選擇：

- **極致減半（深層）**: 256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1 (共 8 層)
- **快速收斂（淺層）**: 256 → 32 → 1 (共 3 層)

**決定層數的關鍵**

決定層數的關鍵不在於「一定要除以 2 除到盡頭」，而在於**資料的複雜度**：

**一般結構化數據（表格類）**：
- 通常 **3 到 5 層隱藏層**是「甜蜜點（Sweet Spot）」
- **太淺（< 3 層）**: 模型可能無法學習到特徵之間的高階非線性關係
  - 例如：A 欄位只有在 B 欄位大於 10 且 C 欄位為負時才重要
- **太深（> 6-8 層）**: 對於非圖像/語音的數據，層數過深容易導致：
  - 梯度消失（Vanishing Gradient）
  - 訓練困難
  - 邊際效益遞減（加了層數但準確率沒提升）

> [!IMPORTANT]
> 對於結構化數據，停在 5 層（例如 256 → 128 → 64 → 32 → 16）是一個非常標準且穩健的深度設計，既有足夠的非線性轉換能力，又不會深到難以訓練。

**為何倒數第二層停在 16？為什麼不是 32 或 8？**

倒數第二層（Penultimate Layer）的角色非常特殊，它是模型在做出最終決策前，對數據的**最終理解（Latent Representation）**。

最後一層 `Dense(1)` 其實只做了一件簡單的事：將倒數第二層的節點進行「線性加權總和」。

**為什麼不繼續減半到 8、4 或 2？**

雖然這樣更「規律」，但有潛在風險：

1. **資訊瓶頸過窄（Information Bottleneck）**
   - 停在 16：最後決策者聽取 16 位不同領域專家的意見來做決定。這 16 個特徵代表了資料的 16 個不同面向
   - 停在 2：強迫模型把所有複雜資訊濃縮成 2 個指標。如果問題很複雜，這 2 個指標可能無法涵蓋所有變數，導致模型產生「以偏概全」的誤判

2. **訓練效率**
   - 為了從 16 降到 8 再降到 4，您必須多加兩層網路
   - 這增加了計算成本和訓練難度
   - 但對於一個最後只是要做「加權總和」的動作來說，從 16 個數加總和從 4 個數加總，對模型來說負擔差異不大

**為什麼不停在 32？**

這也是可以的！事實上，很多模型確實停在 32 或 64。但選擇 16 的理由通常是：

1. **強迫特徵純化**: 從 32 壓縮到 16，是再一次的過濾雜訊。如果您的數據雜訊很多，進一步壓縮到 16 可以讓最終進入輸出層的訊號更乾淨
2. **避免過度擬合**: 倒數第二層的參數量對最後一層的權重有直接影響
   - `32 → 1`: 最後一層有 33 個參數 (32 weights + 1 bias)
   - `16 → 1`: 最後一層有 17 個參數
   - 參數越少，模型越不容易死記硬背

**經驗法則建議**

| 模型表現 | 建議調整 |
|----------|----------|
| **欠擬合（Underfitting）** | 讓倒數第二層寬一點，例如停在 32 或 64 直接輸出 |
| **過度擬合（Overfitting）** | 壓縮到 16，甚至 8，或者在 16 這一層後面加一個 Dropout 層 |

**總結**

這個模型設計停在 16 的邏輯是：
> 「我已經把 256 維的原始數據，層層過濾濃縮成了 16 個最精華的特徵指標。這 16 個指標已經足夠用來計算最終分數（輸出 1），再壓縮下去可能會遺失關鍵資訊；而保留 32 個又可能嫌太多雜訊。」

### 3.4 ReLU深入理論：為何DNN隱藏層全都使用ReLU？

> [!NOTE]
> ReLU 的基本定義和公式請參考 **第2.1.1節**。本節深入探討 ReLU 的理論優勢。

在現代 DNN 的**隱藏層（Hidden Layers）**中，ReLU 幾乎就是「標準配備」。這主要歸功於它解決了傳統激活函數（如 Sigmoid）的致命傷，同時帶來了極高的運算效率。

**理由 1: 解決「梯度消失」問題 (Vanishing Gradient Problem) —— 最重要的理由**

這是 ReLU 能統治深度學習的主因。

**過去的問題 (Sigmoid)**：
- 早期的神經網路喜歡用 Sigmoid 函數（S型曲線）
- 但是 Sigmoid 有一個大缺點：當輸入值很大或很小時，它的曲線會變得非常平緩（飽和區）
- 在這些平緩區域，導數（梯度）幾乎為 0
- 當模型很深時（例如您堆了 5 層），在反向傳播（Backpropagation）更新參數時，這些微小的梯度連乘起來（例如 $0.1 \times 0.1 \times 0.1 \dots$），傳到最前面幾層時就變成了 0
- **結果**：深層網路的前面幾層根本學不到東西，模型訓練不動

**ReLU 的解法**：

ReLU 的公式非常簡單：

$$f(x) = \max(0, x)$$

- 只要 $x > 0$，它的導數（梯度）永遠是 1
- 不管層數堆多深，$1 \times 1 \times 1 \dots = 1$，梯度可以毫無耗損地傳回第一層
- 這讓訓練「深」度神經網路成為可能

**理由 2: 極致的運算速度 (Computational Efficiency)**

神經網路訓練需要進行數百萬、數千萬次的運算：

- **Sigmoid / Tanh**: 需要計算指數函數 ($e^x$)、除法等複雜數學運算，這對電腦的 CPU/GPU 來說消耗較大
  
  $$\text{Sigmoid}(x) = \frac{1}{1 + e^{-x}}$$

- **ReLU**: 只需要做一個簡單的邏輯判斷：「如果小於 0 就變 0，否則保持原樣」
  
  $$\text{ReLU}(x) = \max(0, x)$$

這種簡單的運算讓訓練速度大幅提升，節省了大量的運算資源。

**理由 3: 稀疏激發 (Sparse Activation)**

這是一個比較抽象但很有用的特性：

- **概念**: 在任何時刻，神經網路中並不是所有的神經元都需要「運作」
- **ReLU 的作用**: 因為 ReLU 會把所有負值都變成 0，這意味著在運作時，會有部分的神經元輸出為 0（不被激活）
- **好處**: 這讓網路具有**「稀疏性（Sparsity）」**。這有點像人腦的運作方式——當您在看「紅色」時，負責處理「綠色」的神經元應該休息
- 稀疏性讓模型更有效率，且在一定程度上減少了參數之間的糾纏，能提取出更具區別性的特徵

**ReLU 是完美的嗎？（例外情況）**

雖然 ReLU 是預設首選，但它有一個著名的缺點：**「死亡 ReLU 問題 (Dying ReLU Problem)」**。

- **現象**: 如果學習率（Learning Rate）設太大，或者運氣不好，某些神經元的權重被更新成很大的負數，導致不管輸入什麼數據，進入 ReLU 的值永遠是負的
- **結果**: 該神經元永遠輸出 0，梯度也永遠是 0，這個神經元就像「死掉」了一樣，從此不再更新，失去作用

**解決方案**：

如果您的模型訓練效果不佳，或者發現很多神經元壞死，工程師有時會改用：

- **Leaky ReLU**: 在負數區域不給 0，而是給一個很小的斜率（例如 0.01）
  
  $$f(x) = \begin{cases} x & \text{if } x > 0 \\ 0.01x & \text{if } x \leq 0 \end{cases}$$
  
  這樣就算進入負區，神經元還有一口氣在，還能慢慢學回來

- **GELU / Swish**: 最近在 Transformer (BERT/GPT) 架構中非常流行，它們是 ReLU 的平滑版本，表現通常比 ReLU 更好，但計算成本稍高

**總結**

您的模型使用 ReLU 是完全正確且標準的選擇：
- ✅ **簡單**（算得快）
- ✅ **有效**（解決梯度消失，讓深層網路可訓練）
- ✅ **聰明**（自動關閉不必要的特徵）

> [!TIP]
> 對於大多數 DNN 應用，隱藏層使用 ReLU 是最佳選擇。只有在遇到「死亡 ReLU」問題時，才考慮使用 Leaky ReLU 或其他變體。

---

## 4. DNN/MLP應用場景

### 4.1 適合使用DNN/MLP的情境

1. **非線性關係複雜**: 輸入與輸出之間存在高度非線性關係
2. **特徵交互作用**: 特徵之間有複雜的交互作用
3. **大量數據**: 有足夠的訓練數據支持深度模型
4. **特徵工程困難**: 難以手動設計有效特徵時，DNN可自動學習

### 4.2 化工領域應用案例

#### 4.2.1 製程參數優化
- **應用**: 預測反應器溫度、壓力、流量等操作條件對產品品質的影響
- **優勢**: 可處理多變數、非線性的製程關係

#### 4.2.2 產品品質預測
- **應用**: 根據原料成分與製程條件預測最終產品性質
- **範例**: 紅酒品質預測、聚合物性質預測

#### 4.2.3 設備故障診斷
- **應用**: 透過感測器數據預測設備異常或故障
- **優勢**: 可學習複雜的時間序列模式

#### 4.2.4 分離程序模擬
- **應用**: 蒸餾塔、萃取塔等分離設備的快速模擬
- **優勢**: 比傳統數值模擬快速，適合即時控制

#### 4.2.5 環境排放預測
- **應用**: 預測燃燒程序的污染物排放量
- **範例**: NOx、SOx、CO2排放預測

#### 4.2.6 礦業浮選過程
- **應用**: 預測礦石浮選過程的矽石濃度
- **優勢**: 可整合多種感測器數據進行即時預測

### 4.3 DNN的優勢與限制

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

## 5. TensorFlow/Keras框架介紹

### 5.1 TensorFlow與Keras簡介

**TensorFlow** 是Google開發的開源深度學習框架，提供完整的機器學習生態系統。

**Keras** 是高階神經網路API，現已整合進TensorFlow 2.x (tf.keras)，提供:
- 簡潔易用的介面
- 模組化設計
- 易於擴展
- 支援多種後端

### 5.2 環境安裝

```bash
# 安裝TensorFlow (包含Keras)
pip install tensorflow

# 或安裝特定版本
pip install tensorflow==2.15.0

# 驗證安裝
python -c "import tensorflow as tf; print(tf.__version__)"
```

### 5.3 基本導入

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
```

---

## 6. 使用Keras建立DNN模型

### 6.1 模型架構: Sequential vs Functional API

Keras提供兩種建立模型的方式:

#### 6.1.1 Sequential API (序列模型)

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

#### 6.1.2 Functional API (函數式API)

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

### 6.2 常用層(Layers)

#### 6.2.1 Dense Layer (全連接層)

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

#### 6.2.2 Dropout Layer (隨機失活層)

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

#### 6.2.3 BatchNormalization Layer (批次正規化層)

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


#### 6.2.4 Activation Layer (激活層)

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

### 6.3 權重初始化策略

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

### 6.4 正則化(Regularization)

防止過擬合的技術。

#### 6.4.1 L1/L2正則化

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

## 7. 模型編譯 (Model Compilation)

編譯模型時需要指定**優化器**、**損失函數**和**評估指標**。

### 7.1 model.compile() 方法

```python
model.compile(
    optimizer='adam',           # 優化器
    loss='mse',                 # 損失函數
    metrics=['mae', 'mse']      # 評估指標
)
```

### 7.2 優化器(Optimizers)

優化器決定如何根據梯度更新權重。

#### 7.2.1 Adam (Adaptive Moment Estimation)
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

#### 7.2.2 SGD (Stochastic Gradient Descent)

```python
from tensorflow.keras.optimizers import SGD

optimizer = SGD(
    learning_rate=0.01,
    momentum=0.9,           # 動量
    nesterov=True           # 是否使用Nesterov動量
)
```

#### 7.2.3 RMSprop

```python
from tensorflow.keras.optimizers import RMSprop

optimizer = RMSprop(
    learning_rate=0.001,
    rho=0.9
)
```

#### 7.2.4 AdaGrad, Adadelta, Nadam 等

```python
from tensorflow.keras.optimizers import AdaGrad, Adadelta, Nadam
```

**優化器選擇建議**:
- **首選**: Adam (適用大多數情況)
- **需要更好泛化**: SGD with momentum
- **RNN問題**: RMSprop 或 Adam

### 7.3 損失函數在Keras中的使用

> [!NOTE]
> 損失函數的數學原理和理論基礎請參考 **第1.5節**。本節僅說明在Keras中的使用方式。

#### 7.3.1 回歸問題

**MSE (均方誤差)**:
```python
model.compile(optimizer='adam', loss='mse')
model.compile(optimizer='adam', loss='mean_squared_error')
# 或
from tensorflow.keras.losses import MeanSquaredError
model.compile(optimizer='adam', loss=MeanSquaredError())
```

**MAE (平均絕對誤差)**:
```python
model.compile(optimizer='adam', loss='mae')
model.compile(optimizer='adam', loss='mean_absolute_error')
# 或
from tensorflow.keras.losses import MeanAbsoluteError
model.compile(optimizer='adam', loss=MeanAbsoluteError())
```

**Huber Loss** (對異常值較不敏感):
```python
from tensorflow.keras.losses import Huber
model.compile(optimizer='adam', loss=Huber(delta=1.0))
```

**MSLE (均方對數誤差)** - 適合目標值範圍很大的情況:
```python
model.compile(optimizer='adam', loss='msle')
# 或
from tensorflow.keras.losses import MeanSquaredLogarithmicError
model.compile(optimizer='adam', loss=MeanSquaredLogarithmicError())
```

#### 7.3.2 分類問題

**Binary Crossentropy** (二元分類):
```python
model.compile(optimizer='adam', loss='binary_crossentropy')
# 或
from tensorflow.keras.losses import BinaryCrossentropy
model.compile(optimizer='adam', loss=BinaryCrossentropy())
```
**配合使用**: 輸出層使用 `Sigmoid` 激活函數

**Categorical Crossentropy** (多類別，one-hot標籤):
```python
model.compile(optimizer='adam', loss='categorical_crossentropy')
# 或
from tensorflow.keras.losses import CategoricalCrossentropy
model.compile(optimizer='adam', loss=CategoricalCrossentropy())
```
**標籤格式**: `[[0, 0, 1], [1, 0, 0], [0, 1, 0]]` (one-hot)

**Sparse Categorical Crossentropy** (多類別，整數標籤):
```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
# 或
from tensorflow.keras.losses import SparseCategoricalCrossentropy
model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy())
```
**標籤格式**: `[2, 0, 1]` (整數索引)

#### 7.3.3 快速參考

| 問題類型 | Keras損失函數 | 輸出層激活函數 |
|---------|--------------|---------------|
| 回歸(一般) | `'mse'` | `'linear'` |
| 回歸(有異常值) | `'mae'` 或 `Huber()` | `'linear'` |
| 回歸(大範圍) | `'msle'` | `'linear'` |
| 二元分類 | `'binary_crossentropy'` | `'sigmoid'` |
| 多類別(one-hot) | `'categorical_crossentropy'` | `'softmax'` |
| 多類別(整數) | `'sparse_categorical_crossentropy'` | `'softmax'` |

### 7.4 評估指標(Metrics)

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


### 7.5 模型摘要與視覺化

#### 7.5.1 model.summary() - 文字摘要

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

#### 7.5.2 plot_model() - 圖形化視覺化

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

---

## 8. 模型訓練 (Model Training)

### 8.1 model.fit() 方法

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

### 8.2 重要參數說明

#### 8.2.1 batch_size (批次大小)

**定義**: 每次梯度更新使用的樣本數量

**選擇建議**:
- 小batch (8-32): 訓練穩定但較慢，泛化能力可能較好
- 中batch (32-128): **推薦範圍**
- 大batch (128-256): 訓練快但可能泛化較差

**記憶體限制**: 較大batch需要更多GPU記憶體

#### 8.2.2 epochs (訓練輪數)

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


#### 8.2.3 validation_split vs validation_data

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

#### 8.2.4 verbose (顯示模式)

- `0`: 不顯示訓練過程
- `1`: 顯示進度條 (預設)
- `2`: 每輪顯示一行

### 8.3 Callbacks (回調函數)

Callbacks在訓練過程中的特定時間點執行特定操作。

#### 8.3.1 EarlyStopping (早停)

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

#### 8.3.2 ModelCheckpoint (模型檢查點)

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

#### 8.3.3 ReduceLROnPlateau (動態調整學習率)

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

#### 8.3.4 TensorBoard (訓練視覺化)

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

#### 8.3.5 組合使用多個Callbacks

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

---

## 9. 訓練過程視覺化

### 9.1 繪製訓練曲線

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

### 9.2 解讀訓練曲線

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

---

## 10. 模型評估 (Model Evaluation)

### 10.1 model.evaluate() 方法

```python
# 在測試集上評估
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

print(f'Test Loss: {test_loss:.4f}')
print(f'Test MAE: {test_mae:.4f}')
```

**返回值**:
- 第一個值: 損失函數值
- 後續值: metrics中指定的指標

### 10.2 詳細評估指標計算

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

## 11. 模型預測 (Model Prediction)

### 11.1 model.predict() 方法

```python
# 對測試集進行預測
predictions = model.predict(X_test)

# 對單一樣本預測
single_sample = X_test[0:1]  # 保持2D形狀
prediction = model.predict(single_sample)
print(f'Prediction: {prediction[0][0]:.4f}')
print(f'Actual: {y_test[0]:.4f}')
```

### 11.2 預測結果視覺化

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

## 12. 模型保存與載入

### 12.1 保存完整模型

Keras提供兩種主要的模型保存格式:**Keras格式 (.keras)** 和 **HDF5格式 (.h5)**。

#### 12.1.1 Keras格式 (推薦,TensorFlow 2.x預設)

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

#### 12.1.2 HDF5格式 (舊版,仍支援但不推薦新專案使用)

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

### 12.2 僅保存權重

```python
# 保存權重
model.save_weights('model_weights.h5')

# 載入權重(需先建立相同架構的模型)
new_model = create_model()  # 建立相同架構
new_model.load_weights('model_weights.h5')
```

### 12.3 保存模型架構

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

---

## 13. TensorFlow/Keras vs sklearn MLPRegressor/MLPClassifier

### 13.1 主要差異比較

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

### 13.2 sklearn MLP範例

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

### 13.3 選擇建議

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

### 14.5 超參數調整
建議調整順序:
1. 學習率 (最重要)
2. 網路架構 (層數、神經元數)
3. Batch size
4. 正則化參數 (dropout rate, L2係數)
5. 優化器選擇

---

## 15. 總結

本單元涵蓋了DNN/MLP的完整知識體系:

✅ **理論基礎**: 神經網路數學原理、前向傳播、反向傳播  
✅ **激活函數**: ReLU、Sigmoid、Tanh、Softmax的特性與選擇  
✅ **架構設計**: 漏斗型架構、層數選擇、節點數決定的理論基礎  
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

## 16. 參考資料

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
2. TensorFlow官方文檔: https://www.tensorflow.org/
3. Keras官方文檔: https://keras.io/
4. Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.
5. Chollet, F. (2021). Deep Learning with Python (2nd Edition).

## 17. 相關資源說明

本單元配套提供了以下 Jupyter Notebook 檔案，供讀者進行實作演練與自我挑戰：

- **範例程式碼 (`Unit15_DNN_MLP_Overview.ipynb`)**: 為本講義的配套實作教材。其詳細示範了從環境配置、數據模擬生成、模型構建（Sequential API）、模型編譯、訓練（含 Callbacks 運用）、結果視覺化分析，到模型預測與存檔的完整端到端流程。讀者可以透過此檔案快速掌握 Keras 開發深度學習模型的實戰技巧。

- **課堂作業 (`Unit15_DNN_MLP_Homework.ipynb`)**: 本作業旨在引導學生將理論應用於化學工程領域。學生將處理一組模擬的「化工反應器溫度控制數據」，並依照步驟執行數據標準化、模型架構設計、正則化優化及多項超參數調整實驗。作業中亦包含思考題，旨在鍛鍊學習者對模型表現與訓練細節的深度洞察力。

---

**課程編號**: CHE-AI-114  
**授課教師**: 莊曜禎 助理教授  
**逢甲大學化學工程學系**


