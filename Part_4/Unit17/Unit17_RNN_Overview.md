# Unit 17: 循環神經網路(RNN)概述

## 課程目標
- 理解循環神經網路(RNN)、LSTM、GRU的基本概念與數學原理
- 學會使用TensorFlow/Keras建立、訓練、評估RNN模型
- 掌握時間序列數據處理與序列建模技巧
- 了解RNN在化工領域的應用場景

---

## 1. RNN基礎理論

### 本章學習地圖

> [!IMPORTANT]
> **本章核心問題**: 為什麼需要RNN？它如何處理序列數據？與DNN有什麼根本差異？

**學習目標**:
1. 🎯 **理解RNN的本質**: 認識RNN是專門處理序列數據的神經網路架構
2. 🔄 **掌握循環機制**: 了解RNN如何在時間步之間傳遞資訊
3. 🧠 **深入LSTM與GRU**: 理解如何解決傳統RNN的梯度消失問題
4. 🔧 **建立實作基礎**: 為使用Keras建立序列模型打下理論基礎

**為什麼化工人需要學RNN？**

在化工領域，許多重要問題都涉及**時間序列數據**：
- 反應器溫度、壓力隨時間的變化趨勢
- 批次生產過程的品質預測與監控
- 設備運轉狀態的健康診斷與故障預警
- 化工廠操作參數的動態優化控制

傳統的DNN/MLP無法有效處理這些**具有時間依賴性**的數據，因為：
- ❌ DNN將每個時間點視為獨立樣本，忽略時序關聯
- ❌ DNN無法記憶過去的資訊來預測未來
- ❌ DNN無法處理變長度的序列輸入

**RNN的核心優勢**就是引入**記憶機制**，使網路能夠：
- ✅ 記住過去的資訊（如前10分鐘的溫度變化）
- ✅ 學習時間序列的動態模式（如週期性、趨勢性）
- ✅ 進行序列到序列的映射（如多步預測）

**本章架構**:

```
什麼是RNN？與DNN的差異 (1.1-1.2)
    ↓
RNN的數學原理與前向傳播 (1.3-1.4)
    ↓
RNN的訓練：BPTT算法 (1.5)
    ↓
RNN的問題：梯度消失與爆炸 (1.6)
    ↓
解決方案：LSTM與GRU (1.7-1.8)
```

> [!TIP]
> 學習建議：先理解RNN的循環機制，再深入LSTM/GRU的門控結構。實際應用中，LSTM和GRU是更常用的選擇。

---

### 1.1 什麼是循環神經網路(RNN)?

**循環神經網路(Recurrent Neural Network, RNN)** 是一類專門處理**序列數據**的神經網路架構。與前饋神經網路(DNN)不同，RNN具有**循環連接**，使得網路具有**記憶能力**。

#### RNN與DNN的根本差異

| 特性 | DNN (前饋網路) | RNN (循環網路) |
|-----|---------------|---------------|
| **網路結構** | 單向前饋，無循環 | 具有循環連接 |
| **輸入數據** | 固定維度向量 | 變長序列 |
| **資訊流動** | 僅從輸入到輸出 | 在時間步間傳遞 |
| **記憶能力** | 無記憶 | 具有短期記憶 |
| **參數共享** | 無 | 在所有時間步共享 |
| **適用場景** | 靜態數據預測 | 時間序列、序列到序列 |

**核心概念**: RNN的"循環"(Recurrent)指的是：
- 在處理序列的每個時間步時
- 網路不僅接收當前輸入
- 還接收前一時間步的**隱藏狀態**（記憶）
- 這個隱藏狀態攜帶了過去的資訊

**生活類比**: 
- **DNN**: 像是一次性閱讀一張照片，無上下文
- **RNN**: 像是閱讀一本書，前面的情節會影響對後面內容的理解

### 1.2 RNN的應用場景

RNN及其變體(LSTM, GRU)廣泛應用於各種序列建模任務：

#### 1. 序列預測問題

**多對一 (Many-to-One)**:
- **輸入**: 一個序列
- **輸出**: 單一值
- **應用**: 
  - 時間序列分類（設備故障診斷）
  - 情感分析（評論正負面判斷）
  - 產品品質預測（基於批次操作歷程）

```
輸入: [x₁, x₂, x₃, ..., xₜ] → 輸出: y
範例: [溫度₁, 溫度₂, ..., 溫度₁₀₀] → 產品合格/不合格
```

**多對多 (Many-to-Many, 等長)**:
- **輸入**: 序列
- **輸出**: 等長序列
- **應用**:
  - 時間序列標註
  - 影片逐幀分析
  - 序列異常檢測

```
輸入: [x₁, x₂, x₃, ..., xₜ] → 輸出: [y₁, y₂, y₃, ..., yₜ]
範例: [壓力₁, 壓力₂, ...] → [正常, 正常, 異常, ...]
```

**多對多 (Many-to-Many, 不等長 - Seq2Seq)**:
- **輸入**: 序列
- **輸出**: 不同長度序列
- **應用**:
  - 機器翻譯
  - 文本摘要
  - 對話系統

#### 2. 時間序列預測

**單步預測 (One-Step Ahead)**:
```
已知: t-n, ..., t-2, t-1 → 預測: t
範例: 根據過去24小時的溫度預測下一小時溫度
```

**多步預測 (Multi-Step Ahead)**:
```
已知: t-n, ..., t-2, t-1 → 預測: t, t+1, t+2, ..., t+m
範例: 根據歷史數據預測未來一週的生產參數
```

#### 3. 化工領域特定應用

| 應用類型 | 具體任務 | 輸入序列 | 輸出 |
|---------|---------|---------|------|
| **過程監控** | 批次操作結束時間預測 | 溫度、壓力、流量時間序列 | 剩餘時間 |
| **品質預測** | 產品性質預測 | 反應過程參數軌跡 | 最終產品品質 |
| **故障診斷** | 設備異常檢測 | 感測器數據流 | 正常/異常 |
| **預測控制** | 未來狀態預測 | 歷史操作數據 | 未來多步狀態 |
| **軟感測器** | 難測變數估計 | 易測變數時間序列 | 難測變數值 |
| **剩餘壽命預測** | 設備RUL估計 | 退化指標序列 | 剩餘使用時間 |

> [!NOTE]
> 在化工領域，RNN最常見的應用是**多對一的回歸/分類問題**，即：基於一段時間的過程數據，預測某個關鍵指標或事件。

### 1.3 RNN的數學原理

**核心概念**: RNN通過循環連接實現記憶功能，關鍵在於**隱藏狀態**在時間步之間的傳遞。

#### 基本RNN單元結構

在時間步 $t$ ，RNN單元接收：
- **當前輸入**: $\mathbf{x}_t$  （例如：當前時刻的溫度、壓力）
- **前一時刻隱藏狀態**: $\mathbf{h}_{t-1}$  （記憶）

並產生：
- **當前隱藏狀態**: $\mathbf{h}_t$  （更新的記憶）
- **當前輸出**: $\mathbf{y}_t$  （可選，取決於任務）

#### RNN的前向傳播方程

**隱藏狀態更新**:

$$
\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)
$$

**輸出計算** (如果需要):

$$
\mathbf{y}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y
$$

**參數說明**:
- $\mathbf{W}_{xh}$  : 輸入到隱藏層的權重矩陣 (shape: `[input_dim, hidden_dim]`)
- $\mathbf{W}_{hh}$  : 隱藏層到隱藏層的循環權重矩陣 (shape: `[hidden_dim, hidden_dim]`)
- $\mathbf{W}_{hy}$  : 隱藏層到輸出的權重矩陣 (shape: `[hidden_dim, output_dim]`)
- $\mathbf{b}_h, \mathbf{b}_y$  : 偏差項
- $\mathbf{h}_t$  : 隱藏狀態向量 (shape: `[hidden_dim]`)
- $\mathbf{x}_t$  : 輸入向量 (shape: `[input_dim]`)
- $\mathbf{y}_t$  : 輸出向量 (shape: `[output_dim]`)

**關鍵理解**:

1. **循環連接**: $\mathbf{W}_{hh} \mathbf{h}_{t-1}$  項使得前一時刻的狀態影響當前狀態
2. **參數共享**: 所有時間步使用**相同的權重矩陣** $\mathbf{W}_{xh}, \mathbf{W}_{hh}, \mathbf{W}_{hy}$
3. **tanh激活**: 將隱藏狀態值限制在 $[-1, 1]$  範圍內

#### 展開視圖 (Unfolding in Time)

RNN可以被視為在時間軸上"展開"的深度網路：

```
t=1      t=2      t=3      t=T
 ↓        ↓        ↓        ↓
x₁  →  [RNN] →  [RNN] →  [RNN]  →  最終輸出
      ↓ h₁   ↓ h₂   ↓ h₃   ↓ hₜ
      (y₁)   (y₂)   (y₃)   (yₜ)
```

**化工過程類比**:

想像一個**連續攪拌槽式反應器(CSTR)**:
- **輸入 $\mathbf{x}_t$ **: 每時刻進料的原料濃度
- **隱藏狀態 $\mathbf{h}_t$ **: 反應器內的物質濃度（會受上一時刻影響）
- **輸出 $\mathbf{y}_t$ **: 出料濃度或產率

反應器內的濃度不僅取決於當前進料，還取決於之前的累積狀態！

### 1.4 完整序列處理流程

**問題設定**: 給定一個長度為 $T$  的輸入序列:

$$
\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_T]
$$

目標是得到對應的輸出（分類或回歸）。

**前向傳播步驟**:

1. **初始化隱藏狀態**:
   
$$
\mathbf{h}_0 = \mathbf{0} \quad \text{或隨機初始化}
$$

2. **逐時間步計算** (for t = 1 to T):

$$
\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)
$$

3. **產生最終輸出** (依任務類型):
   - **多對一**: 只使用最後的隱藏狀態 $\mathbf{h}_T$
     
$$
\mathbf{y} = f(\mathbf{W}_{hy} \mathbf{h}_T + \mathbf{b}_y)
$$
   
   - **多對多**: 每個時間步都產生輸出 $\mathbf{y}_t$
     
$$
\mathbf{y}_t = f(\mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y)
$$

**具體數值範例**:

假設：
- 輸入維度 `input_dim = 3` (如: 溫度, 壓力, 流量)
- 隱藏層維度 `hidden_dim = 5`
- 序列長度 `T = 10` (10個時間步)

**步驟1**: 在 $t=1$
```
x₁ = [120°C, 2.5bar, 10L/min]  (shape: [3])
h₀ = [0, 0, 0, 0, 0]           (shape: [5])

h₁ = tanh(W_hh @ h₀ + W_xh @ x₁ + b_h)  → 得到 h₁ (shape: [5])
```

**步驟2**: 在 $t=2$
```
x₂ = [122°C, 2.4bar, 9.8L/min]
h₁ = [0.2, -0.5, 0.8, 0.1, -0.3]  (來自上一步)

h₂ = tanh(W_hh @ h₁ + W_xh @ x₂ + b_h)  → 得到 h₂
```

**重複至 $t=10$**...

**步驟3**: 最終輸出 (以多對一回歸為例)
```
y = W_hy @ h₁₀ + b_y  → 得到預測值 (如: 產品純度 = 95.3%)
```

> [!TIP]
> **關鍵直覺**: 隱藏狀態 $\mathbf{h}_t$  是一個"滾動總結"，它濃縮了從開始到當前時刻的所有資訊。每個新的時間步，都會更新這個總結。

### 1.5 RNN的訓練：BPTT算法

**核心概念**: RNN的訓練使用**時間反向傳播(Backpropagation Through Time, BPTT)**算法，本質上是將RNN展開成深度網路，然後應用標準的反向傳播。

#### 損失函數

對於不同的任務類型，損失函數定義不同：

**多對一問題** (如分類/回歸):

$$
L = \mathcal{L}(\mathbf{y}, \hat{\mathbf{y}})
$$

其中 $\hat{\mathbf{y}}$  基於最終隱藏狀態 $\mathbf{h}_T$  計算。

**多對多問題** (每個時間步都有監督訊號):

$$
L = \frac{1}{T} \sum_{t=1}^{T} \mathcal{L}(\mathbf{y}_t, \hat{\mathbf{y}}_t)
$$

#### BPTT梯度計算

由於RNN在時間上展開，梯度需要從最後一個時間步反向傳播到第一個時間步。

**關鍵公式** (以 $\mathbf{W}_{hh}$ 為例):

$$
\frac{\partial L}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^{T} \frac{\partial L}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hh}}
$$

由於 $\mathbf{h}_t$  依賴於 $\mathbf{h}_{t-1}$ ，梯度會**沿時間向後傳播**：

$$
\frac{\partial L}{\partial \mathbf{h}_t} = \frac{\partial L}{\partial \mathbf{h}_{t+1}} \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t}
$$

**計算流程**:
```
1. 前向傳播: t=1 → t=2 → ... → t=T，計算所有 h_t 和損失 L
2. 反向傳播: t=T → t=T-1 → ... → t=1，計算梯度
3. 參數更新: 使用累積的梯度更新 W_xh, W_hh, W_hy
```

> [!WARNING]
> BPTT的計算複雜度與序列長度 $T$  成正比，長序列會導致：
> 1. 訓練時間增加
> 2. 記憶體需求增加
> 3. 梯度消失/爆炸問題

### 1.6 RNN的關鍵問題

#### 問題1: 梯度消失(Vanishing Gradient)

**現象**: 當序列很長時，早期時間步的梯度會變得極小，導致：
- 網路無法學習長期依賴關係
- 早期時間步的參數幾乎不更新

**數學原因**:

在BPTT中，梯度需要連乘多次：

$$
\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod_{i=k+1}^{t} \frac{\partial \mathbf{h}_i}{\partial \mathbf{h}_{i-1}}
$$

每一項包含 $\mathbf{W}_{hh}$  和 $\tanh'$ ：

$$
\frac{\partial \mathbf{h}_i}{\partial \mathbf{h}_{i-1}} = \text{diag}(\tanh'(\mathbf{z}_i)) \cdot \mathbf{W}_{hh}
$$

由於：
- $\tanh'(x) \in (0, 1]$ ，最大值為1
- 如果 $\mathbf{W}_{hh}$  的最大特徵值 < 1

則連乘 $t-k$ 次後，梯度會**指數級衰減**：

$$
\left\| \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} \right\| \approx \lambda^{t-k} \to 0 \quad \text{(當 } \lambda < 1 \text{ 且 } t-k \text{ 很大)}
$$

**實際影響**:
```
假設序列長度 T = 100
如果梯度每步衰減 0.9倍
則從 t=100 傳回 t=1 的梯度: 0.9^99 ≈ 0.00003
```
→ 早期資訊被"遺忘"

**化工類比**: 
就像一個長管道中的訊號傳遞，每經過一個閥門就衰減一些，最終起點的訊號幾乎無法傳到終點。

#### 問題2: 梯度爆炸(Exploding Gradient)

**現象**: 與梯度消失相反，當 $\mathbf{W}_{hh}$  的特徵值 > 1 時，梯度會**指數級增長**。

**後果**:
- 參數更新過大，導致訓練不穩定
- 損失值突然變成 NaN

**解決方法**: **梯度裁剪(Gradient Clipping)**

```python
# Keras 中
optimizer = Adam(clipnorm=1.0)  # 或 clipvalue=0.5
```

數學表示：

$$
\mathbf{g} \leftarrow \begin{cases}
\mathbf{g} & \text{if } \|\mathbf{g}\| \leq \text{threshold} \\
\frac{\text{threshold}}{\|\mathbf{g}\|} \mathbf{g} & \text{otherwise}
\end{cases}
$$

#### 問題3: 長期依賴問題

**核心挑戰**: 傳統RNN很難學習**相距很遠的時間步之間的依賴關係**。

**範例場景**:
```
時間序列: [x₁, x₂, ..., x₉₈, x₉₉, x₁₀₀]
                ↑               ↑
              關鍵事件         需要預測

如果 x₁ 和 x₁₀₀ 之間有因果關係，傳統RNN很難捕捉
```

**化工實例**:
- 批次反應開始時的催化劑投入量，影響2小時後的產品品質
- 如果序列有120個時間步(每分鐘採樣)，RNN難以建立這種長程依賴

> [!IMPORTANT]
> **這些問題的解決方案**: LSTM(長短期記憶網路) 和 GRU(門控循環單元)，它們通過引入**門控機制**來緩解梯度消失並增強長期記憶能力。

### 1.7 LSTM (Long Short-Term Memory)

**核心概念**: LSTM通過引入**記憶細胞**和**三個門控機制**，解決了傳統RNN的梯度消失和長期依賴問題。

#### 為什麼需要LSTM？

傳統RNN的問題：
- ❌ 隱藏狀態 $\mathbf{h}_t$ 在每個時間步都被完全重寫
- ❌ 難以保持長期資訊
- ❌ 梯度容易消失

LSTM的解決方案：
- ✅ 分離**短期記憶** ($\mathbf{h}_t$ ) 和**長期記憶** ($\mathbf{c}_t$ )
- ✅ 使用**門控機制**選擇性地保留/遺忘/更新資訊
- ✅ 梯度可以通過記憶細胞"快速通道"流動，緩解消失問題

#### LSTM的結構

LSTM在每個時間步維護兩個狀態：
- **記憶細胞狀態** $\mathbf{c}_t$  : 長期記憶（類似傳送帶）
- **隱藏狀態** $\mathbf{h}_t$  : 短期記憶（輸出給下一層）

並使用三個門控單元：
1. **遺忘門 (Forget Gate)** $\mathbf{f}_t$  : 決定從 $\mathbf{c}_{t-1}$  中遺忘多少資訊
2. **輸入門 (Input Gate)** $\mathbf{i}_t$  : 決定將多少新資訊加入 $\mathbf{c}_t$
3. **輸出門 (Output Gate)** $\mathbf{o}_t$  : 決定從 $\mathbf{c}_t$  中輸出多少資訊到 $\mathbf{h}_t$

#### LSTM的數學方程

在時間步 $t$ ，給定輸入 $\mathbf{x}_t$  和前一狀態 $\mathbf{h}_{t-1}, \mathbf{c}_{t-1}$ ：

**1. 遺忘門** (Forget Gate):

$$
\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)
$$

- $\mathbf{f}_t \in (0, 1)^{h}$  : 每個元素在0到1之間
- 接近0：遺忘對應位置的記憶
- 接近1：保留對應位置的記憶

**2. 輸入門** (Input Gate) 和**候選記憶**:

$$
\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)
$$

$$
\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)
$$

- $\mathbf{i}_t \in (0, 1)^{h}$  : 控制新資訊的接受程度
- $\tilde{\mathbf{c}}_t \in (-1, 1)^{h}$  : 候選的新記憶內容

**3. 更新記憶細胞**:

$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
$$

- $\odot$ : 逐元素相乘(element-wise multiplication)
- 第一項：保留的舊記憶（經遺忘門篩選）
- 第二項：加入的新記憶（經輸入門篩選）

**4. 輸出門和隱藏狀態**:

$$
\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)
$$

$$
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
$$

- $\mathbf{o}_t \in (0, 1)^{h}$  : 控制輸出程度
- $\mathbf{h}_t$  : 經過輸出門篩選的記憶細胞內容

**符號說明**:
- $\sigma$  : Sigmoid函數，輸出範圍 (0, 1)
- $[\mathbf{h}_{t-1}, \mathbf{x}_t]$  : 拼接向量，shape = `[hidden_dim + input_dim]`
- $\mathbf{W}_f, \mathbf{W}_i, \mathbf{W}_c, \mathbf{W}_o$  : 權重矩陣
- $\mathbf{b}_f, \mathbf{b}_i, \mathbf{b}_c, \mathbf{b}_o$  : 偏差向量

#### LSTM工作機制直覺理解

**遺忘門**: "我應該忘記多少過去的資訊？"
```
例: 當反應階段改變時，遺忘門可能決定遺忘上一階段的溫度趨勢
```

**輸入門**: "我應該接受多少新資訊？"
```
例: 當檢測到重要事件(如加料)時，輸入門開啟，記錄新資訊
```

**輸出門**: "我應該輸出多少當前記憶？"
```
例: 在批次結束時，輸出門可能開啟，提供完整的歷程摘要
```

**記憶細胞更新**: "平衡保留和更新"
```
c_t = (保留的舊記憶 * 遺忘程度) + (新資訊 * 接受程度)
```

#### LSTM的視覺化

```
                    c_{t-1} ─────────→ c_t
                      │                 │
                      │  [遺忘門]       │  [輸入門]
                      │   ×f_t          │   ×i_t
                      │                 │
                      │                 ⊕──── tanh(候選記憶)
                      │                 
h_{t-1} ──┐          └──────┐          
          │                 │          
x_t ──────┴──→ [三個門控單元] ──→ [輸出門] ──→ h_t
                                  ×o_t
```

**化工過程類比**: LSTM就像一個智慧型緩衝槽

- **記憶細胞 $\mathbf{c}_t$ **: 緩衝槽內的物料濃度（長期積累）
- **遺忘門**: 出料閥門（決定排出多少舊物料）
- **輸入門**: 進料閥門（決定接受多少新物料）
- **輸出門**: 取樣閥門（決定輸出多少資訊給下游）

#### LSTM為何能解決梯度消失？

**關鍵機制**: 記憶細胞的**直通路徑**

在BPTT過程中，梯度可以沿著記憶細胞 $\mathbf{c}_t$ 直接流動：

$$
\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \mathbf{f}_t
$$

與傳統RNN不同：
- ❌ 傳統RNN: 梯度必須經過 $\tanh$ 和 $\mathbf{W}_{hh}$ 的連乘
- ✅ LSTM: 梯度可以通過**接近1的遺忘門**直接傳遞

如果遺忘門學習到保持開啟狀態 ($\mathbf{f}_t \approx 1$ )，則：

$$
\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_0} = \mathbf{f}_t \cdot \mathbf{f}_{t-1} \cdot ... \cdot \mathbf{f}_1 \approx 1
$$

→ 梯度不會消失！

> [!TIP]
> **實踐建議**: 在大多數時間序列任務中，LSTM是比傳統RNN更好的選擇。除非序列非常短（<10步），否則優先考慮LSTM或GRU。

### 1.8 GRU (Gated Recurrent Unit)

**核心概念**: GRU是LSTM的簡化版本，使用**更少的門控機制**（2個門），但性能接近LSTM，且計算更快。

#### GRU與LSTM的差異

| 特性 | LSTM | GRU |
|-----|------|-----|
| **狀態數量** | 2個 ($\mathbf{h}_t, \mathbf{c}_t$) | 1個 ($\mathbf{h}_t$) |
| **門的數量** | 3個 (遺忘、輸入、輸出) | 2個 (重置、更新) |
| **參數量** | 更多 | 更少 (約75%) |
| **計算速度** | 較慢 | 較快 |
| **長期記憶能力** | 稍強 | 稍弱但接近 |

**何時選擇GRU？**
- ✅ 數據集較小，需要減少過擬合風險
- ✅ 需要更快的訓練速度
- ✅ 序列長度中等（不是特別長）
- ✅ 計算資源有限

#### GRU的數學方程

GRU只維護一個隱藏狀態 $\mathbf{h}_t$ ，並使用兩個門：

**1. 重置門 (Reset Gate)**:

$$
\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r)
$$

- 決定要**忽略多少過去資訊**
- 類似LSTM的"部分遺忘"機制

**2. 更新門 (Update Gate)**:

$$
\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z)
$$

- 決定**保留多少舊狀態 vs. 接受多少新狀態**
- 同時扮演LSTM中遺忘門和輸入門的角色

**3. 候選隱藏狀態**:

$$
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_h)
$$

- 計算新的候選狀態
- 重置門 $\mathbf{r}_t$ 控制使用多少過去資訊

**4. 最終隱藏狀態**:

$$
\mathbf{h}_t = (\mathbf{1} - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
$$

- 第一項：保留的舊狀態（比例 $1-\mathbf{z}_t$）
- 第二項：加入的新狀態（比例 $\mathbf{z}_t$）

**關鍵直覺**: 更新門 $\mathbf{z}_t$  決定"新舊混合比例"
```
z_t = 0.2 → 保留80%舊狀態 + 20%新狀態
z_t = 0.8 → 保留20%舊狀態 + 80%新狀態
```

#### GRU工作機制

**重置門**: "重新開始還是延續？"
```
r_t ≈ 0: 忽略過去，重新開始（如批次開始）
r_t ≈ 1: 充分利用過去資訊
```

**更新門**: "保留還是更新？"
```
z_t ≈ 0: 大量更新（如遇到突發事件）
z_t ≈ 1: 大量保留（如穩態運行）
```

#### GRU的視覺化

```
h_{t-1} ──┬───────┐
          │       │ [重置門 r_t]
x_t ──────┴──→ [門控單元] ──┐
                           │
                         [候選狀態] ─→ h_tilde
                           │
          ┌────────────────┴─────[更新門 z_t]
          │                      
h_{t-1} ──┴─→ [1-z_t] ×  ─┬
                          ⊕──→ h_t
h_tilde ─────→ [z_t] ×  ──┘
```

#### GRU的優勢

1. **參數效率**: 
   - LSTM參數量: $4 \times (d_h \times (d_h + d_x) + d_h)$
   - GRU參數量: $3 \times (d_h \times (d_h + d_x) + d_h)$
   - 減少約25%參數

2. **訓練速度**: 
   - 更少的矩陣運算
   - 更快的前向/反向傳播

3. **性能**: 
   - 在許多任務上與LSTM相當
   - 部分任務甚至優於LSTM

> [!NOTE]
> **經驗法則**: 
> - 如果不確定，先嘗試GRU（訓練快，調參易）
> - 如果需要最佳性能，再試LSTM
> - 如果兩者性能相近，選GRU（更高效）

### 1.9 LSTM vs GRU vs 基礎RNN 比較總結

| 特性 | 基礎RNN | LSTM | GRU |
|-----|---------|------|-----|
| **結構複雜度** | 最簡單 | 最複雜 | 中等 |
| **參數量** | 最少 | 最多 | 中等 |
| **訓練速度** | 最快 | 最慢 | 中等 |
| **長期依賴** | ❌ 很差 | ✅ 優秀 | ✅ 良好 |
| **梯度消失** | ❌ 嚴重 | ✅ 已解決 | ✅ 已解決 |
| **適用序列長度** | <20步 | >50步 | 20-100步 |
| **記憶機制** | 單一隱藏狀態 | 雙狀態(h+c) | 單一隱藏狀態 |
| **推薦使用場景** | 短序列、教學 | 超長序列、複雜任務 | 一般任務、首選 |

**實際應用建議**:

```
簡單任務、短序列 (T < 20)
    ↓
基礎RNN 或 GRU

一般時間序列任務 (20 < T < 100)
    ↓
GRU (首選)

複雜任務、長序列 (T > 100)、需要精細記憶控制
    ↓
LSTM

計算資源有限、需要快速迭代
    ↓
GRU
```

**本章小結**

恭喜完成RNN基礎理論學習！您現在理解了：

✅ **RNN的核心**: 循環連接實現序列記憶  
✅ **BPTT算法**: 如何訓練RNN  
✅ **關鍵問題**: 梯度消失與長期依賴  
✅ **解決方案**: LSTM的門控機制  
✅ **簡化版本**: GRU的高效設計  

接下來，我們將學習如何使用Keras快速實現這些模型！

---

## 2. 使用Keras建立RNN模型

### 本章學習地圖

> [!IMPORTANT]
> **本章核心問題**: 如何使用Keras快速建立和配置RNN/LSTM/GRU模型？

**學習目標**:
1. 🎯 **掌握RNN層使用**: 了解Keras中RNN相關層的API
2. 🏗️ **學會建立序列模型**: 掌握不同任務類型的模型架構
3. 🔧 **理解關鍵參數**: 熟悉return_sequences、return_state等重要設定
4. 📊 **處理時間序列數據**: 學會數據準備與形狀轉換

**為什麼要學Keras RNN API？**

Keras將複雜的RNN數學實現封裝成簡單的API：
- ✅ **幾行代碼**: 就能建立LSTM/GRU模型
- ✅ **自動處理**: BPTT、梯度裁剪等細節自動完成
- ✅ **靈活配置**: 支援多種序列任務類型
- ✅ **效能優化**: 底層使用CuDNN加速（GPU上）

**本章架構**:

```
Keras RNN層簡介 (2.1)
    ↓
時間序列數據準備 (2.2)
    ├─ 數據形狀要求
    └─ 序列生成方法
    ↓
建立RNN/LSTM/GRU模型 (2.3)
    ├─ 基礎RNN
    ├─ LSTM實現
    └─ GRU實現
    ↓
不同任務類型的架構 (2.4)
    ├─ 多對一 (Many-to-One)
    ├─ 多對多 (Many-to-Many)
    └─ Seq2Seq
    ↓
關鍵參數詳解 (2.5)
```

---

### 2.1 Keras RNN層簡介

#### 2.1.1 Keras提供的RNN層

Keras在`tensorflow.keras.layers`中提供三種主要的RNN層：

```python
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU
```

| 層類型 | 用途 | 參數量 | 訓練速度 | 推薦場景 |
|--------|------|--------|---------|---------|
| `SimpleRNN` | 基礎RNN | 最少 | 最快 | 教學、極短序列(<10) |
| `LSTM` | 長短期記憶 | 最多 (4x) | 最慢 | 長序列(>50)、複雜任務 |
| `GRU` | 門控循環單元 | 中等 (3x) | 中等 | 一般任務、首選 |

#### 2.1.2 基本使用語法

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 建立簡單的LSTM模型
model = Sequential([
    LSTM(units=64, input_shape=(timesteps, features)),
    Dense(1)  # 輸出層
])
```

**核心參數說明**:

| 參數 | 說明 | 預設值 |
|-----|------|--------|
| `units` | 隱藏層神經元數量 (hidden_dim) | 必須指定 |
| `input_shape` | `(時間步數, 特徵數)` | 第一層必須指定 |
| `return_sequences` | 是否返回完整序列 | `False` |
| `return_state` | 是否返回隱藏狀態 | `False` |
| `activation` | 隱藏層激活函數 | `'tanh'` |
| `recurrent_activation` | 循環連接激活函數 | `'sigmoid'` (LSTM/GRU門控) |
| `dropout` | 輸入dropout比例 | `0.0` |
| `recurrent_dropout` | 循環連接dropout | `0.0` |

---

### 2.2 時間序列數據準備

#### 2.2.1 數據形狀要求

**關鍵概念**: RNN層要求輸入為**3D張量**:

```
輸入形狀: (batch_size, timesteps, features)
```

- `batch_size`: 批次大小（樣本數量）
- `timesteps`: 時間步數（序列長度）
- `features`: 每個時間步的特徵數量

**範例理解**:

假設我們有化工反應器的時間序列數據：
```python
# 原始數據
time | 溫度(°C) | 壓力(bar) | 流量(L/min)
-----|----------|-----------|-------------
0    | 120      | 2.5       | 10.0
1    | 122      | 2.4       | 9.8
2    | 125      | 2.3       | 9.5
...  | ...      | ...       | ...
9    | 130      | 2.0       | 8.0
```

**轉換為RNN輸入**:
- 如果我們有100個批次，每個批次有10個時間步
- 每個時間步有3個特徵（溫度、壓力、流量）
- 形狀為: `(100, 10, 3)`

```python
# 數據形狀範例
import numpy as np

# 生成範例數據
n_samples = 100
timesteps = 10
n_features = 3

X = np.random.randn(n_samples, timesteps, n_features)
print(f"輸入形狀: {X.shape}")  # (100, 10, 3)
```

#### 2.2.2 從表格數據生成序列

**問題**: 原始數據通常是2D表格（時間 × 特徵），如何轉換為3D序列？

**方法1: 滑動視窗法 (Sliding Window)**

最常用的方法，用於單變量或多變量時間序列預測：

```python
def create_sequences(data, window_size):
    """
    將2D數據轉換為3D序列
    
    參數:
        data: 形狀為 (n_timestamps, n_features) 的數組
        window_size: 視窗大小（使用多少歷史時間步）
    
    返回:
        X: 形狀為 (n_samples, window_size, n_features)
        y: 形狀為 (n_samples, n_features) 或 (n_samples,)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])  # 取一個視窗
        y.append(data[i+window_size])     # 對應的標籤
    return np.array(X), np.array(y)

# 範例使用
data = np.random.randn(1000, 3)  # 1000個時間點，3個特徵
window_size = 10

X, y = create_sequences(data, window_size)
print(f"X形狀: {X.shape}")  # (990, 10, 3)
print(f"y形狀: {y.shape}")  # (990, 3)
```

**視覺化理解**:
```
原始數據: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...]
視窗大小 = 3

batch1: X = [0, 1, 2] → y = 3
batch2: X = [1, 2, 3] → y = 4
batch3: X = [2, 3, 4] → y = 5
...
```

**方法2: 批次分割法 (Batch Segmentation)**

適用於批次過程數據，每個批次是獨立樣本：

```python
def batch_to_sequences(data, batch_ids, seq_length):
    """
    將批次數據分割為序列
    
    參數:
        data: 完整數據 (n_total_points, n_features)
        batch_ids: 每個數據點的批次ID
        seq_length: 每個序列的長度
    """
    X, y = [], []
    unique_batches = np.unique(batch_ids)
    
    for batch_id in unique_batches:
        batch_data = data[batch_ids == batch_id]
        if len(batch_data) >= seq_length + 1:
            X.append(batch_data[:seq_length])
            y.append(batch_data[seq_length, 0])  # 預測某個指標
    
    return np.array(X), np.array(y)
```

**範例場景**:
```
批次反應器數據：
Batch1: 100個時間點
Batch2: 120個時間點
Batch3: 90個時間點

轉換為:
樣本1: 前10個時間點 → 預測最終品質
樣本2: 前10個時間點 → 預測最終品質
...
```

#### 2.2.3 數據標準化

**重要**: 時間序列數據在送入RNN前應該標準化！

```python
from sklearn.preprocessing import StandardScaler

# 方法1: 對每個特徵獨立標準化
scaler = StandardScaler()

# 先將3D展平為2D
n_samples, timesteps, n_features = X_train.shape
X_train_2d = X_train.reshape(-1, n_features)

# 擬合並轉換
X_train_scaled_2d = scaler.fit_transform(X_train_2d)

# 重新變回3D
X_train_scaled = X_train_scaled_2d.reshape(n_samples, timesteps, n_features)

# 對測試集只做transform (不重新fit)
X_test_2d = X_test.reshape(-1, n_features)
X_test_scaled = scaler.transform(X_test_2d).reshape(-1, timesteps, n_features)
```

**為什麼要標準化？**
- ✅ 加速收斂
- ✅ 避免某些特徵主導訓練
- ✅ 防止梯度爆炸

> [!WARNING]
> **關鍵注意**: 
> 1. 先分割訓練/測試集，再標準化！
> 2. StandardScaler必須在訓練集上fit，測試集只transform
> 3. 保存scaler以便未來預測時使用

---

### 2.3 建立RNN/LSTM/GRU模型

#### 2.3.1 基礎RNN模型

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# 建立簡單RNN (多對一回歸)
model = Sequential([
    SimpleRNN(units=32, input_shape=(timesteps, n_features)),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()
```

**輸出範例**:
```
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
simple_rnn (SimpleRNN)      (None, 32)                1120      
dense (Dense)               (None, 1)                 33        
=================================================================
Total params: 1,153
```

**參數量計算**:
```
SimpleRNN參數 = units * (units + n_features + 1)
             = 32 * (32 + 3 + 1) = 1,152
```

#### 2.3.2 LSTM模型

```python
from tensorflow.keras.layers import LSTM

# 單層LSTM
model = Sequential([
    LSTM(units=64, input_shape=(timesteps, n_features)),
    Dense(1)
])

# 多層LSTM (注意return_sequences)
model = Sequential([
    LSTM(units=64, return_sequences=True, input_shape=(timesteps, n_features)),
    LSTM(units=32),  # 最後一層不需要return_sequences
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()
```

**多層LSTM輸出**:
```
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
lstm (LSTM)                 (None, 10, 64)            17408     
lstm_1 (LSTM)               (None, 32)                12416     
dense (Dense)               (None, 1)                 33        
=================================================================
Total params: 29,857
```

**參數量計算**:
```
LSTM參數 = 4 * units * (units + n_features + 1)
第一層: 4 * 64 * (64 + 3 + 1) = 17,408
第二層: 4 * 32 * (32 + 64 + 1) = 12,416
```

> [!IMPORTANT]
> **return_sequences=True** 的作用：
> - `True`: 返回所有時間步的輸出 → 形狀 `(batch, timesteps, units)`
> - `False`: 只返回最後一個時間步 → 形狀 `(batch, units)`
> - 規則：**堆疊RNN層時，除了最後一層，前面的層都要設為True**

#### 2.3.3 GRU模型

```python
from tensorflow.keras.layers import GRU

# 單層GRU
model = Sequential([
    GRU(units=64, input_shape=(timesteps, n_features)),
    Dense(1)
])

# 多層GRU + Dropout
model = Sequential([
    GRU(units=128, return_sequences=True, 
        dropout=0.2, recurrent_dropout=0.2,
        input_shape=(timesteps, n_features)),
    GRU(units=64, dropout=0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```


**Dropout參數說明**:
- `dropout`: 應用在輸入連接上（ $\mathbf{x}_t$  → 隱藏層）
- `recurrent_dropout`: 應用在循環連接上（ $\mathbf{h}_{t-1}$  → $\mathbf{h}_t$  ）

**注意事項**:
- 在TensorFlow/Keras中，`dropout`和`recurrent_dropout` 會影響訓練效能，特別是在GPU上。這是因為啟用這些參數時，Keras 會自動關閉CuDNN加速（CuDNN僅支援無dropout的LSTM/GRU），導致模型改用較慢的純TensorFlow實現。
- 這會使GPU訓練速度大幅下降，甚至比CPU還慢，尤其是大模型或長序列時。

**常見警告訊息**:
當你在LSTM/GRU層設定 `dropout` 或 `recurrent_dropout` 時，編譯模型時會看到如下警告：
```
UserWarning: `dropout` or `recurrent_dropout` is non-zero, causing the layer to use the generic RNN implementation. This will significantly slow down training on GPU. To use the faster CuDNN implementation, set both `dropout` and `recurrent_dropout` to 0.
```
或
```
WARNING:tensorflow:Layer lstm/gru will not use cuDNN kernels since it doesn't meet the criteria. It will use a generic GPU kernel as fallback when running on GPU.
```

**建議**:
- 若追求訓練速度，建議將 `dropout=0`、`recurrent_dropout=0`或直接移除，充分利用CuDNN加速。
- 若必須使用dropout正則化，可考慮只在Dense層加入Dropout，或接受較慢的訓練速度。
- 若模型較小或只在CPU上訓練，則影響較小。


**各模型參數量對比**:
```python
# 假設 units=64, n_features=3
SimpleRNN: 64 * (64 + 3 + 1) = 4,352
LSTM:      4 * 64 * (64 + 3 + 1) = 17,408  (4倍)
GRU:       3 * 64 * (64 + 3 + 1) = 13,056  (3倍)
```

---

### 2.4 不同任務類型的模型架構

#### 2.4.1 多對一 (Many-to-One)

**任務**: 輸入序列 → 單一輸出（最常見）

**應用場景**:
- 時間序列分類（設備正常/故障）
- 批次結果預測（最終品質）
- 事件預測（未來會不會發生某事）

**架構設計**:

```python
# 回歸問題
model = Sequential([
    LSTM(64, input_shape=(timesteps, n_features)),
    Dense(1)  # 輸出單一值
])

# 二元分類問題
model = Sequential([
    GRU(64, input_shape=(timesteps, n_features)),
    Dense(1, activation='sigmoid')  # 輸出機率
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 多類別分類
model = Sequential([
    LSTM(64, input_shape=(timesteps, n_features)),
    Dense(32, activation='relu'),
    Dense(n_classes, activation='softmax')  # 輸出n個類別機率
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
```

**數據形狀**:
```python
X_train.shape = (n_samples, timesteps, n_features)  # 例: (1000, 10, 3)
y_train.shape = (n_samples,) 或 (n_samples, 1)     # 例: (1000,)
```

#### 2.4.2 多對多 (Many-to-Many) - 等長

**任務**: 輸入序列 → 等長輸出序列

**應用場景**:
- 序列標註（每個時間點的狀態）
- 異常檢測（每個時間點正常/異常）
- 時間序列去噪

**架構設計**:

```python
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(timesteps, n_features)),
    # 重要：return_sequences=True 返回所有時間步
    TimeDistributed(Dense(1))  # 對每個時間步應用Dense
])

# 或更簡單的寫法（Keras自動處理）
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(timesteps, n_features)),
    Dense(1)  # 自動應用到每個時間步
])
```

**數據形狀**:
```python
X_train.shape = (n_samples, timesteps, n_features)    # (1000, 10, 3)
y_train.shape = (n_samples, timesteps, n_outputs)     # (1000, 10, 1)
```

**實例：異常檢測**:
```python
# 每個時間步輸出0(正常)或1(異常)
model = Sequential([
    GRU(64, return_sequences=True, input_shape=(timesteps, n_features)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')

# 訓練數據
X = sequence_data  # (n_samples, timesteps, n_features)
y = labels         # (n_samples, timesteps, 1) - 每個時間步的標籤
```

#### 2.4.3 多對多 (Seq2Seq) - 不等長

**任務**: 輸入序列 → 不同長度輸出序列

**應用場景**:
- 機器翻譯
- 時間序列預測（預測未來多步）
- 摘要生成

**架構設計** (Encoder-Decoder):

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector

# Encoder-Decoder架構
# Encoder: 將輸入序列編碼為固定長度向量
encoder_input = Input(shape=(input_timesteps, n_features))
encoder = LSTM(64)(encoder_input)

# Decoder: 將編碼向量解碼為輸出序列
decoder = RepeatVector(output_timesteps)(encoder)  # 重複編碼向量
decoder = LSTM(64, return_sequences=True)(decoder)
decoder_output = Dense(n_outputs)(decoder)

model = Model(encoder_input, decoder_output)
model.compile(optimizer='adam', loss='mse')
```

**更簡單的多步預測版本**:

```python
# 預測未來n步
def build_multistep_model(input_timesteps, n_features, output_steps):
    model = Sequential([
        LSTM(64, input_shape=(input_timesteps, n_features)),
        Dense(64, activation='relu'),
        Dense(output_steps)  # 輸出未來n步
    ])
    return model

# 範例：基於過去10步預測未來5步
model = build_multistep_model(input_timesteps=10, n_features=3, output_steps=5)
model.compile(optimizer='adam', loss='mse')

# 數據形狀
X.shape = (n_samples, 10, 3)  # 輸入
y.shape = (n_samples, 5)      # 輸出（未來5步）
```

---

### 2.5 關鍵參數詳解

#### 2.5.1 return_sequences

**最重要的參數之一！**

```python
# return_sequences=False (預設)
layer = LSTM(64, return_sequences=False)
# 輸入: (batch, timesteps, features)
# 輸出: (batch, units) - 只有最後一個時間步

# return_sequences=True
layer = LSTM(64, return_sequences=True)
# 輸入: (batch, timesteps, features)
# 輸出: (batch, timesteps, units) - 所有時間步
```

**使用場景**:

| 場景 | return_sequences | 原因 |
|-----|------------------|------|
| 多對一任務 | `False` | 只需要最後的隱藏狀態 |
| 堆疊RNN層（不是最後一層） | `True` | 下一層需要完整序列 |
| 多對多任務 | `True` | 需要每個時間步的輸出 |
| Seq2Seq Encoder | `False` | 只需要編碼向量 |
| Seq2Seq Decoder | `True` | 生成輸出序列 |

**範例對比**:

```python
# 錯誤：第一層沒有return_sequences=True
model = Sequential([
    LSTM(64, input_shape=(10, 3)),        # 輸出 (batch, 64)
    LSTM(32)  # 錯誤！期望3D輸入但得到2D
])

# 正確
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(10, 3)),  # 輸出 (batch, 10, 64)
    LSTM(32)  # 正確！接收3D輸入
])
```

#### 2.5.2 return_state

**用途**: 獲取最終的隱藏狀態和記憶細胞狀態（LSTM專用）

```python
# LSTM with return_state
lstm_layer = LSTM(64, return_state=True)

# 使用方法1: 在Sequential中（不常用）
output = LSTM(64, return_state=True)(input_layer)
# output是一個tuple: (output, hidden_state, cell_state)

# 使用方法2: 在Functional API中（常用於Encoder-Decoder）
encoder_input = Input(shape=(timesteps, n_features))
encoder_output, state_h, state_c = LSTM(64, return_state=True)(encoder_input)
encoder_states = [state_h, state_c]

# 將encoder狀態傳給decoder
decoder_lstm = LSTM(64, return_sequences=True)
decoder_output = decoder_lstm(decoder_input, initial_state=encoder_states)
```

**應用場景**:
- Seq2Seq模型：將Encoder的最終狀態傳遞給Decoder
- 狀態保持：在批次間保持RNN狀態（在線學習）

#### 2.5.3 stateful

**用途**: 在批次間保持狀態（進階用法）

```python
# Stateful RNN
model = Sequential([
    LSTM(64, stateful=True, 
         batch_input_shape=(batch_size, timesteps, n_features)),
    Dense(1)
])

# 訓練時需要手動重置狀態
for epoch in range(epochs):
    model.fit(X, y, batch_size=batch_size, epochs=1, shuffle=False)
    model.reset_states()  # 在每個epoch後重置
```

**何時使用**:
- 處理超長序列（分批輸入但保持記憶）
- 在線預測（持續更新狀態）
- 通常不需要（99%的情況用不到）

> [!WARNING]
> **注意**: stateful=True時：
> 1. 必須指定`batch_input_shape`（而非`input_shape`）
> 2. 不能使用`shuffle=True`
> 3. 需要手動調用`reset_states()`

#### 2.5.4 Dropout參數

```python
LSTM(units=64, 
     dropout=0.2,              # 應用在輸入上
     recurrent_dropout=0.2     # 應用在循環連接上
)
```

**dropout vs recurrent_dropout**:

```
dropout (0.2):
    x_t → [20% 隨機失活] → LSTM

recurrent_dropout (0.2):
    h_{t-1} → [20% 隨機失活] → LSTM
```

**推薦設定**:
- 小數據集: `dropout=0.2-0.3, recurrent_dropout=0.2`
- 大數據集: `dropout=0.1-0.2, recurrent_dropout=0.1`
- 過擬合嚴重: 增加到0.4-0.5

> [!TIP]
> **經驗建議**: 
> - 先用`dropout=0.2, recurrent_dropout=0.2`作為起點
> - 如果過擬合，增加dropout
> - 如果欠擬合，減少或移除dropout

---

### 2.6 完整建模範例

讓我們整合所有知識，建立一個完整的時間序列預測模型：

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ===== 1. 數據準備 =====
# 生成模擬時間序列數據
def generate_timeseries_data(n_samples=1000, timesteps=20, n_features=5):
    X = np.random.randn(n_samples, timesteps, n_features)
    # 目標：預測下一個時間點的第一個特徵
    y = np.random.randn(n_samples)
    return X, y

X, y = generate_timeseries_data(n_samples=2000, timesteps=30, n_features=10)

# 分割數據
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False  # 時間序列不shuffle!
)

# 標準化 (重要!)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_flat = X_train.reshape(-1, X_train.shape[-1])
X_train_scaled = scaler_X.fit_transform(X_train_flat).reshape(X_train.shape)

X_test_flat = X_test.reshape(-1, X_test.shape[-1])
X_test_scaled = scaler_X.transform(X_test_flat).reshape(X_test.shape)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# ===== 2. 建立模型 =====
def build_rnn_model(rnn_type='LSTM', units=[128, 64], dropout=0.2):
    """
    建立通用RNN模型
    
    參數:
        rnn_type: 'LSTM', 'GRU', 或 'SimpleRNN'
        units: 各層的神經元數量列表
        dropout: Dropout比例
    """
    from tensorflow.keras.layers import LSTM, GRU, SimpleRNN
    
    # 選擇RNN層類型
    RNN_LAYER = {'LSTM': LSTM, 'GRU': GRU, 'SimpleRNN': SimpleRNN}[rnn_type]
    
    model = Sequential()
    
    # 第一層（需要指定input_shape）
    model.add(RNN_LAYER(
        units[0], 
        return_sequences=len(units) > 1,  # 如果有多層則返回序列
        dropout=dropout,
        recurrent_dropout=dropout,
        input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])
    ))
    
    # 中間層
    for i, unit in enumerate(units[1:], 1):
        model.add(RNN_LAYER(
            unit,
            return_sequences=(i < len(units) - 1),  # 最後一層不返回序列
            dropout=dropout,
            recurrent_dropout=dropout
        ))
    
    # 全連接層
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1))  # 輸出層
    
    return model

# 建立LSTM模型
model_lstm = build_rnn_model(rnn_type='LSTM', units=[128, 64], dropout=0.2)

# 建立GRU模型
model_gru = build_rnn_model(rnn_type='GRU', units=[128, 64], dropout=0.2)

# 查看模型結構
model_lstm.summary()

# ===== 3. 編譯模型 =====
model_lstm.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae', 'mse']
)

# ===== 4. 設定Callbacks =====
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
]

# ===== 5. 訓練模型 =====
history = model_lstm.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ===== 6. 評估模型 =====
test_loss, test_mae, test_mse = model_lstm.evaluate(X_test_scaled, y_test_scaled)
print(f"\n測試集性能:")
print(f"MAE: {test_mae:.4f}")
print(f"MSE: {test_mse:.4f}")
print(f"RMSE: {np.sqrt(test_mse):.4f}")

# ===== 7. 預測 =====
y_pred_scaled = model_lstm.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)  # 反標準化

# ===== 8. 視覺化 =====
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

# 子圖1: 訓練歷史
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.title('Training History')
plt.grid(True)

# 子圖2: 預測 vs 實際
plt.subplot(1, 3, 2)
plt.plot(y_test[:100], label='Actual', marker='o', markersize=3)
plt.plot(y_pred[:100], label='Predicted', marker='x', markersize=3)
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()
plt.title('Predictions vs Actual (First 100 samples)')
plt.grid(True)

# 子圖3: 散點圖
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Prediction Scatter Plot')
plt.grid(True)

plt.tight_layout()
plt.savefig('rnn_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== 9. 保存模型 =====
model_lstm.save('lstm_model.h5')
print("\n模型已保存為 lstm_model.h5")
```

**本章小結**

恭喜！您已經掌握了Keras RNN的核心知識：

✅ **RNN層使用**: SimpleRNN、LSTM、GRU的基本語法  
✅ **數據準備**: 3D張量要求、滑動視窗法、標準化  
✅ **模型架構**: 多對一、多對多、Seq2Seq設計  
✅ **關鍵參數**: return_sequences、dropout的正確使用  
✅ **完整流程**: 從數據準備到模型訓練的端到端實現  

**下一步**: 接下來我們將學習模型的編譯、訓練和優化技巧！

---

## 3. 模型編譯與訓練

### 本章學習地圖

> [!IMPORTANT]
> **本章核心問題**: 如何正確配置RNN模型的訓練過程？時間序列模型有哪些特殊考量？

**學習目標**:
1. 🎯 **掌握編譯配置**: 為RNN選擇合適的優化器、損失函數
2. 🏋️ **學會訓練技巧**: 批次大小、Epoch設定、驗證策略
3. 📊 **使用Callbacks**: EarlyStopping、ModelCheckpoint、ReduceLROnPlateau
4. 🔍 **理解特殊考量**: 時間序列訓練的注意事項

**本章架構**:

```
模型編譯 (3.1)
    ├─ 優化器選擇
    ├─ 損失函數
    └─ 評估指標
    ↓
模型訓練 (3.2)
    ├─ model.fit()參數
    ├─ 批次大小與Epoch
    └─ 驗證策略
    ↓
Callbacks使用 (3.3)
    ├─ EarlyStopping
    ├─ ModelCheckpoint
    ├─ ReduceLROnPlateau
    └─ TensorBoard
    ↓
時間序列特殊考量 (3.4)
```

---

### 3.1 模型編譯

#### 3.1.1 基本編譯語法

```python
model.compile(
    optimizer='adam',           # 優化器
    loss='mse',                 # 損失函數
    metrics=['mae', 'mse']      # 評估指標
)
```

#### 3.1.2 RNN推薦的優化器

**1. Adam (首選)**

最常用且效果最好的優化器：

```python
from tensorflow.keras.optimizers import Adam

optimizer = Adam(
    learning_rate=0.001,        # 學習率
    beta_1=0.9,                 # 一階矩估計的指數衰減率
    beta_2=0.999,               # 二階矩估計的指數衰減率
    clipnorm=1.0                # 梯度裁剪（重要！）
)

model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
```

**為什麼Adam適合RNN？**
- ✅ 自適應學習率，對不同參數使用不同的更新步長
- ✅ 對超參數不敏感
- ✅ 內建動量機制，加速收斂
- ✅ 適合處理稀疏梯度

**2. RMSprop (備選)**

專門為RNN設計的優化器：

```python
from tensorflow.keras.optimizers import RMSprop

optimizer = RMSprop(
    learning_rate=0.001,
    rho=0.9,
    clipvalue=1.0               # 梯度裁剪
)
```

**何時使用RMSprop？**
- 當Adam訓練不穩定時
- 早期RNN論文多使用RMSprop
- 適合處理非平穩目標

**3. SGD with Momentum (較少使用)**

```python
from tensorflow.keras.optimizers import SGD

optimizer = SGD(
    learning_rate=0.01,
    momentum=0.9,
    clipnorm=1.0
)
```

**梯度裁剪的重要性**

> [!WARNING]
> RNN訓練中**梯度爆炸**是常見問題，必須使用梯度裁剪！

```python
# 方法1: clipnorm (裁剪梯度的範數)
Adam(learning_rate=0.001, clipnorm=1.0)

# 方法2: clipvalue (裁剪梯度的值)
Adam(learning_rate=0.001, clipvalue=0.5)

# 推薦使用clipnorm=1.0
```
clipnorm 是用於梯度裁剪（gradient clipping）的一個參數，常見於深度學習模型的優化器設定中。它的作用是將梯度向量的 L2 範數限制在指定的最大值（如 clipnorm=1.0）以內，防止梯度爆炸。clipnorm 的設定值通常大於 0，常見範圍為 0.1 到 5.0，具體數值需根據模型和訓練情境調整。過小可能導致學習過慢，過大則無法有效抑制梯度爆炸。


**學習率選擇建議**:

| 任務類型 | 推薦學習率 | 說明 |
|---------|-----------|------|
| 標準時間序列預測 | 0.001 | Adam預設值 |
| 複雜長序列 | 0.0001 | 降低以穩定訓練 |
| 簡單短序列 | 0.01 | 可稍微提高 |
| 微調(Fine-tuning) | 0.0001 | 避免破壞已學習的特徵 |

#### 3.1.3 損失函數選擇

**回歸問題** (最常見):

```python
# 1. MSE - 標準選擇
model.compile(optimizer='adam', loss='mse')

# 2. MAE - 對異常值穩健
model.compile(optimizer='adam', loss='mae')

# 3. Huber Loss - MSE和MAE的折衷
from tensorflow.keras.losses import Huber
model.compile(optimizer='adam', loss=Huber(delta=1.0))

# 4. MSLE - 適合目標值範圍大的情況
model.compile(optimizer='adam', loss='msle')
```

**分類問題**:

```python
# 二元分類
model = Sequential([
    LSTM(64, input_shape=(timesteps, n_features)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 多類別分類
model = Sequential([
    LSTM(64, input_shape=(timesteps, n_features)),
    Dense(n_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

**損失函數選擇指南**:

| 場景 | 推薦損失函數 | 原因 |
|-----|-------------|------|
| 一般時間序列預測 | MSE | 標準選擇，懲罰大誤差 |
| 有異常值的數據 | MAE 或 Huber | 對異常值不敏感 |
| 目標值範圍很大 | MSLE | 關注相對誤差 |
| 二元分類 | Binary Crossentropy | 標準選擇 |
| 多類別分類 | Sparse Categorical CE | 標準選擇 |

#### 3.1.4 評估指標

```python
# 回歸問題
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae', 'mse', 'RootMeanSquaredError']
)

# 分類問題
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'Precision', 'Recall', 'AUC']
)
```

> [!NOTE]
> 評估指標僅用於監控，不影響模型訓練。可以添加多個指標以全面評估模型。
>
> 其中 `RootMeanSquaredError` (RMSE) **不建議作為損失函數**，因為其導數在零點處不連續，可能導致訓練不穩定。實務上，RMSE 常用於**直觀觀察模型預測誤差的單位**（與原始數據單位一致），便於解釋模型表現。建議訓練時以 `mse` 或 `mae` 作為損失函數，並將 RMSE 僅作為評估指標。

---

### 3.2 模型訓練

#### 3.2.1 基本訓練語法

```python
history = model.fit(
    X_train,                    # 訓練數據 (3D)
    y_train,                    # 訓練標籤
    epochs=100,                 # 訓練輪數
    batch_size=32,              # 批次大小
    validation_split=0.2,       # 驗證集比例
    verbose=1,                  # 顯示詳細程度
    shuffle=False               # 時間序列不打亂！
)
```

#### 3.2.2 關鍵參數詳解

**1. epochs (訓練輪數)**

```python
# epochs: 完整遍歷訓練集的次數
history = model.fit(X_train, y_train, epochs=100)
```

**如何選擇？**
- 起點：50-100 epochs
- 使用EarlyStopping自動停止（推薦）
- 觀察訓練曲線：
  - 損失持續下降 → 可增加
  - 驗證損失開始上升 → 過擬合，需提前停止

**2. batch_size (批次大小)**

```python
# batch_size: 每次更新權重使用的樣本數
history = model.fit(X_train, y_train, batch_size=32)
```

**RNN的批次大小選擇**:

| 批次大小 | 優點 | 缺點 | 適用場景 |
|---------|------|------|---------|
| 小 (16-32) | 訓練穩定、泛化好 | 訓練慢 | 數據少、記憶體有限 |
| 中 (64-128) | 平衡 | - | 一般推薦 |
| 大 (256-512) | 訓練快 | 可能泛化差 | 大數據集 |

**實用建議**:
```python
# 根據序列長度調整
if timesteps > 100:
    batch_size = 16  # 長序列使用小批次
elif timesteps > 50:
    batch_size = 32
else:
    batch_size = 64  # 短序列可用大批次
```

**3. validation_split vs validation_data**

```python
# 方法1: 自動分割 (注意：時間序列要小心！)
history = model.fit(
    X_train, y_train,
    validation_split=0.2    # 取最後20%作為驗證集
)

# 方法2: 手動提供驗證集 (推薦)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val)
)
```

> [!WARNING]
> **時間序列的驗證策略**:
> - ❌ 不要使用`shuffle=True`
> - ✅ 使用時間順序分割
> - ✅ 驗證集應該在訓練集之後

```python
# 正確的時間序列分割
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    # 訓練模型
```

**4. shuffle (是否打亂數據)**

```python
# 時間序列：必須設為False！
history = model.fit(
    X_train, y_train,
    shuffle=False    # 保持時間順序
)
```

**為什麼時間序列不能shuffle？**
- 時間序列有時間依賴性
- 打亂會破壞時間關係
- 可能導致數據洩漏（用未來預測過去）

**5. verbose (顯示模式)**

```python
# verbose=0: 不顯示
# verbose=1: 顯示進度條
# verbose=2: 每個epoch顯示一行
history = model.fit(X_train, y_train, verbose=1)
```

---

### 3.3 Callbacks使用

Callbacks是在訓練過程中自動執行的函數，用於監控和控制訓練。

#### 3.3.1 EarlyStopping (提前停止)

**功能**: 當驗證損失不再改善時自動停止訓練

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',         # 監控的指標
    patience=20,                # 容忍多少個epoch沒改善
    restore_best_weights=True,  # 恢復最佳權重
    verbose=1                   # 顯示訊息
)

history = model.fit(
    X_train, y_train,
    epochs=200,
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)
```

**參數說明**:
- `monitor`: 監控的指標（`'val_loss'`, `'val_mae'`, `'loss'`等）
- `patience`: 容忍epoch數（建議10-30）
- `min_delta`: 最小改善量（預設0.0）
- `restore_best_weights`: 是否恢復最佳模型（強烈建議True）

**實用範例**:
```python
# 標準配置
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    min_delta=0.0001,
    restore_best_weights=True,
    verbose=1
)
```

#### 3.3.2 ModelCheckpoint (模型保存)

**功能**: 訓練過程中自動保存最佳模型

```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    filepath='best_model.h5',   # 保存路徑
    monitor='val_loss',         # 監控指標
    save_best_only=True,        # 只保存最佳
    save_weights_only=False,    # 保存完整模型
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint]
)
```

**進階用法** - 按epoch命名：
```python
checkpoint = ModelCheckpoint(
    filepath='model_epoch{epoch:02d}_loss{val_loss:.4f}.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
```

#### 3.3.3 ReduceLROnPlateau (動態調整學習率)

**功能**: 當訓練停滯時降低學習率

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',         # 監控指標
    factor=0.5,                 # 降低倍數 (新lr = 舊lr * factor)
    patience=10,                # 多少epoch後降低
    min_lr=1e-7,                # 最小學習率
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[reduce_lr]
)
```

**工作原理**:
```
初始學習率: 0.001
10 epochs沒改善 → 0.001 * 0.5 = 0.0005
再10 epochs沒改善 → 0.0005 * 0.5 = 0.00025
...
直到 min_lr = 1e-7
```

#### 3.3.4 TensorBoard (訓練視覺化)

**功能**: 實時視覺化訓練過程

```python
from tensorflow.keras.callbacks import TensorBoard
import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,           # 每1個epoch記錄權重分布
    write_graph=True,
    update_freq='epoch'
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[tensorboard_callback]
)

# 啟動TensorBoard
# 在終端執行: tensorboard --logdir=logs/fit
```

#### 3.3.5 組合使用Callbacks

**推薦配置**:

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 定義callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-7,
        verbose=1
    )
]

# 訓練
history = model.fit(
    X_train, y_train,
    epochs=200,                 # 設大一點，讓EarlyStopping決定何時停止
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)
```

**Callbacks執行順序**:
```
每個epoch:
1. ReduceLROnPlateau 檢查是否需要降低學習率
2. ModelCheckpoint 檢查是否需要保存模型
3. EarlyStopping 檢查是否需要停止訓練
```

---

### 3.4 時間序列特殊考量

#### 3.4.1 數據洩漏問題

**問題**: 不當的數據處理可能導致用未來資訊預測過去

**錯誤範例**:
```python
# ❌ 錯誤：先標準化再分割
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 用全部數據fit
X_train, X_test = train_test_split(X_scaled)  # 再分割
# 問題：測試集資訊已洩漏到訓練集
```

**正確做法**:
```python
# ✅ 正確：先分割再標準化
X_train, X_test = train_test_split(X, shuffle=False)  # 時間順序分割
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 只用訓練集fit
X_test_scaled = scaler.transform(X_test)        # 測試集只transform
```

#### 3.4.2 驗證策略

**時間序列交叉驗證**:

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def time_series_cv(model_fn, X, y, n_splits=5):
    """
    時間序列交叉驗證
    
    參數:
        model_fn: 建立模型的函數
        X, y: 數據
        n_splits: 分割數
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        print(f"\n=== Fold {fold}/{n_splits} ===")
        
        # 分割數據
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 標準化
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
        
        X_val_flat = X_val.reshape(-1, X_val.shape[-1])
        X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)
        
        # 建立並訓練模型
        model = model_fn()
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        history = model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val_scaled, y_val),
            verbose=0
        )
        
        # 評估
        score = model.evaluate(X_val_scaled, y_val, verbose=0)
        scores.append(score[1])  # MAE
        print(f"Validation MAE: {score[1]:.4f}")
    
    print(f"\n平均 MAE: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    return scores

# 使用範例
def create_model():
    return Sequential([
        LSTM(64, input_shape=(timesteps, n_features)),
        Dense(1)
    ])

scores = time_series_cv(create_model, X, y, n_splits=5)
```

#### 3.4.3 序列長度的影響

**問題**: 序列越長，訓練越困難

```python
# 短序列 (timesteps < 50)
model = Sequential([
    LSTM(64, input_shape=(20, n_features)),  # 容易訓練
    Dense(1)
])

# 長序列 (timesteps > 100)
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(200, n_features)),
    LSTM(64),  # 多層幫助學習長期依賴
    Dense(1)
])
# 需要更多訓練時間和更小的學習率
```

**建議**:

| 序列長度 | 建議配置 | 原因 |
|---------|---------|------|
| < 20 步 | 單層LSTM/GRU，units=32-64 | 簡單任務 |
| 20-50 步 | 單層或雙層，units=64-128 | 標準配置 |
| 50-100 步 | 雙層，units=128-256 | 需要更強記憶 |
| > 100 步 | 多層 + Attention機制 | 超長序列困難 |

#### 3.4.4 批次大小與序列長度的關係

```python
# 記憶體估算
memory_per_sample = timesteps * n_features * 4 bytes  # float32
batch_memory = batch_size * memory_per_sample

# 根據GPU記憶體調整
if timesteps > 100:
    batch_size = 16   # 長序列用小批次
elif timesteps > 50:
    batch_size = 32
else:
    batch_size = 64   # 短序列可用大批次
```

---

### 3.5 完整訓練範例

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

# ===== 1. 數據準備 =====
# (假設已有 X_train, X_val, X_test, y_train, y_val, y_test)

# 標準化
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, X_train.shape[-1])
X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)

X_val_flat = X_val.reshape(-1, X_val.shape[-1])
X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)

# ===== 2. 建立模型 =====
model = Sequential([
    LSTM(128, return_sequences=True, 
         dropout=0.2, recurrent_dropout=0.2,
         input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

# ===== 3. 編譯模型 =====
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mae', 'RootMeanSquaredError']
)

model.summary()

# ===== 4. 設定Callbacks =====
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='best_lstm_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-7,
        verbose=1
    )
]

# ===== 5. 訓練模型 =====
print("開始訓練...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_val_scaled, y_val),
    callbacks=callbacks,
    shuffle=False,      # 時間序列不打亂
    verbose=1
)

# ===== 6. 訓練結果分析 =====
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 損失曲線
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].legend()
axes[0].set_title('Training & Validation Loss')
axes[0].grid(True)

# MAE曲線
axes[1].plot(history.history['mae'], label='Train MAE')
axes[1].plot(history.history['val_mae'], label='Val MAE')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].set_title('Training & Validation MAE')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()

# ===== 7. 最終評估 =====
# 測試集標準化
X_test_flat = X_test.reshape(-1, X_test.shape[-1])
X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

# 評估
test_results = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\n=== 測試集最終結果 ===")
print(f"Test Loss (MSE): {test_results[0]:.4f}")
print(f"Test MAE: {test_results[1]:.4f}")
print(f"Test RMSE: {test_results[2]:.4f}")

# 預測
y_pred = model.predict(X_test_scaled)

# 預測結果視覺化
plt.figure(figsize=(15, 5))
plt.plot(y_test[:200], label='Actual', marker='o', markersize=2)
plt.plot(y_pred[:200], label='Predicted', marker='x', markersize=2)
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()
plt.title('Predictions vs Actual (First 200 samples)')
plt.grid(True)
plt.savefig('predictions.png', dpi=150)
plt.show()
```

**本章小結**

恭喜！您已經掌握了RNN模型訓練的核心技能：

✅ **模型編譯**: 優化器選擇、損失函數、評估指標  
✅ **訓練配置**: epochs、batch_size、validation_split  
✅ **Callbacks**: EarlyStopping、ModelCheckpoint、ReduceLROnPlateau  
✅ **特殊考量**: 時間序列的驗證策略、數據洩漏防範  

**關鍵要點**:

| 主題 | 核心概念 | 實用建議 |
|-----|---------|---------|
| 優化器 | Adam + 梯度裁剪 | `clipnorm=1.0` |
| 批次大小 | 根據序列長度調整 | 長序列用小批次 |
| Shuffle | 時間序列必須False | 保持時間順序 |
| 驗證 | TimeSeriesSplit | 避免數據洩漏 |
| Callbacks | 組合使用 | EarlyStopping + Checkpoint + ReduceLR |

---

## 4. 模型評估與預測

### 4.1 模型評估

```python
# 評估模型
test_loss, test_mae, test_rmse = model.evaluate(X_test, y_test, verbose=0)

print(f"測試集 MSE: {test_loss:.4f}")
print(f"測試集 MAE: {test_mae:.4f}")
print(f"測試集 RMSE: {test_rmse:.4f}")
```

### 4.2 模型預測

```python
# 單次預測
y_pred = model.predict(X_test)

# 批次預測
y_pred_batch = model.predict(X_test, batch_size=32)
```

### 4.3 模型保存與載入

```python
# 保存完整模型
model.save('rnn_model.h5')

# 只保存權重
model.save_weights('rnn_weights.h5')

# 載入模型
from tensorflow.keras.models import load_model
loaded_model = load_model('rnn_model.h5')

# 載入權重
model.load_weights('rnn_weights.h5')
```

---

## 5. RNN在化工領域的應用

### 5.1 典型應用場景

| 應用類型 | 具體任務 | 模型類型 | 輸入/輸出 |
|---------|---------|---------|----------|
| 過程監控 | 批次操作品質預測 | LSTM | 多對一 |
| 故障診斷 | 設備異常檢測 | GRU | 多對一 |
| 預測控制 | 多步超前預測 | LSTM | 多對多 |
| 軟感測器 | 難測變數估計 | LSTM/GRU | 多對一 |
| RUL預測 | 剩餘壽命估計 | LSTM | 多對一 |

### 5.2 與DNN的對比

| 特性 | DNN | RNN/LSTM/GRU |
|-----|-----|--------------|
| 輸入類型 | 固定維度向量 | 變長序列 |
| 時間依賴 | 無 | 有 |
| 記憶能力 | 無 | 有 |
| 適用數據 | 靜態特徵 | 時間序列 |
| 參數量 | 較少 | 較多 |
| 訓練難度 | 較易 | 較難 |

---

## 本課程總結

### 核心要點回顧

**1. RNN基礎理論**:
- ✅ RNN通過循環連接實現序列記憶
- ✅ LSTM使用門控機制解決梯度消失
- ✅ GRU是LSTM的簡化版本，效率更高

**2. Keras實作**:
- ✅ 數據必須為3D張量: `(batch, timesteps, features)`
- ✅ 堆疊RNN層需要設定 `return_sequences=True`
- ✅ 使用dropout防止過擬合

**3. 訓練技巧**:
- ✅ 時間序列不能shuffle
- ✅ 使用梯度裁剪防止梯度爆炸
- ✅ 組合使用EarlyStopping和ModelCheckpoint

**4. 實際應用**:
- ✅ 化工過程監控與品質預測
- ✅ 設備故障診斷與RUL估計
- ✅ 多步超前預測與動態控制

### 模型選擇建議

```
數據特性決策樹:

序列長度 < 20步？
  ├─ 是 → SimpleRNN 或 GRU
  └─ 否 → 繼續

序列長度 > 100步？
  ├─ 是 → LSTM (多層)
  └─ 否 → GRU (首選)

數據量 < 1000樣本？
  ├─ 是 → GRU + 較強正則化
  └─ 否 → LSTM 或 GRU

計算資源有限？
  ├─ 是 → GRU (比LSTM快25%)
  └─ 否 → LSTM (性能稍好)
```

### 最佳實踐清單

**數據準備**:
- [ ] 先分割訓練/測試集，再標準化
- [ ] 使用滑動視窗生成序列樣本
- [ ] 檢查數據形狀是否為3D
- [ ] 時間序列不使用shuffle

**模型建立**:
- [ ] 堆疊RNN層時設定return_sequences
- [ ] 加入dropout防止過擬合
- [ ] 使用He初始化 (ReLU) 或預設初始化

**模型編譯**:
- [ ] 優化器使用Adam + 梯度裁剪 (clipnorm=1.0)
- [ ] 根據任務選擇合適的損失函數
- [ ] 添加多個評估指標監控訓練

**模型訓練**:
- [ ] 使用EarlyStopping避免過擬合
- [ ] 使用ModelCheckpoint保存最佳模型
- [ ] 使用ReduceLROnPlateau動態調整學習率
- [ ] 設定合理的batch_size (根據序列長度)

**模型評估**:
- [ ] 在獨立測試集上評估
- [ ] 視覺化預測結果
- [ ] 檢查殘差分布

### 延伸學習資源

**進階主題**:
1. Attention機制與Transformer
2. 雙向RNN (Bidirectional RNN)
3. Seq2Seq與Encoder-Decoder架構
4. 時間序列異常檢測
5. 在線學習與增量更新

**推薦閱讀**:
- Keras官方文檔: https://keras.io/
- TensorFlow時間序列教程
- 深度學習專項課程 (Andrew Ng)

---

