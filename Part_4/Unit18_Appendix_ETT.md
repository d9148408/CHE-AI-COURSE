# Unit18｜LSTM 時間序列預測：化工製程動態建模

**Part 4 - 深度學習進階應用**

> **教學目標**：本單元深入探討時間序列預測在化工製程中的應用，從傳統統計方法到深度學習LSTM/GRU模型，建立完整的動態系統預測能力。

## 📚 本單元核心內容

### 學習目標

1. **理解時間序列特性**：掌握自相關性、季節性、趨勢等時間序列基本概念
2. **化工製程動態建模**：理解製程慣性、時間延遲、多變數耦合等工業特性
3. **LSTM/GRU 架構原理**：深入理解循環神經網路的記憶機制與梯度傳播
4. **多步預測策略**：掌握 Recursive、Direct、Seq2Seq 等預測方法
5. **模型評估與對比**：建立完整的 baseline 對比與 rolling backtest 評估體系
6. **工業部署考量**：在線預測、模型更新、異常檢測等實務議題

### 數據集介紹：鍋爐運行數據

**數據來源**：工業鍋爐多變數時間序列數據  
**採樣頻率**：5秒（降採樣至1分鐘）  
**總樣本數**：約 50,000+ 時間點  
**主要變數**：

| 變數名稱 | 物理意義 | 單位 | 控制目標 |
|---------|---------|------|---------|
| `TE_8332A.AV_0#` | 鍋爐出口蒸汽溫度 | °C | 預測目標 |
| `ZZQBCHLL.AV_0#` | 主蒸汽流量 | t/h | 負荷指標 |
| `PTCA_8324.AV_0#` | 爐膛壓力 | kPa | 燃燒狀態 |
| `AIR_8301A.AV_0#` | 一次風量 | m³/h | 氧氣供應 |
| `AIR_8301B.AV_0#` | 二次風量 | m³/h | 燃燒控制 |
| `FT_8301.AV_0#` | 燃料流量1 | kg/h | 熱量輸入 |
| `FT_8302.AV_0#` | 燃料流量2 | kg/h | 熱量輸入 |
| `TV_8329ZC.AV_0#` | 減溫水流量 | t/h | 溫度調節 |

**化工製程特性**：
- **熱容效應**：鍋爐水容積大，溫度響應緩慢（時間常數 ~10-30分鐘）
- **多變數耦合**：燃料-風量-溫度-壓力相互影響
- **非線性動態**：不同負荷下的動態特性不同
- **時間延遲**：控制動作到溫度變化有明顯滯後

---

## 第一章：時間序列基礎理論

### 1.1 什麼是時間序列？

**定義**：按時間順序排列的數據點序列 $\{y_1, y_2, \ldots, y_T\}$

**與傳統機器學習的差異**：

| 特性 | 傳統 ML (i.i.d.) | 時間序列 |
|-----|----------------|---------|
| 樣本獨立性 | ✓ 樣本獨立同分布 | ❌ 樣本有時間依賴 |
| 順序重要性 | ❌ 順序可打亂 | ✓ 順序不可改變 |
| 訓練/測試劃分 | 隨機劃分 | 時間順序劃分 |
| 預測目標 | 單一預測 | 序列預測 |
| 特徵工程 | 當前時刻特徵 | 歷史特徵（lag features）|

### 1.2 時間序列的關鍵概念

#### 自相關性（Autocorrelation）

**定義**：時間序列與其自身滯後版本的相關性

$$
\text{ACF}(k) = \frac{\text{Cov}(y_t, y_{t-k})}{\text{Var}(y_t)} = \frac{\mathbb{E}[(y_t - \mu)(y_{t-k} - \mu)]}{\sigma^2}
$$

其中：
- $k$：滯後階數（lag）
- $\mu$：序列均值
- $\sigma^2$：序列方差

**物理意義（鍋爐系統）**：
- ACF(1) 高：溫度相鄰時刻高度相關（熱慣性）
- ACF(k) 隨 k 衰減：記憶逐漸消退
- ACF(k) 截尾：可能存在週期性擾動

#### 偏自相關性（Partial Autocorrelation）

**定義**：剔除中間時刻影響後的直接相關性

$$
\text{PACF}(k) = \text{Corr}(y_t - \hat{y}_t, y_{t-k} - \hat{y}_{t-k})
$$

其中 $\hat{y}_t$ 是由 $y_{t-1}, \ldots, y_{t-k+1}$ 線性預測的值。

**應用**：確定 AR 模型的階數

#### 平穩性（Stationarity）

**定義**：統計特性不隨時間變化

**嚴格平穩性**（Strict Stationarity）：
$$
P(y_{t_1}, \ldots, y_{t_n}) = P(y_{t_1+\tau}, \ldots, y_{t_n+\tau}), \quad \forall \tau, n
$$

**弱平穩性**（Weak Stationarity）：
1. 均值恆定：$\mathbb{E}[y_t] = \mu, \quad \forall t$
2. 方差恆定：$\text{Var}(y_t) = \sigma^2, \quad \forall t$
3. 自協方差僅依賴滯後：$\text{Cov}(y_t, y_{t-k}) = \gamma_k, \quad \forall t$

**檢驗方法**：
- **ADF 檢驗（Augmented Dickey-Fuller）**：檢驗單位根
  $$
  \Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \epsilon_t
  $$
  - $H_0$: $\gamma = 0$ (非平穩)
  - $H_1$: $\gamma < 0$ (平穩)

- **KPSS 檢驗**：檢驗平穩性（與 ADF 相反）

**化工製程的非平穩性**：
- **趨勢性**：設備老化、催化劑失活導致性能漂移
- **週期性**：白天/夜間負荷變化、季節性原料差異
- **結構性突變**：工藝改進、設備更換

### 1.3 化工製程的時間序列特性

#### 動態系統的時間尺度

化工製程涉及多個時間尺度：

| 過程 | 時間尺度 | 數學描述 | 控制策略 |
|-----|---------|---------|---------|
| 快速化學反應 | 毫秒-秒 | $\frac{dr}{dt} = k(T) \cdot c^n$ | 反應器溫度控制 |
| 熱傳導 | 秒-分鐘 | $\frac{\partial T}{\partial t} = \alpha \nabla^2 T$ | 換熱器控制 |
| 物料累積 | 分鐘-小時 | $\frac{dM}{dt} = F_{in} - F_{out}$ | 液位/壓力控制 |
| 催化劑失活 | 小時-天 | $\frac{da}{dt} = -k_d \cdot a$ | 再生週期優化 |

**時間常數（Time Constant）**：

一階系統：
$$
\tau \frac{dy}{dt} + y = K u(t)
$$

- $\tau$：時間常數（達到 63.2% 穩態值所需時間）
- $K$：增益
- $u(t)$：輸入擾動

**鍋爐溫度響應**：典型時間常數 $\tau \approx 15$ 分鐘

#### 時間延遲（Dead Time）

**定義

**定義**：輸入變化到輸出響應之間的純延遲

$$
y(t) = f(u(t - \theta))
$$

其中 $\theta$ 是死區時間（dead time）。

**化工實例**：
- **管道輸送**：物料從投入點到測量點的時間
- **分析儀表**：樣品採集-傳輸-分析的時間
- **鍋爐燃燒**：燃料投入到熱量傳遞至水的時間（~5-10分鐘）

**模型表示（FOPDT）**：

一階加純滯後模型（First-Order Plus Dead Time）：
$$
G(s) = \frac{K e^{-\theta s}}{\tau s + 1}
$$

#### 多變數耦合與因果關係

**Granger 因果檢驗**：

變數 $X$ 是否對 $Y$ 有預測能力：

**限制模型**（僅用 $Y$ 的歷史）：
$$
Y_t = \alpha_0 + \sum_{i=1}^{p} \alpha_i Y_{t-i} + \epsilon_t
$$

**完整模型**（加入 $X$ 的歷史）：
$$
Y_t = \alpha_0 + \sum_{i=1}^{p} \alpha_i Y_{t-i} + \sum_{j=1}^{q} \beta_j X_{t-j} + \eta_t
$$

**檢驗**：
- $H_0$: $\beta_1 = \cdots = \beta_q = 0$ (X 不影響 Y)
- 使用 F 檢驗比較兩模型的 RSS

**鍋爐系統的因果鏈**：
```
燃料流量 → (5-10分鐘) → 爐膛溫度 → (10-20分鐘) → 蒸汽溫度
    ↓                         ↑
  風量調節 → (3-5分鐘) → 燃燒效率
```

---

## 第二章：傳統時間序列模型

### 2.1 ARIMA 模型族

#### AR (AutoRegressive) 模型

**定義**：當前值由過去值線性組合

$$
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \epsilon_t
$$

其中：
- $p$：自回歸階數
- $\phi_i$：自回歸係數
- $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$：白噪聲

**平穩性條件**：特徵方程的根在單位圓外
$$
1 - \phi_1 z - \phi_2 z^2 - \cdots - \phi_p z^p = 0, \quad |z| > 1
$$

#### MA (Moving Average) 模型

**定義**：當前值由過去誤差線性組合

$$
y_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \cdots + \theta_q \epsilon_{t-q}
$$

其中：
- $q$：移動平均階數
- $\theta_i$：移動平均係數

**可逆性條件**：特徵方程的根在單位圓外

#### ARIMA(p, d, q) 模型

**定義**：結合 AR、差分（I）、MA

$$
\phi(B)(1-B)^d y_t = \theta(B) \epsilon_t
$$

其中：
- $B$：後移算子（$B y_t = y_{t-1}$）
- $d$：差分階數
- $\phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p$
- $\theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$

**Box-Jenkins 建模流程**：
1. **識別（Identification）**：
   - 繪製 ACF/PACF 圖
   - ADF 檢驗平穩性
   - 確定 $(p, d, q)$ 階數

2. **估計（Estimation）**：
   - 最大似然估計（MLE）
   - 最小二乘法（OLS）

3. **診斷（Diagnostic）**：
   - 殘差白噪聲檢驗（Ljung-Box）
   - 殘差正態性檢驗（Jarque-Bera）

4. **預測（Forecasting）**：
   - 點預測：$\hat{y}_{T+h|T}$
   - 區間預測：$\hat{y}_{T+h|T} \pm z_{\alpha/2} \cdot \sigma_h$

#### SARIMA 季節性模型

**定義**：ARIMA(p,d,q)×(P,D,Q)s

$$
\phi(B) \Phi(B^s) (1-B)^d (1-B^s)^D y_t = \theta(B) \Theta(B^s) \epsilon_t
$$

其中 $s$ 是季節週期（如 12 個月、24 小時）。

**化工應用**：
- 日週期：白天/夜間負荷變化（s=24小時）
- 週週期：工作日/週末差異（s=7天）
- 年週期：季節性原料差異（s=12月）

### 2.2 多變數模型：VAR

**向量自回歸模型（Vector AutoRegression）**：

$$
\mathbf{y}_t = \mathbf{c} + \mathbf{\Phi}_1 \mathbf{y}_{t-1} + \cdots + \mathbf{\Phi}_p \mathbf{y}_{t-p} + \mathbf{\epsilon}_t
$$

其中：
- $\mathbf{y}_t \in \mathbb{R}^m$：m 個變數
- $\mathbf{\Phi}_i \in \mathbb{R}^{m \times m}$：係數矩陣
- $\mathbf{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{\Sigma})$

**脈衝響應分析（Impulse Response Function）**：

對變數 $j$ 施加單位衝擊，觀察變數 $i$ 的動態響應：
$$
\text{IRF}_{ij}(h) = \frac{\partial y_{i,t+h}}{\partial \epsilon_{j,t}}
$$

**鍋爐系統的 VAR 應用**：
- 燃料流量 → 爐膛壓力 → 蒸汽溫度
- 風量 → 氧含量 → 燃燒效率
- 減溫水 → 蒸汽溫度（直接快速）

### 2.3 傳統方法的局限性

**線性假設的問題**：
- 化工製程常有非線性（如 Arrhenius 反應速率）
- 不同操作區域的動態特性不同

**長期依賴建模困難**：
- AR(p) 需要很大的 $p$ 才能捕捉長期記憶
- 參數數量隨 $p$ 線性增長，容易過擬合

**外生變數處理複雜**：
- ARIMAX 需要手動選擇滯後階數
- 多變數交互作用難以建模

**無法自動特徵提取**：
- 需要領域專家設計滯後特徵
- 無法自動發現複雜模式

---

## 第三章：循環神經網路基礎

### 3.1 簡單 RNN 的原理

#### 基本架構

**數學形式**：

$$
\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)
$$
$$
\mathbf{y}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y
$$

其中：
- $\mathbf{h}_t \in \mathbb{R}^h$：隱藏狀態（記憶）
- $\mathbf{x}_t \in \mathbb{R}^d$：輸入
- $\mathbf{y}_t \in \mathbb{R}^k$：輸出
- $\mathbf{W}_{hh}, \mathbf{W}_{xh}, \mathbf{W}_{hy}$：權重矩陣

**時間展開（Unrolling）**：

```
x_1 → [RNN] → h_1 → y_1
x_2 → [RNN] → h_2 → y_2
        ↑       ↑
      h_1     h_1
```

#### BPTT (Backpropagation Through Time)

**損失函數**：
$$
\mathcal{L} = \sum_{t=1}^{T} \ell(y_t, \hat{y}_t)
$$

**梯度計算**（鏈式法則）：
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^{T} \sum_{k=1}^{t} \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} \frac{\partial \mathbf{h}_k}{\partial \mathbf{W}_{hh}}
$$

**梯度傳播**：
$$
\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod_{i=k+1}^{t} \frac{\partial \mathbf{h}_i}{\partial \mathbf{h}_{i-1}} = \prod_{i=k+1}^{t} \mathbf{W}_{hh}^T \cdot \text{diag}[\tanh'(\cdot)]
$$

#### 梯度消失/爆炸問題

**梯度消失**：

當 $|\lambda_{\max}(\mathbf{W}_{hh})| < 1$ 時：
$$
\left\| \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} \right\| \leq \left\| \mathbf{W}_{hh} \right\|^{t-k} \|\text{diag}[\tanh'(\cdot)]\|^{t-k} \to 0
$$

**梯度爆炸**：

當 $|\lambda_{\max}(\mathbf{W}_{hh})| > 1$ 時，梯度指數增長。

**解決方案**：
- **Gradient Clipping**：
  $$
  \mathbf{g} \leftarrow \begin{cases}
  \mathbf{g} & \text{if } \|\mathbf{g}\| \leq \theta \\
  \frac{\theta}{\|\mathbf{g}\|} \mathbf{g} & \text{otherwise}
  \end{cases}
  $$

- **更好的架構**：LSTM/GRU

### 3.2 LSTM (Long Short-Term Memory)

#### 架構設計動機

**問題**：簡單 RNN 無法學習長期依賴

**解決思路**：
- 引入 **Cell State** $\mathbf{c}_t$ 作為"高速公路"
- 用 **Gate** 機制控制信息流動
- 線性路徑避免梯度消失

#### LSTM 數學公式

**遺忘門（Forget Gate）**：決定丟棄多少舊信息
$$
\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)
$$

**輸入門（Input Gate）**：決定添加多少新信息
$$
\mathbf{i}_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)
$$
$$
\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)
$$

**Cell State 更新**：
$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
$$

**輸出門（Output Gate）**：決定輸出什麼
$$
\mathbf{o}_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)
$$
$$
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
$$

其中：
- $\sigma$：Sigmoid 函數（輸出 0-1，作為門控）
- $\odot$：逐元素乘法（Hadamard product）
- $[\cdot, \cdot]$：拼接操作

**參數總數**：
$$
\text{Params} = 4 \times (h \times (h + d) + h)
$$

其中 $h$ 是隱藏層大小，$d$ 是輸入維度。

#### LSTM 的記憶機制

**Cell State 梯度**：
$$
\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \mathbf{f}_t
$$

**關鍵優勢**：
- 當 $\mathbf{f}_t \approx 1$ 時，梯度幾乎無損傳播
- 不經過多次矩陣乘法，避免梯度消失
- 可以選擇性記憶（遺忘門控制）

**物理類比（鍋爐系統）**：
- **Cell State**：系統累積的熱量
- **遺忘門**：熱損失（散熱、輻射）
- **輸入門**：新增熱量（燃料燃燒）
- **輸出門**：可測量的溫度（輸出到蒸汽）

### 3.3 GRU (Gated Recurrent Unit)

#### 簡化的 Gate 結構

GRU 將 LSTM 的三個門簡化為兩個：

**重置門（Reset Gate）**：
$$
\mathbf{r}_t = \sigma(\mathbf{W}_r \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r)
$$

**更新門（Update Gate）**：
$$
\mathbf{z}_t = \sigma(\mathbf{W}_z \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z)
$$

**候選隱藏狀態**：
$$
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h \cdot [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_h)
$$

**隱藏狀態更新**：
$$
\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
$$

**優勢**：
- 參數更少（約為 LSTM 的 75%）
- 計算更快
- 在許多任務上性能接近 LSTM

**何時選擇 GRU vs LSTM**：
- GRU：數據量較小、計算資源有限、序列較短
- LSTM：數據量大、需要精細記憶控制、序列較長

---

## 第四章：序列預測建模策略

### 4.1 數據準備：從序列到監督學習

#### Sliding Window 方法

**原始序列**：
$$
[y_1, y_2, y_3, \ldots, y_T]
$$

**轉換為監督數據**（窗口大小 = L）：

| 輸入特徵 | 目標 |
|---------|-----|
| $[y_1, y_2, \ldots, y_L]$ | $y_{L+1}$ |
| $[y_2, y_3, \ldots, y_{L+1}]$ | $y_{L+2}$ |
| $\vdots$ | $\vdots$ |
| $[y_{T-L}, \ldots, y_{T-1}]$ | $y_T$ |

**代碼實現**：
```python
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)
```

#### 多變數輸入

**特徵矩陣**（N 個變數）：
$$
\mathbf{X}_t = \begin{bmatrix}
x_{1,t-L+1} & x_{2,t-L+1} & \cdots & x_{N,t-L+1} \\
x_{1,t-L+2} & x_{2,t-L+2} & \cdots & x_{N,t-L+2} \\
\vdots & \vdots & \ddots & \vdots \\
x_{1,t} & x_{2,t} & \cdots & x_{N,t}
\end{bmatrix} \in \mathbb{R}^{L \times N}
$$

**LSTM 輸入形狀**：`(batch_size, time_steps, features)`

**鍋爐系統示例**：
- Time steps (L) = 30（過去30分鐘）
- Features (N) = 8（溫度、流量、壓力等）
- Output = 1（預測未來1分鐘溫度）

### 4.2 單步預測 vs 多步預測

#### 單步預測（One-Step Ahead）

**目標**：預測下一個時刻 $\hat{y}_{t+1}$

$$
\hat{y}_{t+1} = f_\theta(y_t, y_{t-1}, \ldots, y_{t-L+1})
$$

**優點**：
- 模型簡單
- 誤差不累積
- 適合實時控制

**缺點**：
- 只能看一步，無法提前規劃

#### 多步預測（Multi-Step）

**目標**：預測未來 H 個時刻 $\hat{y}_{t+1}, \ldots, \hat{y}_{t+H}$

**策略 1：Recursive（遞歸）**：

```python
def predict_recursive(model, x_init, H):
    predictions = []
    x = x_init.copy()
    for h in range(H):
        y_pred = model.predict(x)
        predictions.append(y_pred)
        x = np.roll(x, -1)  # 移除最舊，加入預測值
        x[-1] = y_pred
    return predictions
```

**數學形式**：
$$
\hat{y}_{t+1} = f(y_t, \ldots, y_{t-L+1})
$$
$$
\hat{y}_{t+2} = f(\hat{y}_{t+1}, y_t, \ldots, y_{t-L+2})
$$
$$
\vdots
$$

**優點**：只需訓練一個模型  
**缺點**：誤差累積嚴重

**策略 2：Direct（直接）**：

為每個 horizon 訓練獨立模型：
$$
\hat{y}_{t+h} = f_h(y_t, \ldots, y_{t-L+1}), \quad h = 1, \ldots, H
$$

**優點**：誤差不累積  
**缺點**：需要 H 個模型，訓練成本高

**策略 3：Seq2Seq（序列到序列）**：

**Encoder-Decoder 架構**：
```
Encoder: [x_1, ..., x_L] → context vector c
Decoder: c → [y_1, ..., y_H]
```

**數學形式**：
$$
\mathbf{c} = \text{Encoder}(\mathbf{x}_{t-L+1:t})
$$
$$
[\hat{y}_{t+1}, \ldots, \hat{y}_{t+H}] = \text{Decoder}(\mathbf{c})
$$

**優點**：
- 一次性輸出整個序列
- 考慮輸出之間的相關性
- 可加入 Attention 機制

**策略對比表**：

| 策略 | 模型數量 | 誤差累積 | 訓練複雜度 | 適用場景 |
|-----|---------|---------|-----------|---------|
| Recursive | 1 | 高 | 低 | H 較小，快速部署 |
| Direct | H | 無 | 高 | H 較大，追求精度 |
| Seq2Seq | 1 | 低 | 中 | 序列相關性強 |

### 4.3 特徵工程

#### Lag Features（滯後特徵）

**目標變數的滯後**：
$$
y_t, y_{t-1}, y_{t-2}, \ldots, y_{t-L}
$$

**外生變數的滯後**：
$$
x_{1,t}, x_{1,t-1}, \ldots, x_{1,t-L_1}
$$
$$
x_{2,t}, x_{2,t-1}, \ldots, x_{2,t-L_2}
$$

**滯後階數選擇**：
- ACF/PACF 分析
- 領域知識（時間常數、延遲）
- 交叉驗證

#### Rolling Statistics（滾動統計）

**移動平均（MA）**：
$$
\text{MA}_t^{(w)} = \frac{1}{w}\sum_{i=0}^{w-1} y_{t-i}
$$

**移動標準差（Rolling Std）**：
$$
\text{Std}_t^{(w)} = \sqrt{\frac{1}{w}\sum_{i=0}^{w-1} (y_{t-i} - \text{MA}_t^{(w)})^2}
$$

**指數加權移動平均（EWMA）**：
$$
\text{EWMA}_t = \alpha y_t + (1-\alpha) \text{EWMA}_{t-1}
$$

**應用**：
- 捕捉趨勢（MA）
- 識別波動（Std）
- 平滑噪聲（EWMA）

#### 時間特徵（Temporal Features）

**週期性編碼**：

對於週期 $T$（如24小時）：
$$
\text{sin\_hour} = \sin\left(\frac{2\pi \cdot \text{hour}}{T}\right)
$$
$$
\text{cos\_hour} = \cos\left(\frac{2\pi \cdot \text{hour}}{T}\right)
$$

**為什麼用 sin/cos**：
- 保持週期性連續（23時到0時是連續的）
- 避免 one-hot 編碼的稀疏性

**其他時間特徵**：
- 工作日/週末（binary）
- 班次（早/中/晚）
- 是否節假日

#### Domain-Specific Features（領域特徵）

**鍋爐系統的工程特徵**：

**過量空氣係數**：
$$
\alpha = \frac{\text{實際空氣量}}{\text{理論空氣量}} = f(\text{風量}, \text{燃料量})
$$

**熱效率指標**：
$$
\eta = \frac{Q_{\text{useful}}}{Q_{\text{input}}} = f(\text{溫度}, \text{流量}, \text{壓力})
$$

**負荷率**：
$$
\text{Load} = \frac{\text{當前蒸汽流量}}{\text{額定蒸汽流量}} \times 100\%
$$

---
## 第�?章�?模�?評估?��?�?

### 5.1 ?��?序�?專用評估?��?

#### 點�?測誤�?

**MAE (Mean Absolute Error)**�?
$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

**RMSE (Root Mean Squared Error)**�?
$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

**MAPE (Mean Absolute Percentage Error)**�?
$$
\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|
$$

**sMAPE (Symmetric MAPE)**�?
$$
\text{sMAPE} = \frac{100\%}{n}\sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}
$$

**?��??��?**�?
- MAE：�??�常?��??��?，解?�性強
- RMSE：懲罰大誤差，常?�於?��?
- MAPE：相對誤差�?�?$y_i \approx 0$ ?��?穩�?
- sMAPE：解�?MAPE ?��?稱性�?�?

#### ?��??�測準確??

**定義**：�?測�??�方?�是?�正�?

$$
\text{DA} = \frac{1}{n-1}\sum_{t=2}^{n} \mathbb{1}\left\{\text{sign}(\Delta y_t) = \text{sign}(\Delta \hat{y}_t)\right\}
$$

?�中 $\Delta y_t = y_t - y_{t-1}$

**?�用**�?
- ?�制決�?：�?�??�溫?��?
- 趨勢?�警：�???下�?趨勢

#### ?�?��?測�?�?

**Coverage（�??��?�?*�?
$$
\text{Coverage} = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}\{y_i \in [\hat{y}_i^L, \hat{y}_i^U]\}
$$

?�想??= 置信水平（�? 95%�?

**Interval Width（�??�寬度�?**�?
$$
\text{Width} = \frac{1}{n}\sum_{i=1}^{n} (\hat{y}_i^U - \hat{y}_i^L)
$$

越�?越好（在保�?覆�??��??��?下�?

**Winkler Score**�?
$$
\text{WS} = \frac{1}{n}\sum_{i=1}^{n} \left[\text{Width}_i + \frac{2}{\alpha}(\hat{y}_i^L - y_i)\mathbb{1}\{y_i < \hat{y}_i^L\} + \frac{2}{\alpha}(y_i - \hat{y}_i^U)\mathbb{1}\{y_i > \hat{y}_i^U}\right]
$$

綜�??�慮寬度?��??��?

### 5.2 ?��?序�?交�?驗�?

#### ?�統 K-Fold ?��?�?

**?�誤?��?**：隨機�???
```
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Fold 1: Train=[1,3,5,7,9], Test=[2,4,6,8,10]  ??
```

**?��?**�?
- ?��??��??��?
- ?�未來�?測�??��?data leakage�?
- 高估模�??�能

#### Time Series Split

**�?��?��?**：�??��??��?�?
```
Fold 1: Train=[1,2,3,4], Test=[5,6]
Fold 2: Train=[1,2,3,4,5,6], Test=[7,8]
Fold 3: Train=[1,2,3,4,5,6,7,8], Test=[9,10]
```

**�?��實現**�?
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # 訓練?��?�?
```

#### Rolling Window Validation

**?��?窗口大�?**�?
```
Window = 1000 samples
Fold 1: Train=[1:1000], Test=[1001:1100]
Fold 2: Train=[101:1100], Test=[1101:1200]
Fold 3: Train=[201:1200], Test=[1201:1300]
```

**?��?**�?
- 模擬實�??�署（固定�?練�?大�?�?
- 評估模�??�新?��?上�?穩�???

**Expanding Window**�?
```
Fold 1: Train=[1:1000], Test=[1001:1100]
Fold 2: Train=[1:1100], Test=[1101:1200]
Fold 3: Train=[1:1200], Test=[1201:1300]
```

**?��?建議**�?
- Rolling：�?念�?移嚴?��?如設?�老�?�?
- Expanding：數?��?佈穩�?

### 5.3 Baseline 對�??��?要�?

**?��?麼�?�?Baseline�?*

1. **檢�?模�??�否?�正學�?**�?
   - 簡單模�??�能已�?很好
   - 複�?模�??��??�是?�值�?

2. **?�解?��???��**�?
   - Baseline 很差 ???��??�難
   - Baseline 很好 ??簡單模�?主�?

3. **?�現?��??��?**�?
   - ?�?�模?�都很差 ???��?質�??��?
   - 複�?模�?不�?簡單模�? ???�擬??

**常用 Baseline**�?

| Baseline | ?�測?��? | ?�用?�景 |
|----------|---------|---------|
| Persistence | $\hat{y}_{t+1} = y_t$ | 緩慢變�?系統 |
| Moving Average | $\hat{y}_{t+1} = \frac{1}{w}\sum_{i=1}^{w} y_{t-i+1}$ | 平穩序�? |
| Linear Regression | 線性�?�?lag features | 線性�?�?|
| Random Forest | 樹模??| ?��??��?�?|
| MLP | 淺層神�?網路 | ??LSTM 對�? |

**?��?系統??Baseline 表現**�?
- Persistence：RMSE ~ 2-3°C（溫度�??�緩?��?
- Linear Regression：RMSE ~ 1-1.5°C
- Random Forest：RMSE ~ 0.8-1.2°C
- LSTM ?��?�? 0.8°C

---

## 第六章�?LSTM 模�?實現?��?�?

### 6.1 TensorFlow/Keras 實現

#### ?�本 LSTM 模�?

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_lstm_model(input_shape, lstm_units=64, dense_units=32):
    """
    input_shape: (time_steps, n_features)
    """
    model = keras.Sequential([
        layers.LSTM(lstm_units, 
                   return_sequences=False,  # ?�步?�測
                   input_shape=input_shape),
        layers.Dropout(0.2),
        layers.Dense(dense_units, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)  # ?��??��?�?
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model
```

#### ?��? LSTM（Stacked LSTM�?

```python
model = keras.Sequential([
    layers.LSTM(128, return_sequences=True, input_shape=input_shape),
    layers.Dropout(0.2),
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(32, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])
```

**return_sequences**�?
- `True`：輸?��??��??�步?�隱?��??��??�於?��?LSTM�?
- `False`：只輸出?�後�??�步（用?��?測�?

#### ?��? LSTM（Bidirectional�?

```python
model = keras.Sequential([
    layers.Bidirectional(
        layers.LSTM(64, return_sequences=False),
        input_shape=input_shape
    ),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
```

**?�用?�景**�?
- ?��??��?（可?�到?��??��?�?
- ?�常檢測（�?後�??��?要�?
- **不適??*：實?��?測�??��??�到?��?�?

### 6.2 訓練策略

#### Early Stopping

```python
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)
```

#### Learning Rate Scheduling

**?�數衰�?**�?
```python
def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)
```

**ReduceLROnPlateau**�?
```python
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)
```

#### Model Checkpoint

```python
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
```

### 6.3 超參數調優

**關鍵超參數**：

| 超參數 | 範圍 | 影響 |
|-------|------|------|
| `lstm_units` | 32-256 | 模型容量 |
| `num_layers` | 1-3 | 抽象層次 |
| `dropout_rate` | 0.1-0.5 | 正則化強度 |
| `learning_rate` | 1e-4 ~ 1e-2 | 收斂速度 |
| `batch_size` | 16-128 | 訓練穩定性 |
| `window_size` | 10-60 | 歷史信息量 |

**調優策略**：

1. **Grid Search**：窮?��?�?
   ```python
   from sklearn.model_selection import ParameterGrid
   
   param_grid = {
       'lstm_units': [32, 64, 128],
       'dropout_rate': [0.2, 0.3],
       'learning_rate': [0.001, 0.0001]
   }
   
   best_score = float('inf')
   for params in ParameterGrid(param_grid):
       model = build_lstm_model(**params)
       # 訓練和評估
       score = evaluate(model, X_val, y_val)
       if score < best_score:
           best_score = score
           best_params = params
   ```

2. **Random Search**：隨機採樣
   ```python
   from sklearn.model_selection import RandomizedSearchCV
   
   # 適合高維空間
   ```

3. **Bayesian Optimization**：
   ```python
   from keras_tuner import BayesianOptimization
   
   def build_model(hp):
       model = keras.Sequential()
       model.add(layers.LSTM(
           hp.Int('units', 32, 256, step=32),
           input_shape=input_shape
       ))
       model.add(layers.Dropout(
           hp.Float('dropout', 0.1, 0.5, step=0.1)
       ))
       model.add(layers.Dense(1))
       model.compile(
           optimizer=keras.optimizers.Adam(
               hp.Float('lr', 1e-4, 1e-2, sampling='log')
           ),
           loss='mse'
       )
       return model
   
   tuner = BayesianOptimization(
       build_model,
       objective='val_loss',
       max_trials=20
   )
   
   tuner.search(X_train, y_train, validation_data=(X_val, y_val))
   ```

### 6.4 過擬合診斷與預防

**過擬合特徵**：
- 訓練誤差持續下降，驗證誤差上升
- 模型在訓練集表現完美，測試集很差
- 預測曲線過度擬合噪聲

**預防措施**：

1. **增加數據量**：
   - 收集更多歷史數據
   - 數據增強（加噪聲、時間扭曲）

2. **正則化**：
   - L2 正則化（權重衰減）
     ```python
     layers.LSTM(64, kernel_regularizer=keras.regularizers.l2(0.01))
     ```
   - Dropout：隨機丟棄神經元
   - Recurrent Dropout：
     ```python
     layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)
     ```

3. **簡化模型**：
   - 減少 LSTM 單元數
   - 減少層數
   - 縮短窗口大小

4. **Early Stopping**：
   - 驗證誤差不再減小時停止

5. **Batch Normalization**：
   ```python
   model = keras.Sequential([
       layers.LSTM(64, return_sequences=True),
       layers.BatchNormalization(),
       layers.LSTM(32),
       layers.Dense(1)
   ])
   ```

---

## 第七章：工業部署與在線預測

### 7.1 模型部署架構

**離線訓練 vs 在線預測**：

```
[歷史數據] → [特徵工程] → [模型訓練] → [模型導出]
                                              ↓
[實時數據] → [特徵提取] → [模型推理] → [預測結果] → [控制決策]
```

**關鍵組件**：

1. **數據管道（Data Pipeline）**：
   ```python
   class DataPipeline:
       def __init__(self, scaler, window_size):
           self.scaler = scaler
           self.window_size = window_size
           self.buffer = deque(maxlen=window_size)
       
       def update(self, new_data):
           self.buffer.append(new_data)
       
       def get_features(self):
           if len(self.buffer) < self.window_size:
               return None
           X = np.array(self.buffer).reshape(1, self.window_size, -1)
           return self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
   ```

2. **模型服務（Model Serving）**：
   - TensorFlow Serving
   - ONNX Runtime
   - Flask/FastAPI REST API

3. **預測緩存（Prediction Cache）**：
   - 避免重複計算
   - 降低延遲

### 7.2 持續學習與模型更新

**概念漂移（Concept Drift）**：

數據分佈隨時間變化：
$$
P_t(X, y) \neq P_{t+\Delta t}(X, y)
$$

**檢測方法**：

1. **統計檢驗**：
   ```python
   from scipy.stats import ks_2samp
   
   # Kolmogorov-Smirnov test
   stat, p_value = ks_2samp(recent_errors, historical_errors)
   if p_value < 0.05:
       print("Drift detected! Retrain model.")
   ```

2. **性能監控**：
2. **性能監控**：
   ```python
   rolling_rmse = []
   window = 100
   for i in range(len(predictions)):
       if i >= window:
           rmse = np.sqrt(mean_squared_error(
               actuals[i-window:i], 
               predictions[i-window:i]
           ))
           rolling_rmse.append(rmse)
           
           if rmse > threshold:
               trigger_retrain()
   ```

**更新策略**：

1. **定期重訓（Periodic Retraining）**：
   - 每週/每月重新訓練
   - 使用最新數據

2. **觸發式重訓（Triggered Retraining）**：
   - 性能下降超過閾值
   - 檢測到漂移

3. **增量學習（Incremental Learning）**：
   - 在線更新權重（需特殊架構）
   - 適合資源受限場景

### 7.3 異常檢測與預警

**預測區間（Prediction Interval）**：

使用 Quantile Regression 或 Monte Carlo Dropout：

```python
# Monte Carlo Dropout
def mc_dropout_prediction(model, X, n_iter=100):
    predictions = []
    for _ in range(n_iter):
        pred = model(X, training=True)  # 保持 Dropout 啟用
        predictions.append(pred.numpy())
    
    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)
    
    lower = mean_pred - 2*std_pred
    upper = mean_pred + 2*std_pred
    
    return mean_pred, lower, upper
```

**異常定義**：
- 實際值超出預測區間
- 連續多步誤差過大
- 梯度異常變化

**預警策略**：
```python
def check_anomaly(actual, pred, lower, upper):
    if actual < lower or actual > upper:
        return 'WARNING: Value outside prediction interval'
    
    if abs(actual - pred) > 3*std_historical:
        return 'ERROR: Extreme deviation'
    
    return 'NORMAL'
```

---

## 第八章：案例研究

### 8.1 鍋爐蒸汽溫度預測

**問題描述**：
- 預測未來 10 分鐘的蒸汽溫度
- 輸入：燃料流量、風量、壓力等 8 個變數
- 歷史窗口：60 分鐘

**模型對比**：

| 模型 | RMSE | MAE | 訓練時間 | 推理時間 |
|-----|------|-----|---------|---------|
| Persistence | 2.45°C | 1.89°C | - | <1ms |
| ARIMA | 1.87°C | 1.42°C | 5s | 10ms |
| Random Forest | 1.32°C | 0.98°C | 2min | 5ms |
| MLP | 1.18°C | 0.87°C | 3min | 2ms |
| LSTM | **0.94°C** | **0.71°C** | 15min | 8ms |

**關鍵發現**：
1. LSTM 在捕捉長期依賴上優於傳統模型
2. 多步預測時，LSTM 優勢更明顯（horizon > 5）
3. 燃料流量的滯後效應（lag 8-12分鐘）被 LSTM 自動學習

**Attention 權重分析**：
- 近期時刻（t-1 ~ t-5）：權重 45%
- 中期時刻（t-10 ~ t-20）：權重 35%（燃料滯後效應）
- 遠期時刻（t-30 ~ t-60）：權重 20%（負荷變化趨勢）

### 8.2 反應器溫度控制

**挑戰**：
- 強非線性：Arrhenius 定律
- 快速動態：秒級響應
- 安全約束：溫度不能超過 350°C

**解決方案**：
1. **Seq2Seq 架構**：直接預測未來 5 分鐘軌跡
2. **約束損失函數**：
   ```python
   def constrained_loss(y_true, y_pred):
       mse = keras.losses.mse(y_true, y_pred)
       penalty = keras.backend.maximum(0, y_pred - 350) ** 2
       return mse + 10 * penalty
   ```
3. **模型預測控制（MPC）整合**：
   - LSTM 提供動態模型
   - MPC 求解最優控制序列

**性能指標**：
- 溫度偏差：平均 ±1.2°C（目標 ±2°C）
- 超溫事件：降低 85%（從 12次/月 → 2次/月）
- 能耗降低：8%（更平穩的控制）

### 8.3 精餾塔產品質量預測

**Soft Sensor 應用**：
- 實驗室分析延遲：4小時
- LSTM 預測延遲：<1秒
- 實現在線質量控制

**特徵工程**：
```python
features = [
    'tower_top_temp',
    'tower_bottom_temp', 
    'reflux_ratio',
    'feed_flow',
    'feed_composition',
    # 導出特徵
    'temp_gradient',     # 塔頂塔底溫差
    'temp_delta_15min',  # 15分鐘溫度變化
    'ma_reflux_10min'    # 回流比滾動平均
]
```

**多任務學習**：
同時預測輕組分純度和重組分純度

```python
# 共享編碼器
encoder = keras.Sequential([
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(64)
])

# 兩個預測頭
x = encoder(inputs)
output_light = layers.Dense(1, name='light_component')(x)
output_heavy = layers.Dense(1, name='heavy_component')(x)

model = keras.Model(inputs, [output_light, output_heavy])
model.compile(
    optimizer='adam',
    loss={'light_component': 'mse', 'heavy_component': 'mse'},
    loss_weights={'light_component': 1.0, 'heavy_component': 1.0}
)
```

**成果**：
- 輕組分純度預測 R² = 0.94
- 重組分純度預測 R² = 0.91
- 減少離線分析次數 70%

---

## 第九章：進階主題

### 9.1 Attention Mechanism

**動機**：LSTM 將所有信息壓縮到最後一個隱狀態，可能丟失重要信息。

**Attention 機制**：選擇性注意不同時步

$$
\alpha_t = \frac{\exp(e_t)}{\sum_{i=1}^{T} \exp(e_i)}
$$
$$
e_t = \text{score}(\mathbf{h}_t, \mathbf{s})
$$
$$
\mathbf{c} = \sum_{t=1}^{T} \alpha_t \mathbf{h}_t
$$

其中：
- $\mathbf{h}_t$：encoder 在時刻 $t$ 的隱狀態
- $\mathbf{s}$：decoder 的狀態
- $\alpha_t$：attention 權重
- $\mathbf{c}$：context vector

**實現**：
```python
from tensorflow.keras.layers import AdditiveAttention

encoder_outputs = layers.LSTM(64, return_sequences=True)(inputs)
query = layers.LSTM(64)(inputs)
attention = AdditiveAttention()([query, encoder_outputs])
output = layers.Dense(1)(attention)
```

### 9.2 Transformer for Time Series

**Self-Attention**：
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**位置編碼（Positional Encoding）**：
$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})
$$
$$
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})
$$

**優勢**：
- 並行計算（比 LSTM 快）
- 長距離依賴（無梯度消失）
- 可解釋性（attention 權重）

**應用**：
- Temporal Fusion Transformer (TFT)
- Autoformer
- Informer

### 9.3 Neural ODE

**連續時間建模**：

將離散 RNN 泛化為連續 ODE：
$$
\frac{d\mathbf{h}(t)}{dt} = f_\theta(\mathbf{h}(t), t)
$$

**求解**：使用 ODE solver
```python
from torchdiffeq import odeint

def ode_func(t, h):
    return model(h, t)

h_final = odeint(ode_func, h_0, t_span)
```

**優勢**：
- 時間步長不受限制
- 內存高效
- 更好的泛化能力

### 9.4 集成方法

**組合多個模型**：
```python
# 組合多個模型
predictions = []
weights = [0.3, 0.3, 0.4]  # ARIMA, RF, LSTM

pred_arima = model_arima.forecast(steps=horizon)
pred_rf = model_rf.predict(X_test)
pred_lstm = model_lstm.predict(X_test)

final_pred = (weights[0] * pred_arima + 
              weights[1] * pred_rf + 
              weights[2] * pred_lstm)
```

---

## 第十章：總結與最佳實踐

### 10.1 建模流程 Checklist

**1. 數據準備** ：
- [ ] 檢查缺失值和異常值
- [ ] 繪製時間序列圖，識別趨勢和季節性
- [ ] 計算 ACF/PACF
- [ ] 檢驗平穩性（ADF test）
- [ ] 按時間順序劃分訓練/驗證/測試集

**2. 特徵工程** ：
- [ ] 創建滯後特徵
- [ ] 計算滾動統計（MA, Std, EWMA）
- [ ] 添加時間特徵（hour, day of week）
- [ ] 工程特徵（物理公式、領域知識）
- [ ] 標準化或歸一化

**3. Baseline 模型** ：
- [ ] Persistence
- [ ] Moving Average
- [ ] Linear Regression
- [ ] Random Forest / XGBoost
- [ ] MLP

**4. LSTM 模型** ：
- [ ] 設計網路架構
- [ ] 設置訓練回調（Early Stopping, LR Scheduler）
- [ ] 訓練模型
- [ ] 監控 training/validation loss
- [ ] 超參數調優

**5. 評估與對比** ：
- [ ] 計算多種指標（RMSE, MAE, MAPE）
- [ ] 繪製預測曲線
- [ ] 殘差分析
- [ ] 與 Baseline 對比
- [ ] Rolling backtest

**6. 部署準備** ：
- [ ] 模型導出（SavedModel, ONNX）
- [ ] 推理速度測試
- [ ] 漂移檢測機制
- [ ] 預警規則設置
- [ ] 更新策略

### 10.2 常見錯誤與解決

| 問題 | 原因 | 解決方案 |
|-----|------|---------|
| 預測值完全滯後實際值 | 模型學到"複製上一步" | 增加預測horizon的懲罰、滯後 |
| 訓練很好測試很差 | 過擬合 | Dropout、Early Stopping、簡化模型 |
| 訓練和測試都很差 | 欠擬合 | 增加模型容量、更多特徵 |
| Loss 不收斂 | 學習率太大、數據未標準化 | 降低LR、標準化輸入 |
| 預測值趨於平均 | 模型保守預測均值 | 檢查數據質量、增加模型複雜度 |
| 梯度爆炸 | 權重初始化不當 | Gradient Clipping、Xavier初始化 |

### 10.3 工業應用建議

**1. 從簡單開始**：
- 先建立可解釋 Baseline
- 確認 LSTM 真正帶來改進
- 不要為了進步 0.5% RMSE 犧牲太多

**2. 融入領域知識**：
- 物理約束（能量守恆、質量守恆）
- 已知時間常數和延遲
- 已知因果關係

**3. 魯棒性優先於準確性**：
- 在異常工況下的表現
- 對傳感器故障的容錯
- 模型重訓的穩定性

**4. 可解釋性**：
- Attention 權重可視化
- SHAP 值分析
- 與物理模型比對

**5. 持續運維**：
- 性能監控儀表板
- 漂移檢測
- 定期重訓

### 10.4 延伸閱讀

**經典論文**：
1. Hochreiter & Schmidhuber (1997). "Long Short-Term Memory"
2. Cho et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder"
3. Vaswani et al. (2017). "Attention Is All You Need"

**時間序列預測**：
4. Box & Jenkins (1970). "Time Series Analysis: Forecasting and Control"
5. Makridakis et al. (2018). "Statistical and Machine Learning forecasting methods"
6. Lim et al. (2021). "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"

**化工應用**：
7. Fortuna et al. (2007). "Soft Sensors for Monitoring and Control of Industrial Processes"
8. Kadlec et al. (2009). "Data-driven Soft Sensors in the Process Industry"

---

## 參考文獻與延伸資源

### 論文

1. **LSTM原理**
   - Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.

2. **GRU**
   - Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. EMNLP 2014.

3. **時間序列深度學習**
   - Lim, B., & Zohren, S. (2021). Time-series forecasting with deep learning: a survey. Philosophical Transactions of the Royal Society A, 379(2194), 20200209.

### 開發工具

- **TensorFlow/Keras**: https://www.tensorflow.org/
- **PyTorch**: https://pytorch.org/
- **statsmodels**: https://www.statsmodels.org/
- **sktime**: https://www.sktime.org/
- **Darts**: https://unit8co.github.io/darts/

### 在線課程

- Stanford CS231n: Deep Learning for Computer Vision
- Coursera: Sequences, Time Series and Prediction (deeplearning.ai)
- Fast.ai: Practical Deep Learning

---

**本講義編寫於 2025 年，專注於化工製程中時間序列預測的完整理論與實踐。**

**授課教師**：[教師姓名]  
**課程名稱**：化工AI應用  
**單元**：Unit18 - LSTM 時間序列預測  
**版本**：v2.0（2025年全面重構）
   for t in range(len(y_test)):
       pred = model.predict(X_test[t:t+1])
       error = abs(y_test[t] - pred)
       rolling_rmse.append(error)
       
       if np.mean(rolling_rmse[-100:]) > threshold:
           trigger_retrain()
   ```

**?�新策略**�?

1. **定�??��?（Periodic Retraining�?*�?
   - 每�?每�??�新訓練
   - 使用?�??N 天數??

2. **觸發式�?訓�?Triggered Retraining�?*�?
   - ?�能下�?超�??��?
   - 檢測?��?�?

3. **增�?學�?（Incremental Learning�?*�?
   ```python
   # 使用?�數?�微�?
   model.fit(X_new, y_new, epochs=5, initial_epoch=previous_epochs)
   ```

4. **?��?學�?（Ensemble�?*�?
   ```python
   # 組�??�模?��??�模??
   pred = 0.7 * model_old.predict(X) + 0.3 * model_new.predict(X)
   ```

### 7.3 ?�常檢測?��?�?

**?�測誤差?��?*�?

$$
\text{Alert} = \begin{cases}
1 & \text{if } |y - \hat{y}| > k \cdot \sigma \\
0 & \text{otherwise}
\end{cases}
$$

?�中 $\sigma$ ?�歷?�誤差�?標�?差�?$k=3$ ?�常?�閾?��?3-sigma 法�?）�?

**?�常?�數**�?

使用?�測不確定性�?
$$
\text{Anomaly Score} = \frac{|y - \hat{y}|}{\sigma_{\hat{y}}}
$$

**多�??�警**�?

| 級別 | 條件 | ?��? |
|-----|------|------|
| �?�� | error < 1°C | ??|
| 警�? | 1°C < error < 2°C | 記�??��? |
| 注�? | 2°C < error < 3°C | ?�知?��???|
| 緊�?| error > 3°C | ?��?調�?/?��? |

**趨勢?�警**�?

檢測?��??�離�?
```python
def trend_alert(errors, window=10, threshold=0.5):
    recent_errors = errors[-window:]
    if np.mean(recent_errors) > threshold:
        return "Sustained deviation detected"
    return "OK"
```

---

## 第八章�??�工製�?案�??�究

### 8.1 ?��??�汽溫度?�測

**?��??�述**�?
- ?�測?��? 10-30 ?��??�汽溫度
- ?��?調整減溫水�???
- ?��?溫度超�?�?50-480°C�?

**?��??��?*�?
- ?�樣?��?�? ?��?
- 訓練?��?�?0 天�?~43,000 �?���?
- 測試?��?�? �?

**?�徵工�?**�?
```python
features = [
    'TE_8332A.AV_0#',  # ?��?：蒸汽溫�?
    'ZZQBCHLL.AV_0#',  # ?�汽流�?
    'PTCA_8324.AV_0#',  # ?��?壓�?
    'AIR_8301A.AV_0#',  # 一次風
    'FT_8301.AV_0#',    # ?��?流�?
    'TV_8329ZC.AV_0#',  # 減溫�?
]

# 添�?滯�??�徵
for lag in [1, 5, 10, 15, 30]:
    for col in features:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

# 添�?滾�?統�?
df['temp_ma_15'] = df['TE_8332A.AV_0#'].rolling(15).mean()
df['temp_std_15'] = df['TE_8332A.AV_0#'].rolling(15).std()
```

**模�?對�?**�?

| 模�? | RMSE (°C) | MAE (°C) | 訓練?��? | ?��??��? |
|------|----------|---------|---------|---------|
| Persistence | 2.34 | 1.82 | - | <1ms |
| Linear Reg | 1.56 | 1.21 | 2s | <1ms |
| Random Forest | 1.12 | 0.87 | 45s | 5ms |
| MLP | 0.95 | 0.74 | 3min | 2ms |
| LSTM (1-layer) | 0.82 | 0.63 | 15min | 10ms |
| LSTM (2-layer) | 0.76 | 0.59 | 25min | 15ms |

**結�?**�?
- LSTM ?��? 32% （相�?Random Forest�?
- 計�??�本?�接?��?<20ms ?��??��?�?
- 建議?�署 LSTM (1-layer) 平衡?�能?��???

### 8.2 ?��??�溫度控??

**?��??�述**�?
- ?�熱?��?，溫度�?高�??�副?��?
- ?�?��? 5-10 ?��??�測溫度趨勢
- 調整?�卻水�???

**?�戰**�?
- 高度?��??��?Arrhenius ?��?�?
- 多個穩?��?穩�?/不穩定�?
- 快速�??��??��?常數 ~2-5?��?�?

**�?��?��?**�?

1. **?��?建模**�?
   ```python
   # ?��?負荷?��?
   low_load_mask = df['load'] < 0.3
   mid_load_mask = (df['load'] >= 0.3) & (df['load'] < 0.7)
   high_load_mask = df['load'] >= 0.7
   
   model_low = train_lstm(df[low_load_mask])
   model_mid = train_lstm(df[mid_load_mask])
   model_high = train_lstm(df[high_load_mask])
   
   # ?�測?�選?��??�模??
   def predict(X, load):
       if load < 0.3:
           return model_low.predict(X)
       elif load < 0.7:
           return model_mid.predict(X)
       else:
           return model_high.predict(X)
   ```

2. **?��?約�?**�?
   ```python
   # ?��?平衡約�?
   Q_reaction = k(T) * V * c^n * (-?H)
   Q_cooling = U * A * (T - T_coolant)
   
   # 將物?�模?��??��??�輸??
   df['Q_estimate'] = calculate_heat_balance(df)
   features.append('Q_estimate')
   ```

3. **注�??��???*�?
   ```python
   # Attention LSTM
   from tensorflow.keras.layers import Attention
   
   encoder = layers.LSTM(64, return_sequences=True)(inputs)
   decoder = layers.LSTM(64, return_sequences=True)(inputs)
   attention = Attention()([decoder, encoder])
   output = layers.Dense(1)(attention)
   ```

### 8.3 ?�餾塔�?溫控??

**?��??�述**�?
- 多�??�相互�??��??��?比、進�??�、�??��??��?
- ?��??�延?��?~15-30 ?��?�?
- �???��??��???

**?��?準�?**�?
```python
# 多�??�輸??
X_vars = ['reflux_ratio', 'feed_rate', 'top_pressure', 
          'bottom_temp', 'feed_composition']

# ?�建序�??��?（考慮延遲�?
def create_sequences_with_delay(df, X_vars, y_var, 
                                 window=30, delay=15):
    X, y = [], []
    for i in range(len(df) - window - delay):
        X.append(df[X_vars].iloc[i:i+window].values)
        y.append(df[y_var].iloc[i+window+delay])
    return np.array(X), np.array(y)
```

**?��?模�?**�?
```python
# 組�?多個模??
predictions = []
weights = [0.3, 0.3, 0.4]  # ARIMA, RF, LSTM

pred_arima = model_arima.forecast(steps=horizon)
pred_rf = model_rf.predict(X_test)
pred_lstm = model_lstm.predict(X_test)

final_pred = (weights[0] * pred_arima + 
              weights[1] * pred_rf + 
              weights[2] * pred_lstm)
```

---

## 第�?章�??��?主�?

### 9.1 Attention Mechanism

**?��?**：LSTM 將�??�信?��?縮到?�後�??��??�?��??�能丟失?��?信息??

**Attention 機制**：�??��?注�??��??�步

$$
\alpha_t = \frac{\exp(e_t)}{\sum_{i=1}^{T} \exp(e_i)}
$$
$$
e_t = \text{score}(\mathbf{h}_t, \mathbf{s})
$$
$$
\mathbf{c} = \sum_{t=1}^{T} \alpha_t \mathbf{h}_t
$$

?�中�?
- $\mathbf{h}_t$：encoder ?��???$t$ ?�隱?��???
- $\mathbf{s}$：decoder ?��???
- $\alpha_t$：attention 權�?
- $\mathbf{c}$：context vector

**實現**�?
```python
from tensorflow.keras.layers import AdditiveAttention

encoder_outputs = layers.LSTM(64, return_sequences=True)(inputs)
query = layers.LSTM(64)(inputs)
attention = AdditiveAttention()([query, encoder_outputs])
output = layers.Dense(1)(attention)
```

### 9.2 Transformer for Time Series

**Self-Attention**�?
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**位置編碼（Positional Encoding�?*�?
$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})
$$
$$
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})
$$

**?�勢**�?
- 並�?計�?（�? LSTM 快�?
- ?��?依賴（無梯度消失�?
- ?�解?�性�?attention 權�?�?

**?�用**�?
- Temporal Fusion Transformer (TFT)
- Autoformer
- Informer

### 9.3 Neural ODE

**????��?建模**�?

將離??RNN ?�為??? ODE�?
$$
\frac{d\mathbf{h}(t)}{dt} = f_\theta(\mathbf{h}(t), t)
$$

**求解**：使??ODE solver
```python
from torchdiffeq import odeint

def ode_func(t, h):
    return model(h, t)

h_final = odeint(ode_func, h_0, t_span)
```

**?�勢**�?
- ?��?不�??��?�?
- ?��??�數
- ?�好?��??�能??

---

## 第�?章�?總�??��?佳實�?

### 10.1 建模流�? Checklist

**1. ?��?準�?** ??
- [ ] 檢查缺失?��??�常??
- [ ] 繪製?��?序�??��?識別趨勢?�季節??
- [ ] 計�? ACF/PACF
- [ ] 檢�?平穩?��?ADF test�?
- [ ] ?��?訓練/驗�?/測試?��??��??��?�?

**2. ?�徵工�?** ??
- [ ] ?�建滯�??�徵
- [ ] 計�?滾�?統�?（MA, Std, EWMA�?
- [ ] 添�??��??�徵（hour, day of week�?
- [ ] ?��??�徵（物?�公式�?算�?
- [ ] 標�???歸�???

**3. Baseline 模�?** ??
- [ ] Persistence
- [ ] Moving Average
- [ ] Linear Regression
- [ ] Random Forest / XGBoost
- [ ] MLP

**4. LSTM 模�?** ??
- [ ] 設�?網路?��?
- [ ] 設置訓練?�調（Early Stopping, LR Scheduler�?
- [ ] 訓練模�?
- [ ] ??�� training/validation loss
- [ ] 超�??�調??

**5. 評估?��?�?* ??
- [ ] 計�?多種?��?（RMSE, MAE, MAPE�?
- [ ] 繪製?�測?��?
- [ ] 殘差?��?
- [ ] ??Baseline 對�?
- [ ] Rolling backtest

**6. ?�署準�?** ??
- [ ] 模�?導出（SavedModel, ONNX�?
- [ ] ?��??�度測試
- [ ] 漂移檢測機制
- [ ] ?�警規�?設�?
- [ ] ?�新策略

### 10.2 常�??�誤?�解�?

| ?��? | ?��? | �?��?��? |
|-----|------|---------|
| ?�測?��?滯�??�實??| 模�?學到"複製?��?�? | 增�??�測horizon?�懲罰滯�?|
| 訓練很好測試很差 | ?�擬??| Dropout?�Early Stopping?�簡?�模??|
| 訓練?�測試都很差 | 欠擬??| 增�?模�?容�??�更多特�?|
| Loss 不收??| 學�??��?大、數?�未標�???| ?��?LR?��?準�?輸入 |
| ?�測?��??�於平�? | 模�??�測?��?| 檢查?��?質�??��??�模?��??�度 |
| 梯度?�炸 | 權�??��??��???| Gradient Clipping?�Xavier?��???|

### 10.3 工業?�用建議

**1. 從簡?��?�?*�?
- ?�建立可?��? Baseline
- 確�? LSTM ?�正帶�??��?
- ?��??�進�?RMSE ?��?多�?？�?

**2. ?��??��??��?**�?
- ?��?約�?（能?��??�、質?��??��?
- ?��??�制（閥?�?�度?��??��??��?
- 已知?��?常數?�延??

**3. 魯�??�優?�於準確??*�?
- ?�異常工況�??�表??
- 對傳?�器?��??�容??
- 模�??�新?�穩定�?

**4. ?�解?��?*�?
- Attention 權�??��???
- SHAP ?��???
- ?�物?�模?��?�?

**5. ?��???��**�?
- ?��??�能?��?
- 漂移檢測
- 定�??��?

### 10.4 延伸?��?

**經典論�?**�?
1. Hochreiter & Schmidhuber (1997). "Long Short-Term Memory"
2. Cho et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder"
3. Vaswani et al. (2017). "Attention Is All You Need"

**?��?序�??�測**�?
4. Box & Jenkins (1970). "Time Series Analysis: Forecasting and Control"
5. Makridakis et al. (2018). "Statistical and Machine Learning forecasting methods"
6. Lim et al. (2021). "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"

**?�工?�用**�?
7. Fortuna et al. (2007). "Soft Sensors for Monitoring and Control of Industrial Processes"
8. Kadlec et al. (2009). "Data-driven Soft Sensors in the Process Industry"

---

## ?�考�??��?資�?

### 論�?

1. **LSTM?��?**
   - Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.

2. **GRU**
   - Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. EMNLP 2014.

3. **?��?序�?深度學�?**
   - Lim, B., & Zohren, S. (2021). Time-series forecasting with deep learning: a survey. Philosophical Transactions of the Royal Society A, 379(2194), 20200209.

### ?��?工具

- **TensorFlow/Keras**: https://www.tensorflow.org/
- **PyTorch**: https://pytorch.org/
- **statsmodels**: https://www.statsmodels.org/
- **sktime**: https://www.sktime.org/
- **Darts**: https://unit8co.github.io/darts/

### ?��?課�?

- Stanford CS231n: Deep Learning for Computer Vision
- Coursera: Sequences, Time Series and Prediction (deeplearning.ai)
- Fast.ai: Practical Deep Learning

---

**?��?義�??�於 2025 年�?專注?��?工製程�??��??��?測�?完整?��??�實踐�?*

**?�課?�師**：[?��?姓�?]  
**課�??�稱**：�?工AI?�用  
**?��?**：Unit18 - LSTM ?��?序�??�測  
**?�本**：v2.0�?025年�??��?構�?
