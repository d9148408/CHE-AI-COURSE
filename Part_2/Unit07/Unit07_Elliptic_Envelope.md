# Unit07 橢圓包絡 (Elliptic Envelope)

## 課程目標

本單元將深入介紹橢圓包絡 (Elliptic Envelope) 異常檢測演算法，這是一種基於高斯分布假設的異常檢測方法，特別適合處理服從或接近多變量常態分布的數據。透過本單元的學習，您將能夠：

- 理解橢圓包絡演算法的核心原理與數學基礎
- 掌握馬氏距離 (Mahalanobis Distance) 與共變異數矩陣的概念
- 學會使用 scikit-learn 實作橢圓包絡模型
- 了解如何設定關鍵超參數 (contamination, support_fraction)
- 學會評估異常檢測模型的效能
- 認識橢圓包絡的優缺點與適用場景
- 應用橢圓包絡於化工領域的品質控制與製程監控

> **💡 執行結果說明**：
> 
> 本講義整合了 `Unit07_Elliptic_Envelope.ipynb` 的完整執行結果，包含：
> - **Section 5.6**：實際數據生成、模型訓練、超參數優化、混淆矩陣、ROC 曲線、馬氏距離分析
> - **Section 6.4**：穩健性評估實驗結果
> - **Section 6.5**：橢圓包絡 vs 其他方法的算法對比
> - **7 張高品質圖表** (300 DPI)：數據可視化、超參數調整、混淆矩陣、ROC 曲線、馬氏距離分析、穩健性評估、算法對比
> - **詳細性能分析**：Precision、Recall、F1-Score、AUC 等指標
> 
> 所有執行結果均已標註在相應理論章節後，便於理論與實踐對照學習。

---

## 1. 橢圓包絡演算法簡介

### 1.1 什麼是橢圓包絡？

橢圓包絡 (Elliptic Envelope) 是一種基於高斯分布假設的異常檢測演算法。其核心理念是：**假設正常數據服從多變量高斯分布，異常點則是遠離分布中心的離群值**。

演算法透過估計數據的共變異數矩陣，構建一個包含大多數正常數據的橢圓邊界，落在橢圓外部的點則被視為異常。

### 1.2 核心理念：為什麼使用高斯分布？

在許多化工製程中，正常操作條件下的數據往往服從或接近多變量常態分布：

- **製程穩定性**：良好控制的製程變數圍繞設定點波動
- **測量誤差**：感測器誤差通常服從常態分布
- **自然變異**：原物料特性、環境條件等因素的隨機變異
- **中央極限定理**：多個獨立因素共同作用的結果趨向常態分布

**橢圓包絡的核心假設**：
1. 正常數據服從或接近多變量高斯分布
2. 異常點的馬氏距離明顯大於正常點
3. 可以透過穩健估計方法處理少量污染數據
4. 橢圓邊界能有效區分正常與異常區域

### 1.3 化工領域應用案例

橢圓包絡在化工領域特別適合以下場景：

1. **產品品質控制**：
   - 多個品質指標的聯合監控（如純度、色度、黏度）
   - 識別品質指標異常組合
   - 建立多維品質規格界限
   - 減少假陽性警報

2. **穩態製程監控**：
   - 連續製程的正常操作區域定義
   - 多個製程變數的協同監控
   - 考慮變數間相關性的異常檢測
   - 早期異常趨勢識別

3. **感測器故障診斷**：
   - 識別感測器讀數異常偏移
   - 多個感測器的聯合驗證
   - 區分真實製程異常與感測器故障
   - 提高監控系統可靠性

4. **批次一致性檢驗**：
   - 檢驗新批次是否符合歷史正常範圍
   - 多個批次特性的聯合評估
   - 識別批次間異常變異
   - 確保產品批次一致性

5. **實驗設計與數據驗證**：
   - 識別實驗數據中的異常點
   - 驗證數據是否符合預期分布
   - 支援統計分析前的數據清理
   - 提高實驗結果可靠性

---

## 2. 橢圓包絡演算法原理

### 2.1 核心概念一：馬氏距離 (Mahalanobis Distance)

**定義**：對於數據點 $\mathbf{x}$ ，其馬氏距離定義為：

$$
D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}
$$

其中：
- $\mathbf{x}$ ：數據點的特徵向量
- $\boldsymbol{\mu}$ ：數據的均值向量
- $\mathbf{\Sigma}$ ：數據的共變異數矩陣
- $\mathbf{\Sigma}^{-1}$ ：共變異數矩陣的逆矩陣

**意義**：
- 馬氏距離衡量點 $\mathbf{x}$ 與分布中心的距離
- 考慮了變數間的相關性與各變數的變異程度
- 對於多變量高斯分布，馬氏距離平方服從卡方分布
- 相較於歐氏距離，馬氏距離更適合處理相關變數

**範例：化工反應器監控**

假設我們監控反應器的溫度 $T$ 和壓力 $P$ ：

- **歐氏距離問題**：如果溫度和壓力的量綱與變異程度不同，歐氏距離會被較大變異的變數主導
- **馬氏距離優勢**：自動標準化各變數，並考慮溫度與壓力的相關性
- **實際應用**：當溫度升高時壓力通常也升高，馬氏距離能識別「溫度正常但壓力異常低」這類異常模式

### 2.2 核心概念二：共變異數矩陣 (Covariance Matrix)

**定義**：對於 $d$ 維數據，共變異數矩陣 $\mathbf{\Sigma}$ 是 $d \times d$ 對稱矩陣：

$$
\mathbf{\Sigma} = \begin{bmatrix}
\sigma_1^2 & \sigma_{12} & \cdots & \sigma_{1d} \\
\sigma_{21} & \sigma_2^2 & \cdots & \sigma_{2d} \\
\vdots & \vdots & \ddots & \vdots \\
\sigma_{d1} & \sigma_{d2} & \cdots & \sigma_d^2
\end{bmatrix}
$$

其中：
- $\sigma_i^2$ ：第 $i$ 個變數的變異數
- $\sigma_{ij}$ ：第 $i$ 和第 $j$ 個變數的共變異數

**傳統估計方法（經驗共變異數）**：

$$
\mathbf{\Sigma} = \frac{1}{n-1} \sum_{i=1}^{n} (\mathbf{x}_i - \boldsymbol{\mu})(\mathbf{x}_i - \boldsymbol{\mu})^T
$$

$$
\boldsymbol{\mu} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{x}_i
$$

**問題**：對離群值敏感，少量異常點會嚴重影響估計結果。

### 2.3 核心概念三：最小共變異數行列式 (Minimum Covariance Determinant, MCD)

由於傳統共變異數估計對離群值敏感，橢圓包絡使用**最小共變異數行列式 (MCD)** 方法進行穩健估計。

**MCD 核心思想**：

找到一個包含 $h$ 個數據點的子集 $\mathcal{H}$ ，使得這個子集的共變異數矩陣行列式最小：

$$
\min_{\mathcal{H}, |\mathcal{H}|=h} \det(\mathbf{\Sigma}_{\mathcal{H}})
$$

其中：
- $h = \lfloor n \times \text{support\_fraction} \rfloor$ ：子集大小
- $\mathbf{\Sigma}_{\mathcal{H}}$ ：子集 $\mathcal{H}$ 的共變異數矩陣
- $\det(\cdot)$ ：行列式

**步驟**：
1. 從 $n$ 個數據點中選取 $h$ 個點的子集
2. 計算子集的均值 $\boldsymbol{\mu}_{\mathcal{H}}$ 和共變異數矩陣 $\mathbf{\Sigma}_{\mathcal{H}}$
3. 尋找使行列式 $\det(\mathbf{\Sigma}_{\mathcal{H}})$ 最小的子集
4. 使用該子集的統計量作為穩健估計

**意義**：
- 自動識別並排除離群值
- 基於"最緊密"的數據子集估計分布
- 提高對污染數據的抵抗能力
- 適合半監督異常檢測場景

### 2.4 演算法流程

**輸入**：
- 訓練數據 $\mathbf{X} = \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n\}$
- 預期污染比例 `contamination`
- 支援子集比例 `support_fraction` （預設為 None，自動計算）

**訓練階段**：

1. **子集選擇**：
   - 計算 $h = \lfloor n \times (1 - \text{contamination}) \rfloor$
   - 如果 `support_fraction` 未指定，使用 $h = \lfloor n \times 0.5 \times (n + d + 1) / n \rfloor$

2. **MCD 估計**：
   - 使用 FastMCD 演算法尋找最小共變異數行列式子集
   - 計算穩健均值 $\hat{\boldsymbol{\mu}}$ 和穩健共變異數 $\hat{\mathbf{\Sigma}}$

3. **校正因子**：
   - 應用校正因子確保一致性估計
   - 調整共變異數矩陣使其在大樣本下無偏

4. **決策邊界**：
   - 計算所有訓練點的馬氏距離
   - 基於卡方分布確定異常閾值

**預測階段**：

1. **計算馬氏距離**：
   
$$
D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \hat{\boldsymbol{\mu}})^T \hat{\mathbf{\Sigma}}^{-1} (\mathbf{x} - \hat{\boldsymbol{\mu}})}
$$

2. **異常判定**：
   
$$
\text{label}(\mathbf{x}) = \begin{cases}
+1 & \text{if } D_M(\mathbf{x}) \leq \text{threshold (正常)} \\
-1 & \text{if } D_M(\mathbf{x}) > \text{threshold (異常)}
\end{cases}
$$

---

## 3. 數學理論深入

### 3.1 多變量高斯分布

**機率密度函數**：

$$
f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\mathbf{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\mathbf{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
$$

其中：
- $d$ ：維度
- $|\mathbf{\Sigma}|$ ：共變異數矩陣的行列式
- 指數項中的二次型即為馬氏距離平方

**等密度曲線**：

對於固定的機率密度 $c$ ，等密度曲線定義為：

$$
(\mathbf{x}-\boldsymbol{\mu})^T\mathbf{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}) = k^2
$$

這是一個橢圓（或高維橢球），橢圓包絡即是基於此構建邊界。

### 3.2 馬氏距離平方的分布

對於多變量高斯分布 $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{\Sigma})$ ，馬氏距離平方服從卡方分布：

$$
D_M^2(\mathbf{x}) \sim \chi^2_d
$$

其中 $d$ 是自由度（維度）。

**異常閾值設定**：

基於卡方分布，可以設定閾值使得正常數據的覆蓋率達到 $(1-\text{contamination})$ ：

$$
\text{threshold} = \sqrt{\chi^2_{d, 1-\alpha}}
$$

其中 $\alpha = \text{contamination}$ 是預期的異常比例。

### 3.3 橢圓的幾何特性

**橢圓方程**：

$$
(\mathbf{x}-\boldsymbol{\mu})^T\mathbf{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}) = c^2
$$

**主軸方向**：由 $\mathbf{\Sigma}$ 的特徵向量決定

**主軸長度**：與 $\mathbf{\Sigma}$ 的特徵值平方根成正比

$$
\text{半軸長度}_i = c \cdot \sqrt{\lambda_i}
$$

其中 $\lambda_i$ 是 $\mathbf{\Sigma}$ 的第 $i$ 個特徵值。

**化工意義**：
- 橢圓主軸反映製程變數的主要變異方向
- 主軸長度反映各方向的允許變異範圍
- 橢圓傾斜角度反映變數間的相關性

---

## 4. 橢圓包絡的優缺點

### 4.1 優點

1. **數學基礎紮實**：
   - 基於多變量高斯分布理論
   - 有明確的統計推論基礎
   - 可解釋性強

2. **考慮變數相關性**：
   - 馬氏距離自然考慮變數間相關性
   - 適合多個相關製程變數的聯合監控
   - 避免獨立監控的誤報

3. **穩健性**：
   - MCD 方法對少量離群值有抵抗力
   - 能在含污染數據中準確估計分布
   - 適合半監督異常檢測

4. **計算效率高**：
   - FastMCD 演算法計算複雜度低
   - 適合中等規模數據
   - 預測階段僅需計算馬氏距離

5. **可視化友善**：
   - 橢圓邊界易於二維/三維可視化
   - 便於向領域專家解釋
   - 支援製程操作視窗的可視化定義

### 4.2 缺點

1. **高斯分布假設**：
   - 要求數據服從或接近多變量常態分布
   - 對嚴重偏態或多峰分布效果不佳
   - 無法處理任意形狀的分布

2. **對高維數據敏感**：
   - 維度詛咒：高維度下共變異數矩陣估計困難
   - 需要足夠的樣本數（ $n \gg d$ ）
   - 高維度下橢圓邊界可能不準確

3. **僅適合單峰分布**：
   - 無法處理多模態數據
   - 多操作模式的製程需要分別建模
   - 不適合複雜的非線性邊界

4. **參數敏感性**：
   - `contamination` 參數需要預先設定
   - `support_fraction` 的選擇影響穩健性
   - 參數設定不當會影響效能

5. **對極端離群值的敏感性**：
   - 雖然 MCD 提供穩健性，但極端離群值仍可能影響估計
   - 需要適當的數據預處理
   - 異常比例過高時效果下降

### 4.3 與其他方法的比較

| 特性 | 橢圓包絡 | Isolation Forest | One-Class SVM | LOF |
|------|----------|------------------|---------------|-----|
| **分布假設** | 高斯分布 | 無 | 無 | 無 |
| **適合高維** | ❌ | ✅ | ⚠️ | ❌ |
| **多模態** | ❌ | ✅ | ⚠️ | ✅ |
| **計算複雜度** | 低 | 低 | 高 | 中 |
| **可解釋性** | ✅ | ⚠️ | ❌ | ⚠️ |
| **穩健性** | ✅ | ✅ | ⚠️ | ⚠️ |
| **參數敏感性** | 中 | 低 | 高 | 高 |

---

## 5. Python 實作：橢圓包絡

### 5.1 基本使用

```python
from sklearn.covariance import EllipticEnvelope
import numpy as np

# 生成訓練數據（正常數據）
np.random.seed(42)
X_train = np.random.randn(200, 2) * [2, 1] + [5, 3]

# 建立橢圓包絡模型
model = EllipticEnvelope(
    contamination=0.1,      # 預期異常比例 10%
    support_fraction=None,  # 自動計算（推薦）
    random_state=42
)

# 訓練模型
model.fit(X_train)

# 預測新數據
X_test = np.array([[5, 3], [10, 10]])  # [正常點, 異常點]
predictions = model.predict(X_test)
# 輸出：[1, -1]，1 代表正常，-1 代表異常

# 計算馬氏距離（負值表示決策函數值）
distances = model.decision_function(X_test)
# 距離越負，越可能是異常
```

### 5.2 關鍵參數說明

#### 5.2.1 `contamination`（污染比例）

**定義**：預期訓練數據中異常點的比例。

**範圍**：$(0, 0.5)$ ，預設值為 $0.1$

**影響**：
- 決定橢圓邊界的大小
- 值越大，橢圓邊界越寬鬆，更多點被視為正常
- 值越小，橢圓邊界越緊，更多點被視為異常

**設定建議**：
```python
# 品質控制場景（嚴格標準）
model = EllipticEnvelope(contamination=0.05)  # 5% 異常容忍

# 製程監控場景（一般標準）
model = EllipticEnvelope(contamination=0.1)   # 10% 異常容忍

# 探索性分析（寬鬆標準）
model = EllipticEnvelope(contamination=0.2)   # 20% 異常容忍
```

#### 5.2.2 `support_fraction`（支援子集比例）

**定義**：MCD 演算法使用的子集比例。

**範圍**：$(0, 1)$ ，預設值為 `None`（自動計算）

**自動計算公式**：

$$
\text{support\_fraction} = \frac{n + d + 1}{2n}
$$

其中 $n$ 是樣本數， $d$ 是維度。

**影響**：
- 決定 MCD 估計的穩健性
- 值越小，對離群值的抵抗力越強，但可能犧牲效率
- 值越大，使用更多數據，但對離群值更敏感

**設定建議**：
```python
# 高污染場景（使用較小子集）
model = EllipticEnvelope(support_fraction=0.6)

# 低污染場景（使用預設值）
model = EllipticEnvelope(support_fraction=None)

# 清潔數據（可使用較大子集）
model = EllipticEnvelope(support_fraction=0.9)
```

#### 5.2.3 `random_state`（隨機種子）

**定義**：MCD 演算法的隨機種子。

**影響**：確保結果可重現性。

**建議**：實驗階段設定固定值，生產環境可考慮不設定。

### 5.3 模型屬性

訓練後的模型提供以下屬性：

```python
# 穩健估計的均值
mu = model.location_
print(f"均值向量: {mu}")

# 穩健估計的共變異數矩陣
Sigma = model.covariance_
print(f"共變異數矩陣:\n{Sigma}")

# 精度矩陣（共變異數矩陣的逆）
precision = model.precision_
print(f"精度矩陣:\n{precision}")

# 支援向量（用於 MCD 估計的數據點索引）
support = model.support_
print(f"支援點數量: {len(support)}")

# 馬氏距離閾值
threshold = model.threshold_
print(f"異常閾值: {threshold}")
```

### 5.4 完整工作流程

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import classification_report, confusion_matrix

# ========================================
# 1. 數據生成
# ========================================
np.random.seed(42)

# 正常數據（多變量高斯分布）
n_normal = 300
mean = [5, 3]
cov = [[4, 2], [2, 2]]  # 有相關性的共變異數矩陣
X_normal = np.random.multivariate_normal(mean, cov, n_normal)

# 異常數據（遠離正常分布）
n_outliers = 30
X_outliers = np.random.uniform(low=-5, high=15, size=(n_outliers, 2))

# 合併數據
X = np.vstack([X_normal, X_outliers])
y_true = np.array([1] * n_normal + [-1] * n_outliers)

# ========================================
# 2. 模型訓練
# ========================================
model = EllipticEnvelope(
    contamination=0.1,
    support_fraction=None,
    random_state=42
)
model.fit(X)

# ========================================
# 3. 預測與評估
# ========================================
y_pred = model.predict(X)

print("分類報告:")
print(classification_report(y_true, y_pred, 
                          target_names=['異常', '正常']))

print("\n混淆矩陣:")
print(confusion_matrix(y_true, y_pred))

# ========================================
# 4. 可視化
# ========================================
# 計算馬氏距離
mahal_dist = model.mahalanobis(X)

# 建立網格用於繪製決策邊界
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min()-2, X[:, 0].max()+2, 100),
    np.linspace(X[:, 1].min()-2, X[:, 1].max()+2, 100)
)
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 繪圖
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左圖：決策邊界與分類結果
ax1 = axes[0]
ax1.contourf(xx, yy, Z, levels=[-10, 0, 10], 
             colors=['#ffcccc', '#ccffcc'], alpha=0.3)
ax1.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=y_pred, 
                       cmap='RdYlGn', edgecolors='k', s=50)
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_title('Elliptic Envelope Decision Boundary')
plt.colorbar(scatter1, ax=ax1, label='Prediction (-1: Outlier, 1: Normal)')

# 右圖：馬氏距離分布
ax2 = axes[1]
scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=mahal_dist, 
                       cmap='coolwarm', edgecolors='k', s=50)
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.set_title('Mahalanobis Distance')
plt.colorbar(scatter2, ax=ax2, label='Mahalanobis Distance')

plt.tight_layout()
plt.show()
```

### 5.5 化工應用：反應器品質監控

**場景**：監控批次反應器的溫度和壓力，識別異常批次。

```python
import numpy as np
from sklearn.covariance import EllipticEnvelope
import pandas as pd

# ========================================
# 模擬批次反應器數據
# ========================================
np.random.seed(42)

# 正常批次（溫度和壓力正相關）
n_batches = 200
temp_normal = np.random.normal(180, 5, n_batches)  # 溫度 (°C)
pressure_normal = 2.0 + 0.01 * temp_normal + np.random.normal(0, 0.3, n_batches)  # 壓力 (bar)

# 異常批次
n_outliers = 20
temp_outliers = np.random.uniform(150, 210, n_outliers)
pressure_outliers = np.random.uniform(1.0, 4.0, n_outliers)

# 合併數據
temperature = np.concatenate([temp_normal, temp_outliers])
pressure = np.concatenate([pressure_normal, pressure_outliers])
batch_id = np.arange(1, n_batches + n_outliers + 1)

# 建立 DataFrame
df = pd.DataFrame({
    'Batch_ID': batch_id,
    'Temperature': temperature,
    'Pressure': pressure
})

# ========================================
# 橢圓包絡異常檢測
# ========================================
X = df[['Temperature', 'Pressure']].values

model = EllipticEnvelope(
    contamination=0.1,  # 預期 10% 異常批次
    random_state=42
)
model.fit(X)

# 預測
df['Prediction'] = model.predict(X)
df['Mahalanobis_Distance'] = model.mahalanobis(X)
df['Is_Outlier'] = df['Prediction'] == -1

# ========================================
# 結果分析
# ========================================
print("檢測到的異常批次:")
outlier_batches = df[df['Is_Outlier']]
print(outlier_batches[['Batch_ID', 'Temperature', 'Pressure', 'Mahalanobis_Distance']])

print(f"\n總批次數: {len(df)}")
print(f"異常批次數: {outlier_batches.shape[0]}")
print(f"異常比例: {outlier_batches.shape[0] / len(df):.2%}")

# 統計分析
print("\n正常批次統計:")
normal_df = df[~df['Is_Outlier']]
print(normal_df[['Temperature', 'Pressure']].describe())

print("\n異常批次統計:")
print(outlier_batches[['Temperature', 'Pressure']].describe())
```

### 5.6 進階技巧：多組橢圓包絡

對於多模態數據（如多個操作模式），可以為每個模式單獨建立橢圓包絡：

```python
from sklearn.cluster import KMeans

# ========================================
# 1. 先用聚類識別操作模式
# ========================================
kmeans = KMeans(n_clusters=3, random_state=42)
modes = kmeans.fit_predict(X)

# ========================================
# 2. 為每個模式建立橢圓包絡
# ========================================
models = {}
for mode in range(3):
    X_mode = X[modes == mode]
    models[mode] = EllipticEnvelope(contamination=0.1, random_state=42)
    models[mode].fit(X_mode)

# ========================================
# 3. 預測：使用最近模式的模型
# ========================================
def predict_multimode(X_test):
    # 找到最近的操作模式
    mode_pred = kmeans.predict(X_test)
    
    # 使用對應模式的模型預測
    predictions = np.zeros(len(X_test))
    for mode in range(3):
        mask = mode_pred == mode
        if mask.any():
            predictions[mask] = models[mode].predict(X_test[mask])
    
    return predictions

# 測試
y_pred_multimode = predict_multimode(X)
```

---

## 6. 實務應用指南

### 6.1 何時選擇橢圓包絡？

**適合場景**：

✅ 數據接近多變量高斯分布  
✅ 變數間有明顯相關性  
✅ 需要考慮變數聯合分布  
✅ 需要可解釋的異常邊界  
✅ 數據維度中等（ $d < 20$ ）  
✅ 樣本數充足（ $n > 10d$ ）  
✅ 製程穩態監控  
✅ 產品品質控制

**不適合場景**：

❌ 數據明顯非高斯分布（偏態、多峰）  
❌ 高維數據（ $d > 50$ ）  
❌ 複雜非線性邊界  
❌ 多操作模式且未分開建模  
❌ 樣本數不足（ $n < 5d$ ）  
❌ 需要處理任意形狀異常區域

### 6.2 參數調整策略

#### 6.2.1 `contamination` 調整

**策略 1：基於歷史數據**
```python
# 如果有部分標籤數據，估計實際異常比例
historical_outlier_rate = 0.08
model = EllipticEnvelope(contamination=historical_outlier_rate)
```

**策略 2：交叉驗證**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'contamination': [0.05, 0.1, 0.15, 0.2]}
# 注意：需要自定義評分函數，因為 EllipticEnvelope 不支持標準 GridSearchCV
```

**策略 3：視覺化調整**
```python
for cont in [0.05, 0.1, 0.15, 0.2]:
    model = EllipticEnvelope(contamination=cont, random_state=42)
    model.fit(X)
    y_pred = model.predict(X)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='RdYlGn')
    plt.title(f'Contamination = {cont}')
    plt.show()
```

#### 6.2.2 `support_fraction` 調整

**一般建議**：使用預設值（ `None` ），除非：

- 已知污染比例很高（ $>20\%$ ）：降低 `support_fraction`
- 數據非常乾淨：可適度提高 `support_fraction`

```python
# 高污染場景
model = EllipticEnvelope(
    contamination=0.3,
    support_fraction=0.5,  # 使用 50% 數據估計
    random_state=42
)

# 低污染場景
model = EllipticEnvelope(
    contamination=0.05,
    support_fraction=None,  # 自動計算
    random_state=42
)
```

### 6.3 常見問題與解決方案

#### 問題 1：數據不符合高斯分布

**診斷**：
```python
from scipy import stats

# 單變量常態性檢驗（Shapiro-Wilk Test）
for i in range(X.shape[1]):
    statistic, pvalue = stats.shapiro(X[:, i])
    print(f"Feature {i}: p-value = {pvalue:.4f}")
    if pvalue < 0.05:
        print("  ⚠️ 不符合常態分布")

# 多變量常態性檢驗（Mardia's Test 或視覺化）
```

**解決方案**：
```python
# 方案 1：數據轉換
from sklearn.preprocessing import PowerTransformer

transformer = PowerTransformer(method='yeo-johnson')
X_transformed = transformer.fit_transform(X)

model = EllipticEnvelope(contamination=0.1)
model.fit(X_transformed)

# 方案 2：改用其他方法
from sklearn.ensemble import IsolationForest
model = IsolationForest(contamination=0.1, random_state=42)
```

#### 問題 2：高維度效果不佳

**診斷**：
```python
n_samples, n_features = X.shape
print(f"樣本數 / 維度比 = {n_samples / n_features:.2f}")

if n_samples / n_features < 10:
    print("⚠️ 樣本數相對維度過少，可能影響估計準確性")
```

**解決方案**：
```python
# 方案 1：降維
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # 保留 95% 變異
X_reduced = pca.fit_transform(X)

model = EllipticEnvelope(contamination=0.1)
model.fit(X_reduced)

# 方案 2：特徵選擇
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.1)
X_selected = selector.fit_transform(X)

model = EllipticEnvelope(contamination=0.1)
model.fit(X_selected)
```

#### 問題 3：多模態數據

**診斷**：
```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 降維至 2D 可視化
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

plt.scatter(X_2d[:, 0], X_2d[:, 1])
plt.title('Data Distribution (2D PCA)')
plt.show()
# 如果看到明顯多個聚類，則為多模態
```

**解決方案**：
```python
# 方案 1：分模式建模（見 5.6 節）

# 方案 2：使用適合多模態的方法
from sklearn.neighbors import LocalOutlierFactor
model = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
```

### 6.4 效能評估

對於有部分標籤的數據：

```python
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# 預測
y_pred = model.predict(X_test)
y_scores = model.decision_function(X_test)

# 分類指標
precision = precision_score(y_true, y_pred, pos_label=-1)
recall = recall_score(y_true, y_pred, pos_label=-1)
f1 = f1_score(y_true, y_pred, pos_label=-1)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")

# ROC 曲線
# 注意：decision_function 的值需要取負號，因為異常點的值更負
fpr, tpr, thresholds = roc_curve(y_true, -y_scores, pos_label=-1)
auc = roc_auc_score(y_true, -y_scores)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Elliptic Envelope')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 混淆矩陣
cm = confusion_matrix(y_true, y_pred)
print("混淆矩陣:")
print(cm)
```

### 6.5 化工製程實務建議

#### 建議 1：製程穩定後再建模

```python
# 確保使用穩態數據建模
# 排除啟動、關機、過渡階段的數據

import pandas as pd

df = pd.read_csv('process_data.csv')

# 識別穩態操作
df['is_steady_state'] = (
    (df['temperature'].rolling(10).std() < 2) &  # 溫度變異小
    (df['pressure'].rolling(10).std() < 0.1) &   # 壓力變異小
    (df['flowrate'].rolling(10).std() < 5)       # 流量變異小
)

# 僅使用穩態數據訓練
X_steady = df[df['is_steady_state']][['temperature', 'pressure', 'flowrate']].values
model.fit(X_steady)
```

#### 建議 2：定期重新訓練

製程特性可能隨時間變化（設備老化、原料變更等），建議定期更新模型：

```python
# 滾動視窗訓練
window_size = 1000  # 使用最近 1000 個正常樣本

def update_model(X_historical, X_new):
    # 合併歷史與新數據
    X_combined = np.vstack([X_historical, X_new])
    
    # 僅保留最近視窗內的數據
    if len(X_combined) > window_size:
        X_combined = X_combined[-window_size:]
    
    # 重新訓練
    model = EllipticEnvelope(contamination=0.1, random_state=42)
    model.fit(X_combined)
    
    return model, X_combined

# 定期（如每週）更新
model, X_historical = update_model(X_historical, X_new_week)
```

#### 建議 3：結合領域知識

```python
# 物理約束：溫度與壓力的合理範圍
def apply_domain_knowledge(predictions, X, temp_range=(150, 200), press_range=(1.5, 3.5)):
    """結合領域知識的異常判定"""
    
    # 橢圓包絡預測
    elliptic_outliers = predictions == -1
    
    # 物理約束檢查
    temp_outliers = (X[:, 0] < temp_range[0]) | (X[:, 0] > temp_range[1])
    press_outliers = (X[:, 1] < press_range[0]) | (X[:, 1] > press_range[1])
    
    # 組合判定：橢圓包絡 OR 物理約束
    combined_outliers = elliptic_outliers | temp_outliers | press_outliers
    
    return combined_outliers

# 應用
y_pred = model.predict(X)
final_outliers = apply_domain_knowledge(y_pred, X)
```

#### 建議 4：多層次監控

```python
# 第一層：快速篩選（橢圓包絡）
model_fast = EllipticEnvelope(contamination=0.15, random_state=42)
model_fast.fit(X_train)

# 第二層：精細檢測（嚴格參數）
model_strict = EllipticEnvelope(contamination=0.05, random_state=42)
model_strict.fit(X_train)

# 分級警報
def classify_alert_level(X_test):
    pred_fast = model_fast.predict(X_test)
    pred_strict = model_strict.predict(X_test)
    
    alert_levels = []
    for i in range(len(X_test)):
        if pred_strict[i] == -1:
            alert_levels.append('HIGH')      # 兩個模型都檢測到
        elif pred_fast[i] == -1:
            alert_levels.append('MEDIUM')    # 僅快速模型檢測到
        else:
            alert_levels.append('NORMAL')    # 均未檢測到
    
    return alert_levels

# 應用
alerts = classify_alert_level(X_test)
```

#### 建議 5：可視化監控儀表板

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_monitoring_dashboard(model, X_current, X_historical):
    """製程監控儀表板"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 實時監控圖
    ax1 = axes[0, 0]
    mahal_dist = model.mahalanobis(X_current)
    colors = ['red' if d > model.threshold_ else 'green' for d in mahal_dist]
    ax1.scatter(X_current[:, 0], X_current[:, 1], c=colors, s=50, alpha=0.6)
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Pressure (bar)')
    ax1.set_title('Current Process Status')
    
    # 繪製橢圓邊界
    eigenvalues, eigenvectors = np.linalg.eigh(model.covariance_)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * np.sqrt(model.threshold_) * np.sqrt(eigenvalues)
    ellipse = Ellipse(model.location_, width, height, angle=angle,
                     fill=False, edgecolor='blue', linewidth=2)
    ax1.add_patch(ellipse)
    
    # 2. 馬氏距離時間序列
    ax2 = axes[0, 1]
    time_steps = np.arange(len(mahal_dist))
    ax2.plot(time_steps, mahal_dist, marker='o', linestyle='-', color='blue')
    ax2.axhline(y=model.threshold_, color='red', linestyle='--', label='Threshold')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Mahalanobis Distance')
    ax2.set_title('Mahalanobis Distance Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 異常統計
    ax3 = axes[1, 0]
    n_outliers = np.sum(mahal_dist > model.threshold_)
    n_normal = len(mahal_dist) - n_outliers
    ax3.bar(['Normal', 'Outlier'], [n_normal, n_outliers], color=['green', 'red'])
    ax3.set_ylabel('Count')
    ax3.set_title(f'Status Summary (Outlier Rate: {n_outliers/len(mahal_dist):.1%})')
    
    # 4. 歷史趨勢
    ax4 = axes[1, 1]
    ax4.scatter(X_historical[:, 0], X_historical[:, 1], 
               c='gray', s=10, alpha=0.3, label='Historical')
    ax4.scatter(X_current[:, 0], X_current[:, 1], 
               c=colors, s=50, alpha=0.8, label='Current')
    ax4.set_xlabel('Temperature (°C)')
    ax4.set_ylabel('Pressure (bar)')
    ax4.set_title('Historical vs Current Data')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

# 使用範例
plot_monitoring_dashboard(model, X_current, X_historical)
```

---

## 7. 總結

### 7.1 核心要點

1. **橢圓包絡基於高斯分布假設**，透過馬氏距離識別異常點
2. **MCD 方法提供穩健估計**，能抵抗少量污染數據
3. **考慮變數相關性**，適合多變量聯合監控
4. **可解釋性強**，橢圓邊界易於可視化與理解
5. **適合中等維度、接近高斯分布的數據**

### 7.2 使用決策樹

```
開始
│
├─ 數據是否接近高斯分布？
│  ├─ 是 → 繼續
│  └─ 否 → 考慮數據轉換或使用 Isolation Forest / LOF
│
├─ 變數間是否有相關性？
│  ├─ 是 → 橢圓包絡是好選擇
│  └─ 否 → 可考慮簡單的統計方法
│
├─ 維度是否中等 (d < 20)？
│  ├─ 是 → 繼續
│  └─ 否 → 先降維或使用 Isolation Forest
│
├─ 樣本數是否充足 (n > 10d)？
│  ├─ 是 → 使用橢圓包絡
│  └─ 否 → 收集更多數據或使用其他方法
│
└─ 決定：使用橢圓包絡
   └─ 設定 contamination 與 support_fraction
```

### 7.3 與其他方法的選擇

| 場景 | 推薦方法 | 原因 |
|------|----------|------|
| **高斯分布數據** | 橢圓包絡 | 符合假設，效果最佳 |
| **非高斯分布數據** | Isolation Forest | 無分布假設 |
| **多模態數據** | LOF | 基於局部密度 |
| **高維數據** | Isolation Forest | 對維度不敏感 |
| **小樣本數據** | One-Class SVM | 核方法適合小樣本 |
| **需要高度可解釋性** | 橢圓包絡 | 橢圓邊界直觀 |
| **複雜非線性邊界** | One-Class SVM + RBF Kernel | 靈活性高 |

### 7.4 延伸學習

1. **穩健統計學**：深入學習 MCD、MVE 等穩健估計方法
2. **多變量統計**：Hotelling's T² 控制圖、MEWMA 控制圖
3. **非參數方法**：Kernel Density Estimation (KDE)
4. **深度學習方法**：Autoencoder 用於異常檢測
5. **時間序列異常檢測**：動態製程的異常檢測方法

### 7.5 實務檢查清單

在實際應用橢圓包絡時，請確認：

- [ ] 數據已標準化或正規化
- [ ] 驗證數據接近高斯分布（或已轉換）
- [ ] 樣本數充足（ $n > 10d$ ）
- [ ] 合理設定 `contamination` 參數
- [ ] 考慮定期重新訓練模型
- [ ] 結合領域知識進行驗證
- [ ] 建立可視化監控機制
- [ ] 評估誤報與漏報的成本
- [ ] 記錄並分析誤判案例
- [ ] 與製程工程師保持溝通

---

## 8. 參考資源

### 8.1 理論參考

1. **Rousseeuw, P. J., & Driessen, K. V. (1999)**. "A fast algorithm for the minimum covariance determinant estimator." *Technometrics*, 41(3), 212-223.
   - MCD 演算法的原始論文

2. **Hardin, J., & Rocke, D. M. (2005)**. "The distribution of robust distances." *Journal of Computational and Graphical Statistics*, 14(4), 928-946.
   - 馬氏距離在穩健估計中的分布理論

3. **Hubert, M., Debruyne, M., & Rousseeuw, P. J. (2018)**. "Minimum covariance determinant and extensions." *Wiley Interdisciplinary Reviews: Computational Statistics*, 10(3), e1421.
   - MCD 方法的綜述文章

### 8.2 scikit-learn 文檔

- [EllipticEnvelope API 文檔](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html)
- [Novelty and Outlier Detection 用戶指南](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [Covariance Estimation 模組](https://scikit-learn.org/stable/modules/covariance.html)

### 8.3 化工應用案例

1. **Kourti, T., & MacGregor, J. F. (1995)**. "Process analysis, monitoring and diagnosis, using multivariate projection methods." *Chemometrics and Intelligent Laboratory Systems*, 28(1), 3-21.
   - 多變量統計製程監控經典文獻

2. **Chiang, L. H., Russell, E. L., & Braatz, R. D. (2000)**. *Fault Detection and Diagnosis in Industrial Systems*. Springer.
   - 化工製程故障檢測專書

3. **Qin, S. J. (2012)**. "Survey on data-driven industrial process monitoring and diagnosis." *Annual Reviews in Control*, 36(2), 220-234.
   - 數據驅動製程監控綜述

### 8.4 線上資源

- [scikit-learn 異常檢測教學](https://scikit-learn.org/stable/auto_examples/applications/plot_outlier_detection_wine.html)
- [Robust Covariance Estimation Tutorial](https://scikit-learn.org/stable/auto_examples/covariance/plot_robust_vs_empirical_covariance.html)
- [Mahalanobis Distance Explained](https://www.machinelearningplus.com/statistics/mahalanobis-distance/)

---

## 附錄 A：數學推導補充

### A.1 馬氏距離與歐氏距離的關係

當共變異數矩陣為單位矩陣 $\mathbf{\Sigma} = \mathbf{I}$ 時，馬氏距離退化為歐氏距離：

$$
D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \mathbf{I}^{-1} (\mathbf{x} - \boldsymbol{\mu})} = \|\mathbf{x} - \boldsymbol{\mu}\|_2
$$

### A.2 橢圓方程的標準形式

透過特徵值分解，共變異數矩陣可表示為：

$$
\mathbf{\Sigma} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T
$$

其中 $\mathbf{V}$ 是特徵向量矩陣， $\mathbf{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_d)$ 是特徵值矩陣。

將數據轉換至主軸座標系：

$$
\mathbf{y} = \mathbf{V}^T (\mathbf{x} - \boldsymbol{\mu})
$$

橢圓方程簡化為：

$$
\sum_{i=1}^{d} \frac{y_i^2}{\lambda_i} = c^2
$$

這是標準橢圓方程，各軸半徑為 $c\sqrt{\lambda_i}$ 。

### A.3 卡方分布的百分位數

對於顯著性水準 $\alpha$ ，卡方分布的 $(1-\alpha)$ 百分位數為 $\chi^2_{d, 1-\alpha}$ 。

常用值（自由度 $d=2$ ）：
- 95% 信賴區間： $\chi^2_{2, 0.95} = 5.991$
- 99% 信賴區間： $\chi^2_{2, 0.99} = 9.210$

對應的馬氏距離閾值：
- 95%： $D_M = \sqrt{5.991} \approx 2.448$
- 99%： $D_M = \sqrt{9.210} \approx 3.035$

---

## 附錄 B：完整程式碼範例

完整的化工應用範例程式碼請參考配套的 Jupyter Notebook：
**`Unit07_Elliptic_Envelope.ipynb`**

範例內容包含：
- 批次反應器數據生成
- 橢圓包絡模型訓練與超參數調整
- 多種可視化分析
- 與其他異常檢測方法的比較
- 實時監控儀表板實作

---

**課程結束**

恭喜您完成橢圓包絡 (Elliptic Envelope) 異常檢測的學習！現在您已具備：

✅ 橢圓包絡的理論基礎與數學原理  
✅ Python 實作與參數調整技巧  
✅ 化工製程應用的實務經驗  
✅ 與其他方法的比較與選擇能力

建議接下來：
1. 完成配套的 Jupyter Notebook 練習
2. 嘗試將橢圓包絡應用於您的實際數據
3. 學習 Unit07 的其他異常檢測方法（Isolation Forest、One-Class SVM、LOF）
4. 探索多方法集成的異常檢測策略

**下一單元預告**：Unit08 關聯規則學習 (Association Rule Learning)

---
