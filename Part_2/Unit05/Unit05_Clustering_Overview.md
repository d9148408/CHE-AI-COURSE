# Unit05 分群分析總覽 (Clustering Overview)

## 1. 分群分析簡介

### 1.1 什麼是分群分析？

分群分析 (Clustering) 是一種非監督式學習方法，旨在將相似的數據點分組到同一群集 (cluster) 中，使得同一群集內的數據點相似度高，而不同群集之間的數據點相似度低。與監督式學習不同，分群分析不需要預先標記的訓練數據，而是根據數據本身的特徵自動發現數據中的內在結構和模式。

### 1.2 分群分析的核心概念

**相似度與距離度量**

分群分析的基礎在於如何定義數據點之間的「相似度」或「距離」。常用的距離度量包括：

- **歐幾里得距離 (Euclidean Distance)**：最常用的距離度量，適用於連續變數

$$
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$

- **曼哈頓距離 (Manhattan Distance)**：適用於高維度數據或存在異常值的情況

$$
d(x, y) = \sum_{i=1}^{n} |x_i - y_i|
$$

- **餘弦相似度 (Cosine Similarity)**：適用於文本數據或高維稀疏數據

$$
\text{similarity}(x, y) = \frac{x \cdot y}{\|x\| \|y\|}
$$

**群集的品質評估**

良好的分群結果應該具備以下特性：
- **高內聚性 (High Cohesion)**：同一群集內的數據點應該彼此相近
- **低耦合性 (Low Coupling)**：不同群集之間的數據點應該明顯分離
- **可解釋性 (Interpretability)**：群集應該具有明確的物理或化學意義

### 1.3 分群分析的目標

在化學工程領域中，分群分析的主要目標包括：

1. **模式識別**：識別製程中的不同操作模式或產品類型
2. **異常檢測**：發現偏離正常操作的異常模式
3. **數據探索**：理解高維數據的內在結構
4. **特徵提取**：將相似的操作條件歸類，簡化後續分析
5. **決策支持**：為製程優化和控制提供依據

---

## 2. 化工領域應用場景

### 2.1 製程操作模式識別

**應用背景**

化工製程通常具有多種操作模式，例如啟動、正常運行、停機、產品切換等。透過分群分析，可以自動識別這些不同的操作模式，幫助工程師理解製程行為。

**實際案例**

- **反應器多模式操作**：聚合反應器可能在不同溫度、壓力下生產不同等級的產品，分群分析可以識別這些不同的操作區間
- **蒸餾塔操作優化**：根據進料組成和產品規格的變化，蒸餾塔可能在多種操作條件下運行，分群分析可以識別最佳操作窗口
- **批次製程分析**：在批次製程中，每批次的操作軌跡可能略有不同，分群分析可以將相似的批次歸類，找出成功批次的共同特徵

### 2.2 產品品質分類

**應用背景**

化工產品的品質指標通常包含多個維度（如純度、黏度、分子量分布等），分群分析可以將產品自動分類為不同的品質等級。

**實際案例**

- **聚合物產品分級**：根據分子量、分子量分布、熔融指數等指標，將聚合物產品分為不同等級
- **石化產品分類**：根據辛烷值、蒸氣壓、密度等性質，將汽油產品分類
- **精細化學品品質控制**：根據多項品質指標，識別合格品、優等品和不良品

### 2.3 異常操作模式探索

**應用背景**

化工製程中可能出現各種異常情況（設備故障、原料品質異常、操作失誤等），分群分析可以幫助識別這些異常模式。

**實際案例**

- **設備健康監控**：透過分群分析，識別設備正常運行和異常運行的模式
- **製程安全監控**：識別可能導致安全事故的異常操作條件
- **品質偏差分析**：當產品品質出現偏差時，分群分析可以幫助找出異常操作的共同特徵

### 2.4 溶劑與配方篩選

**應用背景**

在新產品開發或製程優化中，需要從大量候選溶劑或配方中篩選出最佳選項。分群分析可以根據物理化學性質將候選對象分類，縮小篩選範圍。

**實際案例**

- **綠色溶劑篩選**：根據環境友善性、溶解能力、成本等指標，將候選溶劑分類
- **催化劑配方優化**：根據催化活性、選擇性、穩定性等指標，將不同配方分類
- **製程添加劑選擇**：根據效能和成本指標，篩選最佳添加劑組合

### 2.5 製程監控與診斷

**應用背景**

現代化工廠產生大量實時數據，分群分析可以用於製程狀態監控和故障診斷。

**實際案例**

- **多變量統計製程控制 (MSPC)**：結合 PCA 和分群分析，監控製程狀態
- **故障診斷系統**：根據歷史故障數據，建立故障模式庫，當新數據出現時，透過分群判斷故障類型
- **能源效率分析**：識別高能耗和低能耗的操作模式，優化能源使用

---

## 3. sklearn 模組中的分群方法

scikit-learn (sklearn) 提供了豐富的分群算法，每種算法都有其特定的適用場景和優缺點。本單元將介紹以下四種主要的分群方法：

### 3.1 K-平均演算法 (K-Means Clustering)

**核心原理**

K-Means 是最常用的分群算法之一，其目標是將數據分為 $K$ 個群集，使得每個群集內的數據點到群集中心的距離總和最小。

**算法步驟**

1. 隨機初始化 $K$ 個群集中心
2. 將每個數據點分配到最近的群集中心
3. 重新計算每個群集的中心（群集內所有點的平均值）
4. 重複步驟 2-3，直到群集中心不再變化或達到最大迭代次數

**目標函數**

$$
J = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2
$$

其中 $C_i$ 是第 $i$ 個群集， $\mu_i$ 是第 $i$ 個群集的中心。

**sklearn 實現**

```python
from sklearn.cluster import KMeans

# 建立模型
kmeans = KMeans(n_clusters=3, random_state=42)

# 訓練模型
kmeans.fit(X)

# 預測群集標籤
labels = kmeans.predict(X)
```

### 3.2 階層式分群 (Hierarchical Clustering)

**核心原理**

階層式分群不需要預先指定群集數量，而是建立一個樹狀結構（樹狀圖，Dendrogram），從中可以選擇不同層級的分群結果。

**兩種策略**

- **凝聚式 (Agglomerative)**：由下而上，從每個數據點作為獨立群集開始，逐步合併最相似的群集
- **分裂式 (Divisive)**：由上而下，從所有數據點作為一個群集開始，逐步分割

**連結方法**

定義群集之間的距離有多種方式：
- **單連結 (Single Linkage)**：兩群集最近點之間的距離
- **全連結 (Complete Linkage)**：兩群集最遠點之間的距離
- **平均連結 (Average Linkage)**：兩群集所有點對之間的平均距離
- **Ward 連結**：合併後群集內變異數增量最小

**sklearn 實現**

```python
from sklearn.cluster import AgglomerativeClustering

# 建立模型
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')

# 訓練並預測
labels = hierarchical.fit_predict(X)
```

### 3.3 基於密度的分群 (DBSCAN)

**核心原理**

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) 根據數據點的密度進行分群，能夠發現任意形狀的群集，並自動識別噪音點。

**關鍵參數**

- **eps ( $\epsilon$ )**：鄰域半徑，定義一個點的鄰域範圍
- **min_samples**：核心點的最小鄰居數量

**點的類型**

- **核心點 (Core Point)**：在 $\epsilon$ 範圍內至少有 min_samples 個鄰居
- **邊界點 (Border Point)**：不是核心點，但在某個核心點的 $\epsilon$ 範圍內
- **噪音點 (Noise Point)**：既不是核心點也不是邊界點

**算法特點**

- 不需要預先指定群集數量
- 能夠識別噪音點（標記為 -1）
- 能夠發現任意形狀的群集
- 對參數敏感，需要仔細調整

**sklearn 實現**

```python
from sklearn.cluster import DBSCAN

# 建立模型
dbscan = DBSCAN(eps=0.5, min_samples=5)

# 訓練並預測
labels = dbscan.fit_predict(X)
```

### 3.4 高斯混合模型 (Gaussian Mixture Models, GMM)

**核心原理**

GMM 假設數據由多個高斯分布混合而成，每個群集對應一個高斯分布。與 K-Means 不同，GMM 是一種「軟分群」方法，為每個數據點計算屬於各群集的機率。

**數學模型**

假設數據由 $K$ 個高斯分布混合生成：

$$
p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)
$$

其中：
- $\pi_k$ 是第 $k$ 個高斯分布的混合係數（權重）
- $\mu_k$ 是第 $k$ 個高斯分布的均值向量
- $\Sigma_k$ 是第 $k$ 個高斯分布的協方差矩陣

**參數估計**

使用 EM (Expectation-Maximization) 算法估計模型參數：
- **E-步驟**：計算每個數據點屬於各群集的後驗機率
- **M-步驟**：根據後驗機率更新模型參數

**sklearn 實現**

```python
from sklearn.mixture import GaussianMixture

# 建立模型
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)

# 訓練模型
gmm.fit(X)

# 預測群集標籤
labels = gmm.predict(X)

# 預測機率
probabilities = gmm.predict_proba(X)
```

---

## 4. 分群方法比較與選擇指南

### 4.1 各方法優缺點對照表

| 方法 | 優點 | 缺點 | 適用場景 |
|------|------|------|----------|
| **K-Means** | • 簡單高效<br>• 易於理解與實現<br>• 適合大規模數據<br>• 收斂快速 | • 需預先指定群集數量<br>• 假設群集為球形<br>• 對初始值敏感<br>• 對異常值敏感 | • 反應器多模式操作識別（已知模式數）<br>• 大規模數據快速分群<br>• 群集形狀接近球形 |
| **Hierarchical** | • 不需預先指定群集數量<br>• 提供樹狀圖視覺化<br>• 可選擇不同層級的分群<br>• 能揭示數據階層結構 | • 計算複雜度高 ( $O(n^2 \log n)$ 至 $O(n^3)$ )<br>• 不適合大規模數據<br>• 一旦合併無法撤銷<br>• 對噪音敏感 | • 產品分類體系建立（需要階層關係）<br>• 小規模數據探索<br>• 需要樹狀圖視覺化 |
| **DBSCAN** | • 不需預先指定群集數量<br>• 能發現任意形狀群集<br>• 能自動識別噪音<br>• 對異常值魯棒 | • 對參數敏感（eps, min_samples）<br>• 密度不均勻時效果差<br>• 高維數據困難<br>• 無法處理密度差異大的群集 | • 異常操作模式探索（未知模式數，有噪音）<br>• 群集形狀不規則<br>• 需要識別異常值 |
| **GMM** | • 提供機率歸屬（軟分群）<br>• 理論基礎扎實<br>• 可處理橢圓形群集<br>• 提供不確定性估計 | • 需預先指定群集數量<br>• 計算複雜度較高<br>• 可能陷入局部最優<br>• 對初始值敏感 | • 多產品品質分布建模（需要機率評估）<br>• 需要不確定性量化<br>• 群集有不同形狀和大小 |

### 4.2 演算法選擇決策樹

```
開始
│
├─ 是否已知群集數量？
│  │
│  ├─ 是 ──┐
│  │       │
│  └─ 否 ──┼─ 是否需要識別噪音點？
│           │  │
│           │  ├─ 是 → DBSCAN
│           │  │
│           │  └─ 否 ──┐
│           │           │
│           └───────────┼─ 數據量大小？
│                       │  │
│                       │  ├─ 小（< 10000） → Hierarchical Clustering
│                       │  │
│                       │  └─ 大（≥ 10000） → 使用 K-Means 嘗試多個 K 值
│                       │
│                       └─ 是否需要機率評估？
│                          │
│                          ├─ 是 → GMM
│                          │
│                          └─ 否 ──┐
│                                   │
│                                   ├─ 數據量大小？
│                                   │  │
│                                   │  ├─ 小（< 10000） → 視需求選擇
│                                   │  │
│                                   │  └─ 大（≥ 10000） → K-Means
│                                   │
│                                   └─ 群集形狀？
│                                      │
│                                      ├─ 球形 → K-Means
│                                      │
│                                      ├─ 橢圓形 → GMM
│                                      │
│                                      └─ 任意形狀 → DBSCAN
```

### 4.3 化工應用場景對照

| 應用場景 | 推薦方法 | 理由 |
|----------|----------|------|
| **反應器多模式操作識別** | K-Means | 通常已知操作模式數量（如啟動、正常、停機），數據量大，需要快速分群 |
| **產品品質分級** | K-Means / GMM | K-Means 適合明確分級，GMM 適合需要機率評估的情況 |
| **製程異常檢測** | DBSCAN | 不需預先知道異常類型數量，能自動識別噪音點 |
| **批次製程軌跡分析** | Hierarchical | 需要理解批次之間的相似性階層關係 |
| **溶劑篩選分類** | K-Means / Hierarchical | K-Means 適合大規模篩選，Hierarchical 適合建立溶劑分類體系 |
| **多產品品質分布建模** | GMM | 需要估計每個產品品質的機率分布和不確定性 |
| **設備健康狀態監控** | K-Means / DBSCAN | K-Means 適合已知狀態類型，DBSCAN 適合探索未知異常模式 |
| **能源效率分析** | K-Means | 需要快速分類高能耗和低能耗操作模式 |

### 4.4 實務選擇建議

**第一步：明確分析目標**
- 是探索性分析還是已有預期的群集？
- 需要識別異常值嗎？
- 需要機率評估嗎？
- 需要可解釋的階層結構嗎？

**第二步：考慮數據特性**
- 數據規模大小（樣本數和特徵數）
- 群集形狀（球形、橢圓形、不規則）
- 是否存在噪音和異常值
- 各群集的密度是否均勻

**第三步：評估計算資源**
- K-Means：最快，適合大規模數據
- DBSCAN：中等速度，適合中等規模數據
- GMM：較慢，適合中小規模數據
- Hierarchical：最慢，僅適合小規模數據

**第四步：多方法比較**

建議在實際應用中嘗試多種方法，透過評估指標和領域知識選擇最佳結果：

```python
# 嘗試多種方法
methods = {
    'K-Means': KMeans(n_clusters=3),
    'GMM': GaussianMixture(n_components=3),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
    'Hierarchical': AgglomerativeClustering(n_clusters=3)
}

results = {}
for name, model in methods.items():
    labels = model.fit_predict(X)
    results[name] = evaluate_clustering(X, labels)
```

---

## 5. 資料前處理

分群分析對數據的尺度和分布非常敏感，適當的資料前處理是獲得良好分群結果的關鍵。

### 5.1 資料標準化 (Standardization)

**目的**

將不同尺度的特徵轉換為相同尺度，使每個特徵的均值為 0，標準差為 1。

**數學公式**

$$
z = \frac{x - \mu}{\sigma}
$$

其中 $\mu$ 是均值， $\sigma$ 是標準差。

**何時使用**

- 特徵具有不同的單位和尺度（如溫度 °C 和壓力 bar）
- 使用基於距離的分群方法（K-Means、Hierarchical、DBSCAN）
- 數據近似常態分布

**sklearn 實現**

```python
from sklearn.preprocessing import StandardScaler

# 建立標準化器
scaler = StandardScaler()

# 訓練並轉換數據
X_scaled = scaler.fit_transform(X)

# 或分步驟
scaler.fit(X)
X_scaled = scaler.transform(X)
```

**化工應用範例**

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 反應器操作數據（不同尺度）
data = pd.DataFrame({
    'Temperature_C': [80, 85, 90, 95, 100],      # 溫度（°C）
    'Pressure_bar': [5.0, 5.5, 6.0, 6.5, 7.0],   # 壓力（bar）
    'FlowRate_L_min': [100, 120, 140, 160, 180], # 流量（L/min）
    'Conversion_pct': [75, 80, 85, 90, 95]       # 轉化率（%）
})

# 標準化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

print("原始數據統計：")
print(data.describe())
print("\n標準化後數據統計：")
print(pd.DataFrame(data_scaled, columns=data.columns).describe())
```

### 5.2 資料正規化 (Normalization)

**目的**

將每個特徵的值縮放到特定範圍（通常是 [0, 1] 或 [-1, 1]）。

**數學公式**

Min-Max 正規化：

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

**何時使用**

- 數據不符合常態分布
- 特徵的範圍差異極大
- 需要保留數據的原始分布形狀
- 對異常值敏感度較低的場景

**sklearn 實現**

```python
from sklearn.preprocessing import MinMaxScaler

# 建立正規化器（縮放至 [0, 1]）
scaler = MinMaxScaler(feature_range=(0, 1))

# 訓練並轉換數據
X_normalized = scaler.fit_transform(X)

# 自訂範圍（例如 [-1, 1]）
scaler = MinMaxScaler(feature_range=(-1, 1))
X_normalized = scaler.fit_transform(X)
```

**化工應用範例**

```python
from sklearn.preprocessing import MinMaxScaler

# 品質指標數據（不同範圍）
quality_data = pd.DataFrame({
    'Purity_pct': [95, 96, 97, 98, 99],           # 純度（%）95-99
    'Viscosity_cP': [10, 15, 20, 25, 30],         # 黏度（cP）10-30
    'MolecularWeight_kDa': [50, 75, 100, 125, 150] # 分子量（kDa）50-150
})

# 正規化至 [0, 1]
scaler = MinMaxScaler()
quality_normalized = scaler.fit_transform(quality_data)

print("正規化後數據範圍：")
print(pd.DataFrame(quality_normalized, columns=quality_data.columns).describe())
```

### 5.3 強健縮放 (Robust Scaling)

**目的**

使用中位數和四分位數進行縮放，對異常值較不敏感。

**數學公式**

$$
x' = \frac{x - \text{median}(x)}{\text{IQR}(x)}
$$

其中 IQR (Interquartile Range) = Q3 - Q1

**何時使用**

- 數據包含異常值
- 數據分布偏態
- 需要保持異常值的相對位置

**sklearn 實現**

```python
from sklearn.preprocessing import RobustScaler

# 建立強健縮放器
scaler = RobustScaler()

# 訓練並轉換數據
X_robust = scaler.fit_transform(X)
```

### 5.4 類別變數編碼

當數據包含類別變數時，需要將其轉換為數值形式。

#### 5.4.1 One-Hot 編碼

**目的**

將類別變數轉換為二元向量，每個類別對應一個維度。

**sklearn 實現**

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# 類別變數數據
categorical_data = np.array([['Reactor_A'], ['Reactor_B'], ['Reactor_A'], 
                              ['Reactor_C'], ['Reactor_B']])

# 建立編碼器
encoder = OneHotEncoder(sparse_output=False)

# 訓練並轉換
encoded_data = encoder.fit_transform(categorical_data)

print("編碼後：")
print(encoded_data)
print("類別名稱：", encoder.categories_)
```

**化工應用範例**

```python
# 包含類別變數的製程數據
process_data = pd.DataFrame({
    'Reactor': ['A', 'B', 'A', 'C', 'B'],
    'Product': ['P1', 'P2', 'P1', 'P3', 'P2'],
    'Temperature': [80, 85, 82, 90, 87],
    'Pressure': [5.0, 5.5, 5.2, 6.0, 5.7]
})

# 分離數值和類別變數
numerical_features = ['Temperature', 'Pressure']
categorical_features = ['Reactor', 'Product']

# 處理類別變數
encoder = OneHotEncoder(sparse_output=False)
categorical_encoded = encoder.fit_transform(process_data[categorical_features])

# 處理數值變數
scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(process_data[numerical_features])

# 合併
X_processed = np.hstack([numerical_scaled, categorical_encoded])
```

#### 5.4.2 Label 編碼

**目的**

將類別變數轉換為整數標籤（0, 1, 2, ...）。

**注意事項**

Label 編碼會引入順序關係，可能不適合分群分析。建議優先使用 One-Hot 編碼。

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(['A', 'B', 'A', 'C', 'B'])
# 輸出：[0, 1, 0, 2, 1]
```

### 5.5 處理缺失值

**常用策略**

```python
from sklearn.impute import SimpleImputer

# 使用均值填補
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# 使用中位數填補（對異常值較強健）
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# 使用最頻繁值填補（適合類別變數）
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = imputer.fit_transform(X)
```

### 5.6 資料前處理完整流程範例

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 定義數值和類別特徵
numerical_features = ['Temperature', 'Pressure', 'FlowRate']
categorical_features = ['Reactor', 'Product']

# 建立數值特徵處理管線
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# 建立類別特徵處理管線
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

# 合併處理管線
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# 應用前處理
X_processed = preprocessor.fit_transform(data)
```

---

## 6. 模型評估指標

分群分析的評估比監督式學習更具挑戰性，因為通常沒有真實標籤。評估指標分為內部評估（無需標籤）和外部評估（需要標籤）。

### 6.1 內部評估指標（無需標籤）

#### 6.1.1 輪廓係數 (Silhouette Score)

**定義**

輪廓係數衡量每個樣本與其所屬群集的相似度，以及與其他群集的分離度。

**數學公式**

對於樣本 $i$ ：

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

其中：
- $a(i)$ ：樣本 $i$ 與同群集內其他樣本的平均距離（內聚度）
- $b(i)$ ：樣本 $i$ 與最近的其他群集樣本的平均距離（分離度）

**取值範圍**

- 範圍：[-1, 1]
- $s(i) \approx 1$ ：樣本分群良好，內聚且遠離其他群集
- $s(i) \approx 0$ ：樣本位於兩個群集的邊界
- $s(i) < 0$ ：樣本可能被分配到錯誤的群集

**sklearn 實現**

```python
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt

# 計算整體輪廓係數
score = silhouette_score(X, labels)
print(f"Silhouette Score: {score:.3f}")

# 計算每個樣本的輪廓係數
sample_scores = silhouette_samples(X, labels)

# 視覺化輪廓圖
def plot_silhouette(X, labels):
    n_clusters = len(np.unique(labels))
    silhouette_vals = silhouette_samples(X, labels)
    
    y_lower = 10
    for i in range(n_clusters):
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()
        
        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, cluster_silhouette_vals,
                          alpha=0.7, label=f'Cluster {i}')
        y_lower = y_upper + 10
    
    plt.xlabel('Silhouette Coefficient')
    plt.ylabel('Cluster')
    plt.axvline(x=silhouette_score(X, labels), color='red', linestyle='--')
    plt.legend()
    plt.show()
```

**化工應用解釋**

在反應器操作模式識別中，高輪廓係數表示不同操作模式（如啟動、正常、停機）被清楚區分。

#### 6.1.2 Davies-Bouldin 指數 (DB Index)

**定義**

DB 指數衡量群集內部的緊密度與群集之間的分離度的比率。

**數學公式**

$$
\text{DB} = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} \left( \frac{s_i + s_j}{d_{ij}} \right)
$$

其中：
- $s_i$ ：群集 $i$ 內樣本到中心的平均距離
- $d_{ij}$ ：群集 $i$ 和 $j$ 中心之間的距離
- $K$ ：群集數量

**取值範圍**

- 範圍：[0, ∞)
- 值越小越好
- 0 表示完美分群

**sklearn 實現**

```python
from sklearn.metrics import davies_bouldin_score

# 計算 DB 指數
db_score = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin Index: {db_score:.3f}")
```

**化工應用解釋**

在產品品質分類中，低 DB 指數表示不同品質等級的產品被明確區分，且每個等級內部的品質一致性高。

#### 6.1.3 Calinski-Harabasz 指數 (CH Index)

**定義**

CH 指數（也稱為方差比準則）衡量群集間離散度與群集內離散度的比率。

**數學公式**

$$
\text{CH} = \frac{\text{trace}(B_k)}{\text{trace}(W_k)} \times \frac{n - K}{K - 1}
$$

其中：
- $B_k$ ：群集間離散矩陣
- $W_k$ ：群集內離散矩陣
- $n$ ：樣本總數
- $K$ ：群集數量

**取值範圍**

- 範圍：[0, ∞)
- 值越大越好
- 高值表示群集緊密且分離良好

**sklearn 實現**

```python
from sklearn.metrics import calinski_harabasz_score

# 計算 CH 指數
ch_score = calinski_harabasz_score(X, labels)
print(f"Calinski-Harabasz Index: {ch_score:.3f}")
```

#### 6.1.4 慣性 / 群集內平方和 (Inertia / WCSS)

**定義**

慣性衡量每個樣本到其所屬群集中心的距離平方和。

**數學公式**

$$
\text{Inertia} = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2
$$

**取值範圍**

- 範圍：[0, ∞)
- 值越小越好（但需注意過擬合）
- 用於手肘法 (Elbow Method) 選擇最佳群集數量

**sklearn 實現**

```python
from sklearn.cluster import KMeans

# K-Means 模型自動計算慣性
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
inertia = kmeans.inertia_

print(f"Inertia: {inertia:.3f}")

# 手肘法：嘗試不同的 K 值
inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# 繪製手肘圖
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()
```

**化工應用解釋**

在溶劑篩選中，使用手肘法可以找到最佳的溶劑分類數量，平衡分類的細緻度和實用性。

### 6.2 外部評估指標（需要真實標籤）

外部評估指標用於有真實標籤的情況（如驗證數據集或基準測試），衡量分群結果與真實標籤的一致性。

#### 6.2.1 調整蘭德指數 (Adjusted Rand Index, ARI)

**定義**

ARI 衡量兩種分群結果的相似度，並調整了隨機分群的影響。

**數學公式**

$$
\text{ARI} = \frac{\text{RI} - E[\text{RI}]}{\max(\text{RI}) - E[\text{RI}]}
$$

**取值範圍**

- 範圍：[-1, 1]
- 1：完美匹配
- 0：隨機分群
- 負值：比隨機分群更差

**sklearn 實現**

```python
from sklearn.metrics import adjusted_rand_score

# 假設有真實標籤
true_labels = [0, 0, 1, 1, 2, 2]
predicted_labels = [0, 0, 1, 1, 1, 2]

# 計算 ARI
ari = adjusted_rand_score(true_labels, predicted_labels)
print(f"Adjusted Rand Index: {ari:.3f}")
```

#### 6.2.2 標準化互信息 (Normalized Mutual Information, NMI)

**定義**

NMI 衡量兩種分群結果之間的互信息，並標準化到 [0, 1] 範圍。

**取值範圍**

- 範圍：[0, 1]
- 1：完美匹配
- 0：完全獨立

**sklearn 實現**

```python
from sklearn.metrics import normalized_mutual_info_score

# 計算 NMI
nmi = normalized_mutual_info_score(true_labels, predicted_labels)
print(f"Normalized Mutual Information: {nmi:.3f}")
```

#### 6.2.3 Fowlkes-Mallows 指數 (FMI)

**定義**

FMI 是精確率和召回率的幾何平均數。

**數學公式**

$$
\text{FMI} = \sqrt{\frac{TP}{TP + FP} \times \frac{TP}{TP + FN}}
$$

**取值範圍**

- 範圍：[0, 1]
- 1：完美匹配
- 0：無重疊

**sklearn 實現**

```python
from sklearn.metrics import fowlkes_mallows_score

# 計算 FMI
fmi = fowlkes_mallows_score(true_labels, predicted_labels)
print(f"Fowlkes-Mallows Index: {fmi:.3f}")
```

### 6.3 穩定性評估

#### 6.3.1 Bootstrap Resampling 穩定性測試

**目的**

評估分群結果對數據擾動的穩定性。

**實現方法**

```python
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np

def clustering_stability(X, n_clusters=3, n_iterations=100):
    """
    使用 Bootstrap 方法評估分群穩定性
    """
    # 基準分群
    base_model = KMeans(n_clusters=n_clusters, random_state=42)
    base_labels = base_model.fit_predict(X)
    
    ari_scores = []
    
    for i in range(n_iterations):
        # Bootstrap 抽樣
        X_resampled, indices = resample(X, np.arange(len(X)), random_state=i)
        
        # 重新分群
        model = KMeans(n_clusters=n_clusters, random_state=42)
        resampled_labels = model.fit_predict(X_resampled)
        
        # 計算與基準的 ARI
        ari = adjusted_rand_score(base_labels[indices], resampled_labels)
        ari_scores.append(ari)
    
    return np.mean(ari_scores), np.std(ari_scores)

# 評估穩定性
mean_ari, std_ari = clustering_stability(X, n_clusters=3)
print(f"Stability (ARI): {mean_ari:.3f} ± {std_ari:.3f}")
```

#### 6.3.2 不同初始化的一致性檢驗

**目的**

評估分群算法對初始化的敏感度（特別是 K-Means 和 GMM）。

**實現方法**

```python
def initialization_consistency(X, n_clusters=3, n_runs=20):
    """
    評估不同初始化的分群一致性
    """
    labels_list = []
    
    for i in range(n_runs):
        model = KMeans(n_clusters=n_clusters, random_state=i)
        labels = model.fit_predict(X)
        labels_list.append(labels)
    
    # 計算所有配對的 ARI
    ari_scores = []
    for i in range(n_runs):
        for j in range(i+1, n_runs):
            ari = adjusted_rand_score(labels_list[i], labels_list[j])
            ari_scores.append(ari)
    
    return np.mean(ari_scores), np.std(ari_scores)

# 評估一致性
mean_ari, std_ari = initialization_consistency(X, n_clusters=3)
print(f"Initialization Consistency (ARI): {mean_ari:.3f} ± {std_ari:.3f}")
```

### 6.4 化工特定考量

在化工領域應用分群分析時，除了上述統計指標，還需要考慮以下方面：

#### 6.4.1 群集的工程可解釋性

- **物理意義**：每個群集是否對應明確的操作狀態或產品類型？
- **特徵重要性**：哪些製程變數對分群結果影響最大？
- **群集特徵**：每個群集的中心或典型特徵是什麼？

```python
# 分析群集特徵
def analyze_cluster_characteristics(data, labels):
    """
    分析每個群集的特徵統計
    """
    df = pd.DataFrame(data, columns=['Temperature', 'Pressure', 'FlowRate'])
    df['Cluster'] = labels
    
    # 每個群集的統計特徵
    cluster_stats = df.groupby('Cluster').agg(['mean', 'std', 'min', 'max'])
    
    print("Cluster Characteristics:")
    print(cluster_stats)
    
    return cluster_stats
```

#### 6.4.2 操作模式切換的合理性

- **轉換頻率**：群集之間的切換是否過於頻繁？
- **轉換路徑**：是否存在不合理的模式跳躍？
- **時序一致性**：時間序列數據的分群是否符合製程動態？

```python
def analyze_cluster_transitions(labels, time_index):
    """
    分析群集切換模式
    """
    transitions = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            transitions.append({
                'time': time_index[i],
                'from': labels[i-1],
                'to': labels[i]
            })
    
    print(f"Total transitions: {len(transitions)}")
    print(f"Transition rate: {len(transitions)/len(labels):.2%}")
    
    return pd.DataFrame(transitions)
```

#### 6.4.3 與製程知識的一致性

- **領域專家驗證**：分群結果是否與專家經驗一致？
- **物理約束**：是否違反已知的物理或化學規律？
- **歷史事件對應**：是否能對應已知的操作事件或故障？

### 6.5 綜合評估範例

```python
def comprehensive_evaluation(X, labels, true_labels=None):
    """
    綜合評估分群結果
    """
    print("="*50)
    print("Clustering Evaluation Report")
    print("="*50)
    
    # 內部指標
    print("\n[Internal Metrics]")
    sil_score = silhouette_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    
    print(f"Silhouette Score:        {sil_score:.3f} (higher is better)")
    print(f"Davies-Bouldin Index:    {db_score:.3f} (lower is better)")
    print(f"Calinski-Harabasz Index: {ch_score:.3f} (higher is better)")
    
    # 外部指標（如果有真實標籤）
    if true_labels is not None:
        print("\n[External Metrics]")
        ari = adjusted_rand_score(true_labels, labels)
        nmi = normalized_mutual_info_score(true_labels, labels)
        fmi = fowlkes_mallows_score(true_labels, labels)
        
        print(f"Adjusted Rand Index:     {ari:.3f}")
        print(f"Normalized Mutual Info:  {nmi:.3f}")
        print(f"Fowlkes-Mallows Index:   {fmi:.3f}")
    
    # 基本統計
    print("\n[Cluster Statistics]")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label}: {count} samples ({count/len(labels):.1%})")
    
    print("="*50)
```

---

## 7. 分群分析實務應用流程

### 7.1 完整工作流程

```
1. 問題定義
   ├─ 明確分析目標（探索、分類、異常檢測等）
   ├─ 確定領域知識和預期結果
   └─ 定義成功標準

2. 數據準備
   ├─ 數據收集與整合
   ├─ 數據清洗（處理缺失值、異常值）
   ├─ 特徵選擇與工程
   └─ 數據前處理（標準化、正規化）

3. 探索性數據分析 (EDA)
   ├─ 數據分布視覺化
   ├─ 特徵相關性分析
   ├─ 初步的群集傾向評估
   └─ 確定適合的距離度量

4. 模型選擇
   ├─ 根據數據特性選擇候選算法
   ├─ 確定關鍵超參數範圍
   └─ 考慮計算資源限制

5. 模型訓練與調整
   ├─ 嘗試多種算法和參數組合
   ├─ 使用評估指標選擇最佳模型
   ├─ 驗證穩定性和可重複性
   └─ 超參數優化

6. 結果分析與驗證
   ├─ 視覺化分群結果
   ├─ 分析群集特徵
   ├─ 領域專家驗證
   └─ 與製程知識對照

7. 應用與監控
   ├─ 將模型應用於新數據
   ├─ 持續監控分群品質
   ├─ 定期重新訓練模型
   └─ 整合至決策支持系統
```

### 7.2 Python 完整實作範例

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

# ===== 1. 問題定義 =====
print("Goal: Identify operating modes in a chemical reactor")
print("Expected clusters: 3 (Startup, Normal, Shutdown)\n")

# ===== 2. 數據準備 =====
# 假設數據已載入
# data = pd.read_csv('reactor_data.csv')

# 模擬數據（實際應用中使用真實數據）
np.random.seed(42)
n_samples = 300

# 模擬三種操作模式
startup = np.random.randn(100, 4) * 0.5 + [70, 4, 80, 60]
normal = np.random.randn(100, 4) * 0.5 + [90, 6, 120, 85]
shutdown = np.random.randn(100, 4) * 0.5 + [60, 3, 60, 50]

X = np.vstack([startup, normal, shutdown])
feature_names = ['Temperature', 'Pressure', 'FlowRate', 'Conversion']

data = pd.DataFrame(X, columns=feature_names)

# 數據清洗（檢查缺失值和異常值）
print(f"Missing values: {data.isnull().sum().sum()}")
print(f"Data shape: {data.shape}\n")

# ===== 3. 探索性數據分析 =====
# 數據分布視覺化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, col in enumerate(feature_names):
    ax = axes[idx//2, idx%2]
    ax.hist(data[col], bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 特徵相關性
correlation_matrix = data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# ===== 4. 數據前處理 =====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# ===== 5. 模型選擇與訓練 =====
models = {
    'K-Means': KMeans(n_clusters=3, random_state=42, n_init=10),
    'GMM': GaussianMixture(n_components=3, random_state=42),
    'Hierarchical': AgglomerativeClustering(n_clusters=3, linkage='ward'),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=10)
}

results = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training {name}...")
    print(f"{'='*50}")
    
    # 訓練模型
    if hasattr(model, 'fit_predict'):
        labels = model.fit_predict(X_scaled)
    else:
        model.fit(X_scaled)
        labels = model.predict(X_scaled)
    
    # 檢查是否有有效的群集
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        print(f"Warning: {name} found only {n_clusters} cluster(s)")
        continue
    
    # 計算評估指標
    sil_score = silhouette_score(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)
    ch_score = calinski_harabasz_score(X_scaled, labels)
    
    results[name] = {
        'labels': labels,
        'silhouette': sil_score,
        'davies_bouldin': db_score,
        'calinski_harabasz': ch_score,
        'n_clusters': n_clusters
    }
    
    print(f"Clusters found:          {n_clusters}")
    print(f"Silhouette Score:        {sil_score:.3f}")
    print(f"Davies-Bouldin Index:    {db_score:.3f}")
    print(f"Calinski-Harabasz Index: {ch_score:.3f}")

# ===== 6. 結果比較 =====
print(f"\n{'='*50}")
print("Model Comparison Summary")
print(f"{'='*50}")

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Silhouette': [results[m]['silhouette'] for m in results.keys()],
    'Davies-Bouldin': [results[m]['davies_bouldin'] for m in results.keys()],
    'Calinski-Harabasz': [results[m]['calinski_harabasz'] for m in results.keys()],
    'N_Clusters': [results[m]['n_clusters'] for m in results.keys()]
})

print(comparison_df.to_string(index=False))

# 選擇最佳模型（根據 Silhouette Score）
best_model = max(results.keys(), key=lambda x: results[x]['silhouette'])
print(f"\nBest Model (by Silhouette Score): {best_model}")

# ===== 7. 視覺化最佳模型結果 =====
best_labels = results[best_model]['labels']

# 使用 PCA 降維至 2D 進行視覺化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 5))

# 原始空間投影
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels, 
                      cmap='viridis', s=50, alpha=0.6, edgecolors='k')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title(f'{best_model} Clustering Result (PCA Projection)')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)

# 群集大小分布
plt.subplot(1, 2, 2)
unique_labels, counts = np.unique(best_labels, return_counts=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
plt.bar(unique_labels, counts, color=colors, edgecolor='black')
plt.xlabel('Cluster')
plt.ylabel('Number of Samples')
plt.title('Cluster Size Distribution')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('clustering_results.png', dpi=300, bbox_inches='tight')
plt.close()

# ===== 8. 群集特徵分析 =====
data['Cluster'] = best_labels
cluster_characteristics = data.groupby('Cluster')[feature_names].agg(['mean', 'std'])

print(f"\n{'='*50}")
print("Cluster Characteristics")
print(f"{'='*50}")
print(cluster_characteristics)

# 視覺化群集特徵
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, feature in enumerate(feature_names):
    ax = axes[idx//2, idx%2]
    data.boxplot(column=feature, by='Cluster', ax=ax)
    ax.set_xlabel('Cluster')
    ax.set_ylabel(feature)
    ax.set_title(f'{feature} by Cluster')
    plt.sca(ax)
    plt.xticks(rotation=0)

plt.suptitle('Feature Distribution by Cluster', y=1.02)
plt.tight_layout()
plt.savefig('cluster_characteristics.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAnalysis complete! Results saved to:")
print("- data_distribution.png")
print("- correlation_matrix.png")
print("- clustering_results.png")
print("- cluster_characteristics.png")
```

### 7.3 最佳實踐建議

#### 7.3.1 數據準備階段

✅ **應該做的**
- 仔細檢查和處理缺失值
- 識別並處理異常值（根據領域知識）
- 確保數據尺度一致（標準化或正規化）
- 選擇與分析目標相關的特徵
- 視覺化數據分布和關係

❌ **應該避免的**
- 盲目刪除異常值（可能是重要資訊）
- 使用未經預處理的原始數據
- 包含無關或冗餘的特徵
- 忽略數據的時間序列特性（如果適用）

#### 7.3.2 模型選擇階段

✅ **應該做的**
- 嘗試多種算法並比較結果
- 根據數據特性選擇合適的距離度量
- 考慮計算資源和時間限制
- 使用多個評估指標綜合判斷
- 驗證模型穩定性

❌ **應該避免的**
- 僅依賴單一評估指標
- 忽視領域知識和實際意義
- 過度追求統計指標而忽略可解釋性
- 使用預設參數而不調整

#### 7.3.3 結果分析階段

✅ **應該做的**
- 視覺化分群結果（2D/3D 投影）
- 分析每個群集的特徵統計
- 尋找群集的物理或化學意義
- 與領域專家討論結果
- 檢查結果的時間一致性

❌ **應該避免的**
- 僅依賴數值指標而不視覺化
- 忽略群集的工程意義
- 過度解讀統計結果
- 忽視不確定性和局限性

#### 7.3.4 應用部署階段

✅ **應該做的**
- 建立模型監控機制
- 定義模型更新策略
- 記錄模型版本和參數
- 提供使用文檔和解釋
- 設計異常處理機制

❌ **應該避免的**
- 部署後完全不監控
- 永久使用固定模型（數據漂移）
- 缺乏可追溯性
- 過度依賴自動化決策

### 7.4 常見問題與解決方案

| 問題 | 可能原因 | 解決方案 |
|------|----------|----------|
| 輪廓係數很低 | 群集重疊嚴重 | 嘗試不同的 K 值、特徵選擇、或不同算法 |
| K-Means 結果不穩定 | 初始化敏感 | 增加 `n_init` 參數、使用 K-Means++ 初始化 |
| DBSCAN 找不到群集 | eps 或 min_samples 設定不當 | 使用 K-distance 圖調整參數 |
| 群集大小嚴重不平衡 | 數據本身不平衡或算法不適合 | 考慮使用 GMM、調整類別權重、或重新採樣 |
| 高維數據分群效果差 | 維度詛咒 | 先進行降維（PCA、UMAP）再分群 |
| 分群結果無法解釋 | 特徵選擇不當 | 重新選擇有物理意義的特徵、諮詢領域專家 |
| 計算時間過長 | 數據量大或算法複雜 | 使用 Mini-Batch K-Means、降採樣、或並行計算 |

---

## 8. 總結

### 8.1 關鍵要點回顧

1. **分群分析是非監督式學習的核心方法**，用於發現數據中的內在結構和模式

2. **不同算法有不同的適用場景**：
   - K-Means：快速、簡單、適合大數據和球形群集
   - Hierarchical：提供階層結構、適合小數據
   - DBSCAN：發現任意形狀群集、自動識別噪音
   - GMM：提供機率評估、適合橢圓形群集

3. **資料前處理至關重要**：標準化、正規化、特徵選擇直接影響分群結果

4. **評估需要多維度**：結合內部指標、外部指標（如有）、穩定性測試和領域知識

5. **化工應用需要特別考慮**：工程可解釋性、操作合理性、與製程知識的一致性

### 8.2 學習路線圖

```
Unit05_Clustering_Overview (本單元)
   ↓
Unit05_K_Means
   ├─ K-Means 算法詳解
   ├─ 手肘法選擇 K 值
   ├─ Mini-Batch K-Means
   └─ 化工案例：反應器操作模式識別
   ↓
Unit05_Hierarchical_Clustering
   ├─ 凝聚式與分裂式分群
   ├─ 樹狀圖解讀
   ├─ 連結方法比較
   └─ 化工案例：產品分類體系建立
   ↓
Unit05_DBSCAN
   ├─ 密度分群原理
   ├─ 參數調整技巧
   ├─ 噪音點處理
   └─ 化工案例：異常操作模式探索
   ↓
Unit05_Gaussian_Mixture_Models
   ├─ GMM 理論基礎
   ├─ EM 算法原理
   ├─ 協方差類型選擇
   └─ 化工案例：多產品品質分布建模
   ↓
Unit05_Clustering_Homework
   └─ 綜合應用練習
```

### 8.3 延伸學習資源

**推薦書籍**
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
- "Python Machine Learning" by Sebastian Raschka

**線上資源**
- scikit-learn 官方文檔：https://scikit-learn.org/stable/modules/clustering.html
- Kaggle 分群教程和競賽
- GitHub 上的化工數據集和範例

**相關單元**
- Unit06：降維方法（PCA、t-SNE、UMAP）
- Unit07：異常檢測方法
- Unit08：關聯規則學習

### 8.4 下一步

完成本單元後，建議：

1. **實作練習**：使用提供的程式碼範例處理實際數據
2. **深入學習**：逐一學習 K-Means、Hierarchical、DBSCAN、GMM 的詳細內容
3. **專案實踐**：選擇一個化工領域的實際問題進行完整分析
4. **持續探索**：關注最新的分群算法和應用（如深度學習方法）

---

**本單元結束**

如有任何問題或需要進一步說明，請參考後續單元的詳細教材或諮詢授課教師。

