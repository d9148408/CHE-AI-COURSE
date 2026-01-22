# Unit06 降維 (Dimensionality Reduction) - 課程概述

## 課程目標

本單元將介紹降維 (Dimensionality Reduction) 技術在化工領域的應用。降維是非監督式學習中一項重要的技術，主要目的是在保留數據主要特徵的前提下，減少數據的維度。這不僅能幫助我們理解高維數據的內在結構，還能提升後續模型的效能與可解釋性。

### 學習目標
- 理解降維的基本概念與原理
- 掌握常見降維方法的數學基礎與應用場景
- 學會使用 `sklearn` 模組進行降維分析
- 了解如何評估降維模型的效果
- 將降維技術應用於化工製程監控與數據視覺化

---

## 1. 降維技術背景理論

### 1.1 什麼是降維？

降維 (Dimensionality Reduction) 是指將高維數據映射到低維空間的過程，同時盡可能保留原始數據的重要資訊。在化工領域，我們經常需要處理來自多個感測器的數據，這些數據可能包含數十甚至數百個變數，降維技術能幫助我們：

1. **簡化數據結構**：降低數據複雜度，便於視覺化與理解
2. **去除冗餘資訊**：消除變數間的相關性與噪音
3. **提升計算效率**：減少後續模型的訓練時間
4. **避免維度詛咒**：在高維空間中，數據點變得稀疏，距離度量失效
5. **特徵萃取**：發現數據的潛在結構與模式

### 1.2 維度詛咒 (Curse of Dimensionality)

當數據維度增加時，會產生以下問題：

- **數據稀疏性**：在高維空間中，數據點之間的距離變大，密度降低
- **計算複雜度**：所需的樣本數量與計算資源隨維度指數增長
- **模型泛化能力下降**：高維空間中容易發生過擬合

**數學表達**：

假設在 $d$ 維超立方體中均勻分布 $n$ 個點，要覆蓋 $p$ 比例的空間，所需的邊長比例為 $l = p^{1/d}$ 。當 $d$ 很大時， $l$ 接近 1，意味著需要覆蓋幾乎整個空間才能捕獲足夠的數據。

### 1.3 降維的數學原理

降維可以視為一個映射函數：

$$
f: \mathbb{R}^D \rightarrow \mathbb{R}^d, \quad d \ll D
$$

其中 $D$ 是原始維度， $d$ 是降維後的維度。

降維方法可分為兩大類：

#### 1.3.1 線性降維

線性降維假設數據的主要變異可以透過線性組合表達：

$$
\mathbf{y} = \mathbf{W}^T \mathbf{x}
$$

其中 $\mathbf{x} \in \mathbb{R}^D$ 是原始數據， $\mathbf{y} \in \mathbb{R}^d$ 是降維後的數據， $\mathbf{W} \in \mathbb{R}^{D \times d}$ 是投影矩陣。

**代表方法**：
- 主成分分析 (PCA)
- 線性判別分析 (LDA)
- 因子分析 (Factor Analysis)

#### 1.3.2 非線性降維

非線性降維能捕捉數據的複雜結構，適用於數據存在非線性關係的情況：

$$
\mathbf{y} = f(\mathbf{x}), \quad f \text{ is nonlinear}
$$

**代表方法**：
- Kernel PCA
- t-SNE (t-distributed Stochastic Neighbor Embedding)
- UMAP (Uniform Manifold Approximation and Projection)
- Autoencoder (深度學習方法)

---

## 2. 降維的應用場景

### 2.1 數據視覺化

高維數據難以直接視覺化，降維可以將數據投影到 2D 或 3D 空間，便於觀察數據分布、群集結構與異常點。

**適用情境**：
- 製程狀態監控面板
- 產品品質分布視覺化
- 多變數關聯性探索

### 2.2 製程監控與故障診斷

透過降維技術，可以將多個製程變數壓縮為少數幾個主成分，用於：
- 製程狀態追蹤
- 異常檢測
- 故障模式識別
- 即時監控系統

**關鍵優勢**：
- 減少監控變數數量
- 提升異常檢測靈敏度
- 降低誤報率

### 2.3 特徵工程

降維可作為特徵工程的一環，用於：
- 去除冗餘特徵
- 消除多重共線性
- 提升模型訓練效率
- 改善模型泛化能力

### 2.4 數據壓縮與儲存

在大規模製程數據的長期儲存中，降維可以：
- 減少儲存空間需求
- 加快數據讀取速度
- 降低數據傳輸成本

### 2.5 探索性數據分析 (EDA)

降維是 EDA 的重要工具，幫助我們：
- 理解數據的內在結構
- 發現隱藏的變數關係
- 識別數據中的主要變異來源

---

## 3. 化工領域應用案例

### 3.1 反應器製程監控

**應用背景**：
某化工廠的連續攪拌槽反應器 (CSTR) 配備 50 個感測器，包括溫度、壓力、流量、濃度等變數。操作人員難以同時監控所有變數。

**解決方案**：
- 使用 PCA 將 50 維數據降至 3 個主成分
- 建立 Hotelling's T² 控制圖與 SPE (Squared Prediction Error) 控制圖
- 當主成分超出控制界限時發出警報

**效益**：
- 降低 40% 的誤報率
- 提前 15 分鐘檢測到異常
- 簡化操作人員的監控負擔

### 3.2 產品品質預測

**應用背景**：
聚合物生產過程中，產品品質受 80 多個製程參數影響，導致品質預測模型訓練困難。

**解決方案**：
- 使用 PCA 提取前 10 個主成分 (解釋 85% 變異數)
- 以主成分作為輸入，建立迴歸模型預測產品品質
- 使用 Loading 分析找出關鍵影響因子

**效益**：
- 模型訓練時間減少 70%
- 預測 $R^2$ 從 0.72 提升至 0.88
- 識別出 5 個關鍵控制參數

### 3.3 批次製程軌跡分析

**應用背景**：
批次製程 (Batch Process) 的操作軌跡為高維時間序列數據，需要比較不同批次的操作模式。

**解決方案**：
- 使用動態 PCA (Dynamic PCA) 分析批次軌跡
- 使用 t-SNE 視覺化批次差異
- 識別成功批次與失敗批次的特徵

**效益**：
- 發現 3 種典型操作模式
- 提升良品率 12%
- 縮短新產品開發週期

### 3.4 觸媒性能評估

**應用背景**：
研究團隊測試 100 種觸媒配方，每種觸媒有 20 個物理化學性質與 5 個性能指標。

**解決方案**：
- 使用 UMAP 將觸媒特性降至 2D
- 在 2D 空間中視覺化性能表現
- 使用 Kernel PCA 捕捉非線性關係

**效益**：
- 快速識別高性能觸媒區域
- 發現性質-性能的非線性關聯
- 縮短篩選時間 50%

### 3.5 製程安全監控

**應用背景**：
石化廠需要即時監控潛在危險狀態，感測器數據多達 200 維。

**解決方案**：
- 使用 Isolation Forest 結合 PCA 進行異常檢測
- 建立多層次監控系統：正常操作、警告、危險
- 使用 UMAP 視覺化歷史事故資料

**效益**：
- 提前預警潛在危險狀態
- 降低緊急停機次數 30%
- 強化安全管理系統

---

## 4. sklearn 降維方法介紹

`scikit-learn` 提供了豐富的降維工具，涵蓋線性與非線性方法。以下介紹本單元將學習的四種主要降維技術。

### 4.1 主成分分析 (Principal Component Analysis, PCA)

#### 基本原理
PCA 是最經典的線性降維方法，透過尋找數據變異數最大的方向，將數據投影到新的正交座標系。

**數學表達**：

給定數據矩陣 $\mathbf{X} \in \mathbb{R}^{n \times D}$ ，PCA 尋找投影向量 $\mathbf{w}_1, \mathbf{w}_2, \ldots, \mathbf{w}_d$ 使得投影後的變異數最大：

$$
\mathbf{w}_1 = \arg\max_{\|\mathbf{w}\|=1} \text{Var}(\mathbf{X}\mathbf{w})
$$

等價於求解協方差矩陣的特徵值問題：

$$
\mathbf{C} = \frac{1}{n}\mathbf{X}^T\mathbf{X}, \quad \mathbf{C}\mathbf{w}_i = \lambda_i \mathbf{w}_i
$$

其中 $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_D$ 為特徵值， $\mathbf{w}_i$ 為對應的特徵向量 (主成分)。

#### 主要特點
- **線性投影**：假設數據的主要變異為線性結構
- **最大變異數**：保留數據變異最大的方向
- **正交性**：主成分之間互不相關
- **可解釋性強**：可透過 Loading 分析各變數的貢獻

#### 化工應用場景
- **製程監控與故障診斷**：建立 T² 與 SPE 控制圖
- **數據壓縮**：減少製程數據儲存空間
- **特徵萃取**：提取關鍵製程變數組合
- **多變數統計製程控制 (MSPC)**：監控多個相關變數

#### sklearn 實現
```python
from sklearn.decomposition import PCA

# 建立 PCA 模型
pca = PCA(n_components=2, random_state=42)

# 訓練並轉換數據
X_pca = pca.fit_transform(X)

# 查看解釋變異數比例
print(pca.explained_variance_ratio_)
```

---

### 4.2 核主成分分析 (Kernel PCA)

#### 基本原理
Kernel PCA 是 PCA 的非線性擴展，透過核函數 (Kernel Function) 將數據映射到高維特徵空間，再進行 PCA。

**數學表達**：

定義核函數 $k(\mathbf{x}_i, \mathbf{x}_j) = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle$ ，其中 $\phi$ 是映射函數。

核矩陣 $\mathbf{K}$ 的第 $(i,j)$ 元素為：

$$
K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)
$$

在特徵空間中進行 PCA，求解：

$$
\mathbf{K}\boldsymbol{\alpha}_i = \lambda_i \boldsymbol{\alpha}_i
$$

#### 常見核函數
1. **線性核**： $k(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j$ (退化為標準 PCA)
2. **多項式核**： $k(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T \mathbf{x}_j + c)^d$ 
3. **RBF 核 (徑向基函數)**： $k(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$ 
4. **Sigmoid 核**： $k(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\gamma \mathbf{x}_i^T \mathbf{x}_j + c)$ 

#### 主要特點
- **非線性降維**：捕捉數據的非線性結構
- **核技巧**：無需顯式計算高維映射
- **靈活性**：可透過選擇不同核函數適應不同數據結構
- **計算成本較高**：核矩陣計算複雜度為 $O(n^2)$ 

#### 化工應用場景
- **非線性製程特徵提取**：處理複雜反應動力學
- **化學反應路徑分析**：捕捉反應進程的非線性軌跡
- **觸媒性能評估**：處理性質-性能的非線性關係
- **複雜製程狀態識別**：捕捉多模式操作的非線性邊界

#### sklearn 實現
```python
from sklearn.decomposition import KernelPCA

# 建立 Kernel PCA 模型 (RBF 核)
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1, random_state=42)

# 訓練並轉換數據
X_kpca = kpca.fit_transform(X)
```

---

### 4.3 t-分布隨機鄰域嵌入 (t-SNE)

#### 基本原理
t-SNE 是一種非線性降維方法，主要用於數據視覺化。它透過保持數據點在高維空間與低維空間的局部相似度，將數據映射到低維空間。

**數學表達**：

在高維空間中，定義點 $i$ 與點 $j$ 的相似度 (條件機率)：

$$
p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}
$$

對稱化： $p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$ 

在低維空間中，使用 t-分布定義相似度：

$$
q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}
$$

最小化 KL 散度 (Kullback-Leibler Divergence)：

$$
KL(P\|Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
$$

#### 主要特點
- **局部結構保持**：專注於保持鄰近點的相似關係
- **視覺化優勢**：在 2D/3D 空間中產生清晰的群集結構
- **非確定性**：每次運行結果可能不同 (需設定隨機種子)
- **計算成本高**：複雜度為 $O(n^2)$ ，不適合大規模數據

#### 化工應用場景
- **高維製程數據視覺化**：呈現操作模式的群集結構
- **批次製程軌跡比較**：視覺化不同批次的差異
- **故障模式識別**：清晰顯示正常與異常狀態的分離
- **產品配方分析**：視覺化配方與性能的關聯

#### sklearn 實現
```python
from sklearn.manifold import TSNE

# 建立 t-SNE 模型
tsne = TSNE(n_components=2, perplexity=30, random_state=42)

# 訓練並轉換數據
X_tsne = tsne.fit_transform(X)
```

**重要參數**：
- `perplexity`：控制局部與全局結構的平衡 (通常 5-50)
- `learning_rate`：優化學習率 (通常 10-1000)
- `n_iter`：優化迭代次數 (至少 1000)

---

### 4.4 均勻流形近似與投影 (UMAP)

#### 基本原理
UMAP (Uniform Manifold Approximation and Projection) 是近年發展的降維方法，基於流形學習與拓撲數據分析理論。相較於 t-SNE，UMAP 更快速且更能保持全局結構。

**數學表達**：

UMAP 假設數據位於 Riemannian 流形上，透過模糊拓撲結構建立高維與低維空間的對應關係。

在高維空間中，構建模糊單純複形 (Fuzzy Simplicial Complex)，邊的權重為：

$$
w(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\frac{d(\mathbf{x}_i, \mathbf{x}_j) - \rho_i}{\sigma_i}\right)
$$

其中 $\rho_i$ 是最近鄰距離， $\sigma_i$ 是正規化參數。

在低維空間中，使用交叉熵損失函數優化：

$$
CE = \sum_{i,j} w_{ij} \log \frac{w_{ij}}{v_{ij}} + (1-w_{ij}) \log \frac{1-w_{ij}}{1-v_{ij}}
$$

#### 主要特點
- **速度快**：複雜度接近 $O(n)$ ，適合大規模數據
- **全局與局部結構兼顧**：比 t-SNE 更好地保持全局結構
- **可用於新數據**：支持 transform 方法 (t-SNE 不支持)
- **理論基礎紮實**：基於嚴謹的數學理論

#### 化工應用場景
- **大規模製程數據降維與視覺化**：處理數百萬筆數據
- **即時製程監控**：快速轉換新數據進行監控
- **多廠區數據整合**：保持全局結構以比較不同廠區
- **長期歷史數據分析**：處理數年的製程記錄

#### sklearn 實現 (需安裝 umap-learn)
```python
import umap

# 建立 UMAP 模型
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)

# 訓練並轉換數據
X_umap = reducer.fit_transform(X)

# 可對新數據進行轉換
X_new_umap = reducer.transform(X_new)
```

**重要參數**：
- `n_neighbors`：控制局部結構的尺度 (2-100)
- `min_dist`：控制低維空間點的緊密程度 (0.0-1.0)
- `metric`：距離度量方式 (euclidean, manhattan, cosine 等)

---

## 5. 降維方法對照表

### 5.1 線性 vs 非線性降維方法比較

| 比較項目 | 線性降維 (PCA) | 非線性降維 (Kernel PCA, t-SNE, UMAP) |
|---------|---------------|--------------------------------------|
| **數據假設** | 線性關係 | 非線性流形結構 |
| **計算複雜度** | 低 $O(D^2 n + D^3)$ | 高 $O(n^2)$ ~ $O(n \log n)$ |
| **可解釋性** | 高 (主成分有明確物理意義) | 低 (難以解釋降維後的維度) |
| **視覺化效果** | 一般 (可能無法分離群集) | 優秀 (群集結構清晰) |
| **全局結構保持** | 優秀 | 一般 (t-SNE 較差，UMAP 較好) |
| **局部結構保持** | 一般 | 優秀 |
| **新數據轉換** | 支持 | Kernel PCA / UMAP 支持，t-SNE 不支持 |
| **穩定性** | 確定性結果 | 隨機性 (需設定隨機種子) |
| **大規模數據** | 適合 | UMAP 適合，t-SNE 不適合 |

### 5.2 各方法優缺點對照表

| 方法 | 優點 | 缺點 | 適用數據規模 |
|------|------|------|--------------|
| **PCA** | • 計算快速<br>• 結果穩定<br>• 可解釋性強<br>• 支持反轉換<br>• 數學基礎紮實 | • 僅捕捉線性關係<br>• 對異常值敏感<br>• 假設主要變異為重要資訊<br>• 可能無法分離非線性群集 | 大規模<br>(> 100 萬樣本) |
| **Kernel PCA** | • 捕捉非線性結構<br>• 核函數選擇靈活<br>• 支持新數據轉換<br>• 理論完善 | • 計算成本高<br>• 核函數與參數選擇困難<br>• 可解釋性差<br>• 記憶體需求大 | 中小規模<br>(< 10 萬樣本) |
| **t-SNE** | • 視覺化效果極佳<br>• 群集分離清晰<br>• 保持局部結構優秀<br>• 適合探索性分析 | • 計算速度慢<br>• 不保持全局結構<br>• 不支持新數據轉換<br>• 參數敏感<br>• 每次結果可能不同 | 小規模<br>(< 1 萬樣本) |
| **UMAP** | • 速度快<br>• 保持全局與局部結構<br>• 支持新數據轉換<br>• 適合大規模數據<br>• 理論基礎嚴謹 | • 參數調整需要經驗<br>• 可解釋性一般<br>• 結果有隨機性<br>• 需額外安裝套件 | 大規模<br>(> 100 萬樣本) |

### 5.3 化工應用場景對照表

| 應用場景 | 推薦方法 | 原因說明 |
|---------|---------|---------|
| **製程監控與故障診斷** | PCA | • 線性降維，解釋性強<br>• 可建立統計控制圖 (T², SPE)<br>• 計算快速，適合即時監控<br>• 主成分有明確物理意義 |
| **非線性製程特徵提取** | Kernel PCA | • 捕捉複雜製程的非線性關係<br>• 適用於反應動力學分析<br>• 可用於後續建模 |
| **高維製程數據視覺化** | t-SNE | • 清晰呈現操作模式群集<br>• 適合探索性分析<br>• 群集邊界清晰<br>• 適合報告與簡報 |
| **大規模製程數據降維** | UMAP | • 處理速度快<br>• 保持全局結構利於跨廠區比較<br>• 可用於即時監控<br>• 支持新數據轉換 |
| **批次製程軌跡分析** | PCA / t-SNE | • PCA 用於動態特徵萃取<br>• t-SNE 用於軌跡視覺化比較 |
| **產品品質預測** | PCA | • 提取主成分作為模型輸入<br>• 減少多重共線性<br>• 提升模型泛化能力 |
| **觸媒篩選與配方優化** | UMAP / Kernel PCA | • 視覺化高維配方空間<br>• 捕捉非線性性質-性能關係 |
| **製程安全監控** | PCA + Isolation Forest | • PCA 降維後進行異常檢測<br>• 提升檢測靈敏度 |

### 5.4 演算法選擇決策樹

```
數據需要降維？
│
├─ 是 → 主要目的是什麼？
│      │
│      ├─ 視覺化 → 數據規模？
│      │          │
│      │          ├─ 小規模 (< 1萬) → t-SNE
│      │          └─ 大規模 (> 1萬) → UMAP
│      │
│      ├─ 製程監控/建模 → 數據關係？
│      │                  │
│      │                  ├─ 線性關係 → PCA
│      │                  └─ 非線性關係 → Kernel PCA 或 UMAP
│      │
│      ├─ 特徵萃取 → 需要解釋性？
│      │            │
│      │            ├─ 是 → PCA
│      │            └─ 否 → Kernel PCA 或 UMAP
│      │
│      └─ 數據壓縮 → PCA
│
└─ 否 → 直接使用原始數據
```

### 5.5 方法選擇指引

#### 選擇 PCA 的情境
✓ 需要解釋降維後的物理意義  
✓ 數據呈現線性關係  
✓ 需要快速計算 (即時監控)  
✓ 需要反轉換重建原始數據  
✓ 用於後續統計製程控制 (SPC)  

#### 選擇 Kernel PCA 的情境
✓ 數據呈現明顯非線性結構  
✓ 用於後續建模 (如迴歸、分類)  
✓ 數據規模中等 (< 10 萬)  
✓ 需要對新數據進行轉換  

#### 選擇 t-SNE 的情境
✓ 主要目的是視覺化  
✓ 數據規模較小 (< 1 萬)  
✓ 探索性數據分析  
✓ 不需要對新數據進行轉換  
✓ 可以接受較長計算時間  

#### 選擇 UMAP 的情境
✓ 數據規模大 (> 1 萬)  
✓ 需要快速計算  
✓ 需要保持全局結構  
✓ 需要對新數據進行轉換  
✓ 用於即時監控系統  

---

## 6. 資料前處理方法

降維演算法對數據尺度敏感，適當的前處理能顯著提升降維效果。以下介紹常用的前處理方法與在 sklearn 中的實現。

### 6.1 為什麼需要資料前處理？

1. **尺度差異**：不同變數的數值範圍可能差異極大 (如溫度 vs 壓力)
2. **變異數主導**：PCA 會被高變異數變數主導
3. **距離計算**：t-SNE 與 UMAP 依賴距離計算，需要統一尺度
4. **數值穩定性**：避免數值運算中的溢位或精度問題

### 6.2 標準化 (Standardization)

#### 原理
將數據轉換為均值為 0、標準差為 1 的分布 (Z-score 標準化)：

$$
z = \frac{x - \mu}{\sigma}
$$

其中 $\mu$ 是均值， $\sigma$ 是標準差。

#### 適用情境
- 數據近似常態分布
- 變數間尺度差異大
- **PCA、Kernel PCA** 的標準前處理
- 存在異常值但影響不嚴重

#### sklearn 實現
```python
from sklearn.preprocessing import StandardScaler

# 建立標準化器
scaler = StandardScaler()

# 訓練並轉換數據
X_scaled = scaler.fit_transform(X)

# 對新數據進行轉換
X_new_scaled = scaler.transform(X_new)
```

#### 化工應用範例
```python
# 製程數據標準化
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 假設有溫度 (300-400K)、壓力 (1-50 bar)、流量 (10-100 L/min)
data = pd.DataFrame({
    'Temperature': [350, 360, 355, 365],
    'Pressure': [10, 25, 15, 30],
    'Flow_Rate': [50, 60, 55, 65]
})

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

print("Original mean:", data.mean().values)
print("Scaled mean:", data_scaled.mean(axis=0))
print("Scaled std:", data_scaled.std(axis=0))
```

---

### 6.3 正規化 (Normalization)

#### 原理
將數據縮放到 [0, 1] 或 [-1, 1] 範圍 (Min-Max Scaling)：

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

#### 適用情境
- 數據分布不是常態分布
- 需要將數據限制在特定範圍
- **t-SNE、UMAP** 的前處理
- 存在明確的最大/最小值界限

#### sklearn 實現
```python
from sklearn.preprocessing import MinMaxScaler

# 建立正規化器 (預設範圍 [0, 1])
scaler = MinMaxScaler(feature_range=(0, 1))

# 訓練並轉換數據
X_normalized = scaler.fit_transform(X)

# 自訂範圍 [-1, 1]
scaler_custom = MinMaxScaler(feature_range=(-1, 1))
X_normalized_custom = scaler_custom.fit_transform(X)
```

#### 化工應用範例
```python
# 將感測器數據正規化到 [0, 1]
from sklearn.preprocessing import MinMaxScaler

# 感測器讀值
sensor_data = pd.DataFrame({
    'Sensor_1': [23.5, 24.1, 23.8, 24.5],
    'Sensor_2': [0.85, 0.92, 0.88, 0.95],
    'Sensor_3': [150, 165, 158, 170]
})

scaler = MinMaxScaler()
sensor_normalized = scaler.fit_transform(sensor_data)

print("Normalized range:", sensor_normalized.min(axis=0), "-", sensor_normalized.max(axis=0))
```

---

### 6.4 穩健標準化 (Robust Scaling)

#### 原理
使用中位數與四分位距 (IQR) 進行標準化，對異常值不敏感：

$$
x' = \frac{x - \text{median}}{\text{IQR}}
$$

其中 $\text{IQR} = Q_3 - Q_1$ (第 75 與第 25 百分位數之差)。

#### 適用情境
- **數據存在大量異常值**
- 製程數據受到間歇性干擾
- 感測器故障導致的異常讀值
- 需要穩健的前處理方法

#### sklearn 實現
```python
from sklearn.preprocessing import RobustScaler

# 建立穩健標準化器
scaler = RobustScaler()

# 訓練並轉換數據
X_robust = scaler.fit_transform(X)
```

#### 化工應用範例
```python
# 處理含異常值的製程數據
from sklearn.preprocessing import RobustScaler
import numpy as np

# 模擬製程數據 (含異常值)
process_data = np.array([
    [100, 50, 25],
    [102, 52, 26],
    [101, 51, 25.5],
    [500, 48, 24],  # 異常值
    [103, 53, 26.5],
    [99, 49, 25.2]
])

# 比較標準化與穩健標準化
standard_scaler = StandardScaler()
robust_scaler = RobustScaler()

X_standard = standard_scaler.fit_transform(process_data)
X_robust = robust_scaler.fit_transform(process_data)

print("Standard scaling - mean:", X_standard.mean(axis=0))
print("Robust scaling - median:", np.median(X_robust, axis=0))
```

---

### 6.5 類別變數編碼

降維演算法需要數值型數據，對於類別變數需要進行編碼。

#### 6.5.1 One-Hot Encoding (獨熱編碼)

將類別變數轉換為二進制向量：

**範例**：產品類型 ['A', 'B', 'C'] → [1,0,0], [0,1,0], [0,0,1]

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# 類別數據
categories = pd.DataFrame({
    'Product_Type': ['A', 'B', 'C', 'A', 'B']
})

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(categories)

print("Encoded shape:", encoded.shape)
print("Categories:", encoder.categories_)
```

#### 6.5.2 Label Encoding (標籤編碼)

將類別轉換為整數：

**範例**：產品類型 ['A', 'B', 'C'] → [0, 1, 2]

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(['A', 'B', 'C', 'A', 'B'])

print("Encoded labels:", labels)
```

⚠️ **注意**：Label Encoding 隱含順序關係 (0 < 1 < 2)，若類別間無順序關係，應使用 One-Hot Encoding。

---

### 6.6 處理缺失值

#### 6.6.1 刪除含缺失值的樣本或變數
```python
import pandas as pd

# 刪除含缺失值的列
df_dropped = df.dropna(axis=0)

# 刪除含缺失值的欄位
df_dropped_cols = df.dropna(axis=1)
```

#### 6.6.2 填補缺失值
```python
from sklearn.impute import SimpleImputer

# 使用均值填補
imputer_mean = SimpleImputer(strategy='mean')
X_imputed = imputer_mean.fit_transform(X)

# 使用中位數填補 (對異常值穩健)
imputer_median = SimpleImputer(strategy='median')
X_imputed = imputer_median.fit_transform(X)

# 使用最常見值填補 (類別變數)
imputer_mode = SimpleImputer(strategy='most_frequent')
X_imputed = imputer_mode.fit_transform(X)
```

#### 6.6.3 進階填補：KNN Imputer
```python
from sklearn.impute import KNNImputer

# 使用 K 近鄰填補
imputer_knn = KNNImputer(n_neighbors=5)
X_imputed = imputer_knn.fit_transform(X)
```

---

### 6.7 資料前處理完整流程範例

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# 1. 載入數據
data = pd.read_csv('process_data.csv')

# 2. 處理缺失值
imputer = SimpleImputer(strategy='median')
data_imputed = imputer.fit_transform(data)

# 3. 識別異常值
from scipy import stats
z_scores = np.abs(stats.zscore(data_imputed))
outlier_mask = (z_scores > 3).any(axis=1)

print(f"Detected {outlier_mask.sum()} outliers")

# 4. 選擇適當的標準化方法
if outlier_mask.sum() > len(data) * 0.05:  # 異常值 > 5%
    scaler = RobustScaler()
    print("Using Robust Scaling (outliers detected)")
else:
    scaler = StandardScaler()
    print("Using Standard Scaling")

# 5. 標準化
data_scaled = scaler.fit_transform(data_imputed)

# 6. 降維
pca = PCA(n_components=0.95)  # 保留 95% 變異數
data_pca = pca.fit_transform(data_scaled)

print(f"Original dimensions: {data_scaled.shape[1]}")
print(f"Reduced dimensions: {data_pca.shape[1]}")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
```

---

### 6.8 前處理方法選擇指引

| 情境 | 推薦方法 | 原因 |
|------|---------|------|
| **數據近似常態分布** | StandardScaler | 符合統計假設，效果最佳 |
| **數據存在大量異常值** | RobustScaler | 對異常值不敏感 |
| **數據分布未知或非常態** | MinMaxScaler | 不依賴分布假設 |
| **需要保留零值資訊** | StandardScaler | 不改變零值位置 |
| **數據有固定範圍** | MinMaxScaler | 保持數據範圍 |
| **用於 PCA** | StandardScaler | 標準前處理 |
| **用於 t-SNE / UMAP** | StandardScaler 或 MinMaxScaler | 兩者皆可 |
| **含類別變數** | OneHotEncoder | 避免隱含順序關係 |
| **含缺失值** | SimpleImputer 或 KNNImputer | 根據缺失比例選擇 |

---

## 7. 模型評估指標

降維屬於非監督式學習，沒有明確的標籤，評估方法與監督式學習不同。以下介紹各類評估指標與在 sklearn 中的實現。

### 7.1 PCA 特定指標

#### 7.1.1 解釋變異數比例 (Explained Variance Ratio)

**定義**：每個主成分解釋的原始數據變異數比例。

**數學表達**：

$$
\text{Explained Variance Ratio}_i = \frac{\lambda_i}{\sum_{j=1}^{D} \lambda_j}
$$

其中 $\lambda_i$ 是第 $i$ 個主成分的特徵值。

**sklearn 實現**：
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=5)
pca.fit(X_scaled)

# 各主成分的解釋變異數比例
explained_var = pca.explained_variance_ratio_
print("Explained variance ratio:", explained_var)

# 累積解釋變異數
cumsum_var = np.cumsum(explained_var)
print("Cumulative explained variance:", cumsum_var)
```

**解讀**：
- 第一主成分通常解釋最多變異數 (例如 40%)
- 前幾個主成分累積解釋 80-95% 變異數即可
- 若第一主成分解釋 > 90%，可能數據結構過於簡單

**化工應用**：
```python
# 決定保留多少主成分
import matplotlib.pyplot as plt

pca_full = PCA()
pca_full.fit(X_scaled)

explained_var = pca_full.explained_variance_ratio_
cumsum_var = np.cumsum(explained_var)

# 找到累積解釋變異數達 95% 的主成分數
n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
print(f"Number of components for 95% variance: {n_components_95}")
```

---

#### 7.1.2 累積解釋變異數 (Cumulative Explained Variance)

**定義**：前 $k$ 個主成分累積解釋的變異數比例。

$$
\text{Cumulative Explained Variance}_k = \sum_{i=1}^{k} \frac{\lambda_i}{\sum_{j=1}^{D} \lambda_j}
$$

**視覺化**：
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

# 左圖：各主成分解釋變異數
plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_var)+1), explained_var, alpha=0.7)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Component')

# 右圖：累積解釋變異數
plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumsum_var)+1), cumsum_var, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.legend()

plt.tight_layout()
plt.show()
```

---

#### 7.1.3 Scree Plot (陡坡圖)

**定義**：顯示各主成分的特徵值大小，用於判斷保留多少主成分。

**sklearn 實現**：
```python
# Scree Plot
eigenvalues = pca_full.explained_variance_

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(eigenvalues)+1), eigenvalues, marker='o')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot')
plt.axhline(y=1, color='r', linestyle='--', label='Kaiser Criterion (eigenvalue=1)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**解讀 (Kaiser Criterion)**：
- 保留特徵值 > 1 的主成分
- 觀察曲線的「拐點」(elbow point)
- 拐點之後的主成分貢獻較少

---

#### 7.1.4 重建誤差 (Reconstruction Error)

**定義**：降維後重建的數據與原始數據之間的差異。

**數學表達**：

$$
\text{Reconstruction Error} = \|\mathbf{X} - \hat{\mathbf{X}}\|_F^2 = \|\mathbf{X} - \mathbf{X}_{\text{PCA}} \mathbf{W}^T\|_F^2
$$

其中 $\|\cdot\|_F$ 是 Frobenius 範數。

**sklearn 實現**：
```python
from sklearn.metrics import mean_squared_error

# PCA 降維
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# 重建原始數據
X_reconstructed = pca.inverse_transform(X_pca)

# 計算重建誤差
reconstruction_error = mean_squared_error(X_scaled, X_reconstructed)
print(f"Reconstruction Error (MSE): {reconstruction_error:.4f}")

# 相對重建誤差
relative_error = reconstruction_error / np.var(X_scaled)
print(f"Relative Reconstruction Error: {relative_error:.4f}")
```

**解讀**：
- 誤差越小，降維保留的資訊越多
- 需要在降維程度與重建誤差之間取得平衡
- 化工應用：監控重建誤差可檢測異常狀態

---

### 7.2 通用降維評估指標

#### 7.2.1 Trustworthiness (信賴度)

**定義**：衡量降維後保持局部鄰近關係的程度。若原本不是近鄰的點在低維空間變成近鄰，信賴度降低。

**數學表達**：

$$
T(k) = 1 - \frac{2}{nk(2n-3k-1)} \sum_{i=1}^{n} \sum_{j \in \mathcal{N}_i^k} (r(i,j) - k)
$$

其中 $\mathcal{N}_i^k$ 是低維空間中點 $i$ 的 $k$ 近鄰， $r(i,j)$ 是高維空間中的排名。

**sklearn 實現**：
```python
from sklearn.manifold import trustworthiness

# 計算信賴度 (k=5 近鄰)
trust_score = trustworthiness(X_scaled, X_reduced, n_neighbors=5)
print(f"Trustworthiness (k=5): {trust_score:.4f}")
```

**解讀**：
- 範圍 [0, 1]，越接近 1 越好
- > 0.9：優秀
- 0.7-0.9：良好
- < 0.7：需改進

---

#### 7.2.2 Continuity (連續性)

**定義**：衡量原本是近鄰的點在降維後是否仍然接近。

**sklearn 實現**：
```python
# sklearn 未直接提供，需自行實現
from sklearn.neighbors import NearestNeighbors

def continuity(X_high, X_low, n_neighbors=5):
    """
    計算降維的連續性指標
    """
    nbrs_high = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X_high)
    nbrs_low = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X_low)
    
    _, indices_high = nbrs_high.kneighbors(X_high)
    _, indices_low = nbrs_low.kneighbors(X_low)
    
    # 移除自身 (第一個近鄰)
    indices_high = indices_high[:, 1:]
    indices_low = indices_low[:, 1:]
    
    # 計算重疊比例
    continuity_scores = []
    for i in range(len(X_high)):
        overlap = len(set(indices_high[i]) & set(indices_low[i]))
        continuity_scores.append(overlap / n_neighbors)
    
    return np.mean(continuity_scores)

cont_score = continuity(X_scaled, X_reduced, n_neighbors=5)
print(f"Continuity (k=5): {cont_score:.4f}")
```

**解讀**：
- 範圍 [0, 1]，越接近 1 越好
- 高連續性：局部結構保持良好

---

#### 7.2.3 Kullback-Leibler Divergence (KL 散度)

**定義**：衡量高維與低維空間中機率分布的差異 (主要用於 t-SNE)。

**數學表達**：

$$
KL(P\|Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
$$

**sklearn 實現**：
```python
from sklearn.manifold import TSNE

# t-SNE 會自動輸出 KL 散度
tsne = TSNE(n_components=2, verbose=1)
X_tsne = tsne.fit_transform(X_scaled)

# KL 散度在訓練過程中輸出
# 最終 KL 散度: tsne.kl_divergence_
print(f"Final KL Divergence: {tsne.kl_divergence_:.4f}")
```

**解讀**：
- 越小越好 (接近 0)
- 典型範圍：1-5
- > 10：可能需要調整參數

---

### 7.3 視覺化品質評估

#### 7.3.1 降維後的 2D/3D 散點圖

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 2D 視覺化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, c=y, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection')
plt.colorbar(label='Class')

plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, c=y, cmap='viridis')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Projection')
plt.colorbar(label='Class')

plt.tight_layout()
plt.show()
```

```python
# 3D 視覺化
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                     c=y, cmap='viridis', alpha=0.6)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('PCA 3D Projection')
plt.colorbar(scatter, label='Class')
plt.show()
```

---

#### 7.3.2 群集分離度 (Cluster Separation)

使用 Silhouette Score 評估降維後群集的分離程度：

```python
from sklearn.metrics import silhouette_score

# 假設有分群標籤
silhouette_original = silhouette_score(X_scaled, labels)
silhouette_reduced = silhouette_score(X_reduced, labels)

print(f"Silhouette Score (Original): {silhouette_original:.3f}")
print(f"Silhouette Score (Reduced): {silhouette_reduced:.3f}")
```

**解讀**：
- 降維後 Silhouette Score 應接近或高於原始數據
- 若降維後分數大幅下降，表示丟失重要結構

---

#### 7.3.3 局部結構保持 (Local Structure Preservation)

視覺化近鄰保持情況：

```python
from sklearn.neighbors import NearestNeighbors

# 選擇一個樣本點
sample_idx = 0

# 高維空間的近鄰
nbrs_high = NearestNeighbors(n_neighbors=10).fit(X_scaled)
distances_high, indices_high = nbrs_high.kneighbors([X_scaled[sample_idx]])

# 低維空間的近鄰
nbrs_low = NearestNeighbors(n_neighbors=10).fit(X_reduced)
distances_low, indices_low = nbrs_low.kneighbors([X_reduced[sample_idx]])

# 計算近鄰重疊
overlap = len(set(indices_high[0]) & set(indices_low[0]))
print(f"Neighbor overlap: {overlap}/10")

# 視覺化
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.3, c='gray')
plt.scatter(X_reduced[sample_idx, 0], X_reduced[sample_idx, 1], 
            c='red', s=100, marker='*', label='Query point')
plt.scatter(X_reduced[indices_low[0], 0], X_reduced[indices_low[0], 1], 
            c='blue', s=50, alpha=0.7, label='Neighbors in low-dim')
plt.title('Neighbors in Reduced Space')
plt.legend()

plt.subplot(1, 2, 2)
# 使用 PCA 將高維投影到 2D 以視覺化
pca_vis = PCA(n_components=2)
X_vis = pca_vis.fit_transform(X_scaled)
plt.scatter(X_vis[:, 0], X_vis[:, 1], alpha=0.3, c='gray')
plt.scatter(X_vis[sample_idx, 0], X_vis[sample_idx, 1], 
            c='red', s=100, marker='*', label='Query point')
plt.scatter(X_vis[indices_high[0], 0], X_vis[indices_high[0], 1], 
            c='green', s=50, alpha=0.7, label='Neighbors in high-dim')
plt.title('Neighbors in Original Space (PCA vis)')
plt.legend()

plt.tight_layout()
plt.show()
```

---

### 7.4 化工應用特定評估

#### 7.4.1 主成分的物理意義可解釋性

分析 Loading (負荷矩陣) 以理解主成分的物理意義：

```python
import pandas as pd

# 取得 Loading 矩陣
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# 建立 DataFrame
feature_names = ['Temperature', 'Pressure', 'Flow_Rate', 'Concentration']
loading_df = pd.DataFrame(
    loadings,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=feature_names
)

print("Loading Matrix:")
print(loading_df)

# 視覺化 Loading
plt.figure(figsize=(10, 6))
plt.imshow(loading_df.T, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(label='Loading')
plt.yticks(range(pca.n_components_), loading_df.columns)
plt.xticks(range(len(feature_names)), feature_names, rotation=45)
plt.title('PCA Loading Matrix')
plt.tight_layout()
plt.show()
```

**解讀**：
- 高正負荷：該變數對主成分貢獻大
- PC1 可能代表「整體操作水平」
- PC2 可能代表「溫度-壓力平衡」
- 結合製程知識驗證解釋

---

#### 7.4.2 Loadings 的工程合理性

驗證主成分是否符合製程物理/化學原理：

```python
# 分析 PC1 的主要貢獻變數
pc1_loadings = loading_df['PC1'].sort_values(ascending=False)
print("PC1 Main Contributors:")
print(pc1_loadings)

# 工程驗證：
# - 溫度與壓力是否呈現預期關係？
# - 流量與濃度是否符合物質平衡？
# - 是否與已知的製程模式一致？
```

---

#### 7.4.3 是否保留製程動態特性

對於時間序列製程數據，檢查降維後是否保留動態特性：

```python
# 時間序列降維評估
import matplotlib.pyplot as plt

# 假設有時間序列數據
time = np.arange(len(X_pca))

plt.figure(figsize=(12, 6))

# 原始數據的第一個變數
plt.subplot(2, 1, 1)
plt.plot(time, X_scaled[:, 0], label='Original Feature 1')
plt.ylabel('Scaled Value')
plt.title('Original Time Series')
plt.legend()
plt.grid(True, alpha=0.3)

# 降維後的第一主成分
plt.subplot(2, 1, 2)
plt.plot(time, X_pca[:, 0], label='PC1', color='red')
plt.xlabel('Time')
plt.ylabel('PC Score')
plt.title('Reduced Dimension Time Series')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**評估重點**：
- 動態趨勢是否保留？
- 週期性變化是否可見？
- 突變點是否仍可識別？

---

### 7.5 綜合評估策略

建立完整的降維評估流程：

```python
def evaluate_dimensionality_reduction(X_original, X_reduced, method_name, labels=None):
    """
    綜合評估降維效果
    """
    print(f"\n{'='*50}")
    print(f"Evaluation Report: {method_name}")
    print(f"{'='*50}")
    
    # 1. 維度減少比例
    reduction_ratio = (1 - X_reduced.shape[1] / X_original.shape[1]) * 100
    print(f"Dimension reduction: {X_original.shape[1]} → {X_reduced.shape[1]} ({reduction_ratio:.1f}%)")
    
    # 2. Trustworthiness
    trust = trustworthiness(X_original, X_reduced, n_neighbors=5)
    print(f"Trustworthiness (k=5): {trust:.4f}")
    
    # 3. Continuity
    cont = continuity(X_original, X_reduced, n_neighbors=5)
    print(f"Continuity (k=5): {cont:.4f}")
    
    # 4. 若有標籤，計算 Silhouette Score
    if labels is not None:
        sil_orig = silhouette_score(X_original, labels)
        sil_reduced = silhouette_score(X_reduced, labels)
        print(f"Silhouette (Original): {sil_orig:.4f}")
        print(f"Silhouette (Reduced): {sil_reduced:.4f}")
        print(f"Silhouette Change: {sil_reduced - sil_orig:+.4f}")
    
    print(f"{'='*50}\n")

# 使用範例
evaluate_dimensionality_reduction(X_scaled, X_pca, "PCA", labels=y)
evaluate_dimensionality_reduction(X_scaled, X_tsne, "t-SNE", labels=y)
evaluate_dimensionality_reduction(X_scaled, X_umap, "UMAP", labels=y)
```

---

### 7.6 評估指標總結表

| 指標 | 適用方法 | 範圍 | 理想值 | sklearn 函數 |
|------|---------|------|--------|-------------|
| **Explained Variance Ratio** | PCA | [0, 1] | 累積 > 0.85 | `pca.explained_variance_ratio_` |
| **Reconstruction Error** | PCA | ≥ 0 | 越小越好 | `mean_squared_error` |
| **Trustworthiness** | 所有方法 | [0, 1] | > 0.9 | `sklearn.manifold.trustworthiness` |
| **Continuity** | 所有方法 | [0, 1] | > 0.9 | 自行實現 |
| **KL Divergence** | t-SNE | ≥ 0 | < 5 | `tsne.kl_divergence_` |
| **Silhouette Score** | 所有方法 | [-1, 1] | > 0.5 | `sklearn.metrics.silhouette_score` |

---

## 8. 課程單元架構

本單元包含以下教學內容與程式演練：

### 8.1 教學講義
- **Unit06_Dimensionality_Reduction_Overview.md** (本文件)
  - 降維的背景理論與數學原理
  - 各種降維方法的詳細介紹
  - 化工領域應用案例
  - 方法選擇指引與對照表
  - 資料前處理方法
  - 模型評估指標

### 8.2 程式演練
每個降維方法都有獨立的講義與程式演練檔案：

1. **Unit06_PCA.md & Unit06_PCA.ipynb**
   - 主成分分析的詳細實作
   - 化工製程監控應用案例
   - T² 與 SPE 控制圖
   - Loading 分析與解釋

2. **Unit06_Kernel_PCA.md & Unit06_Kernel_PCA.ipynb**
   - 核主成分分析的實作
   - 核函數選擇與參數調整
   - 非線性製程特徵提取
   - 與 PCA 的比較分析

3. **Unit06_tSNE.md & Unit06_tSNE.ipynb**
   - t-SNE 的實作與參數調整
   - 高維數據視覺化
   - 批次製程軌跡分析
   - 參數敏感性分析

4. **Unit06_UMAP.md & Unit06_UMAP.ipynb**
   - UMAP 的實作與應用
   - 大規模製程數據降維
   - 與 t-SNE 的比較
   - 新數據轉換應用

### 8.3 課堂作業
- **Unit06_Dimensionality_Reduction_Homework.ipynb**
  - 綜合練習：使用所有降維方法
  - 化工領域真實案例分析
  - 方法比較與評估
  - 結果視覺化與解釋

---

## 9. 學習路徑建議

### 9.1 初學者路徑
1. **理解降維概念**：閱讀本文件第 1-2 節
2. **學習 PCA**：先掌握最基礎的線性降維方法
3. **實作練習**：完成 Unit06_PCA.ipynb
4. **視覺化應用**：學習 t-SNE 進行數據探索
5. **綜合作業**：完成課堂作業

### 9.2 進階學習路徑
1. **深入數學原理**：研讀各方法的數學推導
2. **核方法**：學習 Kernel PCA 處理非線性數據
3. **大規模應用**：掌握 UMAP 用於實際製程數據
4. **方法比較**：理解各方法的優缺點與適用場景
5. **實務應用**：將降維技術整合到製程監控系統

### 9.3 化工應用專題
1. **製程監控**：使用 PCA 建立統計製程控制系統
2. **故障診斷**：結合降維與分群識別異常模式
3. **品質預測**：降維後建立迴歸模型
4. **配方優化**：使用 UMAP 視覺化配方空間
5. **批次分析**：動態 PCA 分析批次製程軌跡

---

## 10. 實務應用注意事項

### 10.1 常見陷阱與解決方案

| 陷阱 | 描述 | 解決方案 |
|------|------|---------|
| **數據未標準化** | 高變異數變數主導降維結果 | 使用 StandardScaler 或 RobustScaler |
| **保留維度過少** | 丟失重要資訊 | 檢查累積解釋變異數 (≥ 85%) |
| **保留維度過多** | 降維效果不明顯 | 根據 Scree Plot 或 Kaiser Criterion 選擇 |
| **t-SNE 參數不當** | 視覺化效果差 | 調整 perplexity (5-50) 與 learning_rate |
| **忽略異常值** | 影響降維結果 | 使用 RobustScaler 或先檢測異常值 |
| **過度解釋 t-SNE** | 將視覺化當作絕對真相 | t-SNE 主要用於探索，不保證全局結構 |
| **忽略時間相關性** | 時間序列數據的順序關係 | 使用動態 PCA 或考慮時間延遲 |

### 10.2 化工實務建議

#### 製程監控系統建置
1. **離線模型訓練**：使用歷史正常操作數據建立 PCA 模型
2. **控制界限設定**：基於統計分布 (如 95% 信賴區間)
3. **即時監控**：新數據投影到主成分空間並計算統計量
4. **警報策略**：連續 N 點超出界限才發出警報 (避免誤報)
5. **模型更新**：定期重新訓練模型以適應製程變化

#### 數據視覺化報告
1. **選擇合適方法**：根據數據規模與目的選擇 (參考決策樹)
2. **多方法比較**：使用 PCA、t-SNE、UMAP 多角度呈現
3. **標註關鍵資訊**：標示異常點、操作模式、時間序列
4. **物理解釋**：結合領域知識解釋降維結果
5. **互動式視覺化**：使用 plotly 等工具建立互動圖表

#### 模型整合應用
1. **降維 + 迴歸**：PCA 萃取特徵後建立預測模型
2. **降維 + 分群**：視覺化群集結構以理解操作模式
3. **降維 + 異常檢測**：在低維空間進行異常檢測
4. **降維 + 時間序列**：動態 PCA 分析製程動態

---

## 11. 延伸閱讀與資源

### 11.1 經典教材
1. **"The Elements of Statistical Learning"** - Hastie, Tibshirani, Friedman
   - 第 14 章：Unsupervised Learning (PCA, ICA, Factor Analysis)
   
2. **"Pattern Recognition and Machine Learning"** - Christopher Bishop
   - 第 12 章：Continuous Latent Variables (PCA, Kernel PCA)

3. **"Dimensionality Reduction: A Comparative Review"** - L.J.P. van der Maaten et al.
   - 全面比較各種降維方法

### 11.2 化工應用文獻
1. **Multivariate Statistical Process Control** - 製程監控領域經典應用
2. **Batch Process Monitoring** - 批次製程的動態 PCA
3. **Fault Detection and Diagnosis** - 故障診斷中的降維應用

### 11.3 線上資源
- **scikit-learn 官方文件**: [Dimensionality Reduction](https://scikit-learn.org/stable/modules/decomposition.html)
- **UMAP 官方文件**: [UMAP Documentation](https://umap-learn.readthedocs.io/)
- **t-SNE 原始論文**: van der Maaten & Hinton (2008)

### 11.4 實用工具
- **scikit-learn**: PCA, Kernel PCA, t-SNE
- **umap-learn**: UMAP 實現
- **plotly**: 互動式視覺化
- **matplotlib / seaborn**: 靜態圖表

---

## 12. 課程總結

### 12.1 核心概念回顧
- 降維是將高維數據映射到低維空間的技術
- 主要目的：視覺化、特徵萃取、數據壓縮、降低計算成本
- 分為線性 (PCA) 與非線性 (Kernel PCA, t-SNE, UMAP) 方法
- 選擇方法需考慮：數據特性、目的、規模、計算資源

### 12.2 方法選擇快速指南
- **製程監控/建模** → PCA
- **非線性特徵提取** → Kernel PCA
- **小規模數據視覺化** → t-SNE  
- **大規模數據視覺化** → UMAP
- **需要解釋性** → PCA
- **純探索分析** → t-SNE / UMAP

### 12.3 下一步學習
完成本單元後，您應該能夠：
✓ 理解降維的數學原理與應用場景  
✓ 使用 sklearn 實作各種降維方法  
✓ 根據數據特性選擇合適的降維技術  
✓ 評估降維模型的效果  
✓ 將降維應用於化工製程監控與分析  

**繼續學習**：
- Unit07：異常檢測 (Anomaly Detection)
- Unit08：關聯規則學習 (Association Rule Learning)
- Unit09：綜合案例研究 (Integrated Case Study)

---

## 附錄 A：數學符號表

| 符號 | 意義 |
|------|------|
| $\mathbf{X}$ | 原始數據矩陣 $(n \times D)$ |
| $\mathbf{x}_i$ | 第 $i$ 個樣本向量 |
| $D$ | 原始維度 |
| $d$ | 降維後維度 |
| $n$ | 樣本數量 |
| $\mathbf{W}$ | 投影矩陣 |
| $\lambda_i$ | 第 $i$ 個特徵值 |
| $\mathbf{w}_i$ | 第 $i$ 個特徵向量 (主成分) |
| $\mathbf{C}$ | 協方差矩陣 |
| $\mu$ | 均值 |
| $\sigma$ | 標準差 |
| $k(\cdot, \cdot)$ | 核函數 |
| $\mathbf{K}$ | 核矩陣 |
| $p_{ij}$ | 高維空間相似度 |
| $q_{ij}$ | 低維空間相似度 |
| $\|\cdot\|_F$ | Frobenius 範數 |

---

## 附錄 B：程式碼速查表

### PCA 基本流程
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. PCA 降維
pca = PCA(n_components=0.95)  # 保留 95% 變異數
X_pca = pca.fit_transform(X_scaled)

# 3. 查看結果
print(f"Components: {pca.n_components_}")
print(f"Explained variance: {pca.explained_variance_ratio_}")
```

### t-SNE 視覺化
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# t-SNE 降維
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# 視覺化
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.colorbar(label='Class')
plt.title('t-SNE Visualization')
plt.show()
```

### UMAP 快速降維
```python
import umap

# UMAP 降維
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# 可對新數據轉換
X_new_umap = reducer.transform(X_new_scaled)
```

---

**課程編號**: CHE-AI-114  
**單元**: Unit06 - 降維 (Dimensionality Reduction)  
**教師**: 莊曜禎 助理教授  
**學期**: 114 學年度第 2 學期  

---

**版本歷史**:
- v1.0 (2026-01-23): 初版完成

**聯絡資訊**:
- 如有問題或建議，請聯絡課程助教或透過課程平台提出

---

