# Unit05 分群分析 (Clustering Analysis)

## 📚 單元簡介

本單元系統性地介紹分群分析 (Clustering Analysis) 在化學工程領域的理論與應用。分群分析是非監督式學習的核心方法，能夠自動發現數據中的內在結構與模式，無需預先標記的訓練數據。本單元涵蓋四種主要的分群演算法，從經典的基於距離方法到先進的基於機率模型，全面掌握分群分析技術。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **理解分群分析的核心概念**：掌握相似度度量、群集品質評估等基本概念
2. **掌握多種分群演算法**：理解不同演算法的原理、優缺點與適用場景
3. **選擇合適的分群方法**：根據數據特性與應用需求選擇最佳演算法
4. **實作分群模型**：使用 scikit-learn 實作各種分群演算法
5. **應用於化工領域**：解決製程操作模式識別、產品品質分類、異常檢測等實際問題

---

## 📖 單元內容架構

### 1️⃣ 總覽篇：分群分析基礎

**檔案**：[Unit05_Clustering_Overview.md](Unit05_Clustering_Overview.md)

**內容重點**：
- 分群分析的定義與核心概念
- 相似度與距離度量方法（歐幾里得距離、曼哈頓距離、餘弦相似度）
- 群集品質評估指標（Silhouette Score、Davies-Bouldin Index、Calinski-Harabasz Index）
- 化工領域應用場景：
  - 製程操作模式識別
  - 產品品質分類
  - 異常操作模式探索
  - 溶劑與配方篩選
- 分群演算法分類：分割式、階層式、密度式、基於模型

**適合讀者**：所有學員，建議先閱讀此篇以建立整體概念

---

### 2️⃣ K-Means 分群演算法

**檔案**：
- 講義：[Unit05_K_Means.md](Unit05_K_Means.md)
- 程式範例：[Unit05_K_Means.ipynb](Unit05_K_Means.ipynb)

**內容重點**：
- **演算法原理**：
  - 迭代優化過程（初始化 → 分配 → 更新）
  - 目標函數：最小化群集內變異數 (WCSS/Inertia)
  - 收斂條件與複雜度分析
  
- **實作技術**：
  - scikit-learn `KMeans` 類別使用
  - 最佳 K 值選擇：Elbow Method、Silhouette Analysis
  - K-Means++ 初始化策略
  - Mini-Batch K-Means 加速技術
  
- **化工應用案例**：
  - 反應器多產品操作模式識別（3種產品品質分類）
  - 基於 PCA 降維的視覺化分析
  - 群集中心的化學意義解讀
  
- **演算法特性**：
  - ✅ 優點：簡單高效、可擴展性好、結果可解釋
  - ❌ 缺點：需預設 K 值、對初始值敏感、假設球形群集

**適合場景**：數據量大、群集形狀近似球形、需要快速分群

---

### 3️⃣ 階層式分群演算法

**檔案**：
- 講義：[Unit05_Hierarchical_Clustering.md](Unit05_Hierarchical_Clustering.md)
- 程式範例：[Unit05_Hierarchical_Clustering.ipynb](Unit05_Hierarchical_Clustering.ipynb)

**內容重點**：
- **演算法原理**：
  - 凝聚式策略 (Agglomerative)：由下而上逐步合併
  - 分裂式策略 (Divisive)：由上而下逐步分裂
  - 連結方法：Single、Complete、Average、Ward
  - 樹狀圖 (Dendrogram) 解讀與切割
  
- **實作技術**：
  - scikit-learn `AgglomerativeClustering` 類別
  - scipy 樹狀圖繪製與分析
  - 距離矩陣計算與視覺化
  - 最佳切割高度選擇方法
  
- **化工應用案例**：
  - 反應器操作模式層次結構分析
  - 不同連結方法的比較與選擇
  - 基於樹狀圖的群集數量決策
  
- **演算法特性**：
  - ✅ 優點：無需預設群集數、建立層次關係、視覺化直觀
  - ❌ 缺點：計算複雜度高 (O(n²log n))、對噪音敏感、無法撤銷合併

**適合場景**：數據量中小、需要層次結構、探索性分析

---

### 4️⃣ DBSCAN 分群演算法

**檔案**：
- 講義：[Unit05_DBSCAN.md](Unit05_DBSCAN.md)
- 程式範例：[Unit05_DBSCAN.ipynb](Unit05_DBSCAN.ipynb)

**內容重點**：
- **演算法原理**：
  - 基於密度的分群思想
  - 核心點、邊界點、噪音點分類
  - 參數：ε (eps) 鄰域半徑、MinPts (min_samples) 最小點數
  - 密度可達性與密度連接概念
  
- **實作技術**：
  - scikit-learn `DBSCAN` 類別使用
  - 最佳參數選擇：K-distance Plot、Silhouette Analysis
  - 噪音點識別與處理
  - 任意形狀群集的視覺化
  
- **化工應用案例**：
  - 反應器異常操作模式探索
  - 自動識別噪音點（異常數據）
  - 非球形群集識別（複雜操作區域）
  - 不同參數設定對結果的影響分析
  
- **演算法特性**：
  - ✅ 優點：自動發現群集數、處理任意形狀、識別噪音點
  - ❌ 缺點：對參數敏感、難以處理密度差異大的數據

**適合場景**：存在噪音、群集密度不均、形狀複雜、異常檢測

---

### 5️⃣ 高斯混合模型

**檔案**：
- 講義：[Unit05_Gaussian_Mixture_Models.md](Unit05_Gaussian_Mixture_Models.md)
- 程式範例：[Unit05_Gaussian_Mixture_Models.ipynb](Unit05_Gaussian_Mixture_Models.ipynb)

**內容重點**：
- **演算法原理**：
  - 機率視角的分群：數據來自多個高斯分布的混合
  - 期望最大化 (EM) 演算法：E-step（期望）+ M-step（最大化）
  - 軟分群：提供數據點屬於各群集的機率
  - 協方差類型：full、tied、diag、spherical
  
- **實作技術**：
  - scikit-learn `GaussianMixture` 類別使用
  - 模型選擇：BIC、AIC 資訊準則
  - 機率評估與異常檢測
  - 橢圓形群集邊界視覺化
  
- **化工應用案例**：
  - 反應器多產品操作模式的機率建模
  - 軟分群與硬分群 (K-Means) 的比較
  - 操作模式歸屬機率分析
  - 基於機率的異常檢測（低機率點）
  
- **演算法特性**：
  - ✅ 優點：軟分群、機率評估、處理橢圓形群集、理論基礎扎實
  - ❌ 缺點：可能陷入局部最優、對初始值敏感、計算成本較高

**適合場景**：需要機率評估、群集重疊、橢圓形群集、不確定性量化

---

### 6️⃣ 實作練習

**檔案**：[Unit05_Clustering_Homework.ipynb](Unit05_Clustering_Homework.ipynb)

**練習內容**：
- 應用多種分群演算法於實際化工數據
- 比較不同演算法的結果與性能
- 參數調整與模型優化
- 結果解讀與化學意義分析

---

## 📊 數據集說明

### 1. 反應器操作數據 (`data/reactor_operation/`)
- 包含多種操作條件下的反應器數據
- 用於操作模式識別與分類

### 2. 反應器模擬數據 (`data/reactor_simulation/`)
- **`multi_product_quality.csv`**：多產品品質數據
  - 模擬不同操作條件下的產品品質指標
  - 包含溫度、壓力、流量、收率、轉化率等特徵
  - 用於產品品質分類與操作模式分析

---

## 🎓 演算法比較與選擇指南

| 演算法 | 群集數量 | 群集形狀 | 噪音處理 | 計算複雜度 | 輸出類型 | 適用場景 |
|--------|---------|---------|---------|-----------|---------|----------|
| **K-Means** | 需預設 | 球形 | 差 | 低 (O(nKt)) | 硬分群 | 大數據、球形群集 |
| **Hierarchical** | 彈性切割 | 任意 | 差 | 高 (O(n²log n)) | 硬分群+樹狀圖 | 探索性分析、層次關係 |
| **DBSCAN** | 自動發現 | 任意 | 優秀 | 中 (O(n log n)) | 硬分群+噪音 | 異常檢測、複雜形狀 |
| **GMM** | 需預設 | 橢圓形 | 中 | 中高 (O(nKt)) | 軟分群+機率 | 機率評估、重疊群集 |

**選擇建議**：
1. **快速分群、大數據** → K-Means
2. **需要層次結構、小數據** → Hierarchical Clustering
3. **存在噪音、異常檢測** → DBSCAN
4. **需要機率評估、群集重疊** → GMM

---

## 💻 實作環境需求

### 必要套件
```python
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scikit-learn >= 1.0.0
scipy >= 1.7.0
```

### 選用套件
```python
plotly >= 5.0.0  # 互動式視覺化
ipywidgets >= 7.6.0  # Jupyter 互動元件
```

---

## 📈 學習路徑建議

### 第一階段：基礎概念建立
1. 閱讀 [Unit05_Clustering_Overview.md](Unit05_Clustering_Overview.md)
2. 理解分群分析的核心概念與應用場景
3. 熟悉相似度度量與群集評估指標

### 第二階段：演算法學習與實作
1. **K-Means**（建議最先學習）
   - 閱讀講義 [Unit05_K_Means.md](Unit05_K_Means.md)
   - 執行 [Unit05_K_Means.ipynb](Unit05_K_Means.ipynb)
   
2. **Hierarchical Clustering**
   - 閱讀講義 [Unit05_Hierarchical_Clustering.md](Unit05_Hierarchical_Clustering.md)
   - 執行 [Unit05_Hierarchical_Clustering.ipynb](Unit05_Hierarchical_Clustering.ipynb)
   
3. **DBSCAN**
   - 閱讀講義 [Unit05_DBSCAN.md](Unit05_DBSCAN.md)
   - 執行 [Unit05_DBSCAN.ipynb](Unit05_DBSCAN.ipynb)
   
4. **GMM**（進階主題）
   - 閱讀講義 [Unit05_Gaussian_Mixture_Models.md](Unit05_Gaussian_Mixture_Models.md)
   - 執行 [Unit05_Gaussian_Mixture_Models.ipynb](Unit05_Gaussian_Mixture_Models.ipynb)

### 第三階段：綜合應用與練習
1. 完成 [Unit05_Clustering_Homework.ipynb](Unit05_Clustering_Homework.ipynb)
2. 比較不同演算法在相同數據集上的表現
3. 嘗試將分群分析應用於自己的化工數據

---

## 🔍 化工領域核心應用

### 1. 製程操作模式識別
- **目標**：自動識別反應器、蒸餾塔等設備的不同操作模式
- **演算法建議**：K-Means、GMM
- **關鍵技術**：特徵工程、PCA 降維、時間序列分群

### 2. 產品品質分類
- **目標**：根據多維品質指標自動分類產品等級
- **演算法建議**：K-Means、GMM、Hierarchical
- **關鍵技術**：品質指標標準化、多變量分析

### 3. 異常操作檢測
- **目標**：識別偏離正常操作的異常狀態
- **演算法建議**：DBSCAN、GMM（機率評估）
- **關鍵技術**：噪音點識別、機率閾值設定

### 4. 批次製程分析
- **目標**：分析批次相似度，識別成功批次特徵
- **演算法建議**：Hierarchical、DBSCAN
- **關鍵技術**：DTW 距離、軌跡相似度

### 5. 溶劑與配方篩選
- **目標**：根據性質將候選對象分類，縮小篩選範圍
- **演算法建議**：Hierarchical、K-Means
- **關鍵技術**：化學性質特徵化、相似度計算

---

## 📝 評估指標總結

### 無監督式評估（無真實標籤）
- **Silhouette Score**：-1 到 1，越接近 1 表示分群越好
- **Davies-Bouldin Index**：越小越好，表示群集內聚性高且分離度大
- **Calinski-Harabasz Index**：越大越好，表示群集間離散度高
- **Inertia (WCSS)**：K-Means 專用，越小越好（需搭配 Elbow Method）

### 監督式評估（有真實標籤時）
- **Adjusted Rand Index (ARI)**：-1 到 1，越接近 1 表示與真實標籤越一致
- **Normalized Mutual Information (NMI)**：0 到 1，越接近 1 表示越一致
- **Homogeneity, Completeness, V-measure**：評估群集純度與完整性

---

## 🚀 進階學習方向

完成本單元後，可以進一步探索：

1. **時間序列分群**：將分群方法應用於動態製程數據
2. **半監督式分群**：結合少量標籤數據改善分群效果
3. **深度學習分群**：Autoencoder + Clustering
4. **模糊分群**：Fuzzy C-Means 等方法
5. **譜分群 (Spectral Clustering)**：處理複雜的非線性結構

---

## 📚 參考資源

### 教科書
1. *Pattern Recognition and Machine Learning* by Christopher M. Bishop（第 9 章）
2. *The Elements of Statistical Learning* by Hastie et al.（第 14 章）
3. *Introduction to Data Mining* by Tan et al.（第 8 章）

### 線上資源
- [scikit-learn Clustering Documentation](https://scikit-learn.org/stable/modules/clustering.html)
- [Python Data Science Handbook - Clustering](https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html)

### 化工領域應用論文
- 製程監控與故障診斷中的分群分析應用
- 批次製程數據分析與品質控制
- 化學資訊學中的分子分群方法

---

## ✍️ 課後習題提示

1. **比較分析**：使用同一數據集比較四種演算法的結果，分析優劣
2. **參數優化**：探討各演算法關鍵參數對結果的影響
3. **實際應用**：將分群分析應用於您的化工數據集
4. **結果詮釋**：從化學工程角度解讀分群結果的物理意義

---

## 📞 技術支援

如有疑問或需要協助，歡迎透過以下方式聯繫：
- 課程討論區
- 指導教授 Office Hour
- 課程助教

---

**版本資訊**：Unit05 v1.0 | 最後更新：2026-01-27
