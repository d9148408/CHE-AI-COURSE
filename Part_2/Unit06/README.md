# Unit06 降維 (Dimensionality Reduction)

## 📚 單元簡介

本單元系統性地介紹降維 (Dimensionality Reduction) 技術在化學工程領域的理論與應用。降維是非監督式學習中一項重要的技術，主要目的是在保留數據主要特徵的前提下，減少數據的維度。這不僅能幫助我們理解高維數據的內在結構，還能提升後續模型的效能與可解釋性。本單元涵蓋從經典的線性方法到先進的非線性方法，全面掌握降維技術。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **理解降維的核心概念**：掌握維度詛咒、特徵提取等基本概念
2. **掌握多種降維演算法**：理解線性與非線性降維方法的原理與差異
3. **選擇合適的降維方法**：根據數據特性與應用需求選擇最佳演算法
4. **實作降維模型**：使用 scikit-learn 和 umap-learn 實作各種降維演算法
5. **應用於化工領域**：解決製程監控、品質預測、高維數據視覺化等實際問題

---

## 📖 單元內容架構

### 1️⃣ 總覽篇：降維基礎

**檔案**：[Unit06_Dimensionality_Reduction_Overview.md](Unit06_Dimensionality_Reduction_Overview.md)

**內容重點**：
- 降維技術的定義與目的
- 維度詛咒 (Curse of Dimensionality) 問題
- 線性降維 vs 非線性降維
- 降維的數學原理與映射函數
- 化工領域應用場景：
  - 數據視覺化
  - 製程監控與故障診斷
  - 特徵工程與模型預處理
  - 軟感測器開發
  - 批次製程分析
- 降維方法分類：PCA、Kernel PCA、t-SNE、UMAP

**適合讀者**：所有學員，建議先閱讀此篇以建立整體概念

---

### 2️⃣ 主成分分析 (PCA) ⭐ 重點教學單元

**檔案**：
- 講義：[Unit06_PCA.md](Unit06_PCA.md)
- 程式範例：[Unit06_PCA.ipynb](Unit06_PCA.ipynb)

**內容重點**：
- **演算法原理**：
  - 變異數最大化的線性投影
  - 協方差矩陣與特徵值分解
  - 主成分、載荷、得分的概念
  - 解釋變異數與累積解釋變異數
  
- **實作技術**：
  - scikit-learn `PCA` 類別使用
  - 最佳主成分數選擇：累積解釋變異數、Scree Plot
  - 主成分載荷 (Loadings) 解讀
  - 得分圖 (Scores Plot) 與 Biplot 視覺化
  - 數據重建與壓縮
  
- **化工應用案例**：
  - **製程監控與故障診斷**：Hotelling's T²、SPE (Q統計量) 控制圖
  - **反應器操作數據降維**：從 52 維降至 2-3 維視覺化
  - **貢獻圖分析**：找出故障根源變數
  - **批次製程監控**：Multi-way PCA
  
- **演算法特性**：
  - ✅ 優點：快速、可解釋、理論扎實、適合製程監控
  - ❌ 缺點：僅能捕捉線性結構、對異常值敏感

**適合場景**：數據近似線性、需要可解釋性、製程監控、品質預測

---

### 3️⃣ 核主成分分析 (Kernel PCA)

**檔案**：
- 講義：[Unit06_Kernel_PCA.md](Unit06_Kernel_PCA.md)
- 程式範例：[Unit06_Kernel_PCA.ipynb](Unit06_Kernel_PCA.ipynb)

**內容重點**：
- **演算法原理**：
  - 核技巧 (Kernel Trick) 與隱式高維映射
  - 核函數：RBF、多項式、Sigmoid
  - 在高維特徵空間中執行 PCA
  - 核矩陣的中心化與特徵值分解
  
- **實作技術**：
  - scikit-learn `KernelPCA` 類別使用
  - 核函數選擇與超參數調整 (gamma, degree, coef0)
  - 反向映射 (Inverse Transform) 與重建誤差
  - 核矩陣視覺化與分析
  
- **化工應用案例**：
  - 非線性製程監控（Arrhenius 關係、相變過程）
  - 複雜反應系統的特徵提取
  - 非線性操作區域識別
  - 與線性 PCA 的比較分析
  
- **演算法特性**：
  - ✅ 優點：捕捉非線性結構、理論扎實、泛化能力強
  - ❌ 缺點：計算成本高、超參數敏感、缺乏物理可解釋性

**適合場景**：數據存在非線性關係、複雜化學反應、相變過程

---

### 4️⃣ t-分布隨機鄰域嵌入 (t-SNE)

**檔案**：
- 講義：[Unit06_tSNE.md](Unit06_tSNE.md)
- 程式範例：[Unit06_tSNE.ipynb](Unit06_tSNE.ipynb)

**內容重點**：
- **演算法原理**：
  - 基於機率分布的降維思想
  - 高維空間：高斯分布建模相似度
  - 低維空間：t-分布建模相似度（解決擁擠問題）
  - KL 散度最小化與梯度下降優化
  - 困惑度 (Perplexity) 參數的作用
  
- **實作技術**：
  - scikit-learn `TSNE` 類別使用
  - 超參數調整：perplexity、learning_rate、n_iter
  - 早期誇大 (Early Exaggeration) 技術
  - 初始化策略：random vs PCA
  - 結果穩定性評估
  
- **化工應用案例**：
  - 多變數製程狀態視覺化（50+ 感測器數據降至 2D）
  - 產品品質數據探索與群集識別
  - 批次製程軌跡視覺化
  - 操作模式轉換分析
  - 異常操作條件識別
  
- **演算法特性**：
  - ✅ 優點：視覺化效果極佳、揭示局部結構、群集分離明顯
  - ❌ 缺點：計算慢、不保留全局結構、不支援新數據投影、結果隨機

**適合場景**：高維數據視覺化、探索性分析、群集結構展示

---

### 5️⃣ 均勻流形逼近與投影 (UMAP)

**檔案**：
- 講義：[Unit06_UMAP.md](Unit06_UMAP.md)
- 程式範例：[Unit06_UMAP.ipynb](Unit06_UMAP.ipynb)

**內容重點**：
- **演算法原理**：
  - 基於流形學習和拓撲數據分析的理論基礎
  - 黎曼幾何與代數拓撲概念
  - 模糊拓撲表示與交叉熵優化
  - 局部與全局結構的平衡
  - 與 t-SNE 的理論聯繫
  
- **實作技術**：
  - umap-learn 套件使用
  - 超參數調整：n_neighbors、min_dist、metric
  - 支援新數據投影（transform 方法）
  - 監督式 UMAP (Supervised UMAP)
  - 參數化 UMAP (Parametric UMAP)
  
- **化工應用案例**：
  - 大規模製程數據降維（百萬級數據點）
  - 即時製程監控系統（支援新數據投影）
  - 高維反應條件空間探索
  - 多批次製程軌跡比較分析
  - 與 t-SNE 的比較：速度、全局結構保持
  
- **演算法特性**：
  - ✅ 優點：速度快、可擴展性強、保留全局結構、支援新數據投影、理論扎實
  - ❌ 缺點：超參數敏感、結果有隨機性（但比 t-SNE 穩定）

**適合場景**：大規模數據、需要新數據投影、即時監控、平衡局部與全局結構

---

### 6️⃣ 實作練習

**檔案**：[Unit06_Dimensionality_Reduction_Homework.ipynb](Unit06_Dimensionality_Reduction_Homework.ipynb)

**練習內容**：
- 應用多種降維演算法於實際化工數據
- 比較不同演算法的降維效果
- 參數調整與模型優化
- 結果解讀與工程意義分析

---

## 📊 數據集說明

### 反應器操作數據 (`data/reactor_operation/`)
- 包含多個感測器的時間序列數據
- 52 個製程變數（溫度、壓力、流量、組成等）
- 用於降維視覺化、製程監控、故障診斷

---

## 🎓 演算法比較與選擇指南

| 演算法 | 類型 | 速度 | 全局結構 | 局部結構 | 新數據投影 | 可解釋性 | 適用場景 |
|--------|------|------|----------|----------|-----------|---------|----------|
| **PCA** | 線性 | 極快 | ✅ 優秀 | ⚠️ 中等 | ✅ 支援 | ✅ 高 | 製程監控、品質預測 |
| **Kernel PCA** | 非線性 | 中等 | ✅ 良好 | ✅ 良好 | ⚠️ 有限 | ⚠️ 中等 | 非線性製程、複雜反應 |
| **t-SNE** | 非線性 | 慢 | ❌ 弱 | ✅ 優秀 | ❌ 不支援 | ❌ 低 | 數據視覺化、探索性分析 |
| **UMAP** | 非線性 | 快 | ✅ 良好 | ✅ 優秀 | ✅ 支援 | ⚠️ 中等 | 大規模數據、即時監控 |

**選擇建議**：
1. **製程監控、故障診斷** → PCA（需要可解釋性）
2. **非線性製程、複雜反應** → Kernel PCA 或 UMAP
3. **數據視覺化、探索性分析** → t-SNE 或 UMAP
4. **大規模數據、即時應用** → UMAP
5. **需要新數據投影** → PCA 或 UMAP

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
umap-learn >= 0.5.0  # UMAP 實作
plotly >= 5.0.0  # 互動式視覺化
ipywidgets >= 7.6.0  # Jupyter 互動元件
```

---

## 📈 學習路徑建議

### 第一階段：基礎概念建立
1. 閱讀 [Unit06_Dimensionality_Reduction_Overview.md](Unit06_Dimensionality_Reduction_Overview.md)
2. 理解降維的目的與維度詛咒問題
3. 了解線性與非線性降維的差異

### 第二階段：演算法學習與實作
1. **PCA**（建議最先學習，化工領域最重要）
   - 閱讀講義 [Unit06_PCA.md](Unit06_PCA.md)
   - 執行 [Unit06_PCA.ipynb](Unit06_PCA.ipynb)
   - 重點掌握 T² 和 SPE 統計量、貢獻圖分析
   
2. **Kernel PCA**
   - 閱讀講義 [Unit06_Kernel_PCA.md](Unit06_Kernel_PCA.md)
   - 執行 [Unit06_Kernel_PCA.ipynb](Unit06_Kernel_PCA.ipynb)
   - 比較與 PCA 的差異
   
3. **t-SNE**
   - 閱讀講義 [Unit06_tSNE.md](Unit06_tSNE.md)
   - 執行 [Unit06_tSNE.ipynb](Unit06_tSNE.ipynb)
   - 掌握視覺化技巧
   
4. **UMAP**（進階主題）
   - 閱讀講義 [Unit06_UMAP.md](Unit06_UMAP.md)
   - 執行 [Unit06_UMAP.ipynb](Unit06_UMAP.ipynb)
   - 比較 t-SNE 與 UMAP 的差異

### 第三階段：綜合應用與練習
1. 完成 [Unit06_Dimensionality_Reduction_Homework.ipynb](Unit06_Dimensionality_Reduction_Homework.ipynb)
2. 比較不同演算法在相同數據集上的表現
3. 嘗試將降維技術應用於自己的化工數據

---

## 🔍 化工領域核心應用

### 1. 製程監控與故障診斷 ⭐ 最重要應用
- **目標**：多變數統計製程控制 (MSPC)
- **演算法建議**：PCA（線性）、Kernel PCA（非線性）
- **關鍵技術**：
  - Hotelling's T² 統計量：監控主成分空間
  - SPE (Q 統計量)：監控殘差空間
  - 貢獻圖：識別故障變數
  - 動態 PCA：處理時間序列數據

### 2. 高維數據視覺化
- **目標**：將 50+ 維製程數據降至 2D/3D 視覺化
- **演算法建議**：t-SNE、UMAP
- **關鍵技術**：超參數調整、群集識別、操作模式分析

### 3. 品質預測前處理
- **目標**：消除共線性、降低模型複雜度
- **演算法建議**：PCA
- **關鍵技術**：主成分回歸 (PCR)、PLS 回歸

### 4. 軟感測器開發
- **目標**：用易測量變數預測難測量品質指標
- **演算法建議**：PCA + 回歸模型
- **關鍵技術**：特徵提取、降維、模型訓練

### 5. 批次製程分析
- **目標**：分析批次軌跡、預測批次品質
- **演算法建議**：Multi-way PCA
- **關鍵技術**：三維數據展開、批次終點檢測

---

## 📝 評估指標總結

### 降維效果評估
- **解釋變異數比例 (Explained Variance Ratio)**：PCA 專用，越高越好
- **重建誤差 (Reconstruction Error)**：評估信息保留程度
- **可信度 (Trustworthiness)**：評估局部鄰域結構保持程度
- **連續性 (Continuity)**：評估新鄰居是否合理

### 視覺化品質評估
- **群集分離度**：不同群集在低維空間的分離程度
- **局部結構保持**：鄰近點在降維後是否仍鄰近
- **全局結構保持**：遠離點在降維後是否仍遠離

### 製程監控專用指標
- **T² 統計量**：主成分空間的馬氏距離
- **SPE (Q 統計量)**：殘差空間的歐氏距離
- **故障檢測率 (FDR)**：正確檢測故障的比例
- **誤報率 (FAR)**：誤判正常為異常的比例

---

## 🚀 進階學習方向

完成本單元後，可以進一步探索：

1. **動態 PCA (DPCA)**：處理時間序列的自相關性
2. **多階 PCA (Multi-Level PCA)**：處理多尺度製程數據
3. **監督式降維**：LDA、Supervised UMAP
4. **深度學習降維**：Autoencoder、Variational Autoencoder (VAE)
5. **張量分解**：Multi-way PCA、Tucker Decomposition
6. **流形學習**：Isomap、Laplacian Eigenmaps、LLE

---

## 📚 參考資源

### 教科書
1. *Multivariate Statistical Process Control* by Richard K. Borden（PCA 在製程監控的應用）
2. *Pattern Recognition and Machine Learning* by Christopher M. Bishop（第 12 章）
3. *The Elements of Statistical Learning* by Hastie et al.（第 14 章）
4. *Dimensionality Reduction: A Comparative Review* by L.J.P. van der Maaten et al.

### 線上資源
- [scikit-learn Dimensionality Reduction](https://scikit-learn.org/stable/modules/decomposition.html)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [Distill.pub - How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)

### 化工領域應用論文
- PCA for Process Monitoring and Fault Diagnosis
- Multi-way PCA for Batch Process Monitoring
- Kernel PCA for Nonlinear Process Monitoring

---

## ✍️ 課後習題提示

1. **比較分析**：使用同一數據集比較四種演算法的降維效果
2. **製程監控實作**：建立 PCA 模型進行製程監控，計算 T² 和 SPE 統計量
3. **參數優化**：探討各演算法關鍵參數對結果的影響
4. **實際應用**：將降維技術應用於您的化工數據集
5. **物理解釋**：解讀主成分的物理意義與工程含義

---

## 📞 技術支援

如有疑問或需要協助，歡迎透過以下方式聯繫：
- 課程討論區
- 指導教授 Office Hour
- 課程助教

---

**課程資訊**
- 課程名稱：AI在化工上之應用
- 課程單元：Unit06 降維 (Dimensionality Reduction)
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-01-28

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---