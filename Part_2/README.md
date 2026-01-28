# Part 2: 非監督式學習 (Unsupervised Learning)

## 📚 Part 2 簡介

本 Part 系統性地介紹非監督式學習 (Unsupervised Learning) 在化學工程領域的理論與應用。非監督式學習是機器學習的重要分支，主要處理**未標記數據**，透過演算法自動發現數據中的內在結構、模式和關聯性。在化工領域，非監督式學習可用於製程操作模式識別、異常檢測、數據視覺化、知識挖掘等多種應用場景。

本 Part 涵蓋五個核心主題：
1. **分群分析 (Clustering)**：自動發現數據中的群組結構
2. **降維 (Dimensionality Reduction)**：在保留主要特徵的前提下降低數據維度
3. **異常檢測 (Anomaly Detection)**：識別與大多數數據顯著不同的樣本
4. **關聯規則學習 (Association Rule Learning)**：發現數據中項目之間的有趣關聯
5. **綜合案例研究 (Integrated Case Study)**：組合使用多種方法進行端到端分析

---

## 🎯 學習目標

完成 Part 2 後，您將能夠：

1. **理解非監督式學習的核心概念**：掌握各類非監督式學習方法的原理與適用場景
2. **掌握多種演算法**：從經典的 K-Means 到先進的 UMAP，理解各演算法的優缺點
3. **選擇合適的方法**：根據數據特性與應用需求選擇最佳分析方法
4. **實作與應用**：使用 Python 和 scikit-learn 等工具實作各種演算法
5. **解決實際問題**：將方法組合應用於化工製程的實際問題

---

## 📖 單元目錄

### [Unit 05: 分群分析 (Clustering Analysis)](Unit05/)

**主題**：使用無監督式演算法自動發現數據中的群組結構

**投影片**：[Unit05_Clustering_Overview.pdf](Unit05/Unit05_Clustering_Overview.pdf)
**單元概述**：[Unit05_Clustering_Overview.md](Unit05/Unit05_Clustering_Overview.md)

**演算法詳解**：
- [K-Means 分群演算法](Unit05/Unit05_K_Means.md) ([Notebook](Unit05/Unit05_K_Means.ipynb))
- [階層式分群](Unit05/Unit05_Hierarchical_Clustering.md) ([Notebook](Unit05/Unit05_Hierarchical_Clustering.ipynb))
- [DBSCAN 密度分群](Unit05/Unit05_DBSCAN.md) ([Notebook](Unit05/Unit05_DBSCAN.ipynb))
- [高斯混合模型](Unit05/Unit05_Gaussian_Mixture_Models.md) ([Notebook](Unit05/Unit05_Gaussian_Mixture_Models.ipynb))

**作業練習**：[Unit05_Clustering_Homework.ipynb](Unit05/Unit05_Clustering_Homework.ipynb)

**應用場景**：
- 製程操作模式識別
- 產品品質分類
- 反應器操作條件分群
- 設備運行狀態分類

---

### [Unit06: 降維 (Dimensionality Reduction)](Unit06/)

**主題**：在保留主要特徵的前提下，減少數據的維度

**投影片**：[Unit06_Dimensionality_Reduction_Overview.pdf](Unit06/Unit06_Dimensionality_Reduction_Overview.pdf)
**單元概述**：[Unit06_Dimensionality_Reduction_Overview.md](Unit06/Unit06_Dimensionality_Reduction_Overview.md)

**演算法詳解**：
- [主成分分析 (PCA)](Unit06/Unit06_PCA.md) ([Notebook](Unit06/Unit06_PCA.ipynb))
- [核主成分分析 (Kernel PCA)](Unit06/Unit06_Kernel_PCA.md) ([Notebook](Unit06/Unit06_Kernel_PCA.ipynb))
- [t-SNE 降維](Unit06/Unit06_tSNE.md) ([Notebook](Unit06/Unit06_tSNE.ipynb))
- [UMAP 降維](Unit06/Unit06_UMAP.md) ([Notebook](Unit06/Unit06_UMAP.ipynb))

**作業練習**：[Unit06_Dimensionality_Reduction_Homework.ipynb](Unit06/Unit06_Dimensionality_Reduction_Homework.ipynb)

**應用場景**：
- 高維數據視覺化
- 製程監控與品質預測
- 特徵提取與選擇
- 數據壓縮與去噪

---

### [Unit07: 異常檢測 (Anomaly Detection)](Unit07/)

**主題**：識別與大多數數據顯著不同的異常樣本

**投影片**：[Unit07_Anomaly_Detection_Overview.pdf](Unit07/Unit07_Anomaly_Detection_Overview.pdf)
**單元概述**：[Unit07_Anomaly_Detection_Overview.md](Unit07/Unit07_Anomaly_Detection_Overview.md)

**演算法詳解**：
- [隔離森林 (Isolation Forest)](Unit07/Unit07_Isolation_Forest.md) ([Notebook](Unit07/Unit07_Isolation_Forest.ipynb))
- [局部離群因子 (LOF)](Unit07/Unit07_LOF.md) ([Notebook](Unit07/Unit07_LOF.ipynb))
- [橢圓包絡線 (Elliptic Envelope)](Unit07/Unit07_Elliptic_Envelope.md) ([Notebook](Unit07/Unit07_Elliptic_Envelope.ipynb))
- [單類支持向量機 (One-Class SVM)](Unit07/Unit07_OneClass_SVM.md) ([Notebook](Unit07/Unit07_OneClass_SVM.ipynb))

**作業練習**：[Unit07_Anomaly_Detection_Homework.ipynb](Unit07/Unit07_Anomaly_Detection_Homework.ipynb)

**應用場景**：
- 製程安全監控
- 產品品質控制
- 設備故障診斷
- 異常操作識別

---

### [Unit08: 關聯規則學習 (Association Rule Learning)](Unit08/)

**主題**：發現數據中項目之間的有趣關聯或相關模式

**投影片**：[Unit08_Association_Rule_Learning_Overview.pdf](Unit08/Unit08_Association_Rule_Learning_Overview.pdf)
**單元概述**：[Unit08_Association_Rule_Learning_Overview.md](Unit08/Unit08_Association_Rule_Learning_Overview.md)

**演算法詳解**：
- [Apriori 演算法](Unit08/Unit08_Apriori_Algorithm.md) ([Notebook](Unit08/Unit08_Apriori_Algorithm.ipynb))
- [FP-Growth 演算法](Unit08/Unit08_FP_Growth_Algorithm.md) ([Notebook](Unit08/Unit08_FP_Growth_Algorithm.ipynb))

**作業練習**：[Unit08_Association_Rule_Learning_Homework.ipynb](Unit08/Unit08_Association_Rule_Learning_Homework.ipynb)

**應用場景**：
- 配方成分協同效應分析
- 製程變數關聯性挖掘
- 操作條件與品質關係分析
- 製程知識發現

---

### [Unit09: 綜合案例研究 (Integrated Case Study)](Unit09/)

**主題**：組合使用多種非監督式學習方法進行端到端分析

**單元概述**：
- [Unit09_Integrated_Case_Study_Overview.md](Unit09/Unit09_Integrated_Case_Study_Overview.md)
- [Unit09_Integrated_Case_Study_Overview.ipynb](Unit09/Unit09_Integrated_Case_Study_Overview.ipynb)

**案例研究**：
- [製程安全異常檢測](Unit09/Unit09_Process_Safety_Anomaly_Detection.md) ([Notebook](Unit09/Unit09_Process_Safety_Anomaly_Detection.ipynb))
- [溶劑篩選案例研究](Unit09/Unit09_Solvent_Screening_Case_Study.md) ([Notebook](Unit09/Unit09_Solvent_Screening_Case_Study.ipynb))
- [進階主題](Unit09/Unit09_Advanced_Topics.md) ([Notebook](Unit09/Unit09_Advanced_Topics.ipynb))

**作業練習**：[Unit09_Integrated_Case_Study_Homework.ipynb](Unit09/Unit09_Integrated_Case_Study_Homework.ipynb)

**學習重點**：
- 方法組合策略
- 端到端分析流程
- 工程判斷能力
- 結果解釋與應用

---

## 🗂️ 數據集

本 Part 使用的數據集包含多種化工製程場景：

- **反應器操作數據** (`reactor_operation/`)：包含溫度、壓力、流量、濃度等製程變數
- **反應器模擬數據** (`reactor_simulation/`)：用於分群分析的模擬數據
- **批次反應器數據** (`batch_reactor/`)：用於異常檢測的批次操作數據
- **溶劑性質數據**：用於關聯規則學習和溶劑篩選案例

---

## 💻 環境需求

### 必要套件

```python
# 核心數據處理
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0

# 機器學習核心
scikit-learn>=1.0.0

# 降維與視覺化
umap-learn>=0.5.0

# 關聯規則學習
mlxtend>=0.19.0
pyfpgrowth>=1.0

# 其他工具
scipy>=1.7.0
joblib>=1.0.0
```

### 安裝指令

```bash
# 使用 pip 安裝
pip install numpy pandas matplotlib seaborn scikit-learn umap-learn mlxtend pyfpgrowth scipy joblib

# 或使用 conda 安裝
conda install numpy pandas matplotlib seaborn scikit-learn scipy joblib
conda install -c conda-forge umap-learn mlxtend
pip install pyfpgrowth
```

---

## 📊 學習路徑建議

### 1. 初學者路徑（按順序學習）

1. **Unit05 分群分析**：從最直觀的分群開始，理解無監督學習的概念
2. **Unit06 降維**：學習如何處理高維數據並進行視覺化
3. **Unit07 異常檢測**：應用於製程安全與品質控制
4. **Unit08 關聯規則學習**：發現變數之間的關聯性
5. **Unit09 綜合案例**：整合所有方法解決實際問題

### 2. 快速應用路徑（針對特定需求）

- **需要製程監控**：Unit05 (分群) → Unit07 (異常檢測) → Unit09 (案例)
- **需要數據視覺化**：Unit06 (降維) → Unit05 (分群) → Unit09 (案例)
- **需要知識挖掘**：Unit08 (關聯規則) → Unit05 (分群) → Unit09 (案例)
- **需要全面理解**：按照 Unit05 → Unit06 → Unit07 → Unit08 → Unit09 順序學習

### 3. 進階研究路徑

1. 完成所有單元的基礎學習
2. 深入研究 Unit09 的進階主題
3. 將方法應用於自己的研究數據
4. 探索方法組合的新策略

---

## 🎓 學習建議

1. **理論與實務並重**：先理解演算法原理，再透過 Notebook 實作
2. **比較不同方法**：同一問題使用不同演算法，比較結果差異
3. **調整超參數**：實驗不同參數設定，理解其對結果的影響
4. **結合領域知識**：用化工專業知識解釋分析結果
5. **完成作業練習**：透過作業鞏固所學，培養問題解決能力

---

## 📝 評估方式

- **單元作業**：每個單元都有對應的作業 Notebook，需完成指定任務
- **綜合專案**：Unit09 的綜合案例需完整實作並撰寫報告
- **學習重點**：
  - 演算法選擇的合理性
  - 參數調整的邏輯性
  - 結果解釋的正確性
  - 程式碼的可讀性與效率

---

## 🔗 相關資源

### 官方文件
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [mlxtend Documentation](http://rasbt.github.io/mlxtend/)

### 推薦閱讀
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*
- Aggarwal, C. C., & Reddy, C. K. (2013). *Data Clustering: Algorithms and Applications*
- Van der Maaten, L., & Hinton, G. (2008). "Visualizing Data using t-SNE"

### 化工應用文獻
- 製程監控與故障診斷相關論文
- 數據驅動的製程優化研究
- 化工大數據分析應用案例

---

## 👥 貢獻與反饋

如有任何問題、建議或發現錯誤，歡迎透過以下方式聯繫：
- 開啟 GitHub Issue
- 提交 Pull Request
- 發送電子郵件

---

---

**課程資訊**
- 課程名稱：AI在化工上之應用
- 課程單元：Part 2 非監督式學習
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-01-28

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---
