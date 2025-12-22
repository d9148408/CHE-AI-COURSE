# Unit15 綜合案例：配方最佳化完整工作流程

## 📋 檔案說明

### 主要檔案
- **Unit15_Integrated_Case_Study.ipynb**：完整 Jupyter Notebook（包含所有 7 個步驟）
- **Unit15_Integrated_Case_Study.md**：理論說明文件

### 輸出目錄
- **Unit15_Results/**：所有程式執行結果（CSV、PNG 等）

---

## 🎯 學習目標

完成本單元後，你將能夠：
1. ✅ 整合 3 種以上的非監督學習技術解決實際問題
2. ✅ 設計完整的資料驅動配方開發流程
3. ✅ 撰寫專業的工程決策報告
4. ✅ 理解各技術在工作流程中的角色與銜接點

---

## 🔬 案例背景

**工程挑戰**：某塗料製造商需要替換高毒性溶劑 **Toluene（甲苯）**

**目標**：
- VOC 排放 < 150 g/L（原配方 220 g/L）
- EHS 等級 ≤ 3（Toluene = 4）
- 光澤度 ≥ 85 GU（原配方 88 GU）
- 成本增幅 < 15%
- 6 個月內完成開發

---

## 📊 完整工作流程（7 個步驟）

### Step 0：環境設置與資料生成
- 導入必要套件
- 生成 50 種溶劑資料庫
- 生成 200 個歷史配方

### Step 1：溶劑家族聚類（K-Means）
- **輸入**：50 種候選溶劑
- **技術**：K-Means（k=4）
- **輸出**：15 種候選溶劑（EHS ≤ 3）
- **縮減率**：70%

### Step 2：配方關聯規則挖掘（Apriori）
- **輸入**：200 個歷史配方
- **技術**：Apriori 算法
- **輸出**：3 個成分約束（必備成分）
- **發現**：BYK-333、Tego270、TiO2

### Step 3：相似度篩選（Jaccard）
- **輸入**：15 種候選溶劑 × 成分約束
- **技術**：Jaccard 相似度
- **輸出**：8 個符合條件的候選配方
- **縮減率**：47%

### Step 4：多目標優化與排序
- **輸入**：8 個候選配方
- **技術**：線性加權複合評分
- **輸出**：Top 3 最終候選
- **縮減率**：63%

### Step 5：PCA 視覺化驗證
- **輸入**：Top 3 候選
- **技術**：PCA 降維
- **輸出**：2D 視覺化，確認候選合理性

### Step 6：DOE 實驗設計與優化
- **輸入**：第一候選
- **技術**：中心組合設計（CCD）+ Random Forest
- **實驗數**：20 個
- **輸出**：最佳配方配比

### Step 7：生產監控（Isolation Forest）
- **輸入**：生產批次資料
- **技術**：Isolation Forest 異常檢測
- **輸出**：異常檢測系統
- **成效**：異常率從 9.4% → 2.6%

---

## 🏆 最終成果

| 指標 | 原配方 (Toluene) | 新配方 | 改善幅度 |
|------|-----------------|--------|---------|
| **VOC** | 220 g/L | ~140 g/L | **-36%** ✅ |
| **EHS** | 4 | 2 | **-50%** ✅ |
| **光澤度** | 88 GU | ~88 GU | 持平 ✅ |
| **開發時間** | 12 個月 | 4 個月 | **-67%** ✅ |
| **實驗數** | 200+ | 20 | **-90%** ✅ |

---

## 🚀 快速開始

### 1. 安裝必要套件

```bash
pip install pandas numpy matplotlib seaborn plotly
pip install scikit-learn mlxtend pyDOE3 scipy
```

### 2. 執行 Notebook

```bash
jupyter notebook Unit15_Integrated_Case_Study.ipynb
```

### 3. 逐步執行

按照 Cell 順序執行（Shift + Enter），觀察每個步驟的輸出。

---

## 📁 輸出檔案清單

執行完成後，`Unit15_Results/` 資料夾將包含：

### Step 1
- `step1_candidate_solvents.csv`
- `step1_all_solvents_clustered.csv`
- `step1_optimal_k_analysis.png`
- `step1_clustering_pca.png`

### Step 2
- `step2_toluene_association_rules.csv`
- `step2_formulations_database.csv`

### Step 3
- `step3_qualified_candidates.csv`
- `step3_all_similarity_results.csv`
- `step3_jaccard_similarity_analysis.png`

### Step 4
- `step4_final_ranking.csv`
- `step4_top3_candidates.csv`

### Step 5
- `step5_pca_visualization.png`

### Step 6
- `step6_doe_experimental_data.csv`
- `step6_optimal_formulation.csv`

### Step 7
- `step7_production_monitoring_data.csv`
- `step7_production_monitoring.png`

---

## 💡 學習重點

### 技術整合
- 如何組合多種非監督學習技術
- 各技術的適用場景與限制
- 工作流程的設計原則

### 實務應用
- 配方開發的系統化流程
- 實驗設計與驗證策略
- 生產監控與品質控制

### 工程思維
- 資料驅動決策的完整流程
- 風險控制與漸進式改良
- 專家知識與 AI 的結合

---

## 📚 相關單元

- **Unit 09**：K-Means 聚類基礎
- **Unit 11**：Isolation Forest 異常檢測
- **Unit 13**：綠色溶劑篩選（Step 1 詳細版）
- **Unit 14**：關聯規則與相似度（Step 2-3 詳細版）

---

## 🎯 課後作業

### 基礎題（必做）
1. 修改候選溶劑數量（50 → 30），觀察結果變化
2. 調整多目標優化權重，找出不同的最佳配方
3. 增加 DOE 實驗因子（如攪拌時間、溫度）

### 進階題（選做）
1. 使用 DBSCAN 替代 K-Means
2. 建立神經網路響應面模型
3. 開發 Streamlit 即時監控儀表板

---

## ❓ 常見問題

### Q1：為什麼選擇這 7 個步驟？
A：這是典型的配方開發流程：篩選 → 驗證 → 優化 → 監控。每個步驟都有明確目標和技術適配。

### Q2：可以跳過某些步驟嗎？
A：不建議。每個步驟都有其作用，跳過可能導致風險增加或結果不佳。

### Q3：實際應用需要調整嗎？
A：是的。實際案例需要根據：
- 可用資料的數量和品質
- 領域專家的經驗
- 時間和成本限制
- 法規和安全要求

### Q4：如何推廣到其他產業？
A：核心流程相同，但需調整：
- 特徵選擇（根據產業特性）
- 約束條件（法規、成本等）
- 評估指標（性能、安全等）

---

## 📧 聯絡資訊

如有問題或建議，請聯繫：
- **課程教師**：[Your Name]
- **Email**：[Your Email]

---

**最後更新**：2025-12-18
