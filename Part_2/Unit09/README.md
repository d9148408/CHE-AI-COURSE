# Unit09 綜合案例研究 (Integrated Case Study)

## 📚 單元簡介

本單元是 Part 2 非監督式學習的綜合實踐單元，透過完整的工業案例展示如何**組合使用**分群 (Clustering)、降維 (Dimensionality Reduction)、異常檢測 (Anomaly Detection) 和關聯規則學習 (Association Rule Learning) 等方法，進行端到端的化工製程數據分析。本單元不僅整合前面所學的技術，更重要的是培養系統性思考、工程判斷和實務應用能力。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **理解工業製程的複雜性**：認識真實化工製程的多變數特性和動態行為
2. **掌握數據分析工作流程**：學會如何系統性地組合多種分析方法
3. **實踐端到端的分析流程**：從原始數據到最終洞察的完整過程
4. **培養工程判斷能力**：根據領域知識解釋分析結果，提出改善建議
5. **掌握方法組合策略**：理解何時使用哪種方法，如何組合發揮最大效用

---

## 📖 單元內容架構

### 1️⃣ 總覽篇：綜合案例研究概述

**檔案**：[Unit09_Integrated_Case_Study_Overview.md](Unit09_Integrated_Case_Study_Overview.md)
（也可參考：[Unit09_Integrated_Case_Study_Overview.ipynb](Unit09_Integrated_Case_Study_Overview.ipynb)）

**內容重點**：
- 為什麼需要綜合案例研究？
  - 單一方法的局限性
  - 方法組合的協同效應
  - 工業應用的複雜性
  
- **田納西-伊士曼製程 (TEP) 簡介**：
  - 背景與重要性（化工製程控制的標準基準測試）
  - 製程描述（反應、分離、循環系統）
  - 主要反應與產物
  - 數據特性（52 個變數、21 種故障場景）
  
- **綜合分析工作流程設計**：
  ```
  階段 1：數據準備與探索性分析
     ↓
  階段 2：降維與特徵提取 (PCA, t-SNE, UMAP)
     ↓
  階段 3：操作模式識別 (Clustering)
     ↓
  階段 4：異常檢測與故障診斷
     ↓
  階段 5：變數關聯分析與洞察提取
     ↓
  階段 6：結果整合與工程建議
  ```
  
- 每個階段的目標、方法、輸出與價值

**適合讀者**：完成 Unit05-08 學習後的所有學員

---

### 2️⃣ 案例一：製程安全與異常檢測

**檔案**：
- 講義：[Unit09_Process_Safety_Anomaly_Detection.md](Unit09_Process_Safety_Anomaly_Detection.md)
- 程式範例：[Unit09_Process_Safety_Anomaly_Detection.ipynb](Unit09_Process_Safety_Anomaly_Detection.ipynb)

**內容重點**：

**1. 製程安全監控的演進**
- 從傳統單變數警報到多變數關聯監控
- AI 異常檢測的優勢與核心概念
- 監督式 vs 非監督式學習在安全監控中的角色

**2. 方法組合：Isolation Forest + PCA/MSPC**
- **Isolation Forest**：
  - 快速全廠掃描，偵測是否有異常發生
  - 異常分數計算與閾值設定
  - 適合處理高維感測器數據
  
- **PCA/MSPC (多變數統計製程控制)**：
  - Hotelling's T² 統計量：監控主成分空間
  - SPE (Q 統計量)：監控殘差空間
  - 貢獻圖分析：找出故障根源變數
  - 可解釋性強，符合工業標準

**3. 實務應用流程**
- **階段 1**：Isolation Forest 進行全廠異常掃描
- **階段 2**：PCA/MSPC 提供詳細診斷與根因分析
- **階段 3**：建立分級告警系統（黃色預警、橙色警告、紅色緊急）
- **階段 4**：處置流程 SOP（確認→降級→取樣→回復/重訓）

**4. 交付成果**
- 監控策略設計（監控哪些 tag、頻率、基準數據）
- 告警規則設定（門檻定義、百分位、rolling 指標）
- 處置流程 SOP
- 模型版本與變更管理

**化工應用價值**：
- 早期預警潛在安全風險
- 降低誤報率，提高告警可靠性
- 快速定位故障根源變數
- 支援操作員決策

---

### 3️⃣ 案例二：溶劑篩選決策支援

**檔案**：
- 講義：[Unit09_Solvent_Screening_Case_Study.md](Unit09_Solvent_Screening_Case_Study.md)
- 程式範例：[Unit09_Solvent_Screening_Case_Study.ipynb](Unit09_Solvent_Screening_Case_Study.ipynb)

**內容重點**：

**1. 溶劑篩選的重要性與挑戰**
- 傳統方法：逐一實驗、高成本、多目標衝突、經驗依賴
- AI 方法優勢：快速縮小候選範圍、發現隱藏模式、多維度評估

**2. 多階段篩選漏斗策略**
```
階段 1：分群分析 (Clustering)
   ↓ 識別溶劑類型，縮小至 30-40 種
階段 2：降維分析 (PCA)
   ↓ 視覺化綜合表現，進一步篩選至 15-20 種
階段 3：關聯規則學習 (Association Rules)
   ↓ 發現優良溶劑的共同特徵模式
最終結果：Top 5 候選溶劑
```

**3. 方法組合與協同效應**
- **Clustering (K-Means / Hierarchical)**：
  - 將 100 種溶劑依物化性質分為 3-5 個家族
  - 識別每個家族的代表性溶劑
  - 快速排除不合適的家族
  
- **PCA (降維與視覺化)**：
  - 將 7-10 個物化性質降至 2D/3D
  - 視覺化溶劑的綜合性能
  - 識別帕累托前沿（多目標最優）
  
- **Association Rules (模式挖掘)**：
  - 發現「優良溶劑」的共同特徵
  - 規則範例：{低沸點, 高溶解度} → {優先候選}
  - 提供可解釋的篩選邏輯

**4. 決策支援輸出**
- Top 5 候選溶劑清單與性質對比
- 決策建議書（優缺點分析、風險評估）
- 視覺化儀表板（互動式探索）
- 後續實驗優先順序

**化工應用價值**：
- 大幅減少實驗次數（從 100 種降至 5 種）
- 縮短產品開發週期
- 降低研發成本
- 提供數據驅動的決策依據

---

### 4️⃣ 進階主題

**檔案**：
- 講義：[Unit09_Advanced_Topics.md](Unit09_Advanced_Topics.md)
- 程式範例：[Unit09_Advanced_Topics.ipynb](Unit09_Advanced_Topics.ipynb)

**內容重點**：
- 時間序列數據的特殊處理
- 在線學習與概念漂移
- 模型部署與監控
- 人機協作與決策支援系統
- 可解釋 AI 在化工領域的應用

---

### 5️⃣ 綜合練習

**檔案**：[Unit09_Integrated_Case_Study_Homework.ipynb](Unit09_Integrated_Case_Study_Homework.ipynb)

**練習內容**：
- 應用多種方法於新的化工案例
- 設計完整的數據分析工作流程
- 撰寫分析報告與決策建議
- 展示與溝通分析結果

---

## 🎓 工作流程設計原則

### 1. 從問題出發，而非從方法出發
- ❌ 錯誤：「我想用 t-SNE 視覺化數據」
- ✅ 正確：「我想識別不同的操作模式，需要降維視覺化幫助理解」

### 2. 組合方法形成互補
| 方法組合 | 協同效應 | 應用場景 |
|---------|---------|----------|
| **PCA + Clustering** | 降維消除噪音 → 提升分群品質 | 高維製程數據分群 |
| **Clustering + Anomaly Detection** | 先分群 → 在各群內檢測異常 | 多模式製程監控 |
| **PCA + Anomaly Detection** | T²/SPE 統計量 → MSPC | 製程監控標準方法 |
| **Clustering + Association Rules** | 在各群內挖掘規則 → 更精準 | 多產品配方優化 |
| **UMAP + Clustering** | 非線性降維 → 更好的群集分離 | 複雜數據視覺化 |

### 3. 由粗到細、迭代精煉
```
第一輪：快速掃描
  - 降維視覺化 → 初步理解數據結構
  - 分群分析 → 識別大類別
  
第二輪：深入分析
  - 異常檢測 → 找出問題點
  - 關聯規則 → 挖掘細節模式
  
第三輪：驗證與精煉
  - 結合領域知識驗證
  - 調整參數精煉結果
```

### 4. 平衡自動化與人工判斷
- **自動化擅長**：大量數據處理、模式識別、重複性任務
- **人工判斷擅長**：異常情況處理、因果推理、決策制定
- **最佳實踐**：AI 提供候選方案 + 人類最終決策

---

## 💻 實作環境需求

### 必要套件（整合所有前面單元）
```python
# 基礎套件
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.4.0
seaborn >= 0.11.0

# 機器學習
scikit-learn >= 1.0.0
scipy >= 1.7.0

# 降維與視覺化
umap-learn >= 0.5.0

# 關聯規則學習
mlxtend >= 0.19.0
```

### 選用套件
```python
# 互動式視覺化
plotly >= 5.0.0
ipywidgets >= 7.6.0

# 進階異常檢測
pyod >= 1.0.0

# 網絡分析
networkx >= 2.6.0
```

---

## 📈 學習路徑建議

### 前提條件
完成 Unit05-08 的學習，掌握：
- Unit05：分群分析
- Unit06：降維技術
- Unit07：異常檢測
- Unit08：關聯規則學習

### 學習順序
1. **理解綜合案例的必要性**
   - 閱讀 [Unit09_Integrated_Case_Study_Overview.md](Unit09_Integrated_Case_Study_Overview.md)
   - 理解方法組合的價值

2. **案例一：製程安全監控**
   - 閱讀 [Unit09_Process_Safety_Anomaly_Detection.md](Unit09_Process_Safety_Anomaly_Detection.md)
   - 執行 [Unit09_Process_Safety_Anomaly_Detection.ipynb](Unit09_Process_Safety_Anomaly_Detection.ipynb)
   - 理解 Isolation Forest + PCA/MSPC 的組合策略

3. **案例二：溶劑篩選**
   - 閱讀 [Unit09_Solvent_Screening_Case_Study.md](Unit09_Solvent_Screening_Case_Study.md)
   - 執行 [Unit09_Solvent_Screening_Case_Study.ipynb](Unit09_Solvent_Screening_Case_Study.ipynb)
   - 理解多階段篩選漏斗策略

4. **進階主題探索**
   - 閱讀 [Unit09_Advanced_Topics.md](Unit09_Advanced_Topics.md)
   - 執行 [Unit09_Advanced_Topics.ipynb](Unit09_Advanced_Topics.ipynb)

5. **綜合練習**
   - 完成 [Unit09_Integrated_Case_Study_Homework.ipynb](Unit09_Integrated_Case_Study_Homework.ipynb)
   - 設計自己的綜合分析工作流程

---

## 🔍 核心案例總結

### 案例一：製程安全監控
- **問題**：如何即時監控化工製程，及早發現安全風險？
- **解決方案**：Isolation Forest（快速掃描）+ PCA/MSPC（詳細診斷）
- **價值**：早期預警、降低誤報、快速定位故障

### 案例二：溶劑篩選
- **問題**：如何從 100 種候選溶劑中快速篩選出最佳選項？
- **解決方案**：Clustering（分類）+ PCA（視覺化）+ Association Rules（模式挖掘）
- **價值**：減少實驗次數、縮短開發週期、降低成本

### 方法組合的關鍵啟示
1. **沒有萬能的方法**：每種方法都有其適用場景和局限性
2. **組合產生協同效應**：1 + 1 > 2
3. **工作流程比單一方法更重要**：如何系統性地組合使用
4. **工程判斷不可或缺**：AI 輔助決策，人類最終判斷

---

## 📝 分析報告撰寫指南

### 報告結構建議
1. **背景與目標** (10%)
   - 製程/問題描述
   - 分析目標與預期效益
   
2. **數據描述** (10%)
   - 數據來源與規模
   - 變數說明
   - 數據品質評估
   
3. **方法與流程** (20%)
   - 分析工作流程設計
   - 各階段使用的方法
   - 方法選擇的理由
   
4. **結果與發現** (40%)
   - 視覺化結果
   - 關鍵發現
   - 模式與規律
   
5. **工程建議** (15%)
   - 實務改善建議
   - 實施計劃
   - 預期效益
   
6. **總結與展望** (5%)
   - 主要結論
   - 局限性
   - 未來工作

### 視覺化最佳實踐
- 使用高品質圖表（300 DPI）
- 添加清晰的標題與圖例
- 使用顏色突出重點
- 適當的字體大小（便於閱讀）
- 中英文標籤（避免中文亂碼）

---

## 🚀 進階學習方向

完成本單元後，可以進一步探索：

1. **深度學習整合**：Autoencoder + Clustering、LSTM + Anomaly Detection
2. **因果推理**：從相關性到因果關係
3. **強化學習**：製程優化與控制
4. **數位孿生 (Digital Twin)**：虛實整合的製程監控
5. **邊緣計算與 IoT**：感測器數據的即時分析
6. **可解釋 AI (XAI)**：提升模型的透明度與可信度

---

## 📚 參考資源

### 田納西-伊士曼製程 (TEP)
- Downs, J. J., & Vogel, E. F. (1993). *A plant-wide industrial process control problem*. Computers & Chemical Engineering.
- [TEP Dataset and Simulator](http://depts.washington.edu/control/LARRY/TE/download.html)

### 工業數據分析
- *Process Analytics: Data-driven Performance, Optimization and Control* by Bhavik R. Bakshi
- *Data-Driven Modeling and Control in Chemical Engineering* by Jie Zhang

### 案例研究方法論
- *Case Study Research: Design and Methods* by Robert K. Yin
- *Applied Data Science in Chemical Engineering* (Journal Articles)

---

## ✍️ 課後習題提示

1. **TEP 完整分析**：使用所有方法分析 TEP 數據，撰寫完整報告
2. **方法比較研究**：比較不同方法組合策略的效果
3. **自定義案例**：將綜合分析應用於您的化工數據
4. **決策支援系統**：設計一個實用的數據分析決策支援系統
5. **批判性思考**：分析某個方法組合的優缺點與適用場景

---

## 📞 技術支援

如有疑問或需要協助，歡迎透過以下方式聯繫：
- 課程討論區
- 指導教授 Office Hour
- 課程助教

---

**版本資訊**：Unit09 v1.0 | 最後更新：2026-01-27

---

## 🎓 Part 2 總結

恭喜您完成 Part 2 非監督式學習的所有單元！您已經掌握：

- **Unit05 分群分析**：K-Means、Hierarchical、DBSCAN、GMM
- **Unit06 降維**：PCA、Kernel PCA、t-SNE、UMAP
- **Unit07 異常檢測**：Isolation Forest、LOF、One-Class SVM、Elliptic Envelope
- **Unit08 關聯規則學習**：Apriori、FP-Growth
- **Unit09 綜合案例**：方法組合與實務應用

您現在具備了：
✅ 扎實的理論基礎
✅ 豐富的實作經驗
✅ 系統性思考能力
✅ 工程應用能力

**下一步建議**：
- 繼續學習 Part 3（監督式學習）
- 將所學應用於實際化工問題
- 參與開源專案，持續精進
- 閱讀最新文獻，追蹤技術發展
