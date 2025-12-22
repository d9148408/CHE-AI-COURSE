# Part 2｜監督式學習（Supervised Learning）

**課程代號**：CHE-AI-101 Part 2  
**課程性質**：化工資料科學與機器學習核心課程  
**更新日期**：2025-12-17  
**版本**：v2.0 (Restructured)

---

## 🎯 課程目標

本 Part 旨在建立化工領域**監督式學習**的完整知識體系，從基礎分類/回歸模型到工業級部署，涵蓋：

1. **分類任務**：決策樹、混淆矩陣、不平衡數據處理
2. **驗證策略**：交叉驗證、時序數據分割、超參數調校
3. **回歸建模**：從線性回歸到非線性熱力學參數擬合
4. **工業應用**：軟感測器開發、模型部署、漂移監控

**核心特色**：
- ✅ 每個算法都有**數學推導** + **工程案例** + **可執行代碼**
- ✅ 強調化工製程數據的**時序性**與**物理約束**
- ✅ 從學術研究延伸到**生產環境部署**
- ✅ 所有數據自包含，支援 **Colab + 本地環境**

---

## 📚 課程單元結構

### 🌳 Unit05｜決策樹分類與模型評估

**檔案**：
- 📄 講義：[Unit05_DecisionTree_Classification.md](Unit05_DecisionTree_Classification.md)
- 💻 實作：[Unit05_DecisionTree_Classification.ipynb](Unit05_DecisionTree_Classification.ipynb)
- 📁 輸出：`P2_Unit05_Results/` (8 files)

**學習目標**：
- 理解決策樹算法（Gini Impurity、Information Gain）
- 掌握 **Confusion Matrix** 深度分析（Precision/Recall/F1）
- 處理不平衡數據（Class Weight、SMOTE）
- 交叉驗證與超參數調校（K-Fold、Grid Search）
- PR Curve 與閾值優化
- **成本敏感學習**：FP vs FN 的錯誤成本分析

**化工案例**：
- 🔬 **Titanic 生存預測**（經典入門案例）
- ⚛️ **CSTR 反應器異常偵測**：溫度-壓力安全邊界
- 💧 **流動模式分類**（課後練習）：Bubble / Slug / Annular Flow

**技術亮點**：
- ✅ 強調 **Recall** 在化工安全系統中的重要性
- ✅ 決策邊界視覺化（垂直切分 vs SOP 規則）
- ✅ 模型可解釋性（Feature Importance）

**預計學習時間**：4-6 小時

---

### 🔄 Unit06｜交叉驗證與模型選擇策略

**檔案**：
- 📄 講義：[Unit06_CV_Model_Selection.md](Unit06_CV_Model_Selection.md)
- 💻 實作：[Unit06_CV_Model_Selection.ipynb](Unit06_CV_Model_Selection.ipynb)
- 📁 輸出：`P2_Unit06_Results/` (2 files)

**學習目標**：
- 理解交叉驗證的數學原理與實務意義
- 掌握不同 CV 策略的適用場景
- **⚠️ 時序數據陷阱**：KFold vs TimeSeriesSplit 的關鍵差異
- GroupKFold 處理批次反應數據
- Nested CV 避免資訊洩漏
- 模型選擇決策樹（演算法選擇指南）

**核心內容**：

| 數據類型 | CV 策略 | 關鍵考量 |
|---------|---------|---------|
| **i.i.d 數據** | KFold / Shuffle Split | 隨機分割即可 |
| **批次數據** | GroupKFold | 同批次不可分開 |
| **時序數據** | TimeSeriesSplit | 訓練必須在測試之前 |

**實戰證明**：
```python
# 化工 DCS 數據範例
KFold (錯誤):        R² = 0.96  ← 未來資訊洩漏！
TimeSeriesSplit (正確): R² = 0.79  ← 真實性能
差異: 17 個百分點的虛假提升
```

**化工案例**：
- 📊 反應器安全邊界視覺化
- 🕐 時序分割示意圖
- 🎯 模型選擇決策流程圖

**技術亮點**：
- ✅ **課程最關鍵單元之一**：避免化工 AI 最常見的錯誤
- ✅ DCS/SCADA 數據的正確驗證方法
- ✅ 模型選擇考慮可解釋性與部署限制

**預計學習時間**：3-4 小時

---

### 📈 Unit07｜工程回歸：從基礎到熱力學建模

**檔案**：
- 📄 講義：[Unit07_Thermodynamic_Fitting.md](Unit07_Thermodynamic_Fitting.md)
- 💻 實作：[Unit07_Thermodynamic_Fitting.ipynb](Unit07_Thermodynamic_Fitting.ipynb)
- 📁 輸出：`P2_Unit07_Results/` (12 files: A01-A04, B01-B08)

**課程結構**：

#### 📘 Part A: 回歸基礎（4 個案例）

**學習目標**：
- 線性回歸的數學原理（最小二乘法、梯度下降）
- 多項式回歸與 Overfitting 問題
- 回歸評估指標（R²、RMSE、MAE、MAPE）
- 殘差分析與模型診斷
- 正則化技術（Ridge、Lasso）

**化工案例**：
- 🔥 **阿瑞尼士方程擬合**：$k = A \exp(-E_a/RT)$（反應速率 vs 溫度）
- 💧 **溶液黏度建模**：濃度 vs 黏度關係

**輸出圖表**：A01-A04 (Arrhenius Plot, Residuals, Polynomial, Regularization)

#### 📗 Part B: 熱力學參數擬合（8 個圖表）

**學習目標**：
- 非線性回歸（`scipy.optimize.curve_fit`）
- Wilson Model 活度係數方程
- VLE (Vapor-Liquid Equilibrium) 數據分析
- 參數不確定性量化（Confidence Interval）
- 參數相關性分析
- 多起點優化避免局部最優

**理論深度**：
```
Modified Raoult's Law: y_i P = x_i γ_i P_i^sat

Wilson Model: ln γ_i = -ln(x_i + Λ_ij x_j) + x_j(...)

參數擬合: minimize Σ(y_exp - y_model)²
```

**化工案例**：
- 🧪 **乙醇-水系統 VLE 擬合**（蒸餾設計基礎）
- 📊 x-y Diagram、Parity Plot、Residual Analysis

**輸出圖表**：B01-B08 (VLE Diagram, Parity Plot, Confidence Ellipse, Multi-start, etc.)

**技術亮點**：
- ✅ 從線性到非線性的完整學習路徑
- ✅ 化工熱力學的嚴謹數學推導
- ✅ 可直接應用於論文研究

**預計學習時間**：6-8 小時

---

### 🏭 Unit08｜工業軟感測器開發與部署

**檔案**：
- 📄 講義：[Unit08_SoftSensor_and_Cheminformatics.md](Unit08_SoftSensor_and_Cheminformatics.md)
- 💻 實作：[Unit08_SoftSensor_and_Cheminformatics.ipynb](Unit08_SoftSensor_and_Cheminformatics.ipynb)
- 📁 輸出：`P2_Unit08_SoftSensor_Results/` (8 files)

**學習目標**：
- 理解軟感測器在化工製程中的角色
- 掌握 **Gradient Boosting** 算法（數學推導 + 直觀理解）
- 集成學習理論（Bagging vs Boosting）
- 時序特徵工程（Lag Features、死區時間）
- TimeSeriesSplit 驗證策略
- 超參數調校與模型監控
- 不確定性量化（Quantile Regression）
- SHAP 可解釋性分析

**⭐ 工業部署完整指南**（課程最大亮點）：

```
開發階段
├─ 數據預處理與特徵工程
├─ 模型訓練與驗證（TimeSeriesSplit）
├─ 超參數調校（RandomizedSearchCV）
└─ 性能評估與可解釋性（SHAP）

部署階段
├─ 模型序列化（Pickle + Joblib + 版本控制）
├─ REST API 開發（Flask 微服務）
├─ OPC UA 整合（化工 DCS 系統對接）
├─ 漂移監控（Residual Monitoring）
├─ 自動再訓練策略
├─ 可觀測性（Prometheus + Grafana）
└─ 災難恢復計畫
```

**化工案例**：
- 🏭 **蒸餾塔品質軟感測器**：用溫度/壓力/流量預測產品純度

**模型比較**：
- 線性回歸 vs Random Forest vs Gradient Boosting vs Neural Network
- 性能 / 可解釋性 / 部署複雜度 / 計算成本 綜合評估

**技術亮點**：
- ✅ 從學術研究到工業部署的**完整閉環**
- ✅ 業界罕見的「最後一哩路」教學
- ✅ 模型卡片（Model Card）、部署檢查清單

**預計學習時間**：8-10 小時

---

### 📖 選讀附錄｜化學資訊學 (Cheminformatics)

**檔案**：
- 📄 講義：[Unit08_Appendix_Cheminformatics.md](Unit08_Appendix_Cheminformatics.md)
- 💻 實作：[Unit08_Appendix_Cheminformatics.ipynb](Unit08_Appendix_Cheminformatics.ipynb)
- 📁 輸出：`P2_Unit08_Cheminfo_Results/` (4 files)

**附錄說明**：
- ⚠️ 本附錄為**選讀內容**，獨立於主課程之外
- ✅ 對製藥/材料化學感興趣的同學可深入學習
- ✅ 不影響 Part 2 主線學習進度

**學習目標**：
- 化學結構的數位化表示（SMILES、Graph）
- 分子描述子計算（LogP、TPSA、Lipinski's Rule）
- RDKit 套件使用與特徵工程
- Morgan Fingerprints 與分子相似度（Tanimoto Coefficient）
- 子結構搜尋（SMARTS 語法）
- QSAR 建模基礎

**與主課程的連結**：
- 延伸 Unit05-08 的監督式學習方法至藥物發現
- 展示特徵工程在化學領域的專業應用

**預計學習時間**：2-3 小時

---

## 🎓 學習路徑建議

### 推薦學習順序

```
1. Unit05 (Decision Tree)
   ↓ 建立分類模型基礎
   
2. Unit06 (Cross-Validation)
   ↓ 掌握驗證策略（時序數據重點）
   
3. Unit07 (Regression Basics + Thermodynamics)
   ↓ 從線性到非線性回歸
   
4. Unit08 (Soft Sensor + Deployment)
   ↓ 工業級應用與部署
   
5. Unit08 Appendix (可選)
   ↓ 化學資訊學進階主題
```

### 先修知識檢查清單

**必備知識**：
- ✅ Python 基礎語法（變數、迴圈、函數、類別）
- ✅ NumPy 陣列操作
- ✅ Pandas DataFrame 基本操作
- ✅ Matplotlib/Seaborn 繪圖基礎

**建議先修**：
- 📚 Part 1 資料前處理單元（`Part_1/Unit04_ML_Preprocessing_Workflow`）
- 📊 基礎統計學（平均數、變異數、相關係數）
- 🧮 線性代數基礎（矩陣乘法、轉置）

**化工專業背景**（有助於但非必須）：
- 化工熱力學（相平衡、活度係數）
- 反應工程（阿瑞尼士方程）
- 單元操作（蒸餾、反應器）

---

## 📊 課程特色與亮點

### ⭐ 核心優勢

1. **時序數據處理專章**（Unit06）
   - 解決化工 AI 最常見的陷阱
   - 實例證明 KFold vs TimeSeriesSplit 性能差異可達 17%

2. **完整部署指南**（Unit08）
   - 從模型訓練到生產環境的完整流程
   - OPC UA、REST API、漂移監控、災難恢復

3. **熱力學建模路徑**（Unit07）
   - 從基礎回歸到非線性熱力學的無縫過渡
   - 可直接應用於學術研究

4. **工程思維培養**
   - 成本效益分析（FP vs FN）
   - 系統整合（DCS/SCADA）
   - 模型可維護性（漂移監控、自動再訓練）

5. **可複現性極高**
   - 所有數據自包含（線上下載或合成）
   - Colab + 本地環境雙支援
   - 路徑管理規範（REPO_ROOT / OUTPUT_DIR）

### 🏆 課程成果

完成本 Part 後，學生將能夠：

- ✅ 建立並評估分類/回歸模型
- ✅ 正確處理化工製程的時序數據
- ✅ 進行熱力學參數擬合與不確定性分析
- ✅ 開發工業級軟感測器
- ✅ 將模型部署到生產環境
- ✅ 實施模型監控與維護策略

---

## 📁 資料夾結構

```
Part_2/
├── README.md                                    # 本課程說明
├── COURSE_REVIEW_REPORT.md                      # 課程審查報告
├── INTEGRATION_REPORT.md                        # 內容整合報告
├── VERIFICATION_REPORT.md                       # 驗證報告
│
├── Unit05_DecisionTree_Classification.md        # 講義：決策樹分類
├── Unit05_DecisionTree_Classification.ipynb     # 實作：決策樹分類
├── P2_Unit05_Results/                           # 輸出檔案 (8 files)
│   ├── 01_confusion_matrix.png
│   ├── 02_feature_importance.png
│   ├── 03_pr_curve.png
│   ├── 04_cv_scores.png
│   ├── 05_grid_search_heatmap.png
│   ├── 06_reactor_boundary.png
│   ├── titanic_tree_model.pkl
│   └── model_card.json
│
├── Unit06_CV_Model_Selection.md                # 講義：交叉驗證
├── Unit06_CV_Model_Selection.ipynb             # 實作：交叉驗證
├── P2_Unit06_Results/                           # 輸出檔案 (2 files)
│   ├── 04_reactor_boundary.png
│   └── 05_timeseries_split.png
│
├── Unit07_Thermodynamic_Fitting.md             # 講義：熱力學擬合
├── Unit07_Thermodynamic_Fitting.ipynb          # 實作：熱力學擬合
├── P2_Unit07_Results/                           # 輸出檔案 (12 files)
│   ├── A01_arrhenius_plot.png                   # Part A: 回歸基礎
│   ├── A02_residual_diagnostics.png
│   ├── A03_polynomial_regression.png
│   ├── A04_regularization.png
│   ├── B01_param_correlation.png                # Part B: 熱力學
│   ├── B02_multistart_params.png
│   ├── B03_vle_diagram.png
│   ├── B04_parity_plot.png
│   ├── B05_residual_analysis.png
│   ├── B06_thermo_properties.png
│   ├── B07_model_comparison.png
│   └── B08_ai_vs_physics.png
│
├── Unit08_SoftSensor_and_Cheminformatics.md    # 講義：軟感測器
├── Unit08_SoftSensor_and_Cheminformatics.ipynb # 實作：軟感測器
├── P2_Unit08_SoftSensor_Results/               # 輸出檔案 (8 files)
│   ├── soft_sensor_analysis.png
│   ├── model_comparison.png
│   ├── distillation_timeseries.png
│   ├── uncertainty_quantification.png
│   ├── shap_importance.png
│   └── shap_summary.png
│
├── Unit08_Appendix_Cheminformatics.md          # 講義：化學資訊學 (選讀)
├── Unit08_Appendix_Cheminformatics.ipynb       # 實作：化學資訊學 (選讀)
└── P2_Unit08_Cheminfo_Results/                 # 輸出檔案 (4 files)
    ├── molecules_grid.png
    ├── substructure_match.png
    ├── molecular_descriptors.csv
    └── similarity_to_aspirin.csv
```

---

## 🔄 內容整合說明

本課程為 2025-12-17 重構版本，整合了以下原始單元：

| 原始單元 | 整合至 | 整合內容 |
|---------|--------|---------|
| Unit02_MachineLearning_Basics | **Unit05** | 決策樹基礎 + 新增 CV、Grid Search、PR Curve、化工案例 |
| Unit03_Thermodynamic_Fitting | **Unit07 Part B** | 熱力學擬合 + 新增參數不確定性、多起點優化 |
| Unit05B_Regression_Basics | **Unit07 Part A** | 回歸基礎 + 阿瑞尼士方程案例 |
| Unit06_Cheminformatics | **Unit08 附錄** | 分子特徵工程（作為選讀內容） |
| Unit08_Soft_Sensor | **Unit08** | 軟感測器 + 新增 GB 理論、SHAP、完整部署指南 |

**整合原則**：
- ✅ 保留所有原始教學內容
- ✅ 新增工業實務進階技術
- ✅ 統一輸出路徑至 `P2_UnitXX_Results/`
- ✅ 所有數據自包含，無需外部檔案

---

## 🚀 快速開始

### 環境設定

**方法一：Google Colab（推薦）**
```python
# 無需安裝，直接在 Colab 中執行各單元的 .ipynb 檔案
# 所有套件會自動安裝
```

**方法二：本地環境**
```bash
# 1. 建立虛擬環境
conda create -n chemeng_ml python=3.10
conda activate chemeng_ml

# 2. 安裝必要套件
pip install numpy pandas matplotlib seaborn scikit-learn scipy

# 3. (Unit08 附錄需要) 安裝 RDKit
conda install -c conda-forge rdkit

# 4. 啟動 Jupyter Notebook
jupyter notebook
```

### 執行順序

1. 開啟 `Part_2/` 資料夾
2. 按照 Unit05 → Unit06 → Unit07 → Unit08 順序學習
3. 每個單元：
   - 📖 先閱讀 `.md` 講義（理解理論）
   - 💻 再執行 `.ipynb` 實作（動手實踐）
   - 📊 檢查 `P2_UnitXX_Results/` 輸出結果

---

## 💡 學習建議

### 對於學生

- 📝 **做筆記**：每個數學公式的物理意義
- 🔬 **動手改**：修改超參數觀察結果變化
- 🤔 **多思考**：為什麼化工數據要用 TimeSeriesSplit？
- 💬 **多討論**：與同學分享模型選擇的考量

### 對於教師

- 🎯 **重點強調**：Unit06 的時序數據處理（易錯點）
- 📊 **課堂討論**：Unit05 的成本分析（FP vs FN）
- 🏭 **實務連結**：Unit08 的部署案例分享
- 📚 **作業設計**：讓學生用實驗室數據建模

### 進階挑戰

完成基礎學習後，可嘗試：

1. **Unit05 進階**：比較 Decision Tree vs Random Forest
2. **Unit06 進階**：實作 Stratified Split 處理不平衡數據
3. **Unit07 進階**：擬合 NRTL Model（另一活度係數模型）
4. **Unit08 進階**：實作 Online Learning（流式數據更新）

---

## 📊 課程評估

根據 [課程審查報告](COURSE_REVIEW_REPORT.md)：

**綜合評分**：**96/100** ⭐⭐⭐⭐⭐

| 評估項目 | 分數 | 評語 |
|---------|------|------|
| 課程結構完整性 | 20/20 | 知識遞進邏輯清晰 |
| 技術深度 | 19/20 | 從基礎到工業部署 |
| 化工專業性 | 19/20 | 案例貼近實務，理論嚴謹 |
| 實作可執行性 | 20/20 | 所有代碼可運行，輸出完整 |
| 教學法設計 | 18/20 | 理論實作平衡，漸進式學習 |

**審查結論**：✅ **課程已完備，可正式投入教學**

---

## 📞 支援與回饋

### 常見問題

**Q: 我沒有化工背景，可以學習本課程嗎？**
A: 可以！課程會解釋所有化工專業術語。建議先學習 Part 1 的數據前處理單元。

**Q: 為什麼 Unit06 只有 2 個輸出檔案？**
A: Unit06 聚焦於**概念理解**（CV 策略選擇），不強調大量視覺化，這是刻意的教學設計。

**Q: Unit08 附錄是必學的嗎？**
A: 不是。Cheminformatics 是選讀內容，適合對製藥/材料化學感興趣的同學。

**Q: 可以跳過 Unit07 Part A 直接學 Part B 嗎？**
A: 不建議。Part A 建立了回歸的基礎概念，是理解 Part B 非線性擬合的必要前置知識。

### 技術支援

- 🐛 程式碼問題：檢查 Python 版本 (推薦 3.9-3.11)
- 📦 套件安裝：使用 `conda` 或 `pip` 最新版本
- 💾 路徑問題：確認工作目錄在專案根目錄

---

## 📖 延伸學習資源

### 推薦教科書

- **機器學習**：*Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (Aurélien Géron)
- **化工熱力學**：*Chemical, Biochemical, and Engineering Thermodynamics* (Sandler)
- **化學資訊學**：*An Introduction to Chemoinformatics* (Leach & Gillet)

### 線上資源

- [Scikit-learn 官方文檔](https://scikit-learn.org/)
- [RDKit 官方文檔](https://www.rdkit.org/)
- [時序數據交叉驗證指南](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)

---

## 📋 更新紀錄

### v2.0 (2025-12-17) - 重構版本
- ✅ 整合 5 個原始單元為 4 個主單元 + 1 個附錄
- ✅ 新增 Unit06 時序數據處理專章
- ✅ 新增 Unit07 Part A 回歸基礎
- ✅ 新增 Unit08 完整部署指南
- ✅ 分離 Cheminformatics 為選讀附錄
- ✅ 統一輸出路徑與檔案命名
- ✅ 所有 notebook 測試通過

### v1.0 (2024) - 原始版本
- 基礎監督式學習內容

---

**🎓 準備好開始學習了嗎？從 [Unit05](Unit05_DecisionTree_Classification.md) 開始吧！**

---

> **課程維護者**：化工資料科學教學團隊  
> **最後更新**：2025-12-17  
> **授權**：教育用途
