# Unit14 模型評估與選擇 (Model Evaluation and Selection)

## 📚 單元簡介

在前四個單元中，我們學習了各種監督式學習模型：線性模型 (Unit10)、非線性模型 (Unit11)、分類模型 (Unit12)、集成學習 (Unit13)。我們掌握了豐富的建模工具，但面對實際問題時，一個關鍵問題出現了：**如何系統性地評估和選擇最適合的模型？**

模型評估與選擇是機器學習建模流程中最關鍵的一環。在工業實務中，選錯模型可能導致安全風險、經濟損失、法規問題或資源浪費。因此，我們需要客觀的評估標準、系統性的比較方法，以及全面的性能考量（不只是準確度，還要考慮成本、穩健性、可解釋性）。

本單元將教導學生如何：
- **全面評估模型性能**：理解各種評估指標的含義、適用場景與限制
- **確保泛化能力**：掌握交叉驗證的進階技巧，避免過擬合
- **診斷模型問題**：識別過擬合與欠擬合，使用學習曲線與驗證曲線
- **公平比較模型**：使用統計方法比較不同模型的性能
- **多目標決策**：在準確度、成本、可解釋性之間做出權衡

本單元是 Part_3 監督式學習的總結與整合，為學員建立完整的機器學習建模能力。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **掌握全面的評估指標**：理解回歸指標 (MSE, RMSE, MAE, R², MAPE) 和分類指標 (Accuracy, Precision, Recall, F1, ROC-AUC) 的適用場景
2. **精通交叉驗證技術**：使用 K-Fold、Stratified K-Fold、Time Series CV、Nested CV 確保模型泛化能力
3. **診斷模型狀態**：使用學習曲線診斷過擬合/欠擬合，使用驗證曲線優化超參數
4. **公平比較模型**：使用統計檢定方法 (t-test, Wilcoxon test) 判斷模型差異顯著性
5. **多目標決策**：在準確度、計算成本、可解釋性、部署難度之間做出最佳權衡

---

## 📖 單元內容架構

### 1️⃣ 總覽篇：模型評估的重要性 ⭐

**檔案**：[Unit14_Model_Evaluation_Overview.md](Unit14_Model_Evaluation_Overview.md)

**內容重點**：
- **模型評估的重要性**：
  - 為什麼模型評估如此重要？
  - 工業實務中選錯模型的風險
  - 化工領域的特殊考量（安全性、可解釋性、實時性、小樣本、多目標）
  
- **回歸模型評估指標**：
  - MSE、RMSE、MAE、R²、MAPE 的定義與適用場景
  - 指標選擇原則與注意事項
  
- **分類模型評估指標**：
  - Accuracy、Precision、Recall、F1-Score、ROC-AUC 的定義與適用場景
  - 混淆矩陣 (Confusion Matrix) 分析
  - 多類別分類指標 (Macro/Micro/Weighted Average)
  
- **交叉驗證進階技巧**：
  - K-Fold CV、Stratified K-Fold、Leave-One-Out CV
  - Time Series CV（時間序列資料的正確驗證方式）
  - Nested CV（超參數調整的無偏估計）
  
- **偏差-方差權衡 (Bias-Variance Tradeoff)**：
  - 理解模型誤差的分解
  - 欠擬合與過擬合的識別與解決
  
- **學習曲線與驗證曲線**：
  - 學習曲線診斷資料量需求
  - 驗證曲線優化超參數
  
- **多目標模型選擇框架**：
  - 準確度 vs. 計算成本 vs. 可解釋性 vs. 部署難度
  - 帕累托前沿 (Pareto Frontier) 概念

**適合讀者**：所有學員，**建議最先閱讀**以建立全面的評估思維

---

### 2️⃣ 超參數調整：Grid Search

**檔案**：
- [Unit14_Hyperparameter_Tuning_GridSearch.ipynb](Unit14_Hyperparameter_Tuning_GridSearch.ipynb)
- [Unit14_Hyperparameter_Tuning_GridSearch_v2.ipynb](Unit14_Hyperparameter_Tuning_GridSearch_v2.ipynb)

**內容重點**：
- **Grid Search 原理**：
  - 窮舉搜索所有超參數組合
  - 交叉驗證評估每組參數
  - 選擇交叉驗證分數最高的組合
  
- **實作技術**：
  - scikit-learn `GridSearchCV` 使用
  - 參數網格設計 (`param_grid`)
  - 評分函數選擇 (`scoring`)
  - 並行化加速 (`n_jobs=-1`)
  - 結果分析 (`cv_results_`, `best_params_`)
  
- **化工應用案例**：
  - 製程模型的超參數優化
  - 不同評分指標的選擇（如優先降低 RMSE 或 MAE）
  
- **優缺點**：
  - ✅ 優點：保證找到最佳組合、簡單直觀、結果可重現
  - ❌ 缺點：計算成本高、維度災難、不適合連續參數

**適合場景**：參數空間小、需要精確最佳參數、計算資源充足

---

### 3️⃣ 超參數調整：Bayesian Optimization

**檔案**：[Unit14_Hyperparameter_Tuning_Bayesian.ipynb](Unit14_Hyperparameter_Tuning_Bayesian.ipynb)

**內容重點**：
- **Bayesian Optimization 原理**：
  - 基於高斯過程的 Surrogate Model
  - Acquisition Function（EI, UCB, PI）指導搜索方向
  - 探索 (Exploration) vs. 利用 (Exploitation) 權衡
  - 順序最佳化策略
  
- **實作技術**：
  - `scikit-optimize` 或 `optuna` 套件使用
  - 搜索空間定義（Integer、Real、Categorical）
  - 最佳化過程視覺化
  - Early Stopping 機制
  
- **化工應用案例**：
  - 高成本實驗的參數優化（每次評估昂貴）
  - 複雜模型的超參數搜索（XGBoost、LightGBM）
  
- **優缺點**：
  - ✅ 優點：搜索效率高、適合高維連續空間、自動平衡探索與利用
  - ❌ 缺點：結果可能不穩定、實作較複雜、需要選擇 Acquisition Function

**適合場景**：參數空間大、評估成本高、連續參數優化

---

### 4️⃣ 模型比較

**檔案**：[Unit14_Model_Comparison.ipynb](Unit14_Model_Comparison.ipynb)

**內容重點**：
- **統計檢定方法**：
  - **配對 t 檢定 (Paired t-test)**：比較兩個模型在多個數據集上的平均性能
  - **Wilcoxon 符號秩檢定**：t 檢定的非參數替代方案
  - **McNemar 檢定**：比較兩個分類模型的錯誤率
  - **Friedman 檢定**：比較多個模型在多個數據集上的性能
  
- **實作技術**：
  - `scipy.stats` 統計檢定函數使用
  - p-value 解讀與顯著性判斷（通常 α=0.05）
  - 效果量 (Effect Size) 計算
  - 多重比較校正 (Bonferroni Correction)
  
- **化工應用案例**：
  - 判斷新模型是否顯著優於現有模型
  - 多個候選模型的排序與篩選
  
- **注意事項**：
  - 交叉驗證折數的選擇影響檢定力
  - 僅看 p-value 不足，需考慮實際差異大小
  - 避免 "p-hacking"（反覆測試直到顯著）

**適合場景**：需要科學證據證明模型差異、多個候選模型的正式比較

---

### 5️⃣ 模型可解釋性基礎

**檔案**：[Unit14_Model_Interpretability_Basics.ipynb](Unit14_Model_Interpretability_Basics.ipynb)

**內容重點**：
- **可解釋性的重要性**：
  - 化工領域的監管要求
  - 操作人員的理解需求
  - 模型調試與改進
  - 建立信任與可信度
  
- **全局解釋方法**：
  - **特徵重要性 (Feature Importance)**：哪些特徵最重要？
  - **部分依賴圖 (Partial Dependence Plot, PDP)**：特徵如何影響預測？
  - **SHAP Summary Plot**：特徵影響的分佈
  
- **局部解釋方法**：
  - **LIME (Local Interpretable Model-agnostic Explanations)**：單一樣本的局部線性近似
  - **SHAP Values (SHapley Additive exPlanations)**：基於博弈論的特徵貢獻分解
  - **Individual Conditional Expectation (ICE)**：單一樣本的特徵效應曲線
  
- **實作技術**：
  - `shap` 套件使用 (TreeExplainer, KernelExplainer)
  - `lime` 套件使用
  - scikit-learn `PartialDependenceDisplay`
  
- **化工應用案例**：
  - 解讀複雜模型的預測邏輯
  - 識別關鍵操作變數
  - 向非技術人員解釋模型決策

**適合場景**：使用黑箱模型（如 XGBoost）但需要解釋、監管要求、建立信任

---

### 6️⃣ 模型選擇流程 (Pipeline)

**檔案**：[Unit14_Model_Selection_Pipeline.ipynb](Unit14_Model_Selection_Pipeline.ipynb)

**內容重點**：
- **完整建模流程**：
  1. 資料載入與探索性分析
  2. 資料預處理 (Pipeline 整合)
  3. 候選模型定義（Linear、Tree、Ensemble）
  4. 交叉驗證評估所有候選模型
  5. 超參數調整最佳候選模型
  6. 最終模型選擇與測試集評估
  7. 模型解釋與特徵重要性分析
  8. 模型儲存與部署準備
  
- **實作技術**：
  - scikit-learn `Pipeline` 整合預處理與模型
  - `ColumnTransformer` 處理混合特徵類型
  - 多模型批量訓練與比較
  - 最佳模型的持久化 (`joblib.dump`)
  
- **化工應用案例**：
  - 標準化的建模工作流程
  - 可重現的模型開發
  - 生產環境部署的最佳實踐

**適合場景**：正式的工業項目、需要可重現性、團隊協作

---

### 7️⃣ 實作練習

**檔案**：[Unit14_Homework.ipynb](Unit14_Homework.ipynb)

**練習內容**：
- 在化工數據集上評估 Unit10-13 學習的多種模型
- 使用交叉驗證公平比較模型性能
- 繪製學習曲線診斷模型狀態
- 使用 Grid Search 或 Bayesian Optimization 調整超參數
- 使用統計檢定判斷模型差異顯著性
- 使用 SHAP 解釋最佳模型的預測邏輯
- 撰寫模型選擇報告，給出最終建議

---

## 📊 數據集說明

### 1. 催化劑篩選數據 (`data/catalyst_screening/`)
- 催化劑配方與反應性能數據
- 用於示範模型評估與選擇流程

### 2. 反應器數據 (`data/reactor_data/` 或 `data/reactor_simulation/`)
- 反應器操作數據與產率預測
- 用於回歸模型評估練習

---

## 🎓 模型評估與選擇決策框架

### 評估維度與權重
| 評估維度 | 化工重要性 | 量化指標 | 權重建議 |
|---------|-----------|---------|---------|
| **預測準確度** | 極高 | RMSE, R², F1-Score | 40% |
| **泛化能力** | 極高 | 交叉驗證分數穩定性 | 20% |
| **可解釋性** | 高 | 模型複雜度、SHAP 可用性 | 15% |
| **計算成本** | 中 | 訓練時間、預測延遲 | 10% |
| **穩健性** | 高 | 對異常值敏感度 | 10% |
| **部署難度** | 中 | 依賴套件數量、模型大小 | 5% |

### 決策流程圖
```
1. 定義問題類型 (回歸/分類/多類別)
   ↓
2. 選擇評估指標 (根據業務目標)
   ↓
3. 訓練候選模型 (Linear, Tree, Ensemble)
   ↓
4. 交叉驗證評估 (5-10 Fold CV)
   ↓
5. 初步篩選 (Top 3 模型)
   ↓
6. 超參數調整 (Grid Search / Bayesian Opt)
   ↓
7. 統計檢定比較 (確認顯著差異)
   ↓
8. 多目標權衡 (準確度、成本、可解釋性)
   ↓
9. 最終模型選擇與測試集驗證
   ↓
10. 模型解釋與文檔化
```

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

### 模型解釋與優化套件
```python
shap >= 0.40.0              # SHAP 解釋工具
lime >= 0.2.0               # LIME 解釋工具
scikit-optimize >= 0.9.0    # 貝氏優化（skopt）
optuna >= 3.0.0             # 超參數優化
```

### 選用套件
```python
joblib >= 1.0.0             # 模型儲存
mlxtend >= 0.19.0           # 機器學習擴展工具
yellowbrick >= 1.3.0        # 視覺化診斷工具
```

---

## 📈 學習路徑建議

### 第一階段：基礎概念建立
1. 閱讀 [Unit14_Model_Evaluation_Overview.md](Unit14_Model_Evaluation_Overview.md)
2. 理解各種評估指標的適用場景與限制
3. 熟悉交叉驗證的進階技巧

### 第二階段：超參數調整實作
1. **Grid Search**（基礎方法）
   - 執行 [Unit14_Hyperparameter_Tuning_GridSearch.ipynb](Unit14_Hyperparameter_Tuning_GridSearch.ipynb)
   - 執行 [Unit14_Hyperparameter_Tuning_GridSearch_v2.ipynb](Unit14_Hyperparameter_Tuning_GridSearch_v2.ipynb)
   - 重點掌握：參數網格設計、並行化、結果分析
   
2. **Bayesian Optimization**（進階方法）
   - 執行 [Unit14_Hyperparameter_Tuning_Bayesian.ipynb](Unit14_Hyperparameter_Tuning_Bayesian.ipynb)
   - 重點掌握：Acquisition Function、搜索效率、收斂監控

### 第三階段：模型比較與解釋
1. **模型統計比較**
   - 執行 [Unit14_Model_Comparison.ipynb](Unit14_Model_Comparison.ipynb)
   - 重點掌握：配對 t 檢定、Wilcoxon 檢定、p-value 解讀
   
2. **模型可解釋性**
   - 執行 [Unit14_Model_Interpretability_Basics.ipynb](Unit14_Model_Interpretability_Basics.ipynb)
   - 重點掌握：SHAP、LIME、PDP、特徵重要性

### 第四階段：完整建模流程
1. **模型選擇 Pipeline**
   - 執行 [Unit14_Model_Selection_Pipeline.ipynb](Unit14_Model_Selection_Pipeline.ipynb)
   - 學習標準化的建模工作流程
   - 理解從資料到部署的完整流程

### 第五階段：綜合應用與練習
1. 完成 [Unit14_Homework.ipynb](Unit14_Homework.ipynb)
2. 在實際化工數據上完整執行評估與選擇流程
3. 撰寫模型選擇報告，給出技術與商業建議
4. 嘗試將整個流程應用於自己的項目

---

## 🔍 化工領域核心應用

### 1. 製程預測模型的評估與部署 ⭐
- **目標**：選擇最適合生產環境的預測模型
- **評估流程**：
  1. 定義業務指標（如預測誤差 <2%）
  2. 候選模型：Linear, Random Forest, XGBoost
  3. 交叉驗證評估（Time Series CV，保持時間順序）
  4. 超參數調整最佳候選模型
  5. 統計檢定確認優勢顯著
  6. 評估計算成本與實時性
  7. 模型解釋與操作人員培訓
  8. A/B 測試驗證實際效益
  
- **關鍵決策**：
  - 準確度 vs. 實時性權衡
  - 黑箱模型 vs. 可解釋性需求
  - 模型更新頻率與成本

### 2. 品質分類模型的選擇與優化
- **目標**：準確分類產品品質等級（A/B/C/D）
- **評估流程**：
  - 類別不平衡處理
  - 多類別評估指標（Macro F1）
  - 誤判成本矩陣設計
  - 閾值調整優化
  - 混淆矩陣分析與錯誤模式識別
  
- **關鍵決策**：
  - Type I vs. Type II Error 的成本權衡
  - 召回率 vs. 精確率優先順序
  - 自動化決策 vs. 人工複審

### 3. 實驗設計的模型驗證
- **目標**：確保模型在未見數據上的泛化能力
- **評估策略**：
  - 留出法 (Holdout) 驗證
  - 時間分割驗證（訓練舊數據，測試新數據）
  - 外部驗證（不同批次、不同設備）
  - 持續監控與模型衰退檢測
  
- **關鍵技術**：
  - 概念飄移 (Concept Drift) 偵測
  - 模型重訓練觸發機制
  - 在線學習策略

### 4. 安全預警系統的模型評估
- **目標**：高召回率（不能漏報危險）同時控制誤報率
- **評估重點**：
  - 召回率 >99% 為底線
  - ROC 曲線與最佳閾值選擇
  - 誤報成本 vs. 漏報成本量化
  - 實時響應速度要求
  
- **關鍵決策**：
  - 閾值設定策略（保守 vs. 平衡）
  - 多級預警架構設計
  - 人工介入機制

### 5. 多目標優化的模型組合
- **目標**：同時優化產率、能耗、品質等多個目標
- **評估框架**：
  - 帕累托前沿分析
  - 多目標評分函數設計
  - 權重敏感度分析
  - 專家知識與數據驅動的結合
  
- **關鍵技術**：
  - Multi-output Regression
  - Ensemble with Multiple Objectives
  - 約束優化

---

## 📝 評估指標完整總結

### 回歸指標選擇建議
| 應用場景 | 推薦指標 | 理由 |
|---------|---------|------|
| 一般預測任務 | RMSE | 懲罰大誤差，符合最小二乘法假設 |
| 對異常值敏感 | MAE | 對大誤差不過度懲罰 |
| 相對誤差重要 | MAPE | 百分比誤差，適合不同量級比較 |
| 解釋模型好壞 | R² | 直觀的變異解釋比例 (0-1) |

### 分類指標選擇建議
| 應用場景 | 推薦指標 | 理由 |
|---------|---------|------|
| 類別平衡 | Accuracy | 簡單直觀 |
| 類別不平衡 | F1-Score | 平衡精確率與召回率 |
| 漏報代價高 | Recall | 優先保證不漏掉正類 |
| 誤報代價高 | Precision | 優先保證正類預測準確 |
| 需要機率輸出 | ROC-AUC | 評估機率排序能力 |
| 多類別任務 | Macro F1 | 類別等權平均 |

---

## 🚀 進階學習方向

完成本單元後，可以進一步探索：

1. **AutoML 自動化建模**：Auto-sklearn、TPOT、H2O AutoML
2. **模型監控與維護**：Concept Drift Detection、Online Learning
3. **A/B 測試與因果推論**：線上實驗設計、因果效應估計
4. **模型壓縮與加速**：模型蒸餾、量化、剪枝、ONNX 部署
5. **公平性與偏見**：Fairness-aware ML、Bias Detection
6. **可信機器學習**：Uncertainty Quantification、Conformal Prediction

---

## 📚 參考資源

### 教科書
1. *An Introduction to Statistical Learning* by James et al. (第 5 章 Resampling Methods)
2. *The Elements of Statistical Learning* by Hastie et al. (第 7 章 Model Assessment and Selection)
3. *Interpretable Machine Learning* by Christoph Molnar (線上免費)

### 線上資源
- [scikit-learn Model Selection 官方文件](https://scikit-learn.org/stable/model_selection.html)
- [SHAP 官方文件與教學](https://shap.readthedocs.io/)
- [Optuna 官方文件](https://optuna.readthedocs.io/)

### 化工領域應用論文
- 機器學習模型在化工製程預測中的評估與比較
- 可解釋 AI 在製程控制中的應用
- 化工數據科學的最佳實踐指南

---

## ✍️ 課後習題提示

1. **評估指標比較**：在同一模型上比較不同評估指標，理解其差異與適用場景
2. **交叉驗證實驗**：比較 K-Fold、Stratified K-Fold、Time Series CV 的結果差異
3. **學習曲線繪製**：繪製學習曲線，判斷是否需要更多資料或更複雜模型
4. **超參數調整對決**：比較 Grid Search 與 Bayesian Optimization 的效率與效果
5. **統計檢定實踐**：使用 t 檢定判斷兩個模型的差異是否顯著（p<0.05）
6. **SHAP 解釋黑箱模型**：對 XGBoost 模型使用 SHAP 解釋預測結果
7. **完整建模報告**：撰寫一份包含模型選擇、評估、解釋的完整技術報告

---

## 📞 技術支援

如有疑問或需要協助，歡迎透過以下方式聯繫：
- 課程討論區
- 指導教授 Office Hour
- 課程助教

---

**版本資訊**：Unit14 v1.0 | 最後更新：2026-01-27

---

## 🎊 Part 3 監督式學習完結總結

恭喜您完成 Part 3 的所有單元！您已經系統性地掌握了：

✅ **Unit10**：線性模型回歸（Linear Regression, Ridge, Lasso, ElasticNet, SGD）  
✅ **Unit11**：非線性模型回歸（Polynomial, Decision Tree, SVM, GPR, Gradient Boosting）  
✅ **Unit12**：分類模型（Logistic Regression, Decision Tree, SVC, Naive Bayes, Gradient Boosting）  
✅ **Unit13**：集成學習（Random Forest, XGBoost, LightGBM, CatBoost, Stacking）  
✅ **Unit14**：模型評估與選擇（交叉驗證、超參數調整、統計檢定、模型解釋）

**您現在具備的核心能力**：
- 🎯 根據問題類型選擇合適的模型
- 🔧 建立、訓練、評估、優化機器學習模型
- 📊 系統性地比較模型性能並做出最佳選擇
- 🧠 解釋黑箱模型的預測邏輯
- 🚀 將模型部署到實際化工應用中

**下一步學習方向**：
- **Part 4**：深度學習（Neural Networks, CNN, RNN, Transformer）
- **Part 5**：時間序列預測、強化學習、遷移學習等進階主題
- **專題實作**：將所學應用於實際化工問題，完成端到端的建模專案

繼續加油，邁向化工 AI 專家之路！💪
