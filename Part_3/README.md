# Part 3: 監督式學習 (Supervised Learning)

## 📚 Part 3 簡介

本 Part 系統性地介紹監督式學習 (Supervised Learning) 在化學工程領域的理論與應用。監督式學習是機器學習中應用最廣泛的分支，主要處理**已標記數據**，透過學習輸入特徵與目標變數之間的映射關係，建立預測模型。在化工領域，監督式學習可用於製程參數預測、產品品質控制、設備故障診斷、操作條件優化等多種應用場景。

本 Part 涵蓋兩大核心任務：
1. **回歸 (Regression)**：預測連續數值（如溫度、產量、濃度）
   - **線性模型 (Unit10)**：從基礎的 OLS 到正則化技術
   - **非線性模型 (Unit11)**：多項式、決策樹、SVM、高斯過程、梯度提升
2. **分類 (Classification)**：預測離散類別（如良品/不良品、安全/危險）
   - **分類模型 (Unit12)**：邏輯迴歸、決策樹、SVM、樸素貝氏、梯度提升
3. **集成學習 (Ensemble Learning)**：組合多個模型提升性能
   - **集成方法 (Unit13)**：Random Forest、XGBoost、LightGBM、CatBoost、Stacking
4. **模型評估與選擇 (Model Evaluation)**：系統性評估與比較模型
   - **評估技術 (Unit14)**：評估指標、交叉驗證、學習曲線、模型比較

---

## 🎯 學習目標

完成 Part 3 後，您將能夠：

1. **理解監督式學習的核心概念**：回歸 vs. 分類、偏差-方差權衡、正則化、超參數調整
2. **掌握豐富的建模工具**：從線性模型到集成學習，涵蓋主流演算法
3. **選擇合適的模型**：根據問題類型、數據特性、性能需求做出最佳選擇
4. **系統性評估模型**：使用科學方法評估、比較、優化模型性能
5. **解決實際問題**：將方法應用於化工製程的預測、控制、優化、診斷

---

## 📖 單元目錄

### [Unit10: 線性模型回歸 (Linear Models Regression)](Unit10/)

**主題**：使用線性關係建模，從基礎的最小二乘法到正則化技術

**單元概述**：[Unit10_Linear_Models_Overview.md](Unit10/Unit10_Linear_Models_Overview.md)

**演算法詳解**：
- [線性回歸 (Linear Regression)](Unit10/Unit10_Linear_Regression.md) ([Notebook](Unit10/Unit10_Linear_Regression.ipynb))
- [Ridge 回歸](Unit10/Unit10_Ridge_Regression.md) ([Notebook](Unit10/Unit10_Ridge_Regression.ipynb))
- [Lasso 回歸](Unit10/Unit10_Lasso_Regression.md) ([Notebook](Unit10/Unit10_Lasso_Regression.ipynb))
- [ElasticNet 回歸](Unit10/Unit10_ElasticNet_Regression.md) ([Notebook](Unit10/Unit10_ElasticNet_Regression.ipynb))
- [隨機梯度下降回歸 (SGD)](Unit10/Unit10_SGD_Regression.md) ([Notebook](Unit10/Unit10_SGD_Regression.ipynb))

**作業練習**：[Unit10_Linear_Models_Homework.ipynb](Unit10/Unit10_Linear_Models_Homework.ipynb)

**核心概念**：
- 最小二乘法 (OLS)
- 正則化：Ridge (L2)、Lasso (L1)、ElasticNet (L1+L2)
- 多重共線性處理
- 特徵縮放與標準化

**應用場景**：
- 產量預測
- 品質控制
- 能耗建模
- 參數優化

---

### [Unit11: 非線性模型回歸 (Non-Linear Models Regression)](Unit11/)

**主題**：捕捉複雜的非線性關係，處理化工領域的複雜現象

**單元概述**：[Unit11_NonLinear_Models_Overview.md](Unit11/Unit11_NonLinear_Models_Overview.md)

**演算法詳解**：
- [多項式回歸 (Polynomial Regression)](Unit11/Unit11_Polynomial_Regression.md) ([Notebook](Unit11/Unit11_Polynomial_Regression.ipynb))
- [決策樹回歸 (Decision Tree)](Unit11/Unit11_Decision_Tree.md) ([Notebook](Unit11/Unit11_Decision_Tree.ipynb))
- [支持向量機回歸 (SVM)](Unit11/Unit11_Support_Vector_Machine.md) ([Notebook](Unit11/Unit11_Support_Vector_Machine.ipynb))
- [高斯過程回歸 (Gaussian Process)](Unit11/Unit11_Gaussian_Process_Regression.md) ([Notebook](Unit11/Unit11_Gaussian_Process_Regression.ipynb))
- [梯度提升樹回歸 (Gradient Boosting)](Unit11/Unit11_Gradient_Boosting_Trees.md) ([Notebook](Unit11/Unit11_Gradient_Boosting_Trees.ipynb))

**作業練習**：[Unit11_NonLinear_Models_Homework.ipynb](Unit11/Unit11_NonLinear_Models_Homework.ipynb)

**核心概念**：
- 非線性關係類型
- 核函數 (Kernel Functions)
- 決策樹分裂準則
- 貝氏機率模型
- 梯度提升原理

**應用場景**：
- 反應動力學建模
- 製程曲線擬合
- 相平衡預測
- 複雜系統建模

---

### [Unit12: 分類模型 (Classification Models)](Unit12/)

**主題**：預測離散類別標籤，支援決策與診斷

**單元概述**：[Unit12_Classification_Models_Overview.md](Unit12/Unit12_Classification_Models_Overview.md)

**演算法詳解**：
- [邏輯迴歸 (Logistic Regression)](Unit12/Unit12_Logistic_Regression.md) ([Notebook](Unit12/Unit12_Logistic_Regression.ipynb))
- [決策樹分類器 (Decision Tree Classifier)](Unit12/Unit12_Decision_Tree_Classifier.md) ([Notebook](Unit12/Unit12_Decision_Tree_Classifier.ipynb))
- [支持向量分類 (Support Vector Classification)](Unit12/Unit12_Support_Vector_Classification.md) ([Notebook](Unit12/Unit12_Support_Vector_Classification.ipynb))
- [高斯樸素貝氏 (Gaussian Naive Bayes)](Unit12/Unit12_Gaussian_Naive_Bayes.md) ([Notebook](Unit12/Unit12_Gaussian_Naive_Bayes.ipynb))
- [梯度提升分類器 (Gradient Boosting Classifier)](Unit12/Unit12_Gradient_Boosting_Classifier.md) ([Notebook](Unit12/Unit12_Gradient_Boosting_Classifier.ipynb))

**實際案例**：
- [水質分類 (Water Quality)](Unit12/Unit12_Example_WaterQuality.md) ([Notebook](Unit12/Unit12_Example_WaterQuality.ipynb))
- [電力故障檢測與分類](Unit12/Unit12_Example_Electrical_Fault_detection_and_classification.md) ([Notebook](Unit12/Unit12_Example_Electrical_Fault_detection_and_classification.ipynb))

**作業練習**：[Unit12_Classification_Models_Homework.ipynb](Unit12/Unit12_Classification_Models_Homework.ipynb)

**核心概念**：
- 激活函數 (Sigmoid, Softmax)
- 決策邊界
- 機率預測
- 多類別分類
- 類別不平衡處理

**應用場景**：
- 品質檢測（良品/不良品）
- 安全監控（正常/異常）
- 故障診斷（故障類型分類）
- 狀態識別（操作模式分類）

---

### [Unit13: 集成學習方法 (Ensemble Learning Methods)](Unit13/)

**主題**：組合多個模型提升預測性能，工業級機器學習的核心技術

**單元概述**：
- [Unit13_Ensemble_Learning_Overview.md](Unit13/Unit13_Ensemble_Learning_Overview.md)
- [Unit13_Ensemble_Learning_Overview.ipynb](Unit13/Unit13_Ensemble_Learning_Overview.ipynb)

**演算法詳解**：
- **Random Forest (隨機森林)**：
  - [分類](Unit13/Unit13_Random_Forest_Classifier.md) ([Notebook](Unit13/Unit13_Random_Forest_Classifier.ipynb))
  - [回歸](Unit13/Unit13_Random_Forest_Regression.md) ([Notebook](Unit13/Unit13_Random_Forest_Regression.ipynb))
- **XGBoost**：
  - [分類](Unit13/Unit13_XGBoost_Classification.md) ([Notebook](Unit13/Unit13_XGBoost_Classification.ipynb))
  - [回歸](Unit13/Unit13_XGBoost_Regression.md) ([Notebook](Unit13/Unit13_XGBoost_Regression.ipynb))
- **LightGBM**：
  - [分類](Unit13/Unit13_LightGBM_Classification.ipynb)
  - [回歸](Unit13/Unit13_LightGBM_Regression.ipynb)
  - [講義](Unit13/Unit13_LightGBM.md)
- **CatBoost**：
  - [分類](Unit13/Unit13_CatBoost_Classification.ipynb)
  - [回歸](Unit13/Unit13_CatBoost_Regression.ipynb)
  - [講義](Unit13/Unit13_CatBoost.md)
- **Stacking (堆疊集成)**：
  - [講義與實作](Unit13/Unit13_Stacking.md) ([Notebook](Unit13/Unit13_Stacking.ipynb))

**作業練習**：[Unit13_Ensemble_Learning_Homework.ipynb](Unit13/Unit13_Ensemble_Learning_Homework.ipynb)

**核心概念**：
- Bagging vs. Boosting vs. Stacking
- 偏差-方差分解
- 多樣性原則
- 弱學習器組合

**應用場景**：
- 高精度預測（Kaggle 競賽）
- 複雜非線性問題
- 穩健的工業級模型
- 特徵重要性分析

---

### [Unit14: 模型評估與選擇 (Model Evaluation and Selection)](Unit14/)

**主題**：系統性評估模型性能，公平比較不同模型，做出最佳選擇

**單元概述**：[Unit14_Model_Evaluation_Overview.md](Unit14/Unit14_Model_Evaluation_Overview.md)

**主題詳解**：
- [回歸評估指標](Unit14/Unit14_Regression_Metrics.md) ([Notebook](Unit14/Unit14_Regression_Metrics.ipynb))
- [分類評估指標](Unit14/Unit14_Classification_Metrics.md) ([Notebook](Unit14/Unit14_Classification_Metrics.ipynb))
- [交叉驗證技術](Unit14/Unit14_Cross_Validation.md) ([Notebook](Unit14/Unit14_Cross_Validation.ipynb))
- [學習曲線與驗證曲線](Unit14/Unit14_Learning_Validation_Curves.md) ([Notebook](Unit14/Unit14_Learning_Validation_Curves.ipynb))
- [模型比較與選擇](Unit14/Unit14_Model_Comparison.md) ([Notebook](Unit14/Unit14_Model_Comparison.ipynb))

**作業練習**：[Unit14_Model_Evaluation_Homework.ipynb](Unit14/Unit14_Model_Evaluation_Homework.ipynb)

**核心概念**：
- **回歸指標**：MSE, RMSE, MAE, R², MAPE
- **分類指標**：Accuracy, Precision, Recall, F1, ROC-AUC
- **交叉驗證**：K-Fold, Stratified K-Fold, Time Series CV, Nested CV
- **診斷工具**：學習曲線、驗證曲線、混淆矩陣
- **統計檢定**：t-test, Wilcoxon test

**學習重點**：
- 選擇正確的評估指標
- 避免過擬合
- 確保泛化能力
- 公平比較模型
- 多目標權衡

---

## 🗂️ 數據集

本 Part 使用的數據集涵蓋多種化工與工業場景：

- **化工製程數據**：反應器、蒸餾塔、熱交換器操作數據
- **產品品質數據**：化學成分、物理性質、性能指標
- **設備監測數據**：溫度、壓力、流量、振動、電流
- **環境監測數據**：水質、空氣品質、排放數據
- **公開數據集**：UCI、Kaggle 等標準數據集

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

# 集成學習框架
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0

# 模型評估與優化
scipy>=1.7.0
joblib>=1.0.0
imbalanced-learn>=0.8.0  # 處理類別不平衡

# 視覺化進階
plotly>=5.3.0  # 互動式圖表
```

### 安裝指令

```bash
# 使用 pip 安裝
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost scipy joblib imbalanced-learn plotly

# 或使用 conda 安裝
conda install numpy pandas matplotlib seaborn scikit-learn scipy joblib
conda install -c conda-forge xgboost lightgbm imbalanced-learn plotly
pip install catboost
```

---

## 📊 學習路徑建議

### 1. 標準學習路徑（按順序學習）

**適合對象**：初學者、系統性學習機器學習

1. **Unit10 線性模型**：建立回歸建模的基礎概念
2. **Unit11 非線性模型**：擴展到複雜的非線性關係
3. **Unit12 分類模型**：從回歸延伸到分類任務
4. **Unit13 集成學習**：學習最強大的工業級方法
5. **Unit14 模型評估**：掌握系統性評估與選擇模型的能力

### 2. 快速應用路徑（針對特定需求）

**適合對象**：有基礎、需要快速解決問題

- **需要快速建模**：Unit10 (線性) → Unit13 (集成) → Unit14 (評估)
- **需要高精度預測**：Unit13 (集成) → Unit14 (評估) → Unit11 (非線性)
- **需要可解釋模型**：Unit10 (線性) → Unit11 (決策樹) → Unit14 (評估)
- **需要分類決策**：Unit12 (分類) → Unit13 (集成分類) → Unit14 (評估)

### 3. 進階研究路徑

**適合對象**：研究生、進階應用者

1. 完成所有單元的基礎學習
2. 深入研究 Unit13 的集成學習進階技術
3. 深入研究 Unit14 的模型評估統計方法
4. 探索超參數優化（Bayesian Optimization、Optuna）
5. 將方法應用於自己的研究數據

---

## 🎓 學習建議

### 1. 理論與實務並重
- 先理解演算法的數學原理（為什麼有效？）
- 再透過 Notebook 實作理解實現細節（如何實作？）
- 最後應用於實際問題（何時使用？）

### 2. 比較不同方法
- 同一問題使用多種演算法，比較性能差異
- 理解各演算法的優缺點與適用場景
- 培養模型選擇的直覺與判斷力

### 3. 調整超參數
- 實驗不同參數設定，觀察對結果的影響
- 使用 GridSearchCV 和 RandomizedSearchCV 系統性調參
- 理解超參數背後的物理/統計意義

### 4. 結合領域知識
- 用化工專業知識解釋模型結果
- 評估模型預測的合理性與可靠性
- 識別模型的限制與適用範圍

### 5. 完成作業練習
- 每個單元都有對應的作業 Notebook
- 透過作業鞏固所學，培養問題解決能力
- 嘗試將方法應用於自己的數據

---

## 📝 評估方式

- **單元作業**：每個單元都有對應的作業 Notebook（40%）
- **期中專題**：完成一個完整的回歸或分類專題（30%）
- **期末專題**：使用集成學習解決實際化工問題（30%）

**評估重點**：
- 演算法選擇的合理性
- 數據處理的完整性
- 模型評估的系統性
- 結果解釋的正確性
- 程式碼的可讀性與效率

---

## 🔗 相關資源

### 官方文件
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)

### 推薦閱讀
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*
- Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"

### 化工應用文獻
- 製程建模與優化相關論文
- 數據驅動的製程控制研究
- 機器學習在化工的應用案例

### 線上課程與教學
- Coursera: Andrew Ng's Machine Learning
- Fast.ai: Practical Machine Learning
- Kaggle Learn: Machine Learning Courses

---

## ⚙️ 與其他 Part 的連接

### 前置知識（來自 Part_1 和 Part_2）
- **Part_1**：Python 基礎、數據處理、探索性分析
- **Part_2**：非監督式學習（分群、降維、異常檢測）

### 後續學習（連接到 Part_4）
- **Part_4**：深度學習（DNN、CNN、RNN）
- 從傳統機器學習過渡到深度學習
- 理解何時使用傳統 ML，何時使用 DL

---

## 👥 貢獻與反饋

如有任何問題、建議或發現錯誤，歡迎透過以下方式聯繫：
- 開啟 GitHub Issue
- 提交 Pull Request
- 發送電子郵件

---

## 📄 授權

本教材遵循 [授權協議名稱] 授權。

---

**更新日期**：2026-01-27
