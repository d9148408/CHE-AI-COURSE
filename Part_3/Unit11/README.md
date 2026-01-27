# Unit11 非線性模型回歸 (Non-Linear Models Regression)

## 📚 單元簡介

在 Unit10 中，我們學習了線性模型假設目標變數與特徵變數之間存在線性關係。然而，化學工程領域中的許多現象本質上是非線性的：化學反應動力學的 Arrhenius 方程、相平衡的 Antoine 方程、吸附等溫線的 Langmuir 方程、製程控制中的閥門特性曲線等，都展現出顯著的非線性特徵。

非線性模型 (Non-Linear Models) 能夠捕捉這些複雜的資料模式，包括曲線關係、交互作用、局部模式和階躍變化。雖然非線性模型的複雜度較高，但在許多實務應用中，它們能提供更準確的預測性能。

本單元涵蓋五種重要的非線性回歸方法：
- **多項式回歸 (Polynomial Regression)**：透過特徵擴展捕捉非線性關係
- **決策樹 (Decision Tree)**：基於樹狀結構的分層決策模型
- **支持向量機 (Support Vector Machine, SVM)**：透過核函數映射到高維空間
- **高斯過程回歸 (Gaussian Process Regression, GPR)**：基於貝氏推論的機率模型
- **梯度提升樹 (Gradient Boosting Trees)**：逐步修正誤差的強大集成方法

透過系統性學習這些方法，學員將能夠處理化工領域中複雜的非線性問題，為後續的分類模型 (Unit12) 和集成學習 (Unit13) 建立堅實基礎。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **理解非線性模型的核心概念**：非線性關係類型、與線性模型的差異、適用場景
2. **掌握多種非線性建模方法**：理解樹狀模型、核方法、機率模型的原理與特性
3. **選擇合適的非線性模型**：根據資料特性（樣本數、特徵數、雜訊程度）和應用需求做出最佳選擇
4. **實作非線性回歸模型**：使用 scikit-learn 建立、訓練、評估和優化非線性模型
5. **應用於化工領域**：解決反應動力學建模、製程曲線擬合、複雜系統預測等實際問題

---

## 📖 單元內容架構

### 1️⃣ 總覽篇：非線性模型基礎 ⭐

**檔案**：[Unit11_NonLinear_Models_Overview.md](Unit11_NonLinear_Models_Overview.md)

**內容重點**：
- 非線性模型的定義與數學表示
- 為什麼需要非線性模型：化工領域的非線性現象
  - 化學反應動力學 (Arrhenius 方程)
  - 相平衡 (Antoine 方程)
  - 吸附等溫線 (Langmuir 等溫式)
  - 製程控制 (閥門特性曲線)
- 非線性模型 vs. 線性模型比較
- 非線性模型分類：參數擴展型、樹狀模型、核方法與機率模型
- sklearn 中的五種非線性模型介紹
- 化工領域應用場景：反應動力學、相平衡預測、製程優化、品質預測

**適合讀者**：所有學員，**建議最先閱讀**以建立整體概念

---

### 2️⃣ 多項式回歸 (Polynomial Regression) ⭐

**檔案**：
- 講義：[Unit11_Polynomial_Regression.md](Unit11_Polynomial_Regression.md)
- 程式範例：[Unit11_Polynomial_Regression.ipynb](Unit11_Polynomial_Regression.ipynb)

**內容重點**：
- **演算法原理**：
  - 透過 `PolynomialFeatures` 將特徵擴展為高次項和交互項
  - 本質上仍是線性模型，但擬合非線性關係
  - 多項式次數選擇的權衡：欠擬合 vs. 過擬合
  - 交互項的物理意義
  
- **實作技術**：
  - scikit-learn `PolynomialFeatures` + `LinearRegression` / `Ridge` 組合
  - 關鍵參數：`degree` (多項式次數)、`interaction_only`、`include_bias`
  - Pipeline 整合特徵轉換與模型訓練
  - 特徵標準化的重要性
  
- **化工應用案例**：
  - **反應速率建模**：溫度、壓力對反應速率的非線性影響
  - **最佳化操作點尋找**：產率曲線的二次建模與最大值求解
  - **熱力學性質關聯**：溫度-密度、溫度-黏度等曲線關係
  
- **演算法特性**：
  - ✅ 優點：簡單易懂、可解釋性佳、適合平滑曲線、可控制複雜度
  - ❌ 缺點：高次項易過擬合、特徵數爆炸性增長、外插能力差、對異常值敏感

**適合場景**：資料呈現平滑曲線、特徵數少、需要可解釋性、有明確的物理關係

---

### 3️⃣ 決策樹回歸 (Decision Tree Regression)

**檔案**：
- 講義：[Unit11_Decision_Tree.md](Unit11_Decision_Tree.md)
- 程式範例：[Unit11_Decision_Tree.ipynb](Unit11_Decision_Tree.ipynb)

**內容重點**：
- **演算法原理**：
  - 基於樹狀結構的遞迴分割 (Recursive Partitioning)
  - 分裂準則：最小化 MSE (均方誤差)
  - 葉節點預測：區域內樣本的平均值
  - 樹的生長與剪枝策略
  
- **實作技術**：
  - scikit-learn `DecisionTreeRegressor` 類別使用
  - 關鍵超參數：`max_depth`、`min_samples_split`、`min_samples_leaf`、`max_features`
  - 樹結構視覺化與解讀
  - 特徵重要性分析
  - 避免過擬合的剪枝技術
  
- **化工應用案例**：
  - **操作模式規則萃取**：自動發現「若溫度>200°C 且壓力<5bar 則...」的決策規則
  - **分段線性建模**：不同操作區間有不同的線性關係
  - **異常檢測與診斷**：識別異常操作條件的路徑
  
- **演算法特性**：
  - ✅ 優點：高度可解釋、自動特徵選擇、處理非線性和交互作用、無需標準化、可處理缺失值
  - ❌ 缺點：容易過擬合、對資料變化敏感、預測不平滑、外推能力差

**適合場景**：需要規則解釋、資料有明顯分段特徵、特徵重要性分析

---

### 4️⃣ 支持向量機回歸 (Support Vector Machine Regression)

**檔案**：
- 講義：[Unit11_Support_Vector_Machine.md](Unit11_Support_Vector_Machine.md)
- 程式範例：[Unit11_Support_Vector_Machine.ipynb](Unit11_Support_Vector_Machine.ipynb)

**內容重點**：
- **演算法原理**：
  - ε-insensitive 損失函數：容許 ±ε 範圍內的預測誤差
  - 核技巧 (Kernel Trick)：將資料映射到高維空間
  - 支持向量 (Support Vectors)：影響模型的關鍵樣本
  - 常用核函數：Linear、RBF (Radial Basis Function)、Polynomial
  
- **實作技術**：
  - scikit-learn `SVR` 類別使用
  - 關鍵超參數：`kernel`、`C` (懲罰參數)、`epsilon`、`gamma` (RBF 核的寬度)
  - 核函數選擇與參數調整
  - 特徵標準化的必要性
  
- **化工應用案例**：
  - **小樣本高維建模**：實驗數據有限但需高精度預測
  - **穩健回歸**：對異常值不敏感的建模需求
  - **複雜非線性關係**：無明確數學形式的系統建模
  
- **演算法特性**：
  - ✅ 優點：小樣本表現優異、對異常值穩健、理論基礎完善、泛化能力強
  - ❌ 缺點：計算成本高、超參數調整困難、缺乏機率輸出、黑箱模型

**適合場景**：小樣本、高維資料、需要穩健預測、複雜非線性關係

---

### 5️⃣ 高斯過程回歸 (Gaussian Process Regression) ⭐

**檔案**：
- 講義：[Unit11_Gaussian_Process_Regression.md](Unit11_Gaussian_Process_Regression.md)
- 程式範例：[Unit11_Gaussian_Process_Regression.ipynb](Unit11_Gaussian_Process_Regression.ipynb)

**內容重點**：
- **演算法原理**：
  - 非參數貝氏模型：視函數為無限維高斯分佈
  - 核函數定義特徵相似度 (Covariance Structure)
  - 提供預測均值與不確定性估計 (標準差)
  - 貝氏推論框架：先驗 + 資料 → 後驗分佈
  
- **實作技術**：
  - scikit-learn `GaussianProcessRegressor` 類別使用
  - 關鍵參數：`kernel` (RBF、Matérn、RationalQuadratic 等)、`alpha` (噪音水平)
  - 核函數設計與組合
  - 不確定性量化與信賴區間
  - 超參數自動優化 (Maximum Likelihood Estimation)
  
- **化工應用案例**：
  - **實驗設計最佳化**：貝氏最佳化 (Bayesian Optimization) 的核心
  - **不確定性量化**：提供預測的信賴區間，支援風險決策
  - **小樣本建模**：充分利用有限的實驗資料
  - **主動學習**：識別最有價值的下一個實驗點
  
- **演算法特性**：
  - ✅ 優點：提供不確定性估計、小樣本表現優異、靈活的核函數、自動超參數優化
  - ❌ 缺點：計算成本極高 (O(n³))、大數據不適用、核函數選擇需領域知識

**適合場景**：小樣本、需要不確定性估計、實驗設計、貝氏最佳化

---

### 6️⃣ 梯度提升樹回歸 (Gradient Boosting Trees Regression)

**檔案**：
- 講義：[Unit11_Gradient_Boosting_Trees.md](Unit11_Gradient_Boosting_Trees.md)
- 程式範例：[Unit11_Gradient_Boosting_Trees.ipynb](Unit11_Gradient_Boosting_Trees.ipynb)

**內容重點**：
- **演算法原理**：
  - Boosting 思想：序列訓練多棵樹，每棵樹修正前一棵的殘差
  - 梯度下降在函數空間的應用
  - 學習率控制收斂速度
  - Shrinkage 策略防止過擬合
  
- **實作技術**：
  - scikit-learn `GradientBoostingRegressor` 類別使用
  - 關鍵超參數：`n_estimators`、`learning_rate`、`max_depth`、`subsample`
  - Early Stopping 機制
  - 特徵重要性分析
  - 學習曲線監控
  
- **化工應用案例**：
  - **高精度預測需求**：產品品質的精準預測
  - **複雜多變數系統**：多個操作變數的綜合影響
  - **特徵交互作用**：自動捕捉變數間的複雜互動
  
- **演算法特性**：
  - ✅ 優點：預測精度極高、自動特徵選擇、處理缺失值、捕捉複雜交互作用
  - ❌ 缺點：訓練時間長、容易過擬合、超參數多、可解釋性低

**適合場景**：追求最高預測精度、資料充足、願意投入調參時間

---

### 7️⃣ 實作練習

**檔案**：[Unit11_NonLinear_Models_Homework.ipynb](Unit11_NonLinear_Models_Homework.ipynb)

**練習內容**：
- 應用五種非線性模型於實際化工數據
- 比較線性模型 (Unit10) 與非線性模型的性能差異
- 分析過擬合與欠擬合現象
- 超參數調整與模型優化
- 特徵重要性與模型可解釋性分析
- 不確定性量化 (GPR)
- 結果解讀與工程意義分析

---

## 📊 數據集說明

### 1. 決策樹專用數據 (`data/decision_tree/`)
- 包含分段特徵的製程數據
- 用於示範決策樹的規則萃取能力

### 2. 模擬製程數據 (`data/simulated_data/`)
- 具有非線性關係的化學反應器數據
- 用於比較五種非線性模型的性能

---

## 🎓 非線性模型比較與選擇指南

| 模型 | 複雜度 | 可解釋性 | 小樣本 | 大數據 | 不確定性估計 | 計算成本 | 適用場景 |
|------|--------|---------|--------|--------|-------------|---------|----------|
| **Polynomial Regression** | 低-中 | 高 | ⚠️ | ✅ | ❌ | 低 | 平滑曲線、物理關係明確 |
| **Decision Tree** | 中 | 極高 | ⚠️ | ✅ | ❌ | 低 | 規則萃取、分段關係 |
| **SVM** | 高 | 低 | ✅ | ❌ | ❌ | 高 | 小樣本、穩健預測 |
| **GPR** | 高 | 中 | ✅ | ❌ | ✅ | 極高 | 實驗設計、不確定性量化 |
| **Gradient Boosting** | 極高 | 低 | ⚠️ | ✅ | ❌ | 高 | 追求最高精度 |

**選擇建議**：
1. **資料呈現平滑曲線且物理關係明確** → Polynomial Regression
2. **需要可解釋的決策規則** → Decision Tree
3. **小樣本且需要穩健預測** → SVM 或 GPR
4. **需要不確定性估計（如實驗設計）** → GPR
5. **追求最高預測精度且資料充足** → Gradient Boosting Trees
6. **大數據且時間受限** → Polynomial Regression 或 Decision Tree
7. **不確定時的通用選擇** → Gradient Boosting Trees (通常表現最佳)

**線性 vs. 非線性選擇原則**：
- 先嘗試線性模型 (Unit10)，建立 baseline 性能
- 若殘差分析顯示明顯的非線性模式，再使用非線性模型
- 考慮可解釋性需求：線性模型 > 多項式回歸 > 決策樹 > SVM/GBT/GPR
- 考慮計算資源：線性模型最快，GPR 最慢

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
graphviz >= 0.16        # 決策樹視覺化
joblib >= 1.0.0         # 模型儲存與載入
tqdm >= 4.62.0          # 進度條顯示
```

---

## 📈 學習路徑建議

### 第一階段：基礎概念建立
1. 閱讀 [Unit11_NonLinear_Models_Overview.md](Unit11_NonLinear_Models_Overview.md)
2. 理解非線性模型的必要性與分類
3. 回顧 Unit10 線性模型的限制

### 第二階段：演算法學習與實作
1. **Polynomial Regression**（建議最先學習，最接近線性模型）
   - 閱讀講義 [Unit11_Polynomial_Regression.md](Unit11_Polynomial_Regression.md)
   - 執行 [Unit11_Polynomial_Regression.ipynb](Unit11_Polynomial_Regression.ipynb)
   - 重點掌握：特徵擴展、多項式次數選擇、過擬合控制
   
2. **Decision Tree**（學習非參數模型的概念）
   - 閱讀講義 [Unit11_Decision_Tree.md](Unit11_Decision_Tree.md)
   - 執行 [Unit11_Decision_Tree.ipynb](Unit11_Decision_Tree.ipynb)
   - 重點掌握：樹結構、分裂準則、剪枝策略、規則解讀
   
3. **Gradient Boosting Trees**（最實用的高精度模型）
   - 閱讀講義 [Unit11_Gradient_Boosting_Trees.md](Unit11_Gradient_Boosting_Trees.md)
   - 執行 [Unit11_Gradient_Boosting_Trees.ipynb](Unit11_Gradient_Boosting_Trees.ipynb)
   - 重點掌握：Boosting 原理、超參數調整、Early Stopping
   
4. **Support Vector Machine**（進階：核方法）
   - 閱讀講義 [Unit11_Support_Vector_Machine.md](Unit11_Support_Vector_Machine.md)
   - 執行 [Unit11_Support_Vector_Machine.ipynb](Unit11_Support_Vector_Machine.ipynb)
   - 重點掌握：核函數、C 和 gamma 參數、特徵標準化
   
5. **Gaussian Process Regression**（進階：機率模型與不確定性）
   - 閱讀講義 [Unit11_Gaussian_Process_Regression.md](Unit11_Gaussian_Process_Regression.md)
   - 執行 [Unit11_Gaussian_Process_Regression.ipynb](Unit11_Gaussian_Process_Regression.ipynb)
   - 重點掌握：貝氏推論、核函數設計、不確定性量化

### 第三階段：綜合應用與練習
1. 完成 [Unit11_NonLinear_Models_Homework.ipynb](Unit11_NonLinear_Models_Homework.ipynb)
2. 比較線性模型與五種非線性模型的性能
3. 分析各模型的優缺點與適用場景
4. 練習根據資料特性選擇最適合的模型
5. 嘗試將非線性模型應用於自己的化工數據

---

## 🔍 化工領域核心應用

### 1. 反應動力學建模 ⭐
- **目標**：建立反應速率與溫度、壓力、催化劑濃度的非線性關係
- **模型建議**：
  - 已知 Arrhenius 形式 → Polynomial Regression (對數轉換後線性化)
  - 複雜反應網絡 → Gradient Boosting Trees
  - 小樣本實驗數據 → GPR
- **關鍵技術**：
  - 物理約束整合（反應速率為正）
  - 參數可解釋性（活化能、頻率因子）
  - 外推能力評估

### 2. 相平衡與熱力學性質預測
- **目標**：預測蒸氣壓、溶解度、分配係數等熱力學性質
- **模型建議**：Polynomial Regression 或 GPR（配合物理知識設計核函數）
- **關鍵技術**：
  - 溫度、壓力範圍的外推
  - 相轉變點的捕捉
  - 多組分系統建模

### 3. 製程曲線擬合與優化
- **目標**：擬合產率-溫度曲線、轉化率-停留時間曲線，找出最佳操作點
- **模型建議**：Polynomial Regression（二次或三次）+ 數值優化
- **關鍵技術**：
  - 曲線平滑性約束
  - 極值點求解
  - 靈敏度分析

### 4. 複雜系統品質預測
- **目標**：多變數製程的產品品質精準預測
- **模型建議**：Gradient Boosting Trees 或 SVM
- **關鍵技術**：
  - 特徵交互作用捕捉
  - 特徵重要性排序
  - 模型可解釋性工具（SHAP、LIME）

### 5. 實驗設計與貝氏最佳化 ⭐
- **目標**：在有限實驗預算下，快速找到最佳實驗條件
- **模型建議**：GPR（提供不確定性估計，支援 Acquisition Function）
- **關鍵技術**：
  - Expected Improvement (EI) 準則
  - 探索 vs. 利用權衡
  - 序列實驗設計

---

## 📝 評估指標總結

### 回歸模型評估指標（與 Unit10 相同）
- **均方誤差 (MSE)**：$\frac{1}{n}\sum(y_i - \hat{y}_i)^2$
- **均方根誤差 (RMSE)**：$\sqrt{\text{MSE}}$
- **平均絕對誤差 (MAE)**：$\frac{1}{n}\sum|y_i - \hat{y}_i|$
- **決定係數 (R²)**：$1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$
- **平均絕對百分比誤差 (MAPE)**：$\frac{100\%}{n}\sum\frac{|y_i - \hat{y}_i|}{|y_i|}$

### 非線性模型特殊評估
- **外推能力**：測試集超出訓練範圍時的表現
- **過擬合診斷**：訓練集與測試集性能差距
- **不確定性校準**：GPR 的預測區間覆蓋率
- **特徵重要性一致性**：不同隨機種子下的穩定性

---

## 🚀 進階學習方向

完成本單元後，可以進一步探索：

1. **深度學習回歸**：多層神經網絡 (MLP)、卷積神經網絡 (CNN)
2. **集成學習進階**：Random Forest、XGBoost、LightGBM、CatBoost (Unit13)
3. **貝氏最佳化實務**：實驗設計自動化、超參數調整
4. **混合物理-數據驅動模型**：結合機理知識與機器學習
5. **時間序列非線性模型**：LSTM、GRU、Temporal Convolutional Networks
6. **因果推論**：超越相關性的因果關係建模

---

## 📚 參考資源

### 教科書
1. *An Introduction to Statistical Learning* by James, Witten, Hastie, Tibshirani (第 7, 8, 9 章)
2. *The Elements of Statistical Learning* by Hastie, Tibshirani, Friedman (第 9, 10, 14 章)
3. *Gaussian Processes for Machine Learning* by Rasmussen & Williams

### 線上資源
- [scikit-learn Supervised Learning 官方文件](https://scikit-learn.org/stable/supervised_learning.html)
- [Polynomial Regression and Overfitting](https://towardsdatascience.com/)
- [A Visual Exploration of Gaussian Processes](https://distill.pub/)

### 化工領域應用論文
- 反應動力學參數估計與模型選擇
- 熱力學性質預測的機器學習方法
- 貝氏最佳化在化工實驗設計中的應用

---

## ✍️ 課後習題提示

1. **模型比較**：在同一數據集上比較五種非線性模型與 Unit10 線性模型的性能
2. **過擬合診斷**：使用學習曲線分析多項式回歸的過擬合現象
3. **規則萃取**：從決策樹中提取製程操作規則並驗證合理性
4. **不確定性分析**：使用 GPR 預測並繪製 95% 信賴區間
5. **化工應用**：將梯度提升樹應用於實際化工數據，分析特徵重要性

---

## 📞 技術支援

如有疑問或需要協助，歡迎透過以下方式聯繫：
- 課程討論區
- 指導教授 Office Hour
- 課程助教

---

**版本資訊**：Unit11 v1.0 | 最後更新：2026-01-27
