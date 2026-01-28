# Unit10 線性模型回歸 (Linear Models Regression)

## 📚 單元簡介

線性模型 (Linear Models) 是機器學習中最基礎且應用最廣泛的預測模型之一，在化學工程領域扮演著關鍵角色。無論是製程參數與產品品質的關係建模、反應速率的預測，還是能耗與操作條件的相關性分析，線性模型都是首選的建模工具。

線性模型假設目標變數與特徵變數之間存在線性關係，其核心優勢在於：
- **可解釋性強**：模型參數直接反映各特徵的影響程度
- **計算效率高**：訓練速度快，適合大規模資料集
- **泛化能力好**：在資料符合線性假設時，模型穩定可靠

本單元涵蓋從最基礎的普通最小二乘法 (OLS) 到各種正則化技術 (Regularization)，包括 Ridge、Lasso、ElasticNet 以及適合大規模數據的隨機梯度下降法 (SGD)。透過系統性學習這些方法，學員將能夠根據數據特性選擇最適合的線性模型，並解決過擬合、特徵選擇等實務問題。

本單元是監督式學習的基礎，為後續的非線性模型 (Unit11)、分類模型 (Unit12) 和集成學習 (Unit13) 奠定堅實的理論與實作基礎。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **理解線性模型的核心概念**：線性關係假設、損失函數 (MSE)、最小二乘法原理
2. **掌握多種正則化技術**：理解 Ridge、Lasso、ElasticNet 的數學原理與適用場景
3. **選擇合適的線性模型**：根據資料特性 (特徵相關性、樣本數、多重共線性) 做出最佳選擇
4. **實作線性回歸模型**：使用 scikit-learn 建立、訓練、評估和優化線性模型
5. **應用於化工領域**：解決產量預測、品質控制、參數優化等實際問題

---

## 📖 單元內容架構

### 1️⃣ 總覽篇：線性模型基礎 ⭐

**投影片檔案**：[Unit10_Linear_Models_Overview.pdf](Unit10_Linear_Models_Overview.pdf)
**講義檔案**：[Unit10_Linear_Models_Overview.md](Unit10_Linear_Models_Overview.md)

**內容重點**：
- 線性模型的定義與數學表示
- 線性模型的基本假設（線性關係、獨立性、同質變異性、常態分佈、無多重共線性）
- 損失函數：均方誤差 (MSE) 與最小化原理
- 正則化技術的必要性與理論基礎
- 化工領域應用場景：
  - 產量預測建模
  - 能耗分析與優化
  - 品質參數關係建模
  - 反應速率預測
- sklearn 中的五種線性模型分類與比較

**適合讀者**：所有學員，**建議最先閱讀**以建立整體概念

---

### 2️⃣ 線性回歸 (Linear Regression) ⭐

**檔案**：
- 投影片檔案：[Unit10_Linear_Regression.pdf](Unit10_Linear_Regression.pdf)
- 講義檔案：[Unit10_Linear_Regression.md](Unit10_Linear_Regression.md)
- 程式範例：[Unit10_Linear_Regression.ipynb](Unit10_Linear_Regression.ipynb)

**內容重點**：
- **演算法原理**：
  - 最小二乘法 (Ordinary Least Squares, OLS)
  - 正規方程式 (Normal Equation) 求解
  - 無正則化項，最直接的線性擬合
  
- **實作技術**：
  - scikit-learn `LinearRegression` 類別使用
  - 關鍵參數：`fit_intercept`、`n_jobs`
  - 模型訓練與預測流程
  - 係數解讀與特徵重要性分析
  
- **化工應用案例**：
  - **反應器產量預測**：根據溫度、壓力、催化劑濃度預測反應產率
  - **能耗分析**：建立操作參數與能源消耗的關係模型
  - **品質控制**：產品品質參數與製程變數的線性關係建模
  
- **演算法特性**：
  - ✅ 優點：計算速度最快、可解釋性最強、實作簡單、具有解析解
  - ❌ 缺點：無法處理多重共線性、對異常值敏感、容易過擬合（特徵數多時）

**適合場景**：特徵數量適中、特徵間無嚴重共線性、資料品質良好的情況

---

### 3️⃣ Ridge 回歸 (Ridge Regression)

**檔案**：
- 投影片檔案：[Unit10_Ridge_Regression.pdf](Unit10_Ridge_Regression.pdf)
- 講義檔案：[Unit10_Ridge_Regression.md](Unit10_Ridge_Regression.md)
- 程式範例：[Unit10_Ridge_Regression.ipynb](Unit10_Ridge_Regression.ipynb)

**內容重點**：
- **演算法原理**：
  - L2 正則化：在損失函數中加入權重平方和懲罰項
  - 損失函數：$L(\mathbf{w}) = \text{MSE} + \alpha \sum_{i=1}^{n} w_i^2$
  - 參數收縮效應：將所有權重向零收縮，但不會完全為零
  - 多重共線性問題的解決
  
- **實作技術**：
  - scikit-learn `Ridge` 類別使用
  - 關鍵超參數：`alpha` (正則化強度)、`solver` (求解器選擇)
  - 交叉驗證選擇最佳 alpha：`RidgeCV`
  - 正則化路徑視覺化
  
- **化工應用案例**：
  - **製程變數高度相關時的建模**：多個溫度、壓力感測器測量值高度相關
  - **穩健預測**：減少模型對單一特徵的過度依賴
  - **小樣本情境**：實驗數據有限時防止過擬合
  
- **演算法特性**：
  - ✅ 優點：解決多重共線性、提升模型穩定性、所有特徵都保留、適合特徵相關的情況
  - ❌ 缺點：不具特徵選擇功能、解釋性略降、需調整 alpha 參數

**適合場景**：特徵間存在多重共線性、需要所有特徵參與建模、樣本數小於特徵數

---

### 4️⃣ Lasso 回歸 (Lasso Regression)

**檔案**：
- 投影片檔案：[Unit10_Lasso_Regression.pdf](Unit10_Lasso_Regression.pdf)
- 講義檔案：[Unit10_Lasso_Regression.md](Unit10_Lasso_Regression.md)
- 程式範例：[Unit10_Lasso_Regression.ipynb](Unit10_Lasso_Regression.ipynb)

**內容重點**：
- **演算法原理**：
  - L1 正則化：在損失函數中加入權重絕對值和懲罰項
  - 損失函數：$L(\mathbf{w}) = \text{MSE} + \alpha \sum_{i=1}^{n} |w_i|$
  - 稀疏性特性：可將部分權重壓縮至零，實現自動特徵選擇
  - 內建特徵選擇機制
  
- **實作技術**：
  - scikit-learn `Lasso` 類別使用
  - 關鍵超參數：`alpha` (正則化強度)、`max_iter` (最大迭代次數)
  - 交叉驗證選擇最佳 alpha：`LassoCV`
  - 特徵選擇分析：識別非零權重特徵
  - Lasso Path 視覺化
  
- **化工應用案例**：
  - **關鍵變數識別**：從數十個製程變數中自動篩選最重要的影響因素
  - **感測器優化**：確定哪些感測器對預測最有貢獻，降低設備成本
  - **高維資料建模**：特徵數遠多於樣本數時的穩健建模
  
- **演算法特性**：
  - ✅ 優點：自動特徵選擇、提升模型可解釋性、產生稀疏模型、降低模型複雜度
  - ❌ 缺點：特徵相關時選擇不穩定、可能遺漏重要特徵、需要特徵標準化

**適合場景**：特徵數量多、希望自動選擇重要特徵、需要簡潔模型

---

### 5️⃣ ElasticNet 回歸 (ElasticNet Regression)

**檔案**：
- 投影片檔案：[Unit10_ElasticNet_Regression.pdf](Unit10_ElasticNet_Regression.pdf)
- 講義檔案：[Unit10_ElasticNet_Regression.md](Unit10_ElasticNet_Regression.md)
- 程式範例：[Unit10_ElasticNet_Regression.ipynb](Unit10_ElasticNet_Regression.ipynb)

**內容重點**：
- **演算法原理**：
  - 結合 L1 和 L2 正則化：同時具備 Ridge 和 Lasso 的優點
  - 損失函數：$L(\mathbf{w}) = \text{MSE} + \alpha \rho \sum |w_i| + \alpha (1-\rho) \sum w_i^2$
  - `l1_ratio` 參數控制 L1 與 L2 的比例
  - 在特徵相關且數量多時表現優異
  
- **實作技術**：
  - scikit-learn `ElasticNet` 類別使用
  - 關鍵超參數：`alpha` (正則化強度)、`l1_ratio` (L1/L2 混合比例)
  - 交叉驗證同時優化兩個參數：`ElasticNetCV`
  - 正則化路徑分析
  
- **化工應用案例**：
  - **複雜製程建模**：特徵多且相關的情況 (如批次製程的多個時間點測量)
  - **穩健特徵選擇**：在特徵群組相關時保留群組效應
  - **預測性維護**：從多種設備參數中選擇關鍵監測指標
  
- **演算法特性**：
  - ✅ 優點：結合 Ridge 和 Lasso 優點、處理相關特徵群組、選擇穩定、適用廣泛
  - ❌ 缺點：兩個超參數需調整、計算成本較高、相對複雜

**適合場景**：特徵多且相關、需要特徵選擇但要保持穩定性、Ridge 和 Lasso 都不理想時

---

### 6️⃣ 隨機梯度下降回歸 (SGD Regression)

**檔案**：
- 投影片檔案：[Unit10_SGD_Regression.pdf](Unit10_SGD_Regression.pdf)
- 講義檔案：[Unit10_SGD_Regression.md](Unit10_SGD_Regression.md)
- 程式範例：[Unit10_SGD_Regression.ipynb](Unit10_SGD_Regression.ipynb)

**內容重點**：
- **演算法原理**：
  - 增量學習 (Incremental Learning)：每次只使用一個樣本更新權重
  - 梯度下降優化：$\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla L$
  - 支援多種損失函數和正則化組合
  - 適合超大規模資料集
  
- **實作技術**：
  - scikit-learn `SGDRegressor` 類別使用
  - 關鍵超參數：`loss`、`penalty`、`alpha`、`learning_rate`、`eta0`
  - 線上學習與部分擬合 (`partial_fit`)
  - 學習率衰減策略
  
- **化工應用案例**：
  - **串流數據建模**：即時製程數據的線上模型更新
  - **大規模歷史資料分析**：處理數百萬筆製程歷史記錄
  - **資源受限環境**：記憶體有限時的模型訓練
  
- **演算法特性**：
  - ✅ 優點：處理超大數據、支援線上學習、記憶體占用小、訓練速度快
  - ❌ 缺點：對超參數敏感、需要特徵標準化、收斂可能不穩定、調參較困難

**適合場景**：資料量非常大、需要線上學習、計算資源受限

---

### 7️⃣ 實作練習

**檔案**：[Unit10_Linear_Models_Homework.ipynb](Unit10_Linear_Models_Homework.ipynb)

**練習內容**：
- 應用多種線性模型於實際化工數據
- 比較不同模型的預測性能 (MSE, R², MAE)
- 分析正則化參數對模型的影響
- 特徵選擇與重要性分析
- 超參數調整與模型優化
- 結果解讀與工程意義分析

---

## 📊 數據集說明

### 1. 模擬製程數據 (`data/simulated_data/`)
- 化學反應器操作數據，包含溫度、壓力、流量、催化劑濃度等變數
- 目標變數：反應產率、產品品質指標、能耗等
- 用於示範各種線性模型的訓練與評估

---

## 🎓 線性模型比較與選擇指南

| 模型 | 正則化 | 特徵選擇 | 多重共線性處理 | 計算效率 | 可解釋性 | 適用場景 |
|------|--------|---------|---------------|---------|---------|----------|
| **Linear Regression** | 無 | 無 | ❌ | 極高 | 極高 | 特徵少、無共線性、資料乾淨 |
| **Ridge** | L2 | 無 | ✅ | 高 | 高 | 特徵相關、需保留所有特徵 |
| **Lasso** | L1 | ✅ | ⚠️ | 中 | 高 | 特徵多、需自動選擇 |
| **ElasticNet** | L1+L2 | ✅ | ✅ | 中 | 高 | 特徵多且相關、穩健選擇 |
| **SGD** | 可選 | 視正則化 | 視正則化 | 極高 | 中 | 大規模數據、線上學習 |

**選擇建議**：
1. **特徵少且品質好** → Linear Regression
2. **特徵間高度相關** → Ridge Regression
3. **特徵多且需自動選擇** → Lasso Regression
4. **特徵多且相關，需穩健選擇** → ElasticNet Regression
5. **資料量極大或需線上學習** → SGD Regression
6. **不確定時的通用選擇** → ElasticNet (可退化為 Ridge 或 Lasso)

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
joblib >= 1.0.0        # 模型儲存與載入
tqdm >= 4.62.0         # 進度條顯示
```

---

## 📈 學習路徑建議

### 第一階段：基礎概念建立
1. 閱讀 
   - 投影片 [Unit10_Linear_Models_Overview.pdf](Unit10_Linear_Models_Overview.pdf)
   - 講義 [Unit10_Linear_Models_Overview.md](Unit10_Linear_Models_Overview.md)
2. 理解線性模型的數學原理與基本假設
3. 熟悉均方誤差 (MSE) 損失函數與正則化概念

### 第二階段：演算法學習與實作
1. **Linear Regression**（建議最先學習，最基礎）
   - 投影片 [Unit10_Linear_Regression.pdf](Unit10_Linear_Regression.pdf)
   - 講義 [Unit10_Linear_Regression.md](Unit10_Linear_Regression.md)
   - 執行 [Unit10_Linear_Regression.ipynb](Unit10_Linear_Regression.ipynb)
   - 重點掌握：最小二乘法、係數解讀、模型評估
   
2. **Ridge Regression**（理解 L2 正則化）
   - 投影片 [Unit10_Ridge_Regression.pdf](Unit10_Ridge_Regression.pdf)
   - 講義 [Unit10_Ridge_Regression.md](Unit10_Ridge_Regression.md)
   - 執行 [Unit10_Ridge_Regression.ipynb](Unit10_Ridge_Regression.ipynb)
   - 重點掌握：多重共線性處理、alpha 參數選擇、RidgeCV 使用
   
3. **Lasso Regression**（理解 L1 正則化與特徵選擇）
   - 投影片 [Unit10_Lasso_Regression.pdf](Unit10_Lasso_Regression.pdf)
   - 講義 [Unit10_Lasso_Regression.md](Unit10_Lasso_Regression.md)
   - 執行 [Unit10_Lasso_Regression.ipynb](Unit10_Lasso_Regression.ipynb)
   - 重點掌握：稀疏性、自動特徵選擇、Lasso Path
   
4. **ElasticNet Regression**（整合 L1 和 L2）
   - 投影片 [Unit10_ElasticNet_Regression.pdf](Unit10_ElasticNet_Regression.pdf) 
   - 講義 [Unit10_ElasticNet_Regression.md](Unit10_ElasticNet_Regression.md)
   - 執行 [Unit10_ElasticNet_Regression.ipynb](Unit10_ElasticNet_Regression.ipynb)
   - 重點掌握：l1_ratio 參數、穩健特徵選擇
   
5. **SGD Regression**（進階主題：大規模數據）
   - 投影片 [Unit10_SGD_Regression.pdf](Unit10_SGD_Regression.pdf)
   - 講義 [Unit10_SGD_Regression.md](Unit10_SGD_Regression.md)
   - 執行 [Unit10_SGD_Regression.ipynb](Unit10_SGD_Regression.ipynb)
   - 重點掌握：線上學習、partial_fit、學習率調整

### 第三階段：綜合應用與練習
1. 完成 [Unit10_Linear_Models_Homework.ipynb](Unit10_Linear_Models_Homework.ipynb)
2. 比較五種模型在相同數據集上的表現
3. 練習根據資料特性選擇最適合的模型
4. 嘗試將線性模型應用於自己的化工數據

---

## 🔍 化工領域核心應用

### 1. 產量與品質預測 ⭐
- **目標**：根據操作條件 (溫度、壓力、流量、停留時間) 預測反應產率或產品品質
- **模型建議**：
  - 若特徵少且關係明確 → Linear Regression
  - 若感測器測量值相關 → Ridge Regression
  - 若需識別關鍵操作變數 → Lasso 或 ElasticNet
- **關鍵技術**：
  - 特徵工程：交互作用項 (如溫度×壓力)
  - 模型診斷：殘差分析、多重共線性檢測
  - 係數解讀：理解各變數的影響方向與強度

### 2. 能耗分析與優化
- **目標**：建立操作參數與能源消耗的關係模型，找出節能操作點
- **模型建議**：Ridge 或 ElasticNet (操作變數通常相關)
- **關鍵技術**：
  - 標準化處理：確保不同單位的變數可比較
  - 靈敏度分析：計算各變數對能耗的偏導數
  - 最佳化應用：在約束條件下尋找最小能耗操作點

### 3. 品質參數關係建模
- **目標**：建立多個品質指標之間的數學關係 (如黏度、密度、折射率)
- **模型建議**：Linear Regression (品質參數通常線性相關)
- **關鍵技術**：
  - 軟感測器 (Soft Sensor) 設計
  - 品質推論 (Quality Inference)
  - 線上監測與警報

### 4. 感測器優化與降維
- **目標**：從數十個感測器中篩選最重要的幾個，降低設備成本
- **模型建議**：Lasso Regression (自動特徵選擇)
- **關鍵技術**：
  - 特徵重要性排序
  - 冗餘感測器識別
  - 成本效益分析

### 5. 即時製程監控
- **目標**：基於即時數據流持續更新預測模型
- **模型建議**：SGD Regression (支援線上學習)
- **關鍵技術**：
  - 串流數據處理
  - 模型增量更新
  - 概念飄移偵測

---

## 📝 評估指標總結

### 回歸模型評估指標
- **均方誤差 (MSE)**：$\frac{1}{n}\sum(y_i - \hat{y}_i)^2$，懲罰大誤差
- **均方根誤差 (RMSE)**：$\sqrt{\text{MSE}}$，與目標變數同單位，易解讀
- **平均絕對誤差 (MAE)**：$\frac{1}{n}\sum|y_i - \hat{y}_i|$，對異常值較不敏感
- **決定係數 (R²)**：$1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$，表示模型解釋變異的比例 (0-1)
- **平均絕對百分比誤差 (MAPE)**：$\frac{100\%}{n}\sum\frac{|y_i - \hat{y}_i|}{|y_i|}$，相對誤差百分比

### 化工實務評估考量
- **預測準確度**：滿足工業要求的精度 (如 ±2%)
- **模型穩定性**：不同批次資料上的表現一致性
- **計算成本**：即時系統的響應時間要求
- **可維護性**：模型更新與參數調整的難易度

---

## 🚀 進階學習方向

完成本單元後，可以進一步探索：

1. **廣義線性模型 (GLM)**：處理非常態分佈的目標變數 (如 Poisson、Gamma 分佈)
2. **混合效應模型 (Mixed Effects Models)**：處理多層次結構資料
3. **貝氏線性回歸 (Bayesian Linear Regression)**：提供預測的不確定性估計
4. **核嶺回歸 (Kernel Ridge Regression)**：線性模型的非線性擴展
5. **時間序列回歸 (Time Series Regression)**：考慮時間相依性的線性模型
6. **分散式線性回歸**：Spark MLlib 處理超大規模資料

---

## 📚 參考資源

### 教科書
1. *An Introduction to Statistical Learning* by James, Witten, Hastie, Tibshirani (第 3, 6 章)
2. *The Elements of Statistical Learning* by Hastie, Tibshirani, Friedman (第 3, 18 章)
3. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* by Aurélien Géron (第 4 章)

### 線上資源
- [scikit-learn Linear Models 官方文件](https://scikit-learn.org/stable/modules/linear_model.html)
- [Ridge and Lasso Regression: A Complete Guide with Python Scikit-Learn](https://towardsdatascience.com/)
- [Regularization Techniques in Machine Learning](https://www.analyticsvidhya.com/)

### 化工領域應用論文
- 製程優化與軟感測器設計應用
- 品質預測與控制系統開發
- 能源管理與節能分析

---

## ✍️ 課後習題提示

1. **模型比較**：在同一數據集上訓練五種模型，比較預測性能與訓練時間
2. **正則化效果**：視覺化不同 alpha 值對 Ridge/Lasso 係數的影響
3. **特徵選擇**：使用 Lasso 從 50 個特徵中自動選擇最重要的 10 個
4. **多重共線性**：建立具有強相關特徵的資料集，比較 OLS vs Ridge 的穩定性
5. **化工應用**：將學到的模型應用於實際化工數據，解讀係數的工程意義

---

## 📞 技術支援

如有疑問或需要協助，歡迎透過以下方式聯繫：
- 課程討論區
- 指導教授 Office Hour
- 課程助教

---

**課程資訊**
- 課程名稱：AI在化工上之應用
- 課程單元：Unit10 線性模型回歸 (Linear Models Regression)
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-01-28

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---
