# Unit10 線性模型回歸總覽 | Linear Models Regression Overview

---

## 課程目標

本單元將深入介紹線性模型回歸 (Linear Regression Models) 的理論基礎與實務應用，專注於 scikit-learn 模組提供的各種線性回歸方法。學生將學習：

- 理解線性模型的數學原理與假設條件
- 掌握不同正則化技術 (Regularization) 的原理與應用
- 學習使用 sklearn 建立、訓練與評估線性模型
- 了解線性模型在化工領域的實際應用案例

---

## 1. 線性模型基礎理論

### 1.1 什麼是線性模型？

線性模型 (Linear Model) 是機器學習中最基礎且應用最廣泛的預測模型之一。其核心概念是假設目標變數 (Target Variable) 與特徵變數 (Features) 之間存在**線性關係**。

對於具有 $n$ 個特徵的資料集，線性模型的預測函數可表示為：

$$
\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_n x_n = w_0 + \sum_{i=1}^{n} w_i x_i
$$

或以向量形式表示：

$$
\hat{y} = \mathbf{w}^T \mathbf{x} + b
$$

其中：
- $\hat{y}$ : 預測值 (Predicted value)
- $\mathbf{x} = [x_1, x_2, \ldots, x_n]^T$ : 特徵向量 (Feature vector)
- $\mathbf{w} = [w_1, w_2, \ldots, w_n]^T$ : 權重向量 (Weight vector)
- $b$ (或 $w_0$ ): 偏置項 (Bias term / Intercept)

### 1.2 線性模型的優點

線性模型在實務應用中具有以下優勢：

1. **可解釋性強**：模型參數 (權重) 直接反映各特徵對預測結果的影響程度
2. **計算效率高**：訓練速度快，適合大規模資料集
3. **泛化能力好**：在資料符合線性假設時，模型不易過擬合
4. **穩定性高**：對雜訊和異常值有一定的容忍度 (特別是加入正則化後)
5. **理論基礎完善**：統計學理論支撐，具有嚴謹的數學證明

### 1.3 線性模型的基本假設

使用線性模型時，需要資料滿足以下基本假設：

1. **線性關係** (Linearity)：特徵與目標變數之間存在線性關係
2. **獨立性** (Independence)：觀測值之間互相獨立
3. **同質變異性** (Homoscedasticity)：殘差的變異數為常數
4. **常態分佈** (Normality)：殘差服從常態分佈
5. **無多重共線性** (No Multicollinearity)：特徵之間不存在高度相關性

### 1.4 損失函數：均方誤差 (MSE)

線性回歸模型的訓練目標是最小化預測值與實際值之間的誤差。最常用的損失函數是**均方誤差 (Mean Squared Error, MSE)**：

$$
\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2 = \frac{1}{m} \sum_{i=1}^{m} (y_i - \mathbf{w}^T \mathbf{x}_i - b)^2
$$

其中 $m$ 是訓練樣本數。

最小化 MSE 的目標可以寫成：

$$
\min_{\mathbf{w}, b} \frac{1}{m} \sum_{i=1}^{m} (y_i - \mathbf{w}^T \mathbf{x}_i - b)^2
$$

---

## 2. sklearn 中的線性模型方法

scikit-learn 提供了多種線性回歸模型，主要差異在於**正則化技術 (Regularization)** 的不同。正則化是防止模型過擬合的重要手段，通過在損失函數中加入懲罰項來限制模型複雜度。

### 2.1 線性回歸 (Linear Regression)

**模型**：`sklearn.linear_model.LinearRegression`

**特點**：
- 最基本的線性回歸模型，無正則化項
- 使用最小二乘法 (Ordinary Least Squares, OLS) 求解
- 適用於特徵數量較少、多重共線性不嚴重的情況

**損失函數**：

$$
L(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} (y_i - \mathbf{w}^T \mathbf{x}_i)^2
$$

**主要參數**：
- `fit_intercept`: 是否計算截距項 (預設 True)
- `normalize`: 是否對特徵進行標準化 (已棄用，建議使用 Pipeline)
- `n_jobs`: 並行計算的工作數 (-1 表示使用所有處理器)

### 2.2 嶺回歸 (Ridge Regression)

**模型**：`sklearn.linear_model.Ridge`

**特點**：
- 加入 **L2 正則化**項 (權重平方和的懲罰)
- 可有效處理多重共線性問題
- 不會將權重縮減為零，所有特徵都會保留

**損失函數**：

$$
L(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \alpha \sum_{j=1}^{n} w_j^2
$$

其中 $\alpha$ 是正則化強度參數：
- $\alpha = 0$ : 退化為普通線性回歸
- $\alpha$ 越大：權重被壓縮得越小，模型越簡單

**主要參數**：
- `alpha`: 正則化強度 (預設 1.0)
- `fit_intercept`: 是否計算截距項
- `solver`: 求解器選擇 ('auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga')

### 2.3 Lasso 回歸 (Lasso Regression)

**模型**：`sklearn.linear_model.Lasso`

**特點**：
- 加入 **L1 正則化**項 (權重絕對值和的懲罰)
- 可將不重要特徵的權重壓縮為**零**，具有**特徵選擇**功能
- 適用於高維度資料且只有少數特徵重要的情況

**損失函數**：

$$
L(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \alpha \sum_{j=1}^{n} |w_j|
$$

**主要參數**：
- `alpha`: 正則化強度 (預設 1.0)
- `fit_intercept`: 是否計算截距項
- `max_iter`: 最大迭代次數 (預設 1000)
- `selection`: 特徵更新策略 ('cyclic' 或 'random')

### 2.4 彈性網路回歸 (Elastic Net Regression)

**模型**：`sklearn.linear_model.ElasticNet`

**特點**：
- 結合 **L1 和 L2 正則化**的優點
- 可以同時進行特徵選擇和處理多重共線性
- 適用於特徵數量遠大於樣本數的情況

**損失函數**：

$$
L(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \alpha \rho \sum_{j=1}^{n} |w_j| + \frac{\alpha (1-\rho)}{2} \sum_{j=1}^{n} w_j^2
$$

其中：
- $\alpha$ : 正則化強度
- $\rho$ : L1 和 L2 的混合比例 (0 ≤ $\rho$ ≤ 1)
  - $\rho = 0$ : 純 L2 (Ridge)
  - $\rho = 1$ : 純 L1 (Lasso)

**主要參數**：
- `alpha`: 正則化強度 (預設 1.0)
- `l1_ratio`: L1 懲罰的比例 (預設 0.5，對應 $\rho$ )
- `fit_intercept`: 是否計算截距項
- `max_iter`: 最大迭代次數

### 2.5 隨機梯度下降回歸 (SGD Regressor)

**模型**：`sklearn.linear_model.SGDRegressor`

**特點**：
- 使用**隨機梯度下降 (Stochastic Gradient Descent)** 演算法求解
- 適合**大規模資料集**和線上學習 (Online Learning)
- 可選擇不同的損失函數和正則化方式
- 訓練速度快，但可能需要調整學習率等參數

**損失函數 (以平方損失為例)**：

$$
L(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \alpha \cdot \text{Penalty}
$$

其中 Penalty 可以是：
- `'l2'`: Ridge 正則化
- `'l1'`: Lasso 正則化
- `'elasticnet'`: Elastic Net 正則化

**主要參數**：
- `loss`: 損失函數類型 ('squared_error', 'huber', 'epsilon_insensitive' 等)
- `penalty`: 正則化類型 ('l2', 'l1', 'elasticnet', None)
- `alpha`: 正則化強度 (預設 0.0001)
- `learning_rate`: 學習率策略 ('constant', 'optimal', 'invscaling', 'adaptive')
- `max_iter`: 最大迭代次數 (預設 1000)
- `eta0`: 初始學習率

### 2.6 模型比較總結

| 模型 | 正則化 | 特徵選擇 | 適用場景 | 計算複雜度 |
|------|--------|----------|----------|------------|
| Linear Regression | 無 | 否 | 簡單線性問題 | 低 |
| Ridge | L2 | 否 | 多重共線性 | 低 |
| Lasso | L1 | **是** | 高維稀疏資料 | 中 |
| Elastic Net | L1 + L2 | **是** | 高維+共線性 | 中 |
| SGD Regressor | 可選 | 依設定 | **大規模資料** | 低 |

---

## 3. 資料前處理技術

### 3.1 為什麼需要資料前處理？

在使用線性模型之前，適當的資料前處理可以：
1. **提升模型性能**：統一特徵量級，加速模型收斂
2. **改善數值穩定性**：避免數值計算誤差
3. **滿足模型假設**：確保模型基本假設成立

### 3.2 特徵縮放 (Feature Scaling)

#### 3.2.1 標準化 (Standardization)

**目的**：將特徵轉換為均值為 0、標準差為 1 的分佈

**公式**：

$$
x_{\text{scaled}} = \frac{x - \mu}{\sigma}
$$

其中 $\mu$ 是均值， $\sigma$ 是標準差。

**sklearn 實現**：
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**適用情況**：
- 特徵服從或近似常態分佈
- 使用正則化的線性模型 (Ridge, Lasso, Elastic Net)
- 使用梯度下降演算法 (SGD)

#### 3.2.2 正規化 (Normalization / Min-Max Scaling)

**目的**：將特徵縮放到指定範圍 (通常是 [0, 1])

**公式**：

$$
x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

**sklearn 實現**：
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

**適用情況**：
- 特徵不服從常態分佈
- 需要保留零值
- 神經網路模型

### 3.3 類別變數編碼 (Categorical Encoding)

#### 3.3.1 獨熱編碼 (One-Hot Encoding)

**目的**：將類別變數轉換為二進位向量

**sklearn 實現**：
```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X_categorical)
```

**範例**：
```
原始類別: ['A', 'B', 'C', 'A']
編碼結果:
[[1, 0, 0],
 [0, 1, 0],
 [0, 0, 1],
 [1, 0, 0]]
```

#### 3.3.2 標籤編碼 (Label Encoding)

**目的**：將類別轉換為整數

**sklearn 實現**：
```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
```

**注意**：標籤編碼會引入順序關係，對於無順序的類別變數應使用獨熱編碼。

### 3.4 處理缺失值

**常見方法**：

1. **刪除缺失值**：
```python
df.dropna()  # 刪除含缺失值的行
```

2. **均值/中位數填補**：
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
X_imputed = imputer.fit_transform(X)
```

3. **前向/後向填補** (時間序列資料)：
```python
df.fillna(method='ffill')  # 前向填補
df.fillna(method='bfill')  # 後向填補
```

### 3.5 Pipeline 整合

**建議使用 Pipeline 整合前處理步驟**：

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

**優點**：
- 避免資料洩漏 (Data Leakage)
- 程式碼簡潔易維護
- 便於參數調整和交叉驗證

---

## 4. 模型評估方法

### 4.1 評估指標 (Evaluation Metrics)

#### 4.1.1 均方誤差 (Mean Squared Error, MSE)

$$
\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
$$

**sklearn 實現**：
```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)
```

#### 4.1.2 均方根誤差 (Root Mean Squared Error, RMSE)

$$
\text{RMSE} = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2}
$$

**sklearn 實現**：
```python
import numpy as np

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

#### 4.1.3 平均絕對誤差 (Mean Absolute Error, MAE)

$$
\text{MAE} = \frac{1}{m} \sum_{i=1}^{m} |y_i - \hat{y}_i|
$$

**sklearn 實現**：
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
```

#### 4.1.4 決定係數 (R-squared, $R^2$ )

$$
R^2 = 1 - \frac{\sum_{i=1}^{m} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{m} (y_i - \bar{y})^2}
$$

其中 $\bar{y}$ 是目標變數的平均值。

- $R^2$ 範圍：$(-\infty, 1]$
- $R^2 = 1$ : 完美預測
- $R^2 = 0$ : 模型預測能力等於使用平均值
- $R^2 < 0$ : 模型表現比平均值還差

**sklearn 實現**：
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
```

### 4.2 交叉驗證 (Cross-Validation)

**目的**：更穩健地評估模型性能，避免過擬合

#### 4.2.1 K 折交叉驗證 (K-Fold CV)

將資料分為 $K$ 個子集，每次用 $K-1$ 個子集訓練，1 個子集驗證，重複 $K$ 次。

**sklearn 實現**：
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"R² scores: {scores}")
print(f"Mean R²: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

#### 4.2.2 留一法交叉驗證 (Leave-One-Out CV, LOOCV)

每次只用一個樣本作為驗證集，其餘作為訓練集。

**sklearn 實現**：
```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring='r2')
```

### 4.3 超參數調整 (Hyperparameter Tuning)

#### 4.3.1 網格搜尋 (Grid Search)

遍歷所有參數組合，找出最佳參數。

**sklearn 實現**：
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best R² score: {grid_search.best_score_:.3f}")
```

#### 4.3.2 隨機搜尋 (Random Search)

隨機採樣參數組合，計算效率更高。

**sklearn 實現**：
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

param_dist = {
    'alpha': uniform(loc=0.001, scale=100)
}

random_search = RandomizedSearchCV(
    Ridge(), 
    param_dist, 
    n_iter=100, 
    cv=5, 
    scoring='r2',
    random_state=42
)
random_search.fit(X_train, y_train)
```

---

## 5. 線性模型在化工領域的應用

### 5.1 應用場景

線性模型在化工領域有廣泛的應用，包括：

#### 5.1.1 製程參數預測
- **目標**：預測反應器溫度、壓力、轉化率等關鍵製程變數
- **特徵**：原料組成、操作條件、催化劑性質
- **範例**：預測化學反應產率

#### 5.1.2 品質控制 (Quality Control)
- **目標**：預測產品品質指標 (純度、黏度、強度等)
- **特徵**：製程參數、原料批次、環境條件
- **範例**：預測聚合物分子量分布

#### 5.1.3 能源消耗預測
- **目標**：預測蒸餾塔、反應器等單元的能源需求
- **特徵**：進料流量、操作溫度、回流比
- **範例**：預測蒸餾塔能耗

#### 5.1.4 設備健康監測
- **目標**：預測設備異常、剩餘壽命
- **特徵**：振動、溫度、壓力等感測器資料
- **範例**：預測泵浦軸承壽命

#### 5.1.5 物性估算 (Property Estimation)
- **目標**：預測物理化學性質 (沸點、密度、溶解度等)
- **特徵**：分子結構描述符、溫度、壓力
- **範例**：QSPR (Quantitative Structure-Property Relationship) 模型

### 5.2 實際應用案例

#### 案例 1：催化裂解反應產率預測

**背景**：某煉油廠希望根據進料組成和操作條件預測催化裂解反應的汽油產率。

**資料特徵**：
- 進料密度、黏度、芳香烴含量
- 反應溫度、壓力、空速
- 催化劑活性指標

**模型選擇**：Ridge Regression (處理特徵間相關性)

**結果**： $R^2 = 0.92$ ，可有效指導操作優化

#### 案例 2：聚合物性質預測

**背景**：預測聚合物的玻璃轉化溫度 (Tg)。

**資料特徵**：
- 單體結構描述符
- 聚合條件 (溫度、壓力、時間)
- 分子量、分子量分布

**模型選擇**：Lasso Regression (特徵選擇，找出關鍵結構因子)

**結果**：識別出 15 個關鍵結構描述符， $R^2 = 0.88$

#### 案例 3：蒸餾塔操作優化

**背景**：建立蒸餾塔操作參數與產品純度的關係模型。

**資料特徵**：
- 進料流量、組成
- 回流比、加熱功率
- 塔板數、壓力

**模型選擇**：Elastic Net (平衡特徵選擇與多重共線性)

**結果**：模型 RMSE < 0.5%，成功應用於自動控制系統

### 5.3 線性模型的限制與進階方法

雖然線性模型功能強大，但在以下情況下可能表現不佳：

1. **非線性關係顯著**：考慮使用多項式回歸、決策樹、神經網路
2. **複雜交互作用**：考慮加入交互項或使用非線性模型
3. **時間序列資料**：考慮使用 ARIMA、LSTM 等時序模型
4. **高維度稀疏資料**：考慮使用深度學習模型

---

## 6. 學習資源與延伸閱讀

### 6.1 官方文檔

- [scikit-learn Linear Models 官方文檔](https://scikit-learn.org/stable/modules/linear_model.html)
- [scikit-learn User Guide - Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html)

### 6.2 延伸學習主題

1. **進階正則化技術**：
   - Group Lasso
   - Sparse Group Lasso
   - Fused Lasso

2. **穩健回歸 (Robust Regression)**：
   - Huber Regression
   - RANSAC Regression
   - Theil-Sen Estimator

3. **貝氏線性回歸 (Bayesian Linear Regression)**：
   - Bayesian Ridge
   - Automatic Relevance Determination (ARD)

4. **廣義線性模型 (Generalized Linear Models, GLM)**：
   - Poisson Regression
   - Gamma Regression
   - Tweedie Regression

---

## 7. 本單元學習路徑

完成本單元後，您將依序學習以下主題：

1. **Unit10_Linear_Regression**：普通線性回歸的詳細理論與實作
2. **Unit10_Ridge_Regression**：L2 正則化的原理與應用
3. **Unit10_Lasso_Regression**：L1 正則化與特徵選擇
4. **Unit10_ElasticNet_Regression**：混合正則化策略
5. **Unit10_SGD_Regression**：大規模資料的梯度下降方法
6. **Unit10_Linear_Models_Homework**：綜合練習與模型比較

每個子主題都包含詳細的數學推導、程式碼實作和化工領域的應用案例，請按順序學習以獲得最佳效果。

---

## 8. 課前準備

在開始本單元之前，請確保您已經：

1. ✅ 安裝 Python 3.8 以上版本
2. ✅ 安裝必要套件：
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn scipy
   ```
3. ✅ 複習基礎數學知識：
   - 線性代數 (向量、矩陣運算)
   - 微積分 (偏微分、梯度)
   - 統計學 (均值、變異數、相關係數)
4. ✅ 熟悉 Python 基礎語法和 NumPy、Pandas 操作

---

## 9. 總結

本單元介紹了線性模型回歸的核心概念：

- 線性模型假設目標與特徵間存在線性關係，具有可解釋性強、計算效率高等優點
- sklearn 提供多種線性模型，主要差異在於正則化技術 (無、L2、L1、L1+L2)
- 適當的資料前處理 (標準化、編碼) 可顯著提升模型性能
- 使用交叉驗證和超參數調整可獲得更穩健的模型
- 線性模型在化工領域有廣泛應用，從製程預測到品質控制

接下來的各子單元將深入探討每種線性模型的細節與實作，請繼續學習！

---

**版權聲明**：本教材由逢甲大學化學工程學系莊曜禎助理教授編寫，僅供教學使用。

**課程代碼**：CHE-AI-114  
**更新日期**：2026 年 1 月
