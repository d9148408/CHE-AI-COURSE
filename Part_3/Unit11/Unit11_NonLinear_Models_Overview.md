# Unit11 非線性模型回歸總覽 | Non-Linear Models Regression Overview

---

## 課程目標

本單元將深入介紹非線性模型回歸 (Non-Linear Regression Models) 的理論基礎與實務應用，專注於 scikit-learn 模組提供的各種非線性回歸方法。學生將學習：

- 理解非線性模型的數學原理與適用條件
- 掌握樹狀模型、支持向量機、高斯過程等方法的核心概念
- 學習使用 sklearn 建立、訓練與評估非線性模型
- 了解非線性模型在化工領域的實際應用案例
- 比較線性與非線性模型的優缺點與選擇策略

---

## 1. 非線性模型基礎理論

### 1.1 什麼是非線性模型？

非線性模型 (Non-Linear Model) 是指目標變數與特徵變數之間存在**非線性關係**的預測模型。與線性模型不同，非線性模型可以捕捉更複雜的資料模式，包括：

- **曲線關係**：如指數、對數、多項式關係
- **交互作用**：特徵之間的複雜互動效應
- **局部模式**：資料在不同區域有不同的行為特徵
- **階躍變化**：突然的轉折點或臨界值效應

數學上，非線性模型的預測函數可以表示為：

$$
\hat{y} = f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n)
$$

其中 $f$ 是非線性函數，無法表示成特徵的線性組合。

### 1.2 為什麼需要非線性模型？

在化工領域中，許多物理化學現象本質上就是非線性的：

#### 1.2.1 化學反應動力學
- Arrhenius 方程：反應速率與溫度呈指數關係

$$
k = A e^{-\frac{E_a}{RT}}
$$

#### 1.2.2 相平衡
- Antoine 方程：蒸氣壓與溫度的非線性關係

$$
\log_{10} P = A - \frac{B}{C + T}
$$

#### 1.2.3 吸附等溫線
- Langmuir 等溫式：吸附量與濃度的非線性關係

$$
q = \frac{q_{\max} K C}{1 + K C}
$$

#### 1.2.4 製程控制
- 閥門特性曲線、pH 緩衝效應等都呈現顯著的非線性特徵

### 1.3 非線性模型 vs. 線性模型

| 特性 | 線性模型 | 非線性模型 |
|------|----------|------------|
| **適用關係** | 線性關係 | 任意複雜關係 |
| **模型複雜度** | 低 | 高 |
| **可解釋性** | 高（權重直觀） | 中至低 |
| **計算效率** | 高 | 中至低 |
| **過擬合風險** | 低 | 高（需正則化） |
| **資料需求** | 較少 | 較多 |
| **泛化能力** | 中等 | 需謹慎調參 |

### 1.4 非線性模型的分類

本單元將介紹的非線性模型可分為三大類：

#### 1.4.1 參數擴展型
- **多項式回歸 (Polynomial Regression)**：透過增加特徵的高次項來捕捉非線性關係

#### 1.4.2 樹狀模型 (Tree-based Models)
- **決策樹 (Decision Tree)**：透過樹狀結構進行分層決策
- **隨機森林 (Random Forest)**：多棵決策樹的集成學習
- **梯度提升樹 (Gradient Boosting Trees)**：逐步修正誤差的集成學習

#### 1.4.3 核方法與機率模型
- **支持向量機 (Support Vector Machine, SVM)**：透過核函數映射到高維空間
- **高斯過程回歸 (Gaussian Process Regression, GPR)**：基於貝氏推論的機率模型

---

## 2. sklearn 中的非線性模型方法

### 2.1 多項式回歸 (Polynomial Regression)

**模型**：`sklearn.preprocessing.PolynomialFeatures` + `sklearn.linear_model.LinearRegression`

**核心概念**：
多項式回歸本質上仍是線性模型，但透過將特徵擴展為高次項和交互項，可以擬合非線性關係。

對於特徵 $x$ ，二次多項式回歸的形式為：

$$
\hat{y} = w_0 + w_1 x + w_2 x^2
$$

對於兩個特徵 $x_1, x_2$ ，二次多項式包含：

$$
\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_1^2 + w_4 x_1 x_2 + w_5 x_2^2
$$

**主要參數**：
- `degree`: 多項式次數（1 = 線性，2 = 二次，3 = 三次...）
- `include_bias`: 是否包含截距項（預設 True）
- `interaction_only`: 是否僅包含交互項（預設 False）

**優點**：
- 實作簡單，基於線性模型
- 可解釋性較好
- 適合捕捉平滑的曲線關係

**缺點**：
- 高次項容易過擬合
- 特徵數量隨次數呈指數增長
- 外推能力差

**適用場景**：
- 特徵與目標呈現明顯的曲線關係
- 需要保持一定的可解釋性
- 資料範圍有限，不需要外推

---

### 2.2 決策樹 (Decision Tree)

**模型**：`sklearn.tree.DecisionTreeRegressor`

**核心概念**：
決策樹透過一系列的 if-then-else 規則將特徵空間遞迴地分割成多個區域，每個區域內的樣本被賦予相同的預測值。

**分割策略**：
決策樹在每個節點選擇最佳的特徵和分割點，使得分割後的子節點「純度」最高。對於回歸問題，常用的分割標準是**均方誤差 (MSE)**：

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \bar{y})^2
$$

在每個分割點，目標是最小化加權平均的子節點 MSE：

$$
\text{Cost} = \frac{N_{\text{left}}}{N} \text{MSE}_{\text{left}} + \frac{N_{\text{right}}}{N} \text{MSE}_{\text{right}}
$$

**主要參數**：
- `max_depth`: 樹的最大深度（控制複雜度）
- `min_samples_split`: 內部節點分裂所需的最小樣本數（預設 2）
- `min_samples_leaf`: 葉節點所需的最小樣本數（預設 1）
- `max_features`: 尋找最佳分割時考慮的最大特徵數
- `criterion`: 分割品質的評估標準（'squared_error', 'friedman_mse', 'absolute_error', 'poisson'）

**優點**：
- 高度可解釋，可視化呈現決策規則
- 不需要特徵縮放
- 自動處理特徵選擇
- 可處理非線性和交互作用

**缺點**：
- 容易過擬合（特別是深樹）
- 對訓練資料微小變化敏感
- 預測結果為階梯函數，不夠平滑
- 外推能力弱

**適用場景**：
- 需要高度可解釋性
- 資料包含類別型特徵
- 特徵之間有複雜交互作用

---

### 2.3 隨機森林 (Random Forest)

**模型**：`sklearn.ensemble.RandomForestRegressor`

**核心概念**：
隨機森林是**決策樹的集成學習 (Ensemble Learning)** 方法，透過以下策略提升預測性能：

1. **Bagging (Bootstrap Aggregating)**：從訓練資料中隨機抽樣（有放回）建立多棵樹
2. **隨機特徵選擇**：每次分割時只考慮隨機選取的部分特徵
3. **平均預測**：最終預測為所有樹預測值的平均

數學上，隨機森林的預測為：

$$
\hat{y} = \frac{1}{T} \sum_{t=1}^{T} f_t(\mathbf{x})
$$

其中 $T$ 是樹的數量， $f_t$ 是第 $t$ 棵決策樹的預測函數。

**主要參數**：
- `n_estimators`: 樹的數量（預設 100，越多越好但計算慢）
- `max_depth`: 每棵樹的最大深度
- `min_samples_split`: 內部節點分裂所需的最小樣本數
- `min_samples_leaf`: 葉節點所需的最小樣本數
- `max_features`: 每次分割考慮的最大特徵數（'sqrt', 'log2', int, float）
- `bootstrap`: 是否使用 bootstrap 抽樣（預設 True）
- `oob_score`: 是否計算袋外 (out-of-bag) 分數（預設 False）

**特徵重要性**：
隨機森林可計算每個特徵的重要性，基於該特徵在所有樹中降低不純度的平均值。

**優點**：
- 預測準確度高，泛化能力強
- 不易過擬合（相比單棵決策樹）
- 可處理高維資料
- 可評估特徵重要性
- 訓練可平行化

**缺點**：
- 可解釋性降低（相比單棵樹）
- 訓練和預測速度較慢
- 模型檔案較大
- 對雜訊和異常值敏感度仍存在

**適用場景**：
- 資料量充足
- 需要高預測準確度
- 特徵數量多
- 需要評估特徵重要性

---

### 2.4 梯度提升樹 (Gradient Boosting Trees)

**模型**：`sklearn.ensemble.GradientBoostingRegressor`

**核心概念**：
梯度提升是另一種集成學習方法，採用**逐步修正誤差 (Boosting)** 的策略：

1. 訓練第一棵樹預測目標值
2. 訓練第二棵樹預測第一棵樹的殘差（誤差）
3. 訓練第三棵樹預測前兩棵樹的殘差
4. 重複以上步驟...

最終預測為所有樹的加權和：

$$
\hat{y} = f_0 + \eta \sum_{t=1}^{T} f_t(\mathbf{x})
$$

其中 $\eta$ 是學習率 (learning rate)， $f_0$ 是初始預測（通常是目標值的平均）。

**梯度下降的視角**：
梯度提升實際上是在函數空間中進行梯度下降，每棵新樹都在損失函數的負梯度方向上擬合資料。

**主要參數**：
- `n_estimators`: 提升階段的數量（樹的數量）
- `learning_rate`: 學習率 $\eta$ （預設 0.1，控制每棵樹的貢獻）
- `max_depth`: 每棵樹的最大深度（通常設為較小值，如 3-5）
- `subsample`: 用於訓練每棵樹的樣本比例（< 1 可減少過擬合）
- `loss`: 損失函數（'squared_error', 'absolute_error', 'huber', 'quantile'）
- `min_samples_split`: 內部節點分裂所需的最小樣本數
- `min_samples_leaf`: 葉節點所需的最小樣本數

**優點**：
- 預測準確度極高（常在競賽中獲勝）
- 可處理混合型特徵（數值+類別）
- 對異常值較穩健（使用 Huber 損失）
- 可透過學習率控制過擬合

**缺點**：
- 訓練速度慢（無法平行化）
- 需要仔細調參
- 對雜訊敏感（可能過擬合）
- 可解釋性低

**適用場景**：
- 追求最高預測準確度
- 資料質量高、雜訊少
- 有充足的調參時間
- 不需要即時預測

**進階變體**：
- **XGBoost**：高效能實作，支援平行化和正則化
- **LightGBM**：基於直方圖的快速演算法
- **CatBoost**：自動處理類別特徵

---

### 2.5 支持向量機 (Support Vector Machine, SVM)

**模型**：`sklearn.svm.SVR` (Support Vector Regression)

**核心概念**：
SVM 透過**核函數 (Kernel Function)** 將資料映射到高維空間，在高維空間中尋找最佳的超平面進行回歸。

對於線性不可分的資料，核技巧 (Kernel Trick) 允許我們在原始空間中計算高維空間的內積：

$$
K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)
$$

其中 $\phi$ 是映射函數， $K$ 是核函數。

**常用核函數**：

1. **線性核 (Linear Kernel)**：
   $$K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j$$

2. **多項式核 (Polynomial Kernel)**：
   $$K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T \mathbf{x}_j + r)^d$$

3. **徑向基函數核 (RBF Kernel / Gaussian Kernel)**：
   $$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$

4. **Sigmoid 核**：
   $$K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\gamma \mathbf{x}_i^T \mathbf{x}_j + r)$$

**SVM 回歸目標**：
SVM 回歸 (SVR) 目標是找到一個函數，使得大部分訓練樣本與該函數的偏差不超過 $\epsilon$ ：

$$
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{m} (\xi_i + \xi_i^*)
$$

$$
\text{subject to} \quad |y_i - f(\mathbf{x}_i)| \leq \epsilon + \xi_i
$$

其中 $\xi_i$ 是鬆弛變數 (slack variables)， $C$ 是懲罰參數。

**主要參數**：
- `kernel`: 核函數類型（'linear', 'poly', 'rbf', 'sigmoid'）
- `C`: 正則化參數（預設 1.0，越大越重視訓練誤差）
- `epsilon`: $\epsilon$ -不敏感損失的參數（預設 0.1）
- `gamma`: RBF、poly、sigmoid 核的係數（'scale', 'auto', float）
- `degree`: 多項式核的次數（預設 3）

**優點**：
- 可透過核函數處理非線性關係
- 對高維資料表現良好
- 記憶體效率高（僅儲存支持向量）
- 對異常值較不敏感（ $\epsilon$ -不敏感損失）

**缺點**：
- 訓練時間長（特別是大資料集）
- 核函數和參數選擇困難
- 模型可解釋性低
- 對特徵縮放敏感（需標準化）

**適用場景**：
- 中小型資料集（< 10,000 樣本）
- 高維特徵空間
- 需要穩健的預測
- 資料存在複雜的非線性模式

---

### 2.6 高斯過程回歸 (Gaussian Process Regression, GPR)

**模型**：`sklearn.gaussian_process.GaussianProcessRegressor`

**核心概念**：
高斯過程是一種**非參數貝氏方法**，將函數視為服從高斯分佈的隨機過程。GPR 不僅提供預測值，還提供預測的**不確定性 (uncertainty)**。

**高斯過程定義**：
函數 $f(\mathbf{x})$ 服從高斯過程，記作：

$$
f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))
$$

其中：
- $m(\mathbf{x})$ 是均值函數（通常設為 0）
- $k(\mathbf{x}, \mathbf{x}')$ 是核函數（協方差函數），描述不同輸入點之間的相關性

**預測分佈**：
給定訓練資料 $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{m}$ 和新輸入 $\mathbf{x}_*$ ，GPR 提供預測分佈：

$$
p(f_* | \mathbf{x}_*, \mathcal{D}) = \mathcal{N}(\mu_*, \sigma_*^2)
$$

其中：

$$
\mu_* = \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y}
$$

$$
\sigma_*^2 = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*
$$

- $\mathbf{K}$ 是訓練資料的核矩陣
- $\mathbf{k}_*$ 是新輸入與訓練資料的核向量
- $\sigma_n^2$ 是雜訊變異數

**常用核函數**：

1. **RBF 核 (Radial Basis Function)**：
   $$k(\mathbf{x}_i, \mathbf{x}_j) = \sigma^2 \exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2l^2}\right)$$
   
2. **Matérn 核**：更靈活，可控制平滑度

3. **有理二次核 (Rational Quadratic)**：RBF 核的多尺度版本

4. **週期核 (Periodic)**：適用於週期性資料

**主要參數**：
- `kernel`: 核函數（預設為 RBF 核）
- `alpha`: 對角線上的雜訊水平（預設 1e-10）
- `n_restarts_optimizer`: 核函數參數優化的重啟次數（預設 0）
- `normalize_y`: 是否標準化目標值（預設 False）

**優點**：
- 提供預測的**不確定性量化**
- 適合小資料集
- 自動特徵選擇（透過核函數參數）
- 可融入先驗知識（透過核函數設計）
- 插值性能優異

**缺點**：
- 計算複雜度高： $O(n^3)$ （不適合大資料集）
- 記憶體需求大：需儲存核矩陣
- 核函數選擇和參數調整困難
- 高維資料表現不佳（維度詛咒）

**適用場景**：
- 小資料集（< 1,000 樣本）
- 需要不確定性量化（如實驗設計、貝氏優化）
- 平滑函數擬合
- 主動學習 (Active Learning)
- 資料取得成本高昂的工程應用

---

### 2.7 模型比較總結

| 模型 | 複雜度 | 可解釋性 | 預測準確度 | 訓練速度 | 不確定性量化 | 適用資料量 |
|------|--------|----------|------------|----------|--------------|------------|
| 多項式回歸 | 低-中 | ★★★★☆ | ★★☆☆☆ | ★★★★★ | ✗ | 中-大 |
| 決策樹 | 中 | ★★★★★ | ★★★☆☆ | ★★★★☆ | ✗ | 小-中 |
| 隨機森林 | 高 | ★★☆☆☆ | ★★★★☆ | ★★★☆☆ | △ (OOB) | 中-大 |
| 梯度提升樹 | 高 | ★★☆☆☆ | ★★★★★ | ★★☆☆☆ | ✗ | 中-大 |
| SVM | 中-高 | ★☆☆☆☆ | ★★★★☆ | ★★☆☆☆ | ✗ | 小-中 |
| GPR | 高 | ★☆☆☆☆ | ★★★★☆ | ★☆☆☆☆ | ★★★★★ | 小 |

**選擇建議**：
- **追求準確度**：梯度提升樹 > 隨機森林 > GPR
- **需要可解釋**：決策樹 > 多項式回歸
- **小資料集**：GPR > SVM > 決策樹
- **大資料集**：隨機森林 > 梯度提升樹
- **需要不確定性**：GPR（唯一選擇）
- **即時預測**：多項式回歸 > 決策樹

---

## 3. 資料前處理技術

非線性模型對資料前處理的需求因模型而異。本節介紹常用的前處理方法及其適用性。

### 3.1 特徵縮放的必要性

| 模型 | 是否需要特徵縮放？ | 原因 |
|------|-------------------|------|
| 多項式回歸 | **需要** | 高次項數值範圍極大，影響數值穩定性 |
| 決策樹 | **不需要** | 基於分割點，對尺度不敏感 |
| 隨機森林 | **不需要** | 同決策樹 |
| 梯度提升樹 | **不需要** | 同決策樹 |
| SVM | **強烈需要** | 核函數計算距離，對尺度極度敏感 |
| GPR | **強烈需要** | 核函數基於距離，尺度影響預測 |

### 3.2 標準化 (Standardization)

將特徵轉換為均值 0、標準差 1：

$$
x_{\text{scaled}} = \frac{x - \mu}{\sigma}
$$

**sklearn 實現**：
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 使用訓練集的參數
```

**適用**：SVM、GPR、多項式回歸

### 3.3 正規化 (Min-Max Normalization)

縮放到 [0, 1] 範圍：

$$
x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

**sklearn 實現**：
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

**適用**：SVM（RBF 核）、GPR

### 3.4 類別變數編碼

#### 3.4.1 獨熱編碼 (One-Hot Encoding)

**適用模型**：所有模型（樹狀模型可選）

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop避免共線性
X_encoded = encoder.fit_transform(X_categorical)
```

#### 3.4.2 目標編碼 (Target Encoding)

用目標變數的統計量替換類別（適用於高基數類別特徵）：

```python
category_means = train_data.groupby('category')['target'].mean()
train_data['category_encoded'] = train_data['category'].map(category_means)
```

**注意**：需防止資料洩漏，建議使用交叉驗證。

### 3.5 處理缺失值

```python
from sklearn.impute import SimpleImputer

# 數值特徵：均值/中位數填補
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_numerical)

# 類別特徵：眾數填補或新類別
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = imputer.fit_transform(X_categorical)
```

### 3.6 異常值處理

**檢測方法**：
- **IQR 法**：超出 $[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$ 
- **Z-score**：$|z| > 3$ 視為異常值

**處理策略**：
1. 刪除異常值（謹慎使用）
2. 限幅 (Clipping / Winsorization)
3. 使用穩健模型（如 Huber 損失的梯度提升）

```python
# IQR 限幅
Q1 = df['feature'].quantile(0.25)
Q3 = df['feature'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df['feature_clipped'] = df['feature'].clip(lower, upper)
```

---

## 4. 模型評估方法

### 4.1 評估指標

非線性模型使用與線性模型相同的回歸評估指標：

#### 4.1.1 均方誤差 (MSE) 與均方根誤差 (RMSE)

$$
\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2, \quad \text{RMSE} = \sqrt{\text{MSE}}
$$

```python
from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
```

#### 4.1.2 平均絕對誤差 (MAE)

$$
\text{MAE} = \frac{1}{m} \sum_{i=1}^{m} |y_i - \hat{y}_i|
$$

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
```

#### 4.1.3 決定係數 ( $R^2$  Score)

$$
R^2 = 1 - \frac{\sum_{i=1}^{m} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{m} (y_i - \bar{y})^2}
$$

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
```

#### 4.1.4 平均絕對百分比誤差 (MAPE)

$$
\text{MAPE} = \frac{100\%}{m} \sum_{i=1}^{m} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
$$

```python
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

### 4.2 交叉驗證 (Cross-Validation)

#### 4.2.1 K 折交叉驗證

```python
from sklearn.model_selection import cross_val_score

model = RandomForestRegressor(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print(f"R² scores: {scores}")
print(f"Mean R²: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

#### 4.2.2 時間序列交叉驗證

對於時間序列資料，需使用 `TimeSeriesSplit` 避免資料洩漏：

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
```

### 4.3 超參數調整 (Hyperparameter Tuning)

#### 4.3.1 網格搜尋 (Grid Search)

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best R² score: {grid_search.best_score_:.3f}")
```

#### 4.3.2 隨機搜尋 (Random Search)

對於參數空間大的模型（如 SVM、GPR），隨機搜尋更高效：

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_dist,
    n_iter=100,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
```

### 4.4 過擬合檢測與防止

#### 4.4.1 學習曲線 (Learning Curve)

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='r2'
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation score')
plt.xlabel('Training Set Size')
plt.ylabel('R² Score')
plt.legend()
plt.title('Learning Curve')
plt.show()
```

#### 4.4.2 驗證曲線 (Validation Curve)

觀察超參數對訓練和驗證分數的影響：

```python
from sklearn.model_selection import validation_curve

param_range = [10, 50, 100, 200, 500]
train_scores, val_scores = validation_curve(
    RandomForestRegressor(random_state=42),
    X, y,
    param_name='n_estimators',
    param_range=param_range,
    cv=5,
    scoring='r2'
)

plt.plot(param_range, train_scores.mean(axis=1), label='Training score')
plt.plot(param_range, val_scores.mean(axis=1), label='Validation score')
plt.xlabel('n_estimators')
plt.ylabel('R² Score')
plt.legend()
plt.show()
```

---

## 5. 非線性模型在化工領域的應用

### 5.1 應用場景

#### 5.1.1 複雜反應系統建模
- **場景**：多步驟反應、催化反應、聚合反應
- **特點**：反應機理複雜，難以用簡單方程描述
- **推薦模型**：隨機森林、梯度提升樹、GPR
- **案例**：預測聚合物分子量分布、反應選擇性

#### 5.1.2 非線性製程控制
- **場景**：pH 控制、結晶過程、分批反應器
- **特點**：系統動態呈現高度非線性
- **推薦模型**：SVM、GPR（提供不確定性）
- **案例**：pH 中和過程控制、結晶粒徑控制

#### 5.1.3 品質預測與優化
- **場景**：產品物性、感官品質、藥效預測
- **特點**：多因素交互作用、局部最優
- **推薦模型**：隨機森林（特徵重要性）、XGBoost（高準確度）
- **案例**：藥物溶解度預測、食品質地預測

#### 5.1.4 故障診斷與異常檢測
- **場景**：設備異常、製程偏移、產品缺陷
- **特點**：正常與異常的邊界複雜
- **推薦模型**：決策樹（可解釋）、隨機森林、梯度提升樹
- **案例**：泵浦故障診斷、蒸餾塔異常檢測

#### 5.1.5 分子性質預測 (QSPR/QSAR)
- **場景**：物理化學性質、生物活性、毒性預測
- **特點**：高維分子描述符、非線性結構-性質關係
- **推薦模型**：隨機森林、梯度提升樹、SVM
- **案例**：沸點預測、辛醇-水分配係數預測、藥物活性預測

### 5.2 實際應用案例

#### 案例 1：催化反應產率優化（隨機森林）

**背景**：某化工廠希望建立催化反應產率的預測模型，並識別關鍵操作變數。

**資料特徵**：
- 反應溫度 (200-300°C)
- 反應壓力 (10-50 bar)
- 催化劑負載量 (1-5 wt%)
- 進料流量 (50-200 L/h)
- 氫氣比例 (0.5-2.0)
- 催化劑使用時間 (0-1000 h)

**模型選擇**：隨機森林回歸

**實作流程**：
1. 資料分割：80% 訓練，20% 測試
2. 特徵不需標準化（樹狀模型）
3. 超參數調整：`n_estimators=200`, `max_depth=15`
4. 評估特徵重要性

**結果**：
- 測試集 $R^2 = 0.94$ ， RMSE = 1.2%
- 關鍵特徵：反應溫度 (35%)、催化劑使用時間 (28%)、壓力 (20%)
- 應用：透過模型指導操作優化，產率提升 3.5%

#### 案例 2：蒸餾塔操作建模（梯度提升樹）

**背景**：建立蒸餾塔頂產品純度的預測模型，用於製程監控與控制。

**資料特徵**：
- 進料流量、組成（5 個組分）
- 回流比
- 加熱功率
- 塔頂壓力
- 冷卻水溫度

**模型選擇**：XGBoost (梯度提升樹)

**挑戰**：
- 動態過程，存在時間延遲
- 操作條件變化範圍大

**處理方法**：
- 加入時間滯後特徵 (lag features)：前 1、2、3 小時的操作變數
- 滾動窗口特徵：過去 6 小時的平均值、標準差

**結果**：
- 測試集 $R^2 = 0.97$ ， MAE = 0.3%
- 預測提前 15 分鐘，支援前饋控制
- 產品純度波動降低 40%

#### 案例 3：藥物溶解度預測（SVM + 分子描述符）

**背景**：預測藥物分子在水中的溶解度，用於藥物設計篩選。

**資料特徵**：
- 分子描述符：分子量、氫鍵供體/受體數、LogP、拓撲極性表面積 (TPSA)、旋轉鍵數等
- 特徵數量：> 200 個描述符

**模型選擇**：SVM (RBF 核)

**資料前處理**：
1. 移除低變異數特徵
2. 標準化所有特徵
3. Lasso 特徵選擇（降至 50 個關鍵描述符）

**超參數調整**：
- `C`: [0.1, 1, 10, 100]
- `gamma`: ['scale', 0.001, 0.01, 0.1]
- 最佳參數：`C=10`, `gamma=0.01`

**結果**：
- 測試集 $R^2 = 0.89$ ， RMSE = 0.52 log units
- 關鍵描述符：LogP、TPSA、氫鍵數
- 應用：虛擬篩選，候選分子數量減少 80%

#### 案例 4：反應器溫度控制（高斯過程回歸 + 貝氏優化）

**背景**：最佳化批次反應器的溫度曲線，最大化產率。

**挑戰**：
- 每次實驗成本高（時間、原料）
- 溫度曲線參數空間大（10 維）

**模型選擇**：高斯過程回歸（GPR）

**優勢**：
- 提供預測不確定性
- 支援**貝氏優化 (Bayesian Optimization)**：基於不確定性主動選擇下次實驗條件

**實作**：
1. 初始隨機實驗：20 次
2. 建立 GPR 模型
3. 使用 Expected Improvement (EI) 準則選擇下次實驗點
4. 更新 GPR 模型
5. 重複步驟 3-4

**結果**：
- 50 次實驗後找到最佳溫度曲線
- 產率從 78% 提升至 92%
- 相比隨機搜尋，實驗次數減少 60%

---

## 6. 非線性模型的選擇策略

### 6.1 決策流程圖

```
開始
  ↓
是否需要不確定性量化？
  ├─ 是 → 高斯過程回歸 (GPR)
  └─ 否 ↓
資料量級？
  ├─ 小 (< 1,000) → SVM 或 GPR
  ├─ 中 (1,000 - 100,000) → 隨機森林 或 梯度提升樹
  └─ 大 (> 100,000) → 隨機森林 或 線性模型
          ↓
是否需要高度可解釋性？
  ├─ 是 → 決策樹 或 多項式回歸
  └─ 否 ↓
是否追求最高準確度？
  ├─ 是 → 梯度提升樹 (XGBoost / LightGBM)
  └─ 否 → 隨機森林（平衡準確度與速度）
```

### 6.2 模型組合策略

#### 6.2.1 模型融合 (Model Ensemble)

結合多個模型的預測以提升性能：

```python
# 簡單平均
y_pred_avg = (y_pred_rf + y_pred_gbm + y_pred_svm) / 3

# 加權平均（基於驗證集性能）
w1, w2, w3 = 0.4, 0.4, 0.2
y_pred_weighted = w1 * y_pred_rf + w2 * y_pred_gbm + w3 * y_pred_svm
```

#### 6.2.2 Stacking

使用元學習器 (meta-learner) 組合基模型：

```python
from sklearn.ensemble import StackingRegressor

estimators = [
    ('rf', RandomForestRegressor(n_estimators=100)),
    ('gbm', GradientBoostingRegressor(n_estimators=100)),
    ('svr', SVR(kernel='rbf'))
]

stacking = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge()
)

stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)
```

---

## 7. 學習資源與延伸閱讀

### 7.1 官方文檔

- [scikit-learn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [scikit-learn SVM Documentation](https://scikit-learn.org/stable/modules/svm.html)
- [scikit-learn Gaussian Processes](https://scikit-learn.org/stable/modules/gaussian_process.html)

### 7.2 進階主題

1. **進階梯度提升**：
   - [XGBoost Documentation](https://xgboost.readthedocs.io/)
   - [LightGBM Documentation](https://lightgbm.readthedocs.io/)
   - [CatBoost Documentation](https://catboost.ai/docs/)

2. **貝氏優化**：
   - `scikit-optimize` 套件
   - `BayesianOptimization` 套件

3. **自動機器學習 (AutoML)**：
   - TPOT (Tree-based Pipeline Optimization Tool)
   - Auto-sklearn

### 7.3 延伸應用

- **製程軟感測器 (Soft Sensor)**：使用非線性模型估算難以直接量測的變數
- **適應性建模 (Adaptive Modeling)**：模型隨時間更新以適應製程變化
- **多目標優化**：同時優化產率、成本、環境影響

---

## 8. 本單元學習路徑

完成本單元後，您將依序學習以下主題：

1. **Unit11_Polynomial_Regression**：多項式回歸的詳細理論與實作
2. **Unit11_Decision_Tree**：決策樹的分裂策略與視覺化
3. **Unit11_Random_Forest**：集成學習與特徵重要性分析
4. **Unit11_Gradient_Boosting_Trees**：提升方法與超參數調整
5. **Unit11_Support_Vector_Machine**：核技巧與參數選擇
6. **Unit11_Gaussian_Process_Regression**：貝氏方法與不確定性量化
7. **Unit11_NonLinear_Models_Homework**：綜合練習與模型比較

每個子主題都包含詳細的數學推導、程式碼實作和化工領域的應用案例，請按順序學習以獲得最佳效果。

---

## 9. 課前準備

在開始本單元之前，請確保您已經：

1. ✅ 完成 **Unit10 線性模型回歸**
2. ✅ 安裝必要套件：
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn scipy
   ```
3. ✅ 複習數學知識：
   - 微積分（梯度、最優化）
   - 線性代數（核函數、矩陣運算）
   - 機率統計（高斯分佈、貝氏定理）
4. ✅ 熟悉 sklearn 基本操作：
   - 資料分割、標準化
   - 模型訓練、預測、評估

---

## 10. 總結

本單元介紹了非線性模型回歸的核心概念：

- **多項式回歸**：透過特徵擴展捕捉曲線關係，適合平滑函數擬合
- **決策樹**：高度可解釋的分層決策模型，適合複雜交互作用
- **隨機森林**：決策樹的集成學習，平衡準確度與穩健性
- **梯度提升樹**：逐步修正誤差的集成方法，提供最高預測準確度
- **支持向量機**：透過核函數處理非線性，適合中小型資料集
- **高斯過程回歸**：貝氏方法提供不確定性量化，適合小資料集和貝氏優化

**關鍵要點**：
- 非線性模型可捕捉複雜的資料模式，但需更多資料和謹慎調參
- 樹狀模型（RF、GBM）是工業應用首選，兼顧準確度與實用性
- SVM 和 GPR 需要特徵縮放，計算成本較高
- 模型選擇應考慮：資料量、可解釋性需求、計算資源、預測準確度

非線性模型在化工領域有廣泛應用，從反應建模到製程優化，從品質預測到故障診斷。掌握這些方法將大幅提升您解決實際工程問題的能力！

接下來的各子單元將深入探討每種非線性模型的細節與實作，請繼續學習！

---

**版權聲明**：本教材由逢甲大學化學工程學系莊曜禎助理教授編寫，僅供教學使用。

**課程代碼**：CHE-AI-114  
**更新日期**：2026 年 1 月

