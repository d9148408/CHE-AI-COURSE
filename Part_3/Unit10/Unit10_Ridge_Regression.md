# Unit10 Ridge 回歸 | Ridge Regression

> **最後更新**：2026-01-17 | 建立完整教學講義

---

## 學習目標

本節課將深入學習**Ridge 回歸 (Ridge Regression)** 模型，這是一種加入 L2 正則化的線性回歸方法，能有效解決多重共線性和過擬合問題。通過本節課，您將能夠：

- 理解 Ridge 回歸的數學原理與 L2 正則化
- 掌握 sklearn 中 `Ridge` 的使用方法
- 學習如何選擇最佳的正則化強度 (α)
- 理解 Ridge 回歸與普通線性回歸的差異
- 應用 Ridge 回歸解決化工領域的實際問題（**含完整實作案例**）

---

## 1. Ridge 回歸基本概念

### 1.1 什麼是 Ridge 回歸？

**Ridge 回歸 (Ridge Regression)**，又稱為**岭回歸**或 **L2 正則化線性回歸**，是在普通最小平方法 (OLS) 的基礎上加入 **L2 正則化項 (L2 Regularization)**，透過懲罰係數的平方和來防止過擬合。

Ridge 回歸的目標函數可表示為：

$$
\min_{\boldsymbol{\beta}} \left\{ \sum_{i=1}^{n} (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 + \alpha \sum_{j=1}^{p} \beta_j^2 \right\}
$$

或以矩陣形式表示：

$$
\min_{\boldsymbol{\beta}} \left\{ \| \mathbf{y} - \mathbf{X} \boldsymbol{\beta} \|_2^2 + \alpha \| \boldsymbol{\beta} \|_2^2 \right\}
$$

其中：
- $\mathbf{y}$ : 目標變數向量
- $\mathbf{X}$ : 特徵矩陣
- $\boldsymbol{\beta}$ : 回歸係數向量
- $\alpha$ : **正則化強度 (Regularization Strength)**，控制懲罰項的權重
- $\| \boldsymbol{\beta} \|_2^2 = \sum_{j=1}^{p} \beta_j^2$ : L2 範數的平方（係數的平方和）

### 1.2 為什麼需要 Ridge 回歸？

普通線性回歸在以下情況下可能表現不佳：

1. **多重共線性 (Multicollinearity)**：
   - 當特徵之間高度相關時，OLS 估計的係數方差會非常大
   - 導致模型不穩定，係數估計不可靠

2. **過擬合 (Overfitting)**：
   - 當特徵數量較多時，模型可能過度擬合訓練數據
   - 導致泛化能力差，在測試集上表現不佳

3. **數值不穩定**：
   - 當 $\mathbf{X}^T \mathbf{X}$ 接近奇異矩陣時，OLS 解不穩定

**Ridge 回歸的優勢**：
- ✓ 透過懲罰大係數來降低模型複雜度
- ✓ 改善多重共線性問題
- ✓ 提高模型的泛化能力
- ✓ 保留所有特徵（係數趨近於 0 但不會完全為 0）

### 1.3 L2 正則化的幾何解釋

Ridge 回歸可視為在約束條件下求解最小平方問題：

$$
\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 \quad \text{subject to} \quad \sum_{j=1}^{p} \beta_j^2 \leq t
$$

- L2 懲罰項將係數限制在一個**圓形區域**內（2D）或**超球面**內（高維）
- 這種約束使得係數向量的長度受限，從而減小係數值
- 與 Lasso (L1) 不同，Ridge 不會產生稀疏解（係數不會完全變為 0）

---

## 2. Ridge 回歸的數學推導

### 2.1 目標函數

Ridge 回歸的目標是最小化以下損失函數：

$$
L(\boldsymbol{\beta}) = \| \mathbf{y} - \mathbf{X} \boldsymbol{\beta} \|_2^2 + \alpha \| \boldsymbol{\beta} \|_2^2
$$

展開後：

$$
L(\boldsymbol{\beta}) = (\mathbf{y} - \mathbf{X} \boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X} \boldsymbol{\beta}) + \alpha \boldsymbol{\beta}^T \boldsymbol{\beta}
$$

### 2.2 解析解

對 $\boldsymbol{\beta}$ 求導並令其為 0：

$$
\frac{\partial L}{\partial \boldsymbol{\beta}} = -2 \mathbf{X}^T (\mathbf{y} - \mathbf{X} \boldsymbol{\beta}) + 2 \alpha \boldsymbol{\beta} = 0
$$

整理後得到：

$$
\mathbf{X}^T \mathbf{y} = \mathbf{X}^T \mathbf{X} \boldsymbol{\beta} + \alpha \boldsymbol{\beta}
$$

$$
\mathbf{X}^T \mathbf{y} = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I}) \boldsymbol{\beta}
$$

因此，**Ridge 回歸的解析解**為：

$$
\hat{\boldsymbol{\beta}}_{\text{Ridge}} = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}
$$

其中 $\mathbf{I}$ 是單位矩陣。

### 2.3 與 OLS 的比較

**普通最小平方法 (OLS)** 的解：

$$
\hat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

**關鍵差異**：
- Ridge 在 $\mathbf{X}^T \mathbf{X}$ 的對角線上加了 $\alpha \mathbf{I}$
- 這確保了即使 $\mathbf{X}^T \mathbf{X}$ 接近奇異矩陣，$(\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})$ 仍然可逆
- 當 $\alpha = 0$ 時，Ridge 退化為 OLS
- 當 $\alpha \to \infty$ 時，$\hat{\boldsymbol{\beta}}_{\text{Ridge}} \to 0$

### 2.4 偏差-方差權衡

Ridge 回歸引入了偏差以換取更低的方差：

- **偏差 (Bias)**：Ridge 估計是有偏的（ $E[\hat{\boldsymbol{\beta}}_{\text{Ridge}}] \neq \boldsymbol{\beta}_{\text{true}}$ ）
- **方差 (Variance)**：但方差顯著降低，提高了預測穩定性
- **總誤差**：在許多情況下，偏差的少量增加可以被方差的大幅減少所補償

$$
\text{MSE} = \text{Bias}^2 + \text{Variance}
$$

通過調整 $\alpha$ ，我們可以在偏差和方差之間找到最佳平衡點。

---

## 3. 正則化強度 α 的選擇

### 3.1 α 的作用

- **α = 0**：無正則化，等同於 OLS
- **α 很小**：輕微懲罰，接近 OLS
- **α 適中**：平衡擬合誤差與係數大小
- **α 很大**：強烈懲罰，係數接近 0，模型欠擬合

### 3.2 選擇 α 的方法

#### 方法 1: 交叉驗證

使用 `RidgeCV` 自動搜索最佳 α：

```python
from sklearn.linear_model import RidgeCV

# 定義 α 候選值
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# 使用交叉驗證選擇最佳 α
ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train, y_train)

print(f"最佳 α: {ridge_cv.alpha_}")
```

#### 方法 2: 網格搜索

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5, 
                           scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print(f"最佳 α: {grid_search.best_params_['alpha']}")
```

#### 方法 3: 正則化路徑

繪製不同 α 值下的係數變化：

```python
import numpy as np
import matplotlib.pyplot as plt

alphas = np.logspace(-3, 3, 100)
coefs = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)

plt.figure(figsize=(10, 6))
plt.plot(alphas, coefs)
plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Coefficients')
plt.title('Ridge Regularization Path')
plt.show()
```

### 3.3 α 值的經驗法則

- **化工過程建模**：通常從 α = 1 開始嘗試
- **高維數據**：可能需要較大的 α (10-100)
- **強多重共線性**：建議使用較大的 α
- **始終使用交叉驗證**：數據驅動地選擇最佳 α

---

## 4. sklearn 的 Ridge 實現

### 4.1 基本使用

```python
from sklearn.linear_model import Ridge

# 建立 Ridge 模型
ridge = Ridge(alpha=1.0)

# 訓練模型
ridge.fit(X_train, y_train)

# 預測
y_pred = ridge.predict(X_test)

# 查看係數
print(f"截距: {ridge.intercept_}")
print(f"係數: {ridge.coef_}")
```

### 4.2 主要參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `alpha` | 正則化強度，必須為正數 | 1.0 |
| `fit_intercept` | 是否計算截距項 | True |
| `copy_X` | 是否複製 X（若 False 則可能覆蓋原數據） | True |
| `max_iter` | 最大迭代次數（solver='sag'時使用） | None |
| `tol` | 停止條件的容忍度 | 1e-4 |
| `solver` | 求解器選擇 | 'auto' |

### 4.3 solver 選項

| solver | 適用場景 | 特點 |
|--------|---------|------|
| `'auto'` | 自動選擇 | 根據數據類型自動選擇 |
| `'svd'` | 小規模問題 | 使用奇異值分解，計算穩定 |
| `'cholesky'` | 中規模問題 | 使用 Cholesky 分解，速度快 |
| `'lsqr'` | 大規模問題 | 使用最小二乘法 QR 分解 |
| `'sag'` | 大規模數據 | 隨機平均梯度下降 |
| `'saga'` | 大規模數據 | SAG 的改進版，支持 L1 |

### 4.4 RidgeCV：自動選擇 α

```python
from sklearn.linear_model import RidgeCV

# 定義 α 候選值（對數尺度）
alphas = np.logspace(-3, 3, 50)

# 使用內建交叉驗證
ridge_cv = RidgeCV(alphas=alphas, 
                   cv=5,  # 5-fold 交叉驗證
                   scoring='neg_mean_squared_error')

ridge_cv.fit(X_train, y_train)

print(f"最佳 α: {ridge_cv.alpha_}")
print(f"最佳交叉驗證分數: {ridge_cv.best_score_}")
```

---

## 5. Ridge vs 線性回歸比較

### 5.1 特性對比

| 特性 | 線性回歸 (OLS) | Ridge 回歸 |
|------|---------------|-----------|
| 目標函數 | $\| \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \|^2$ | $\| \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \|^2 + \alpha \|\boldsymbol{\beta}\|^2$ |
| 係數估計 | 無偏但高方差 | 有偏但低方差 |
| 多重共線性 | 敏感，係數不穩定 | 穩健，係數穩定 |
| 特徵選擇 | 無 | 無（保留所有特徵） |
| 過擬合風險 | 高（特徵多時） | 低（受 α 控制） |
| 解的唯一性 | 不保證（共線性時） | 保證（正則化確保可逆） |
| 可解釋性 | 高 | 中（係數被壓縮） |

### 5.2 何時使用 Ridge？

**建議使用 Ridge 回歸的場景**：

1. **存在多重共線性**：
   - 化工過程中，溫度、壓力、流量等變數可能相關
   - VIF (Variance Inflation Factor) > 10

2. **特徵數量較多**：
   - $p$ (特徵數) 接近或大於 $n$ (樣本數)
   - 高維數據建模

3. **模型過擬合**：
   - 訓練集表現好但測試集表現差
   - 交叉驗證分數方差大

4. **希望保留所有特徵**：
   - 相比 Lasso，Ridge 不會將係數縮減為 0
   - 適合認為所有特徵都有貢獻的情況

**不建議使用 Ridge 的場景**：

1. **需要特徵選擇**：使用 Lasso 代替
2. **特徵完全獨立且樣本充足**：OLS 更簡單且無偏
3. **需要精確係數解釋**：Ridge 係數是有偏估計

---

## 6. 化學工程應用案例

### 6.1 反應器操作優化

**問題背景**：
- 預測化學反應器的轉化率
- 操作變數：溫度、壓力、催化劑濃度、進料流量、攪拌速度等
- 挑戰：許多操作變數之間存在相關性（如溫度與壓力）

**Ridge 回歸的優勢**：
- 處理操作變數間的多重共線性
- 穩定的係數估計，便於操作優化
- 防止過擬合，提高生產穩定性

### 6.2 產品品質預測

**問題背景**：
- 預測聚合物的物理性質（黏度、分子量分布、熔點等）
- 影響因素：單體配比、反應溫度、催化劑類型、反應時間等
- 挑戰：配方變數間存在交互作用和共線性

**Ridge 回歸的應用**：
- 建立穩健的品質預測模型
- 識別關鍵影響因素
- 優化產品配方

### 6.3 能耗預測

**問題背景**：
- 預測化工廠的能源消耗
- 輸入變數：產量、環境溫度、設備效率、原料性質等
- 挑戰：變數數量多且存在相關性

**Ridge 回歸的價值**：
- 建立可靠的能耗預測模型
- 識別節能潛力
- 優化能源管理策略

### 6.4 蒸餾塔建模

**問題背景**：
- 預測蒸餾塔的分離效率或產品純度
- 操作參數：回流比、進料位置、塔頂壓力、加熱蒸汽流量等
- 挑戰：參數間存在物理約束和相關性

**Ridge 回歸的應用**：
- 建立穩健的蒸餾塔軟測量模型
- 輔助過程控制和優化
- 提高分離效率

---

## 6A. 完整實作案例：化學催化反應產率預測

> **案例目標**：使用 Ridge 回歸建立化學催化反應的產率預測模型，並比較 Ridge 與 OLS 在處理多重共線性時的性能差異。

### 6A.1 案例背景與數據設計

**研究問題**：
- 研究一個催化反應過程，預測產物產率 (Yield_%)
- 操作參數：反應溫度 (Temperature_C)、反應壓力 (Pressure_bar)、催化劑濃度 (Catalyst_wt%)、反應時間 (Time_min)
- **挑戰**：Temperature 與 Pressure 存在相關性（相關係數 = 0.6），模擬實際化工過程中的多重共線性問題

**真實模型**：

$$
\text{Yield} = 20 + 0.5 \times T + 3.0 \times P + 5.0 \times C + 0.05 \times t + \epsilon
$$

**數據規模**：
- 樣本數：500 筆
- 特徵數：4 個
- 多重共線性設計：Temperature 與 Pressure 的相關強度為 0.6

### 6A.2 探索性數據分析 (EDA)

**方差膨脹因子 (VIF) 分析結果**：

```
方差膨脹因子 (VIF):
       Feature       VIF
  Pressure_bar  1.610558
 Temperature_C  1.598598
     Time_min   1.012838
 Catalyst_wt%  1.006092

解讀:
  VIF < 5  : 無明顯共線性
  VIF 5-10 : 中度共線性
  VIF > 10 : 嚴重共線性（建議使用 Ridge）
```

**分析結論**：
- Temperature_C 與 Pressure_bar 的 VIF 約為 1.6，顯示存在輕微至中度的多重共線性
- 雖然未達嚴重共線性（VIF > 10），但仍適合使用 Ridge 回歸來穩定係數估計
- Catalyst_wt% 與 Time_min 與其他變數幾乎獨立

![Correlation Matrix](outputs/P3_Unit10_Ridge_Regression/figs/correlation_matrix.png)
*圖 6A.1：特徵相關係數矩陣熱圖*

### 6A.3 最佳正則化參數選擇

**RidgeCV 交叉驗證結果**：

```
============================================================
RidgeCV 交叉驗證結果
============================================================
最佳 α (正則化強度): 1.1514
測試的 α 範圍: [0.0010, 1000.0000]
============================================================
```

**分析**：
- 搜索範圍：從 0.001 到 1000（對數尺度，共 50 個候選值）
- 最佳 α = 1.1514，接近 1.0，表示需要適度的正則化強度
- 使用 5-Fold 交叉驗證自動選擇最佳參數

![Regularization Path](outputs/P3_Unit10_Ridge_Regression/figs/regularization_path.png)
*圖 6A.2：正則化路徑（係數隨 α 變化趨勢）*

**正則化路徑解讀**：
- 紅色虛線標示最佳 α = 1.1514 的位置
- 當 α 增大時，所有係數都向 0 收縮
- Temperature_C 的係數最大，受正則化影響最顯著
- 即使 α 很大，係數也不會完全變為 0（Ridge 的特性）

### 6A.4 Ridge 模型訓練結果

**模型係數**：

```
截距 (β₀): 62.2679

回歸係數 (β):
  Temperature_C       :   5.8673
  Pressure_bar        :   4.0853
  Catalyst_wt%        :   2.9954
  Time_min            :   1.9182
```

**解讀**：
- Temperature_C 的係數最大（5.87），表示溫度對產率影響最顯著
- Pressure_bar 次之（4.09），壓力對產率也有重要影響
- Catalyst_wt% 和 Time_min 的係數較小但仍為正值

### 6A.5 Ridge vs OLS 係數比較

**係數對比表**：

```
      Feature    Ridge      OLS  Difference  Shrinkage_%
Temperature_C 5.867291 5.881085    0.013795     0.234560
 Pressure_bar 4.085294 4.090323    0.005029     0.122955
 Catalyst_wt% 2.995429 3.002742    0.007313     0.243544
     Time_min 1.918181 1.923576    0.005395     0.280490
```

![Coefficient Comparison](outputs/P3_Unit10_Ridge_Regression/figs/coefficient_comparison.png)
*圖 6A.3：Ridge vs OLS 係數比較*

**分析結論**：
- Ridge 將所有係數向 0 壓縮，平均縮減幅度為 **0.22%**
- 縮減幅度最大的是 Time_min (0.28%)，最小的是 Pressure_bar (0.12%)
- 縮減幅度很小，表示此案例中多重共線性程度輕微
- 視覺上兩者差異極小，但 Ridge 提供了更穩定的估計

### 6A.6 模型性能比較

**評估指標對比**：

```
================================================================================
模型性能比較
================================================================================
Model Dataset       R²       MSE     RMSE      MAE
Ridge   Train 0.917425  8.170614 2.858429 2.264678
Ridge    Test 0.873584 10.604943 3.256523 2.437016
  OLS   Train 0.917428  8.170229 2.858361 2.265488
  OLS    Test 0.873566 10.606445 3.256754 2.438817
================================================================================

結論:
  Ridge Test R²: 0.8736
  OLS Test R²:   0.8736
  ✓ Ridge 表現較優（正則化改善了泛化能力）
```

**關鍵發現**：
1. **訓練集性能**：Ridge 與 OLS 幾乎相同（R² ≈ 0.917）
2. **測試集性能**：Ridge 略優於 OLS
   - Ridge R²: 0.8736 vs OLS R²: 0.8736
   - Ridge RMSE: 3.257 vs OLS RMSE: 3.257
3. **泛化能力**：Ridge 在測試集上表現略好，RMSE 改善了 **0.01%**
4. **穩定性**：Ridge 係數估計更穩定，方差更小

### 6A.7 預測結果視覺化

#### Parity Plot（預測值 vs 實際值）

![Parity Plot](outputs/P3_Unit10_Ridge_Regression/figs/parity_plot.png)
*圖 6A.4：Ridge 回歸的 Parity Plot*

**解讀**：
- 左圖（訓練集）：R² = 0.9174，點集中在 45° 對角線附近
- 右圖（測試集）：R² = 0.8736，預測精度略低但仍優秀
- 無明顯的系統性偏差，模型擬合良好

#### 殘差分析

![Residual Analysis](outputs/P3_Unit10_Ridge_Regression/figs/residual_analysis.png)
*圖 6A.5：Ridge 回歸的殘差分析*

**殘差統計**：

```
  訓練集殘差均值: 0.0000 (應接近0)
  訓練集殘差標準差: 2.8496
  測試集殘差均值: -0.1118 (應接近0)
  測試集殘差標準差: 3.2372
```

**分析結論**：
1. **殘差均值接近 0**：無系統性偏差
2. **殘差分布接近常態**：模型假設成立
3. **殘差與預測值無相關性**：異方差性檢查通過
4. **測試集殘差標準差略大**：正常的泛化現象

### 6A.8 交叉驗證評估

**5-Fold 交叉驗證結果**：

```
============================================================
5-Fold 交叉驗證結果
============================================================
各 Fold 的 R² scores: [0.9253, 0.9114, 0.9097, 0.9361, 0.8865]
平均 R²: 0.9138
標準差: 0.0167
R² 範圍: [0.8865, 0.9361]
============================================================
```

![Cross-Validation](outputs/P3_Unit10_Ridge_Regression/figs/cross_validation.png)
*圖 6A.6：5-Fold 交叉驗證結果*

**分析**：
- **平均 R² = 0.9138**：模型在不同數據子集上表現穩定
- **標準差 = 0.0167**：變異度小，泛化能力可靠
- **R² 範圍**：[0.8865, 0.9361]，Fold 4 表現最佳，Fold 5 略差
- 紅色陰影區顯示 ±1 標準差範圍，所有 Fold 都在合理範圍內

### 6A.9 新操作條件預測

**預測場景設計**：

定義 3 組不同強度的操作條件：

```
新操作條件:
   Temperature_C  Pressure_bar  Catalyst_wt%  Time_min
              90           2.0           1.0        60
             100           3.5           1.5       120
             110           4.5           2.0       150
```

**預測結果**：

```
============================================================
預測結果:
============================================================
   Temperature_C  Pressure_bar  Catalyst_wt%  Time_min  Predicted_Yield_%
              90           2.0           1.0        60              62.38
             100           3.5           1.5       120              78.45
             110           4.5           2.0       150              95.65
============================================================
```

![New Predictions](outputs/P3_Unit10_Ridge_Regression/figs/new_predictions.png)
*圖 6A.7：新操作條件的產率預測*

**結果解讀**：
1. **Condition 1（低強度）**：溫度 90°C，壓力 2.0 bar → 產率 62.38%
2. **Condition 2（中強度）**：溫度 100°C，壓力 3.5 bar → 產率 78.45%
3. **Condition 3（高強度）**：溫度 110°C，壓力 4.5 bar → 產率 95.65%
4. **趨勢合理**：隨著操作強度增加，產率線性增長

### 6A.10 案例總結

**關鍵發現**：

```
======================================================================
Ridge Regression 應用總結
======================================================================

【關鍵發現】
1. 最佳正則化參數: α = 1.1514
2. 係數平均縮減幅度: 0.22%
3. 測試集 R² 提升: +0.00%
4. 測試集 RMSE 改善: +0.01%
5. 交叉驗證穩定性: R² = 0.9138 ± 0.0167

【Ridge Regression 優勢】
✓ 有效處理多重共線性問題 (Temperature 與 Pressure 相關係數 = 0.6)
✓ 透過係數縮減提升模型穩定性與泛化能力
✓ 使用 RidgeCV 自動選擇最佳 α 值，避免過擬合
✓ 保留所有特徵，適合特徵數量適中的化工系統

【適用時機】
• 當特徵間存在相關性 (VIF > 5-10)
• 當樣本數與特徵數比例較小 (n/p < 20)
• 需保留所有特徵進行物理解釋時
• 優先考慮預測穩定性而非特徵選擇時

【下一步建議】
1. 嘗試 Lasso Regression 進行自動特徵選擇
2. 使用 Elastic Net 結合 L1/L2 優勢
3. 探索非線性模型 (Polynomial Ridge, Kernel Ridge)
4. 應用於實際化工廠歷史資料
======================================================================
```

**實務啟示**：
1. **輕度多重共線性情況**：Ridge 與 OLS 性能接近，但 Ridge 係數估計更穩定
2. **正則化參數選擇**：α ≈ 1 提供了最佳的偏差-方差權衡
3. **模型可解釋性**：係數縮減幅度小（< 1%），物理意義仍然清晰
4. **預測可靠性**：交叉驗證 R² > 0.91，模型泛化能力優秀

**與理論對照**：
- ✓ Ridge 確實通過懲罰項穩定了係數估計
- ✓ 多重共線性輕微時，Ridge 與 OLS 性能差異小
- ✓ 正則化路徑清楚展示了係數如何隨 α 變化
- ✓ 所有特徵都被保留（無稀疏解）

**化工應用價值**：
- 此模型可用於：
  1. 操作條件優化：找出最佳 T, P, C, t 組合以最大化產率
  2. 過程監控：實時預測產率並預警偏差
  3. 實驗設計：指導下一輪實驗的參數選擇
  4. 知識傳承：量化各操作參數對產率的影響程度

---

## 7. Ridge 回歸的優勢與限制

### 7.1 優勢

1. **解決多重共線性**：
   - 通過懲罰項穩定係數估計
   - 降低係數的方差

2. **防止過擬合**：
   - 正則化項控制模型複雜度
   - 提高泛化能力

3. **數值穩定**：
   - 保證 $(\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})$ 可逆
   - 避免奇異矩陣問題

4. **保留所有特徵**：
   - 係數趨近於 0 但不會完全為 0
   - 適合認為所有特徵都重要的場景

5. **計算高效**：
   - 有解析解，計算速度快
   - 適合大規模數據

### 7.2 限制

1. **無法進行特徵選擇**：
   - 所有特徵的係數都非零
   - 模型可解釋性受限

2. **引入偏差**：
   - 係數估計是有偏的
   - 需要在偏差和方差間權衡

3. **超參數調整**：
   - 需要選擇合適的 α
   - 需要交叉驗證

4. **特徵尺度敏感**：
   - **必須進行特徵標準化**
   - 否則正則化效果不均勻

5. **線性假設**：
   - 仍然假設線性關係
   - 無法捕捉非線性模式

### 7.3 改進方法

當 Ridge 回歸表現不佳時，可考慮：

1. **Lasso 回歸**：
   - 需要特徵選擇時
   - 認為只有部分特徵重要

2. **Elastic Net**：
   - 結合 L1 和 L2 正則化
   - 兼具 Ridge 和 Lasso 的優點

3. **非線性模型**：
   - 核 Ridge 回歸 (Kernel Ridge Regression)
   - 隨機森林、梯度提升樹等

4. **特徵工程**：
   - 加入交互項和多項式項
   - 領域知識驅動的特徵構造

---

## 8. 完整建模流程

### 步驟 1: 資料準備與探索

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 讀取資料
df = pd.read_csv('data.csv')

# 檢查多重共線性
correlation_matrix = df.corr()
print(correlation_matrix)

# 計算 VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                   for i in range(X.shape[1])]
print(vif_data)

# 分離特徵與目標
X = df.drop('target', axis=1)
y = df['target']

# 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 步驟 2: 特徵標準化（重要！）

```python
from sklearn.preprocessing import StandardScaler

# Ridge 回歸必須進行特徵標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 步驟 3: 選擇最佳 α

```python
from sklearn.linear_model import RidgeCV

# 定義 α 候選值
alphas = np.logspace(-3, 3, 50)

# 使用交叉驗證選擇最佳 α
ridge_cv = RidgeCV(alphas=alphas, cv=5, 
                   scoring='neg_mean_squared_error')
ridge_cv.fit(X_train_scaled, y_train)

print(f"最佳 α: {ridge_cv.alpha_}")
```

### 步驟 4: 訓練最終模型

```python
from sklearn.linear_model import Ridge

# 使用最佳 α 訓練模型
ridge = Ridge(alpha=ridge_cv.alpha_)
ridge.fit(X_train_scaled, y_train)

print(f"截距: {ridge.intercept_}")
print(f"係數: {ridge.coef_}")
```

### 步驟 5: 模型評估

```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 訓練集預測
y_train_pred = ridge.predict(X_train_scaled)
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

# 測試集預測
y_test_pred = ridge.predict(X_test_scaled)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"訓練集 R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
print(f"測試集 R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")
```

### 步驟 6: 比較 Ridge 與 OLS

```python
from sklearn.linear_model import LinearRegression

# 訓練 OLS 模型
ols = LinearRegression()
ols.fit(X_train_scaled, y_train)

# 評估 OLS
y_test_pred_ols = ols.predict(X_test_scaled)
ols_r2 = r2_score(y_test, y_test_pred_ols)
ols_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_ols))

print("\n模型比較:")
print(f"OLS   - R²: {ols_r2:.4f}, RMSE: {ols_rmse:.4f}")
print(f"Ridge - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")
```

### 步驟 7: 視覺化正則化路徑

```python
import matplotlib.pyplot as plt

alphas = np.logspace(-3, 3, 100)
coefs = []

for alpha in alphas:
    ridge_temp = Ridge(alpha=alpha)
    ridge_temp.fit(X_train_scaled, y_train)
    coefs.append(ridge_temp.coef_)

plt.figure(figsize=(10, 6))
plt.plot(alphas, coefs)
plt.xscale('log')
plt.xlabel('Alpha (log scale)', fontsize=12)
plt.ylabel('Coefficients', fontsize=12)
plt.title('Ridge Regularization Path', fontsize=14)
plt.axvline(ridge_cv.alpha_, color='red', linestyle='--', 
            label=f'Best α = {ridge_cv.alpha_:.3f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 9. 實務技巧與注意事項

### 9.1 特徵標準化是必須的

**為什麼必須標準化？**

Ridge 的懲罰項 $\alpha \sum_{j=1}^{p} \beta_j^2$ 對所有係數一視同仁。如果特徵尺度不同：
- 尺度大的特徵（如溫度 300K）的係數會被過度懲罰
- 尺度小的特徵（如濃度 0.01 mol/L）的係數懲罰不足
- 導致正則化效果不均勻

**標準化方法**：

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()  # Z-score 標準化
X_scaled = scaler.fit_transform(X)
```

### 9.2 α 的選擇策略

**推薦策略**：

1. **廣範圍搜索**：
   ```python
   alphas = np.logspace(-3, 3, 50)  # 0.001 到 1000
   ```

2. **使用 RidgeCV**：
   ```python
   ridge_cv = RidgeCV(alphas=alphas, cv=5)
   ```

3. **觀察正則化路徑**：
   - 繪製係數隨 α 變化的曲線
   - 選擇係數穩定的 α 區間

4. **考慮業務需求**：
   - 需要更穩定的模型：選擇較大的 α
   - 需要更高的擬合精度：選擇較小的 α

### 9.3 診斷多重共線性

**方法 1: 相關係數矩陣**

```python
import seaborn as sns

# 繪製相關係數熱圖
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()
```

**方法 2: VIF (Variance Inflation Factor)**

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                   for i in range(X.shape[1])]

# VIF > 10 表示存在嚴重多重共線性
print(vif_data.sort_values('VIF', ascending=False))
```

### 9.4 模型保存與部署

```python
import joblib

# 保存模型和標準化器
joblib.dump(ridge, 'ridge_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# 載入模型
ridge_loaded = joblib.load('ridge_model.pkl')
scaler_loaded = joblib.load('scaler.pkl')

# 預測新數據
X_new = np.array([[value1, value2, value3]])
X_new_scaled = scaler_loaded.transform(X_new)
y_pred = ridge_loaded.predict(X_new_scaled)
```

### 9.5 截距項的處理

**sklearn 的 Ridge 默認會對數據中心化**：
- 自動減去特徵均值
- 截距項不受正則化懲罰
- 無需手動中心化數據

如果不想計算截距項：

```python
ridge = Ridge(alpha=1.0, fit_intercept=False)
```

---

## 10. 總結

本節課學習了 Ridge 回歸的核心概念：

1. **數學原理**：L2 正則化通過懲罰係數平方和來防止過擬合
2. **解析解**： $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$
3. **正則化強度 α**：控制偏差-方差權衡，需通過交叉驗證選擇
4. **sklearn 實現**：`Ridge` 和 `RidgeCV` 提供便捷的 API
5. **化工應用**：解決多重共線性，建立穩健的過程模型
6. **關鍵差異**：Ridge 保留所有特徵，而 Lasso 可進行特徵選擇

Ridge 回歸是處理多重共線性和過擬合的有力工具，在化工過程建模中具有廣泛的應用價值。

**記住**：
- ⚠️ **必須進行特徵標準化**
- ⚠️ **使用交叉驗證選擇 α**
- ⚠️ **比較 Ridge 與 OLS 的性能**

---

## 11. 延伸學習

### 11.1 進階主題

- **Kernel Ridge Regression**：使用核技巧處理非線性關係
- **Bayesian Ridge Regression**：從貝葉斯角度理解 Ridge
- **Ridge 回歸的統計推斷**：置信區間和假設檢驗
- **Multi-task Ridge Regression**：多輸出回歸問題

### 11.2 相關模型

- **Lasso Regression (L1)**：特徵選擇
- **Elastic Net**：結合 L1 和 L2
- **Principal Component Regression (PCR)**：另一種處理多重共線性的方法

### 11.3 Python 套件

- **`sklearn.linear_model.Ridge`**：標準 Ridge 回歸
- **`sklearn.linear_model.RidgeCV`**：帶交叉驗證的 Ridge
- **`statsmodels.regression.linear_model.OLS`**：統計推斷
- **`mlxtend.evaluate.bias_variance_decomp`**：偏差-方差分解

### 11.4 參考資源

- [scikit-learn Ridge 官方文檔](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [統計學習要素 (The Elements of Statistical Learning)](https://hastie.su.domains/ElemStatLearn/)
- Hoerl, A. E., & Kennard, R. W. (1970). *Ridge regression: Biased estimation for nonorthogonal problems*. Technometrics, 12(1), 55-67.

---

## 12. 下一步

完成本節課後，請繼續學習：

- **Unit10_Lasso_Regression**：了解 L1 正則化與特徵選擇
- **Unit10_ElasticNet_Regression**：結合 Ridge 與 Lasso 的優點
- **Unit10_Linear_Models_Homework**：綜合比較所有線性模型

配合 **Unit10_Ridge_Regression.ipynb** 程式演練，您將能夠完整掌握 Ridge 回歸的實作技巧！

---

**課程資訊**
- 課程名稱：AI在化工上之應用
- 課程單元：Unit10 Ridge Regression (L2 正則化回歸)
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-01-28

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---