# Unit14 超參數調整 | Hyperparameter Tuning

**逢甲大學 化學工程學系**  
**課程名稱**: AI在化工上之應用  
**課程代碼**: CHE-AI-114  
**授課教師**: 莊曜禎 助理教授

---

## 📚 課程大綱 (Table of Contents)

1. [單元簡介](#1-單元簡介)
2. [超參數 vs 模型參數](#2-超參數-vs-模型參數)
3. [Grid Search 網格搜索](#3-grid-search-網格搜索)
4. [Random Search 隨機搜索](#4-random-search-隨機搜索)
5. [Bayesian Optimization 貝氏最佳化](#5-bayesian-optimization-貝氏最佳化)
6. [進階搜索技巧](#6-進階搜索技巧)
7. [搜索空間設計](#7-搜索空間設計)
8. [化工應用案例](#8-化工應用案例)
9. [總結與最佳實踐](#9-總結與最佳實踐)

---

## 1. 單元簡介

### 1.1 學習目標

超參數調整 (Hyperparameter Tuning) 是模型優化的關鍵步驟，直接影響模型性能的上限。

**核心學習目標**：
1. 理解超參數與模型參數的根本差異
2. 掌握三種主流超參數搜索方法的原理與實作
3. 學會設計合理的搜索空間，避免資源浪費
4. 應用進階技巧加速搜索過程
5. 在化工場景中選擇最適合的調參策略

### 1.2 為什麼超參數調整如此重要？

**案例對比**：Random Forest 預測反應器產率

| 超參數配置 | Test R² | 訓練時間 | 說明 |
|----------|---------|---------|------|
| 默認參數 | 0.78 | 2.3 秒 | `RandomForestRegressor()` |
| 手動調整 | 0.83 | 5.1 秒 | 憑經驗設定 n_estimators=200 |
| Grid Search | 0.87 | 45 秒 | 搜索 36 種組合 |
| Bayesian Opt | 0.88 | 18 秒 | 智能搜索 50 次 |

**關鍵發現**：
- ✅ 正確調參可提升 10-15% 性能
- ✅ Bayesian Optimization 效率遠高於 Grid Search
- ⚠️ 過度調參可能過擬合驗證集

### 1.3 化工領域的調參挑戰

| 挑戰 | 化工特點 | 影響 |
|------|---------|------|
| **小樣本** | 實驗成本高，數據量有限 | 易過擬合，需嚴格驗證 |
| **計算資源** | 生產環境硬體受限 | 無法嘗試過於複雜的模型 |
| **即時性需求** | 線上控制需快速推理 | 需平衡準確度與速度 |
| **多目標** | 產率、能耗、品質需兼顧 | 單一超參數組合難以滿足 |
| **領域知識** | 化學反應機制已知 | 可利用先驗知識縮小搜索空間 |

### 1.4 本單元架構

```
Unit14 超參數調整
│
├── 基礎概念
│   ├── 超參數 vs 模型參數
│   └── 調參策略總覽
│
├── 經典方法
│   ├── Grid Search (窮舉法)
│   ├── Random Search (隨機法)
│   └── 方法比較與選擇
│
├── 進階技巧
│   ├── Bayesian Optimization (智能搜索)
│   ├── Halving Search (加速技巧)
│   └── 多保真度優化
│
└── 實務應用
    ├── 搜索空間設計
    ├── 化工案例實戰
    └── 避坑指南
```

---

## 2. 超參數 vs 模型參數

### 2.1 核心差異

| 比較維度 | 模型參數 (Parameters) | 超參數 (Hyperparameters) |
|---------|---------------------|------------------------|
| **定義** | 模型從數據中學習的變量 | 訓練前人為設定的配置 |
| **學習方式** | 自動優化（梯度下降等） | 需要手動調整或搜索 |
| **數量** | 通常很多（百萬級） | 通常較少（個位數到十幾個） |
| **保存** | 保存在模型文件中 | 訓練腳本或配置文件 |
| **例子** | 線性回歸的係數 $w$ 和截距 $b$ | 學習率、正則化強度 $\alpha$ |

### 2.2 常見模型的超參數

#### 線性模型 (Ridge, Lasso)

```python
from sklearn.linear_model import Ridge

model = Ridge(
    alpha=1.0,           # 🔧 超參數：正則化強度
    fit_intercept=True,  # 🔧 超參數：是否擬合截距
    max_iter=1000        # 🔧 超參數：最大迭代次數
)

# 訓練後的模型參數
# model.coef_  → 📊 模型參數：特徵係數
# model.intercept_  → 📊 模型參數：截距
```

#### Random Forest

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,        # 🔧 超參數：樹的數量
    max_depth=10,            # 🔧 超參數：樹的最大深度
    min_samples_split=2,     # 🔧 超參數：分裂所需最小樣本數
    min_samples_leaf=1,      # 🔧 超參數：葉節點最小樣本數
    max_features='sqrt',     # 🔧 超參數：每次分裂考慮的特徵數
    random_state=42          # 🔧 超參數：隨機種子
)

# 訓練後的模型參數（隱藏在樹結構中）
# 每棵樹的分裂點、閾值等 → 📊 模型參數
```

#### Support Vector Machine

```python
from sklearn.svm import SVR

model = SVR(
    kernel='rbf',      # 🔧 超參數：核函數類型
    C=1.0,             # 🔧 超參數：懲罰參數
    epsilon=0.1,       # 🔧 超參數：ε-insensitive 參數
    gamma='scale'      # 🔧 超參數：RBF 核的寬度
)

# 訓練後的模型參數
# model.support_vectors_  → 📊 模型參數：支持向量
# model.dual_coef_  → 📊 模型參數：對偶係數
```

### 2.3 超參數的分類

#### 類型 1: 模型結構超參數

影響模型複雜度和表達能力。

| 模型 | 超參數 | 影響 |
|------|--------|------|
| Random Forest | `n_estimators`, `max_depth` | 樹的數量和深度 |
| Neural Network | 層數, 神經元數 | 網路容量 |
| Polynomial | `degree` | 多項式階數 |

**調參原則**：
- 過大 → 過擬合風險 ↑
- 過小 → 欠擬合風險 ↑
- 需通過驗證集找平衡點

#### 類型 2: 正則化超參數

控制模型複雜度懲罰，防止過擬合。

| 模型 | 超參數 | 作用 |
|------|--------|------|
| Ridge | `alpha` | L2 正則化強度 |
| Lasso | `alpha` | L1 正則化強度 |
| ElasticNet | `alpha`, `l1_ratio` | L1/L2 混合 |
| Random Forest | `min_samples_split` | 間接正則化 |

**調參原則**：
- `alpha` ↑ → 正則化 ↑ → 模型更簡單
- 小數據集需較強正則化

#### 類型 3: 優化超參數

影響訓練過程的收斂速度和穩定性。

| 超參數 | 作用 | 典型範圍 |
|--------|------|---------|
| `learning_rate` | 梯度下降步長 | 0.001 - 0.1 |
| `max_iter` | 最大迭代次數 | 100 - 10000 |
| `batch_size` | 批次大小 | 32 - 512 |
| `early_stopping` | 提前停止 | True/False |

#### 類型 4: 特定算法超參數

| 模型 | 超參數 | 說明 |
|------|--------|------|
| SVM | `kernel`, `gamma` | 核函數選擇與參數 |
| XGBoost | `subsample`, `colsample_bytree` | 採樣比例 |
| K-Means | `n_clusters` | 聚類數 |

### 2.4 化工案例：催化劑性能預測

**場景**：預測催化劑轉化率

**模型**：Random Forest Regressor

**超參數與其影響**：

```python
# 超參數組合 1: 默認
model_1 = RandomForestRegressor()
# 結果: Train R²=0.99, Val R²=0.72 → 過擬合

# 超參數組合 2: 減少複雜度
model_2 = RandomForestRegressor(
    n_estimators=50,        # 減少樹的數量
    max_depth=5,            # 限制深度
    min_samples_split=10    # 增加分裂門檻
)
# 結果: Train R²=0.85, Val R²=0.81 → 改善泛化 ✅

# 超參數組合 3: 過度簡化
model_3 = RandomForestRegressor(
    n_estimators=10,
    max_depth=2
)
# 結果: Train R²=0.68, Val R²=0.65 → 欠擬合
```

**結論**：超參數直接決定模型在 過擬合-最佳-欠擬合 譜系中的位置。

### 2.5 超參數調整的目標

**錯誤目標** ❌：
- 最大化訓練集性能
- 找到"最複雜"的模型

**正確目標** ✅：
- 最大化**驗證集**性能（泛化能力）
- 在性能與成本間取得平衡

**多目標考量**：

$$
\text{Score} = w_1 \cdot \text{Accuracy} - w_2 \cdot \text{Training Time} - w_3 \cdot \text{Model Size}
$$

化工實務中需要平衡：
- 預測準確度
- 訓練/推理速度
- 模型可解釋性
- 硬體資源消耗

---

## 3. Grid Search 網格搜索

### 3.1 原理

Grid Search（網格搜索）是一種**窮舉式**超參數搜索方法。

**核心概念**：
1. 為每個超參數定義候選值列表
2. 生成所有可能的超參數組合（笛卡爾積）
3. 對每個組合訓練模型並評估性能
4. 選擇性能最優的組合

**數學表示**：

給定超參數空間：

$$
\Theta = \{\theta_1, \theta_2, \ldots, \theta_k\}
$$

每個超參數的候選值：

$$
\theta_i \in \{v_i^1, v_i^2, \ldots, v_i^{n_i}\}
$$

總搜索次數：

$$
N_{\text{total}} = \prod_{i=1}^{k} n_i
$$

### 3.2 Sklearn 實作

#### 基本用法

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# 1. 定義模型
model = RandomForestRegressor(random_state=42)

# 2. 定義超參數搜索空間
param_grid = {
    'n_estimators': [50, 100, 200],           # 3 種選擇
    'max_depth': [5, 10, 15, None],           # 4 種選擇
    'min_samples_split': [2, 5, 10]           # 3 種選擇
}

# 總共: 3 × 4 × 3 = 36 種組合

# 3. 設定 GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,                      # 5-fold cross-validation
    scoring='r2',              # 評估指標
    n_jobs=-1,                 # 使用所有 CPU 核心
    verbose=2,                 # 顯示進度
    return_train_score=True    # 記錄訓練分數
)

# 4. 執行搜索
grid_search.fit(X_train, y_train)

# 5. 查看結果
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Best estimator: {grid_search.best_estimator_}")
```

#### 輸出解析

```
Fitting 5 folds for each of 36 candidates, totalling 180 fits
Best parameters: {'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 200}
Best CV score: 0.8534
```

**計算說明**：
- 36 種超參數組合
- 每種組合做 5-fold CV
- 總共訓練 36 × 5 = 180 個模型

### 3.3 搜索結果分析

#### 結果 DataFrame

```python
import pandas as pd

# 將搜索結果轉為 DataFrame
results_df = pd.DataFrame(grid_search.cv_results_)

# 選擇關鍵欄位
key_columns = [
    'param_n_estimators',
    'param_max_depth',
    'param_min_samples_split',
    'mean_test_score',
    'std_test_score',
    'mean_fit_time',
    'rank_test_score'
]

results_summary = results_df[key_columns].sort_values(
    'rank_test_score'
)

print(results_summary.head(10))
```

**輸出示例**：

| n_estimators | max_depth | min_samples_split | mean_test_score | std_test_score | mean_fit_time | rank |
|--------------|-----------|-------------------|-----------------|----------------|---------------|------|
| 200 | 10 | 5 | 0.8534 | 0.0234 | 3.45 | 1 |
| 200 | 15 | 5 | 0.8512 | 0.0256 | 4.12 | 2 |
| 100 | 10 | 5 | 0.8489 | 0.0241 | 1.78 | 3 |

#### 視覺化：Heatmap

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 固定 n_estimators=200，繪製 max_depth vs min_samples_split 熱圖
pivot_data = results_df[
    results_df['param_n_estimators'] == 200
].pivot(
    index='param_max_depth',
    columns='param_min_samples_split',
    values='mean_test_score'
)

plt.figure(figsize=(8, 6))
sns.heatmap(
    pivot_data,
    annot=True,
    fmt='.3f',
    cmap='viridis',
    cbar_kws={'label': 'Mean Test R²'}
)
plt.title('Grid Search Results: max_depth vs min_samples_split\n(n_estimators=200)')
plt.xlabel('min_samples_split')
plt.ylabel('max_depth')
plt.tight_layout()
plt.show()
```

### 3.4 優點與缺點

#### ✅ 優點

1. **完整性**：保證找到搜索空間內的最優組合
2. **可並行化**：不同組合可同時訓練（`n_jobs=-1`）
3. **簡單直觀**：易於理解和實作
4. **可重現**：給定搜索空間，結果唯一

#### ❌ 缺點

1. **計算成本高**：組合數呈指數增長
   
   $$
   N = n_1 \times n_2 \times \cdots \times n_k
   $$
   
   例如：10 個超參數，每個 5 種選擇 → $5^{10} = 9,765,625$ 種組合

2. **維度詛咒**：超參數越多，搜索越不現實

3. **離散化損失**：連續超參數需要離散化，可能錯過最優值

4. **資源浪費**：在不重要的超參數上也會耗費相同資源

### 3.5 Grid Search vs Manual Tuning

**手動調參**：

```python
# 工程師憑經驗嘗試
試驗 1: n_estimators=100, max_depth=10  → R²=0.82
試驗 2: n_estimators=200, max_depth=10  → R²=0.84
試驗 3: n_estimators=200, max_depth=15  → R²=0.83
# 停止搜索，採用試驗 2
```

**Grid Search**：

```python
# 系統化搜索
搜索空間: 
    n_estimators=[50, 100, 150, 200, 250]
    max_depth=[5, 10, 15, 20, None]
結果: n_estimators=250, max_depth=12 → R²=0.87
```

**差異總結**：

| 方法 | 搜索次數 | 最優解保證 | 人力成本 | 適用場景 |
|------|---------|-----------|---------|---------|
| 手動調參 | 3-10 次 | ❌ 不保證 | 高 | 快速驗證 |
| Grid Search | 25-100 次 | ✅ 空間內最優 | 低 | 系統化優化 |

### 3.6 化工案例：蒸餾塔溫度預測

**問題**：使用 SVR 預測蒸餾塔頂溫度

**超參數搜索空間**：

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 建立 Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR())
])

# Grid Search 配置
param_grid = {
    'svr__C': [0.1, 1, 10, 100],              # 懲罰參數
    'svr__epsilon': [0.01, 0.1, 0.5],         # ε-insensitive
    'svr__gamma': ['scale', 'auto', 0.1, 1]   # RBF 核寬度
}

# 總共: 4 × 3 × 4 = 48 種組合

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',  # MAE (化工更關注絕對誤差)
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
```

**結果分析**：

```python
print(f"Best parameters: {grid_search.best_params_}")
# Best parameters: {'svr__C': 10, 'svr__epsilon': 0.1, 'svr__gamma': 'scale'}

print(f"Best MAE: {-grid_search.best_score_:.3f} °C")
# Best MAE: 1.234 °C

# 在測試集上評估
y_pred = grid_search.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
print(f"Test MAE: {test_mae:.3f} °C")
# Test MAE: 1.187 °C (滿足 ±2°C 控制要求)
```

**化工意義**：
- MAE < 2°C：滿足工業控制精度
- `C=10`：適度懲罰，平衡擬合與泛化
- `epsilon=0.1`：容忍 0.1°C 誤差（合理的測量不確定度）

### 3.7 實務技巧

#### 技巧 1: 粗搜 → 精搜

```python
# 第一輪：粗搜索
param_grid_coarse = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [5, 10, 20, None]
}
# 結果: n_estimators=200, max_depth=10 最優

# 第二輪：在最優區域細搜
param_grid_fine = {
    'n_estimators': [150, 175, 200, 225, 250],
    'max_depth': [8, 9, 10, 11, 12]
}
```

#### 技巧 2: 優先調整重要超參數

**重要性排序**（Random Forest 為例）：
1. `n_estimators`, `max_depth` → 影響最大
2. `min_samples_split`, `min_samples_leaf` → 中等影響
3. `max_features` → 較小影響
4. `bootstrap`, `oob_score` → 邊際影響

**策略**：先固定次要參數，只搜索關鍵參數。

#### 技巧 3: 使用對數尺度

對於範圍跨度大的超參數（如學習率、正則化強度），使用對數刻度：

```python
import numpy as np

# ❌ 線性刻度（不推薦）
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}

# ✅ 對數刻度（推薦）
param_grid = {
    'alpha': np.logspace(-3, 2, 6)  # [0.001, 0.01, 0.1, 1, 10, 100]
}
```

**原因**：`alpha` 從 0.001 → 0.01 的變化，與 1 → 10 的變化，對模型影響相似。

#### 技巧 4: 監控訓練-驗證差距

```python
results_df = pd.DataFrame(grid_search.cv_results_)

# 計算過擬合程度
results_df['overfit_gap'] = (
    results_df['mean_train_score'] - results_df['mean_test_score']
)

# 找出過擬合最少且性能好的組合
results_df['score_adjusted'] = (
    results_df['mean_test_score'] - 0.1 * results_df['overfit_gap']
)

best_idx = results_df['score_adjusted'].idxmax()
print(results_df.loc[best_idx, ['params', 'mean_test_score', 'overfit_gap']])
```

### 3.8 何時使用 Grid Search？

| 場景 | 是否適用 | 原因 |
|------|---------|------|
| 超參數 ≤ 3 個 | ✅ 適用 | 搜索空間可控 |
| 超參數 ≥ 5 個 | ❌ 不適用 | 組合爆炸 |
| 已知大致最優區域 | ✅ 適用 | 精細搜索 |
| 完全未知 | ❌ 不適用 | Random Search 更高效 |
| 計算資源充足 | ✅ 適用 | 可並行化 |
| 小數據集 | ✅ 適用 | 訓練快速 |

**推薦場景**：
- 模型訓練時間 < 1 分鐘
- 超參數數量 ≤ 4 個
- 需要完整掃描某個二維超參數空間

---

## 4. Random Search 隨機搜索

### 4.1 原理

Random Search（隨機搜索）從超參數空間中**隨機抽樣**，而非窮舉所有組合。

**核心概念**：
1. 定義每個超參數的**分布**（而非離散值列表）
2. 從分布中隨機抽樣 $N$ 次
3. 訓練並評估每個隨機組合
4. 選擇性能最優的組合

**數學表示**：

給定超參數分布：

$$
\theta_i \sim p_i(\theta_i)
$$

隨機抽樣 $N$ 次：

$$
\{\boldsymbol{\theta}^{(1)}, \boldsymbol{\theta}^{(2)}, \ldots, \boldsymbol{\theta}^{(N)}\} \sim P(\boldsymbol{\Theta})
$$

**與 Grid Search 的關鍵區別**：
- Grid Search: 離散化 + 窮舉
- Random Search: 連續分布 + 隨機抽樣

### 4.2 為什麼 Random Search 更高效？

#### 理論基礎

**Bergstra & Bengio (2012)** 研究指出：

> "在高維空間中，往往只有少數幾個超參數真正重要，Random Search 能更高效地探索這些關鍵維度。"

**視覺化說明**：

假設有 2 個超參數，但只有 $\theta_1$ 重要：

```
Grid Search (9 次嘗試):
    θ₂ │ × × ×
       │ × × ×
       │ × × ×
       └────────── θ₁
       只有 3 種不同的 θ₁ 值

Random Search (9 次嘗試):
    θ₂ │   ×  ×
       │ ×   × ×
       │ × ×   ×
       └────────── θ₁
       有 9 種不同的 θ₁ 值 → 更好的覆蓋
```

**效率比較**：

| 超參數數 | Grid (每維 5 值) | Random (同樣次數) | Random 優勢 |
|---------|-----------------|-----------------|-----------|
| 2 | 25 次 | 25 次 | 相當 |
| 3 | 125 次 | 25 次 | **5倍** |
| 4 | 625 次 | 25 次 | **25倍** |
| 5 | 3125 次 | 25 次 | **125倍** |

### 4.3 Sklearn 實作

#### 基本用法

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint, uniform
import numpy as np

# 1. 定義模型
model = RandomForestRegressor(random_state=42)

# 2. 定義超參數分布
param_distributions = {
    'n_estimators': randint(50, 500),           # 整數均勻分布 [50, 500)
    'max_depth': [5, 10, 15, 20, None],         # 離散選擇
    'min_samples_split': randint(2, 20),        # 整數均勻分布 [2, 20)
    'min_samples_leaf': randint(1, 10),         # 整數均勻分布 [1, 10)
    'max_features': uniform(0.1, 0.9)           # 連續均勻分布 [0.1, 1.0)
}

# 3. 設定 RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=100,            # 隨機抽樣 100 次
    cv=5,                  # 5-fold cross-validation
    scoring='r2',
    n_jobs=-1,
    verbose=2,
    random_state=42,       # 可重現性
    return_train_score=True
)

# 4. 執行搜索
random_search.fit(X_train, y_train)

# 5. 查看結果
print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.4f}")
```

**輸出示例**：

```
Fitting 5 folds for each of 100 candidates, totalling 500 fits
Best parameters: {'max_depth': 15, 'max_features': 0.6234, 
                  'min_samples_leaf': 2, 'min_samples_split': 5, 
                  'n_estimators': 347}
Best CV score: 0.8612
```

### 4.4 分布類型

#### 常用分布 (scipy.stats)

```python
from scipy.stats import randint, uniform, loguniform

param_distributions = {
    # 1. 整數均勻分布
    'n_estimators': randint(50, 500),          # [50, 499]
    
    # 2. 連續均勻分布
    'max_features': uniform(0.1, 0.9),         # [0.1, 1.0)
    
    # 3. 對數均勻分布
    'learning_rate': loguniform(1e-4, 1e-1),   # [0.0001, 0.1]
    
    # 4. 離散選擇
    'criterion': ['gini', 'entropy'],
    
    # 5. 對數整數分布（自定義）
    'max_depth': [2**i for i in range(1, 8)]  # [2, 4, 8, 16, 32, 64, 128]
}
```

#### 對數分布的重要性

對於學習率、正則化強度等跨度大的超參數，**必須使用對數分布**：

```python
# ❌ 錯誤：線性分布
'alpha': uniform(0.0001, 100)
# 90% 的採樣點集中在 [10, 100]，小值區域探索不足

# ✅ 正確：對數分布
'alpha': loguniform(1e-4, 1e2)
# 均勻探索 [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
```

### 4.5 Random Search vs Grid Search

#### 實驗對比

**設定**：Random Forest，4 個超參數

```python
# Grid Search (4×4×4×4 = 256 組合)
param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'max_features': [0.3, 0.5, 0.7, 'sqrt']
}

# Random Search (100 次隨機抽樣)
param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': randint(2, 20),
    'max_features': uniform(0.1, 0.9)
}
```

**結果**：

| 方法 | 搜索次數 | 最優 R² | 訓練時間 | 覆蓋範圍 |
|------|---------|---------|---------|---------|
| Grid Search | 256 | 0.8534 | 45 秒 | 256 種組合 |
| Random Search | 100 | 0.8612 | 18 秒 | 探索更廣 |

**結論**：Random Search 用更少的嘗試，找到更好的結果。

#### 選擇決策樹

```
開始
  │
  ├─ 超參數 ≤ 3 個？
  │   ├─ 是 → Grid Search（完整掃描）
  │   └─ 否 → 繼續
  │
  ├─ 已知大致最優區域？
  │   ├─ 是 → Grid Search（精細搜索）
  │   └─ 否 → Random Search（廣泛探索）
  │
  └─ 計算預算 < 50 次？
      ├─ 是 → Random Search（高效採樣）
      └─ 否 → Bayesian Optimization（智能搜索）
```

### 4.6 化工案例：反應動力學參數估計

**問題**：使用 XGBoost 預測反應轉化率

**挑戰**：
- XGBoost 有 10+ 個重要超參數
- Grid Search 不現實（$5^{10} = 9,765,625$ 組合）

**Random Search 方案**：

```python
import xgboost as xgb
from scipy.stats import uniform, randint, loguniform

# 定義超參數分布
param_distributions = {
    # 樹結構
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 15),
    'min_child_weight': randint(1, 10),
    
    # 採樣
    'subsample': uniform(0.5, 0.5),              # [0.5, 1.0]
    'colsample_bytree': uniform(0.5, 0.5),       # [0.5, 1.0]
    
    # 學習率
    'learning_rate': loguniform(1e-3, 1e-1),     # [0.001, 0.1]
    
    # 正則化
    'gamma': uniform(0, 5),
    'reg_alpha': loguniform(1e-3, 10),           # L1
    'reg_lambda': loguniform(1e-3, 10)           # L2
}

# Random Search
random_search = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=42, tree_method='hist'),
    param_distributions,
    n_iter=200,        # 200 次隨機嘗試
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)
```

**結果分析**：

```python
print(f"Best parameters: {random_search.best_params_}")
# Best parameters: {
#     'colsample_bytree': 0.7234, 'gamma': 2.1456,
#     'learning_rate': 0.0234, 'max_depth': 7,
#     'min_child_weight': 3, 'n_estimators': 456,
#     'reg_alpha': 0.1234, 'reg_lambda': 1.2345,
#     'subsample': 0.8123
# }

print(f"Best RMSE: {np.sqrt(-random_search.best_score_):.4f}")
# Best RMSE: 0.0234 (轉化率單位)

# 測試集評估
y_pred = random_search.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {test_rmse:.4f}")
# Test RMSE: 0.0256
```

**化工意義**：
- RMSE < 3%：滿足工藝預測要求
- `learning_rate=0.0234`：避免過擬合小數據集
- `max_depth=7`：捕捉反應機制的非線性

### 4.7 進階技巧

#### 技巧 1: 預算分配策略

```python
# 第一階段：粗搜索（廣度）
random_search_broad = RandomizedSearchCV(
    model, param_distributions,
    n_iter=50,         # 50 次粗搜
    cv=3,              # 3-fold（節省時間）
    n_jobs=-1
)
random_search_broad.fit(X_train, y_train)

# 第二階段：在最優區域精搜（深度）
best_params = random_search_broad.best_params_

# 縮小搜索空間
param_distributions_refined = {
    'n_estimators': randint(
        int(best_params['n_estimators'] * 0.8),
        int(best_params['n_estimators'] * 1.2)
    ),
    'max_depth': [best_params['max_depth'] - 1,
                  best_params['max_depth'],
                  best_params['max_depth'] + 1]
}

random_search_refined = RandomizedSearchCV(
    model, param_distributions_refined,
    n_iter=30,         # 30 次精搜
    cv=5,              # 5-fold（更準確）
    n_jobs=-1
)
random_search_refined.fit(X_train, y_train)
```

#### 技巧 2: 監控搜索過程

```python
import pandas as pd
import matplotlib.pyplot as plt

# 取得搜索歷史
results_df = pd.DataFrame(random_search.cv_results_)

# 繪製搜索軌跡
plt.figure(figsize=(10, 6))
plt.scatter(
    range(len(results_df)),
    results_df['mean_test_score'],
    c=results_df['mean_fit_time'],
    cmap='viridis',
    alpha=0.6
)
plt.colorbar(label='Training Time (s)')
plt.axhline(results_df['mean_test_score'].max(), 
            color='r', linestyle='--', label='Best Score')
plt.xlabel('Iteration')
plt.ylabel('CV Score (R²)')
plt.title('Random Search Progress')
plt.legend()
plt.tight_layout()
plt.show()
```

#### 技巧 3: 多目標優化

在化工場景中，需平衡**準確度**與**計算成本**：

```python
results_df = pd.DataFrame(random_search.cv_results_)

# 定義綜合評分
results_df['composite_score'] = (
    0.7 * results_df['mean_test_score'] +           # 70% 權重：準確度
    0.2 * (1 - results_df['mean_fit_time'] / 
           results_df['mean_fit_time'].max()) +     # 20% 權重：速度
    0.1 * (1 - results_df['param_n_estimators'] / 
           results_df['param_n_estimators'].max())  # 10% 權重：模型複雜度
)

# 找出綜合最優
best_idx = results_df['composite_score'].idxmax()
best_balanced_params = results_df.loc[best_idx, 'params']

print(f"Balanced best params: {best_balanced_params}")
print(f"Score: {results_df.loc[best_idx, 'mean_test_score']:.4f}")
print(f"Time: {results_df.loc[best_idx, 'mean_fit_time']:.2f}s")
```

### 4.8 實務建議

#### n_iter 如何設定？

| 超參數數量 | 建議 n_iter | 理由 |
|-----------|------------|------|
| 1-2 | 20-50 | 低維空間，快速覆蓋 |
| 3-5 | 50-100 | 中維空間，充分探索 |
| 6-10 | 100-300 | 高維空間，需更多採樣 |
| 10+ | 200-500 | 考慮 Bayesian Optimization |

**經驗法則**：

$$
n_{\text{iter}} \approx 10 \times k^{1.5}
$$

其中 $k$ 是超參數數量。

#### 何時使用 Random Search？

| 場景 | 推薦度 | 原因 |
|------|-------|------|
| 超參數 ≥ 4 個 | ⭐⭐⭐⭐⭐ | 效率遠超 Grid Search |
| 連續超參數為主 | ⭐⭐⭐⭐⭐ | 充分利用分布採樣 |
| 計算預算有限 | ⭐⭐⭐⭐⭐ | 可控制 n_iter |
| 不確定最優區域 | ⭐⭐⭐⭐ | 廣泛探索 |
| 需要完整掃描 | ⭐⭐ | Grid Search 更適合 |

### 4.9 小結

**Random Search 的核心優勢**：
1. ✅ 高維空間下效率遠超 Grid Search
2. ✅ 支持連續分布，不需離散化
3. ✅ 可控制計算預算（`n_iter`）
4. ✅ 更好地探索重要超參數

**適用場景**：
- 超參數數量 ≥ 4
- 完全不確定最優區域
- 計算資源有限
- 需要快速得到"足夠好"的結果

**下一步**：當 Random Search 仍需太多次嘗試時，考慮更智能的 Bayesian Optimization。

---

## 5. Bayesian Optimization 貝氏最佳化

### 5.1 原理

Bayesian Optimization（貝氏最佳化）是一種**智能搜索**方法，利用前面的嘗試結果，來指導後續的探索。

**核心思想**：
1. 建立目標函數的**機率模型**（通常是 Gaussian Process）
2. 利用歷史觀測更新模型
3. 根據模型預測，選擇**最有希望**的下一個點
4. 平衡**探索**（Exploration）與**利用**（Exploitation）

**與前兩種方法的比較**：

| 方法 | 搜索策略 | 智能程度 | 適用場景 |
|------|---------|---------|---------|
| Grid Search | 窮舉所有組合 | 無智能 | 低維 + 充足資源 |
| Random Search | 隨機採樣 | 低智能 | 中維 + 有限資源 |
| Bayesian Opt | 利用歷史信息 | 高智能 | 高維 + 昂貴評估 |

### 5.2 數學框架

#### 目標函數

我們要優化的是超參數 $\boldsymbol{\theta}$ 對模型性能 $f(\boldsymbol{\theta})$ 的影響：

$$
\boldsymbol{\theta}^* = \arg\max_{\boldsymbol{\theta}} f(\boldsymbol{\theta})
$$

其中 $f(\boldsymbol{\theta})$ 是**黑盒函數**（評估成本高，如需訓練模型）。

#### 代理模型 (Surrogate Model)

使用 Gaussian Process 建立 $f$ 的機率模型：

$$
f(\boldsymbol{\theta}) \sim \mathcal{GP}(\mu(\boldsymbol{\theta}), k(\boldsymbol{\theta}, \boldsymbol{\theta}'))
$$

給定歷史觀測 $\mathcal{D} = \{(\boldsymbol{\theta}_i, f_i)\}_{i=1}^{t}$，可預測新點的均值和不確定度：

$$
\mu_t(\boldsymbol{\theta}) = \mathbb{E}[f(\boldsymbol{\theta}) \mid \mathcal{D}]
$$

$$
\sigma_t^2(\boldsymbol{\theta}) = \text{Var}[f(\boldsymbol{\theta}) \mid \mathcal{D}]
$$

#### 獲取函數 (Acquisition Function)

決定下一個採樣點，常用的有：

**1. Expected Improvement (EI)**：

$$
\text{EI}(\boldsymbol{\theta}) = \mathbb{E}[\max(f(\boldsymbol{\theta}) - f^*, 0)]
$$

其中 $f^*$ 是當前最優值。

**2. Upper Confidence Bound (UCB)**：

$$
\text{UCB}(\boldsymbol{\theta}) = \mu_t(\boldsymbol{\theta}) + \kappa \sigma_t(\boldsymbol{\theta})
$$

$\kappa$ 控制探索-利用平衡。

### 5.3 演算法流程

```
1. 隨機初始化 n₀ 個點，評估 f(θ)
2. For t = n₀+1 to T:
   a. 用 {θᵢ, fᵢ} 訓練 Gaussian Process
   b. 計算 Acquisition Function: α(θ)
   c. 找到最大化 α 的點: θₜ = argmax α(θ)
   d. 評估 fₜ = f(θₜ)
   e. 更新歷史數據: D ← D ∪ {(θₜ, fₜ)}
3. 返回 θ* = argmax fᵢ
```

### 5.4 Optuna 實作

Optuna 是目前最流行的 Bayesian Optimization 框架。

#### 安裝

```bash
pip install optuna
```

#### 基本範例

```python
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

# 1. 定義目標函數
def objective(trial):
    # 定義超參數搜索空間
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0)
    }
    
    # 訓練模型
    model = RandomForestRegressor(**params, random_state=42, n_jobs=1)
    
    # 交叉驗證
    scores = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    
    return scores.mean()

# 2. 創建研究對象
study = optuna.create_study(
    direction='maximize',         # 最大化 R²
    sampler=optuna.samplers.TPESampler(seed=42),  # TPE 採樣器
    pruner=optuna.pruners.MedianPruner()          # 中位數剪枝器
)

# 3. 執行優化
study.optimize(objective, n_trials=100, n_jobs=1)

# 4. 查看結果
print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

**輸出示例**：

```
[I 2024-01-15 10:30:45,123] Trial 0 finished with value: 0.8234
[I 2024-01-15 10:31:12,456] Trial 1 finished with value: 0.8456
...
[I 2024-01-15 10:45:33,789] Trial 99 finished with value: 0.8512

Best trial: 67
Best value: 0.8734
Best params: {'max_depth': 12, 'max_features': 0.6234, 
              'min_samples_leaf': 2, 'min_samples_split': 5, 
              'n_estimators': 347}
```

### 5.5 Optuna 進階功能

#### 視覺化優化過程

```python
import optuna.visualization as vis

# 1. 優化歷史
fig = vis.plot_optimization_history(study)
fig.show()

# 2. 參數重要性
fig = vis.plot_param_importances(study)
fig.show()

# 3. 平行坐標圖
fig = vis.plot_parallel_coordinate(study)
fig.show()

# 4. 切片圖（單個超參數的影響）
fig = vis.plot_slice(study)
fig.show()

# 5. 等高線圖（兩個超參數的交互）
fig = vis.plot_contour(study, params=['n_estimators', 'max_depth'])
fig.show()
```

#### 剪枝 (Pruning)

提前終止不promising的trial，節省時間：

```python
def objective_with_pruning(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20)
    }
    
    model = RandomForestRegressor(**params, random_state=42)
    
    # 逐個 fold 評估，允許剪枝
    for fold in range(5):
        train_idx, val_idx = kfold_splits[fold]
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model.fit(X_tr, y_tr)
        score = model.score(X_val, y_val)
        
        # 報告中間結果
        trial.report(score, fold)
        
        # 如果表現不佳，提前終止
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return score

study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=10,    # 前 10 個 trial 不剪枝
        n_warmup_steps=2        # 前 2 個 fold 不剪枝
    )
)

study.optimize(objective_with_pruning, n_trials=100)
```

#### 多目標優化

```python
def multi_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20)
    }
    
    model = RandomForestRegressor(**params, random_state=42)
    
    # 目標 1: 最大化 R²
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    r2_mean = scores.mean()
    
    # 目標 2: 最小化推理時間
    model.fit(X_train, y_train)
    start = time.time()
    _ = model.predict(X_test)
    inference_time = time.time() - start
    
    return r2_mean, inference_time  # 返回多個目標

# 多目標優化
study = optuna.create_study(
    directions=['maximize', 'minimize']  # 第一個最大化，第二個最小化
)

study.optimize(multi_objective, n_trials=100)

# 查看 Pareto Front
print("Pareto-optimal trials:")
for trial in study.best_trials:
    print(f"Trial {trial.number}: R²={trial.values[0]:.4f}, "
          f"Time={trial.values[1]:.4f}s")
```

### 5.6 Hyperopt 替代方案

Hyperopt 是另一個流行的 Bayesian Optimization 框架。

#### 安裝與基本用法

```bash
pip install hyperopt
```

```python
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# 1. 定義搜索空間
space = {
    'n_estimators': hp.quniform('n_estimators', 50, 500, 1),
    'max_depth': hp.quniform('max_depth', 3, 20, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
    'max_features': hp.uniform('max_features', 0.1, 1.0)
}

# 2. 定義目標函數
def objective(params):
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])
    params['min_samples_split'] = int(params['min_samples_split'])
    
    model = RandomForestRegressor(**params, random_state=42, n_jobs=1)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    # Hyperopt 最小化目標，所以取負號
    return {'loss': -scores.mean(), 'status': STATUS_OK}

# 3. 執行優化
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,      # Tree-structured Parzen Estimator
    max_evals=100,
    trials=trials,
    rstate=np.random.default_rng(42)
)

print(f"Best params: {best}")
print(f"Best score: {-trials.best_trial['result']['loss']:.4f}")
```

### 5.7 化工案例：催化劑配方優化

**問題**：優化催化劑組成以最大化轉化率

**挑戰**：
- 實驗成本極高（每次 > 10,000 元）
- 只能進行有限次實驗（< 50 次）
- 需要智能搜索策略

**Bayesian Optimization 方案**：

```python
import optuna
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

# 催化劑配方參數
def objective(trial):
    # 催化劑組成 (mol%)
    metal_loading = trial.suggest_float('metal_loading', 0.5, 5.0)  # 金屬負載量
    promoter_ratio = trial.suggest_float('promoter_ratio', 0.0, 0.5)  # 助劑比例
    calcination_temp = trial.suggest_int('calcination_temp', 400, 800)  # 焙燒溫度 (°C)
    
    # 反應條件
    reaction_temp = trial.suggest_int('reaction_temp', 200, 400)  # 反應溫度 (°C)
    pressure = trial.suggest_float('pressure', 1.0, 10.0)  # 壓力 (bar)
    
    # 建立特徵向量
    X_new = np.array([[
        metal_loading, promoter_ratio, calcination_temp,
        reaction_temp, pressure
    ]])
    
    # 使用已訓練的模型預測轉化率
    predicted_conversion = model.predict(X_new)[0]
    
    return predicted_conversion

# 創建研究對象
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=10  # 前 10 次隨機探索
    )
)

# 執行優化（模擬 50 次實驗）
study.optimize(objective, n_trials=50)

# 最優配方
print("=== 最優催化劑配方 ===")
print(f"金屬負載量: {study.best_params['metal_loading']:.2f} mol%")
print(f"助劑比例: {study.best_params['promoter_ratio']:.2f}")
print(f"焙燒溫度: {study.best_params['calcination_temp']} °C")
print(f"反應溫度: {study.best_params['reaction_temp']} °C")
print(f"壓力: {study.best_params['pressure']:.2f} bar")
print(f"預測轉化率: {study.best_value:.2f}%")
```

**結果分析**：

```python
# 參數重要性分析
importance = optuna.importance.get_param_importances(study)
print("\n=== 參數重要性排序 ===")
for param, imp in importance.items():
    print(f"{param}: {imp:.3f}")

# 輸出示例:
# reaction_temp: 0.456  ← 最關鍵
# metal_loading: 0.289
# pressure: 0.158
# calcination_temp: 0.067
# promoter_ratio: 0.030  ← 影響最小
```

**化工意義**：
- 反應溫度是最關鍵參數（與化學動力學一致）
- 只需 50 次嘗試，找到接近最優的配方
- 相比隨機搜索，節省 60% 實驗成本

### 5.8 Bayesian Optimization vs Random Search

#### 效率對比實驗

**設定**：Random Forest，6 個超參數，目標 R² > 0.90

| 方法 | 達到目標的嘗試次數 | 總時間 | 最優 R² |
|------|------------------|--------|---------|
| Random Search | 78 次 | 45 分鐘 | 0.9012 |
| Bayesian Opt (Optuna) | 32 次 | 20 分鐘 | 0.9034 |

**結論**：Bayesian Optimization 在相同預算下，找到更好的解。

#### 收斂曲線

```python
import matplotlib.pyplot as plt

# Random Search 結果
random_scores = [隨機試驗的分數列表]
random_best = np.maximum.accumulate(random_scores)

# Bayesian Optimization 結果
bayesian_scores = [trial.value for trial in study.trials]
bayesian_best = np.maximum.accumulate(bayesian_scores)

plt.figure(figsize=(10, 6))
plt.plot(random_best, label='Random Search', alpha=0.7)
plt.plot(bayesian_best, label='Bayesian Optimization', linewidth=2)
plt.axhline(0.90, color='r', linestyle='--', label='Target: R²=0.90')
plt.xlabel('Number of Trials')
plt.ylabel('Best Score (R²)')
plt.title('Convergence Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**典型曲線**：
- Random Search: 鋸齒狀，緩慢改進
- Bayesian Opt: 快速收斂，有明確趨勢

### 5.9 實務建議

#### 何時使用 Bayesian Optimization？

| 場景 | 推薦度 | 原因 |
|------|-------|------|
| 評估成本高（訓練時間 > 10 分鐘） | ⭐⭐⭐⭐⭐ | 核心優勢 |
| 超參數 ≥ 5 個 | ⭐⭐⭐⭐⭐ | 高維搜索 |
| 預算嚴格限制（< 50 次） | ⭐⭐⭐⭐⭐ | 樣本效率高 |
| 實驗成本昂貴（化工實驗） | ⭐⭐⭐⭐⭐ | 減少試驗次數 |
| 模型訓練快速（< 1 秒） | ⭐⭐ | Random Search 足夠 |

#### 常見陷阱

**1. 過度依賴初始化**：

```python
# ❌ 不推薦：只有 5 個初始點
study.optimize(objective, n_trials=50)
# 前 5 個是隨機的，可能陷入局部最優

# ✅ 推薦：至少 10-20 個初始點
study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(
        n_startup_trials=20  # 前 20 個隨機探索
    )
)
```

**2. 忽略不確定度**：

```python
# 評估多次，取平均
scores = []
for _ in range(3):
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    scores.append(score)
return np.mean(scores)  # 更穩定的估計
```

**3. 搜索空間設定不當**：

```python
# ❌ 搜索空間過大
'learning_rate': trial.suggest_float('learning_rate', 1e-10, 1.0)
# 大部分區域無意義

# ✅ 合理範圍
'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
```

### 5.10 小結

**Bayesian Optimization 的核心優勢**：
1. ✅ 樣本效率極高（相比 Random Search 節省 50-70%）
2. ✅ 利用歷史信息，智能選擇下一個試驗點
3. ✅ 適合昂貴的評估（訓練時間長、實驗成本高）
4. ✅ 提供豐富的診斷工具（重要性分析、收斂曲線）

**適用場景**：
- 模型訓練時間 > 10 分鐘
- 化工實驗等昂貴評估
- 預算嚴格限制（< 100 次嘗試）
- 超參數數量 ≥ 5

**框架選擇**：
- **Optuna**：功能最全，社群活躍，視覺化豐富（推薦）
- **Hyperopt**：老牌框架，文獻引用多，但語法較複雜

---

## 6. 進階搜索技巧

### 6.1 Halving Search

**Successive Halving** 是一種加速策略，通過逐步淘汰表現差的候選者，集中資源在有希望的組合上。

#### 原理

```
初始: 64 個候選者，每個用 10% 數據評估
  ↓ 淘汰最差的一半
第 2 輪: 32 個候選者，每個用 20% 數據評估
  ↓ 淘汰最差的一半
第 3 輪: 16 個候選者，每個用 40% 數據評估
  ↓ 淘汰最差的一半
第 4 輪: 8 個候選者，每個用 80% 數據評估
  ↓ 選出最優
最終: 1 個候選者，用 100% 數據評估
```

**計算效率**：

傳統方法：64 個候選者 × 100% 數據 = **6400 單位**

Halving 方法：

$$
64 \times 10\% + 32 \times 20\% + 16 \times 40\% + 8 \times 80\% = 25.6 \text{ 單位}
$$

**加速比**：6400 / 25.6 ≈ **250 倍** ✅

#### Sklearn 實作

```python
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.ensemble import RandomForestRegressor

# Halving Grid Search
model = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 20]
}

halving_grid = HalvingGridSearchCV(
    model,
    param_grid,
    factor=3,                # 每輪保留 1/3
    resource='n_samples',    # 逐步增加樣本數
    max_resources='auto',    # 最終使用全部數據
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42
)

halving_grid.fit(X_train, y_train)

print(f"Best params: {halving_grid.best_params_}")
print(f"Best score: {halving_grid.best_score_:.4f}")
print(f"Number of candidates: {halving_grid.n_candidates_}")
print(f"Number of resources (samples): {halving_grid.n_resources_}")
```

#### Halving Random Search

```python
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': randint(2, 20),
    'max_features': uniform(0.1, 0.9)
}

halving_random = HalvingRandomSearchCV(
    model,
    param_distributions,
    n_candidates='exhaust',  # 盡可能多的初始候選者
    factor=3,
    resource='n_samples',
    max_resources='auto',
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42
)

halving_random.fit(X_train, y_train)
```

### 6.2 早停 (Early Stopping)

對於支持增量訓練的模型（如 XGBoost），可以提前終止無希望的試驗。

#### XGBoost 範例

```python
import xgboost as xgb

def objective_with_early_stopping(trial):
    params = {
        'n_estimators': 1000,  # 設定大值
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'early_stopping_rounds': 50  # 50 輪無改善則停止
    }
    
    model = xgb.XGBRegressor(**params, random_state=42)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # 實際訓練的輪數（可能遠少於 1000）
    best_iteration = model.best_iteration
    score = model.score(X_val, y_val)
    
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective_with_early_stopping, n_trials=50)
```

### 6.3 多保真度優化 (Multi-Fidelity Optimization)

使用不同保真度（樣本量、訓練輪數、分辨率等）加速搜索。

#### 策略

| 保真度級別 | 樣本量 | CV Folds | 訓練輪數 | 用途 |
|-----------|--------|----------|---------|------|
| 低 | 10% | 2 | 10 | 初步篩選 |
| 中 | 30% | 3 | 50 | 精選候選 |
| 高 | 100% | 5 | 1000 | 最終評估 |

#### 實作範例

```python
def multi_fidelity_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20)
    }
    
    # 第一階段：快速評估（30% 數據，3-fold）
    X_sample, _, y_sample, _ = train_test_split(
        X_train, y_train, train_size=0.3, random_state=42
    )
    
    model = RandomForestRegressor(**params, random_state=42, n_jobs=1)
    scores = cross_val_score(model, X_sample, y_sample, cv=3, scoring='r2')
    score_low = scores.mean()
    
    # 如果表現太差，提前終止
    if score_low < 0.70:
        return score_low
    
    # 第二階段：完整評估（100% 數據，5-fold）
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    score_high = scores.mean()
    
    return score_high

study = optuna.create_study(direction='maximize')
study.optimize(multi_fidelity_objective, n_trials=100)
```

### 6.4 暖啟動 (Warm Start)

利用之前的搜索結果，加速新一輪優化。

```python
# 第一輪優化
study = optuna.create_study(direction='maximize', study_name='rf_tuning')
study.optimize(objective, n_trials=50)

# 保存結果
import joblib
joblib.dump(study, 'study_checkpoint.pkl')

# 第二輪優化（續接）
study_loaded = joblib.load('study_checkpoint.pkl')
study_loaded.optimize(objective, n_trials=50)  # 再優化 50 次

print(f"Total trials: {len(study_loaded.trials)}")  # 100 次
```

### 6.5 平行分佈式優化

```python
import optuna

# 創建共享研究對象（使用資料庫）
study = optuna.create_study(
    study_name='distributed_optimization',
    storage='sqlite:///optuna_study.db',  # 共享資料庫
    direction='maximize',
    load_if_exists=True  # 如果已存在則載入
)

# 在多台機器或多個進程上同時運行
study.optimize(objective, n_trials=100)
```

**多進程範例**：

```python
from multiprocessing import Pool

def run_optimization(worker_id):
    study = optuna.load_study(
        study_name='distributed_optimization',
        storage='sqlite:///optuna_study.db'
    )
    study.optimize(objective, n_trials=25)

if __name__ == '__main__':
    with Pool(4) as pool:  # 4 個工作進程
        pool.map(run_optimization, range(4))
```

### 6.6 化工案例：加速蒸餾塔控制器調參

**場景**：調整 PID 控制器參數以最小化溫度波動

**挑戰**：每次模擬需要 5 分鐘

**Halving Random Search 方案**：

```python
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

# 模擬控制器性能（實務中需調用模擬器）
def evaluate_controller(Kp, Ki, Kd, simulation_time):
    """
    simulation_time: 模擬時長（秒）
    返回: 溫度標準差（越小越好）
    """
    # 實際應用中會調用 Aspen/HYSYS 等
    # 此處簡化為快速評估
    return some_simulation_function(Kp, Ki, Kd, simulation_time)

# 使用 Halving 策略
param_distributions = {
    'Kp': uniform(0.1, 10),
    'Ki': uniform(0.01, 1),
    'Kd': uniform(0.001, 0.1)
}

# 初始用短時間模擬，逐步增加
halving_search = HalvingRandomSearchCV(
    CustomController(),
    param_distributions,
    resource='simulation_time',  # 逐步增加模擬時長
    min_resources=10,            # 最小 10 秒
    max_resources=300,           # 最大 5 分鐘
    factor=3,
    cv=3,
    scoring='neg_std',           # 最小化標準差
    random_state=42
)

halving_search.fit(X_dummy, y_dummy)
```

**效果**：
- 傳統方法：100 個組合 × 5 分鐘 = **8.3 小時**
- Halving Search：加速 **50 倍** → **10 分鐘**

---

## 7. 搜索空間設計

### 7.1 搜索空間的重要性

> "Garbage in, garbage out" — 搜索空間設計不當，再好的算法也無濟於事。

**常見錯誤**：

```python
# ❌ 錯誤 1: 範圍過大
'learning_rate': uniform(1e-10, 10)  # 大部分區域無意義

# ❌ 錯誤 2: 範圍過小
'n_estimators': [90, 95, 100, 105, 110]  # 可能錯過最優區域

# ❌ 錯誤 3: 刻度不當
'alpha': [0.001, 0.01, 0.1, 1, 10]  # 應使用對數刻度

# ❌ 錯誤 4: 忽略相關性
# max_depth 和 min_samples_leaf 相互影響，需同時調整
```

### 7.2 連續 vs 離散

| 超參數類型 | 搜索空間設計 | 範例 |
|-----------|------------|------|
| 連續型 | 使用分布（uniform, loguniform） | learning_rate, alpha |
| 整數型 | 使用 randint 或整數分布 | n_estimators, max_depth |
| 類別型 | 列舉所有選項 | kernel, criterion |

**Optuna 範例**：

```python
def objective(trial):
    # 連續型
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
    
    # 整數型
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    
    # 類別型
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
    
    # 條件型（依賴其他超參數）
    if kernel == 'poly':
        degree = trial.suggest_int('degree', 2, 5)
    else:
        degree = None
    
    return train_and_evaluate(learning_rate, n_estimators, kernel, degree)
```

### 7.3 對數尺度的應用

**何時使用對數尺度？**

當超參數範圍跨越多個數量級時（如 0.001 → 100）。

| 超參數 | 線性範圍 | 對數範圍 |
|--------|---------|---------|
| learning_rate | ❌ | ✅ [1e-4, 1e-1] |
| alpha (正則化) | ❌ | ✅ [1e-5, 1e2] |
| C (SVM) | ❌ | ✅ [1e-3, 1e3] |
| n_estimators | ✅ [50, 500] | ❌ |
| max_depth | ✅ [3, 20] | ❌ |

**Python 實作**：

```python
# Optuna
'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)

# Sklearn (Random Search)
from scipy.stats import loguniform
'learning_rate': loguniform(1e-4, 1e-1)

# 手動生成對數刻度
import numpy as np
np.logspace(-4, -1, 10)  # [0.0001, ..., 0.1]
```

### 7.4 常見模型的推薦搜索空間

#### Random Forest

```python
# Grid Search 版本
param_grid_rf = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Random/Bayesian Search 版本
param_dist_rf = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)
}
```

#### XGBoost

```python
param_dist_xgb = {
    # 樹結構
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 10),
    'min_child_weight': randint(1, 10),
    
    # 學習率
    'learning_rate': loguniform(1e-3, 1e-1),
    
    # 採樣
    'subsample': uniform(0.5, 0.5),           # [0.5, 1.0]
    'colsample_bytree': uniform(0.5, 0.5),    # [0.5, 1.0]
    
    # 正則化
    'gamma': uniform(0, 5),
    'reg_alpha': loguniform(1e-4, 10),        # L1
    'reg_lambda': loguniform(1e-4, 10)        # L2
}
```

#### Support Vector Machine

```python
param_dist_svm = {
    'C': loguniform(1e-2, 1e3),               # 懲罰參數
    'epsilon': loguniform(1e-3, 1),           # ε-tube 寬度
    'gamma': loguniform(1e-4, 1),             # RBF 核寬度
    'kernel': ['rbf', 'poly', 'sigmoid']
}
```

#### Gradient Boosting

```python
param_dist_gb = {
    'n_estimators': randint(50, 500),
    'learning_rate': loguniform(1e-3, 1),
    'max_depth': randint(3, 10),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'subsample': uniform(0.5, 0.5),
    'max_features': uniform(0.5, 0.5)
}
```

### 7.5 利用領域知識縮小空間

#### 化工案例：反應溫度預測

**場景**：預測化學反應的最優溫度

**物理約束**：
- 反應溫度範圍: 50-300°C（已知）
- 壓力範圍: 1-50 bar（已知）
- 某些特徵組合物理上不可行

**優化搜索空間**：

```python
def objective_with_constraints(trial):
    # 基於化學動力學，縮小學習率範圍
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1e-1)
    # （經驗：太小的學習率在此問題無效）
    
    # 已知樹深度超過 15 沒有幫助（反應機制不那麼複雜）
    max_depth = trial.suggest_int('max_depth', 3, 15)
    
    # 溫度和壓力必須滿足物理約束
    temp = trial.suggest_float('temp', 50, 300)
    pressure = trial.suggest_float('pressure', 1, 50)
    
    # 無效組合：高溫低壓（溶劑會汽化）
    if temp > 200 and pressure < 5:
        return float('-inf')  # 懲罰無效組合
    
    return train_and_evaluate(...)
```

### 7.6 條件超參數

某些超參數只在特定條件下有意義。

```python
def objective_conditional(trial):
    # 選擇模型類型
    model_type = trial.suggest_categorical('model_type', ['rf', 'xgb', 'svm'])
    
    if model_type == 'rf':
        n_estimators = trial.suggest_int('rf_n_estimators', 50, 500)
        max_depth = trial.suggest_int('rf_max_depth', 5, 30)
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
    
    elif model_type == 'xgb':
        n_estimators = trial.suggest_int('xgb_n_estimators', 50, 500)
        learning_rate = trial.suggest_loguniform('xgb_learning_rate', 1e-3, 1e-1)
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
    
    else:  # svm
        C = trial.suggest_loguniform('svm_C', 1e-2, 1e2)
        gamma = trial.suggest_loguniform('svm_gamma', 1e-4, 1)
        model = SVR(C=C, gamma=gamma)
    
    return cross_val_score(model, X_train, y_train, cv=5).mean()
```

### 7.7 避坑指南

#### 陷阱 1: 不切實際的範圍

```python
# ❌ 錯誤：max_depth 設為 100（幾乎一定過擬合）
'max_depth': randint(1, 100)

# ✅ 正確：基於數據量和經驗
# 對於 n=1000 的數據集
'max_depth': randint(3, 20)
```

#### 陷阱 2: 忽略計算成本

```python
# ❌ 錯誤：n_estimators 過大浪費時間
'n_estimators': randint(50, 10000)  # 訓練 10000 棵樹沒必要

# ✅ 正確：合理上限
'n_estimators': randint(50, 500)
```

#### 陷阱 3: 過度離散化

```python
# ❌ 錯誤：刻度太密集
'alpha': [0.1, 0.11, 0.12, 0.13, ..., 1.0]  # 90 個值

# ✅ 正確：對數刻度 + 合理數量
'alpha': np.logspace(-2, 1, 10)  # [0.01, 0.02, ..., 10]
```

---

## 8. 化工應用案例

### 8.1 案例 1：精餾塔溫度軟測量模型

**背景**：
- 目標：預測精餾塔第 15 層溫度
- 輸入：進料流量、回流比、再沸器熱負荷等 12 個變量
- 數據：1200 筆歷史數據
- 要求：MAE < 1°C，推理時間 < 100ms

**模型選擇**：XGBoost（兼顧準確度與速度）

**超參數調整策略**：

```python
import xgboost as xgb
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 8),  # 限制深度防過擬合
        'learning_rate': trial.suggest_loguniform('learning_rate', 5e-3, 5e-1),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10),
        'gamma': trial.suggest_uniform('gamma', 0, 5)
    }
    
    model = xgb.XGBRegressor(
        **params,
        random_state=42,
        tree_method='hist',  # 加速訓練
        n_jobs=1
    )
    
    # 時間序列交叉驗證
    scores = []
    for train_idx, val_idx in TimeSeriesSplit(n_splits=5).split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        scores.append(-mae)  # 負號：最小化 MAE
    
    return np.mean(scores)

# Bayesian Optimization
study = optuna.create_study(
    direction='maximize',  # 最大化 -MAE（即最小化 MAE）
    sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=20)
)

study.optimize(objective, n_trials=100, n_jobs=1)

# 最優模型
best_params = study.best_params
final_model = xgb.XGBRegressor(**best_params, random_state=42)
final_model.fit(X_train, y_train)

# 測試集評估
y_pred = final_model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
print(f"Test MAE: {test_mae:.3f} °C")

# 推理速度測試
import time
start = time.time()
_ = final_model.predict(X_test[:1000])
inference_time = (time.time() - start) / 1000 * 1000  # ms per sample
print(f"Inference time: {inference_time:.2f} ms")
```

**結果**：
- Test MAE: **0.87°C** ✅（滿足 < 1°C）
- Inference time: **0.23 ms** ✅（滿足 < 100 ms）
- 最優參數：`{'n_estimators': 287, 'max_depth': 6, 'learning_rate': 0.0342, ...}`

### 8.2 案例 2：反應器產率優化

**背景**：
- 目標：最大化化學反應產率
- 輸入：反應溫度、壓力、催化劑負載量、停留時間
- 挑戰：實驗成本高（每次 > 10,000 元），只能進行 30 次實驗
- 要求：找到產率 > 85% 的操作條件

**策略**：Bayesian Optimization with Gaussian Process

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import optuna

# 已有的少量實驗數據（10 筆初始數據）
initial_experiments = pd.DataFrame({
    'temperature': [200, 220, 250, 180, 230, 240, 210, 260, 190, 225],
    'pressure': [5, 8, 10, 3, 7, 9, 6, 12, 4, 8],
    'catalyst_loading': [2.0, 2.5, 3.0, 1.5, 2.2, 2.8, 2.1, 3.5, 1.8, 2.6],
    'residence_time': [10, 15, 20, 8, 12, 18, 11, 22, 9, 14],
    'yield': [72, 78, 81, 65, 76, 80, 74, 79, 68, 77]  # %
})

# 訓練 Gaussian Process 代理模型
X_init = initial_experiments[['temperature', 'pressure', 'catalyst_loading', 'residence_time']]
y_init = initial_experiments['yield']

kernel = C(1.0, (1e-3, 1e3)) * RBF([10, 2, 0.5, 5], (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
gp.fit(X_init, y_init)

# 使用 Optuna 找下一個實驗點
def suggest_next_experiment(trial):
    temp = trial.suggest_float('temperature', 180, 280)
    pressure = trial.suggest_float('pressure', 3, 15)
    catalyst = trial.suggest_float('catalyst_loading', 1.5, 4.0)
    residence = trial.suggest_float('residence_time', 8, 25)
    
    X_new = np.array([[temp, pressure, catalyst, residence]])
    
    # Upper Confidence Bound
    mean, std = gp.predict(X_new, return_std=True)
    ucb = mean + 2.0 * std  # kappa=2.0
    
    return ucb[0]

study = optuna.create_study(direction='maximize')
study.optimize(suggest_next_experiment, n_trials=20)  # 建議 20 個新實驗

# 最有希望的實驗條件
best_condition = study.best_params
print("=== 建議的下一個實驗條件 ===")
print(f"Temperature: {best_condition['temperature']:.1f} °C")
print(f"Pressure: {best_condition['pressure']:.1f} bar")
print(f"Catalyst Loading: {best_condition['catalyst_loading']:.2f} wt%")
print(f"Residence Time: {best_condition['residence_time']:.1f} min")

# 預測產率
X_pred = np.array([[
    best_condition['temperature'],
    best_condition['pressure'],
    best_condition['catalyst_loading'],
    best_condition['residence_time']
]])
predicted_yield, uncertainty = gp.predict(X_pred, return_std=True)
print(f"Predicted Yield: {predicted_yield[0]:.1f}% ± {uncertainty[0]:.1f}%")
```

**迭代流程**：
1. 用 10 筆初始數據訓練 GP
2. 找 UCB 最大的點 → 進行實驗
3. 更新 GP 模型 → 再找下一個點
4. 重複直到達到目標或預算用盡

**結果**：
- 第 25 次實驗達到產率 **86.3%** ✅
- 相比隨機實驗節省 **40%** 成本

### 8.3 案例 3：催化劑篩選加速

**背景**：
- 目標：從 500 種催化劑配方中找出最優組合
- 評估方法：高通量實驗（HTE）
- 限制：只能測試 50 種配方
- 優化目標：轉化率、選擇性、穩定性（多目標）

**多目標 Bayesian Optimization**：

```python
def multi_objective_catalyst(trial):
    # 催化劑組成（mol%）
    metal_A = trial.suggest_float('metal_A', 0, 100)
    metal_B = trial.suggest_float('metal_B', 0, 100 - metal_A)
    support_type = trial.suggest_categorical('support_type', ['Al2O3', 'SiO2', 'TiO2', 'ZrO2'])
    
    # 製備條件
    calcination_temp = trial.suggest_int('calcination_temp', 400, 800)
    calcination_time = trial.suggest_int('calcination_time', 2, 8)
    
    # 模擬實驗（實務中調用 HTE 設備）
    conversion = simulate_conversion(metal_A, metal_B, support_type, calcination_temp, calcination_time)
    selectivity = simulate_selectivity(metal_A, metal_B, support_type, calcination_temp, calcination_time)
    stability = simulate_stability(metal_A, metal_B, support_type, calcination_temp, calcination_time)
    
    return conversion, selectivity, stability  # 三個目標

# 多目標優化
study = optuna.create_study(
    directions=['maximize', 'maximize', 'maximize']
)

study.optimize(multi_objective_catalyst, n_trials=50)

# Pareto 最優解
print(f"Number of Pareto-optimal solutions: {len(study.best_trials)}")
for i, trial in enumerate(study.best_trials[:5]):  # 顯示前 5 個
    print(f"\n=== Solution {i+1} ===")
    print(f"Conversion: {trial.values[0]:.1f}%")
    print(f"Selectivity: {trial.values[1]:.1f}%")
    print(f"Stability: {trial.values[2]:.1f} hours")
    print(f"Parameters: {trial.params}")
```

**結果**：
- 找到 8 個 Pareto 最優配方
- 最優配方：轉化率 92%, 選擇性 88%, 穩定性 120 小時
- 相比全面篩選，節省 **90%** 實驗次數

### 8.4 案例總結

| 案例 | 模型 | 調參方法 | 關鍵挑戰 | 成果 |
|------|------|---------|---------|------|
| 精餾塔軟測量 | XGBoost | Bayesian Opt | 速度與準確度平衡 | MAE 0.87°C, 0.23ms |
| 反應器產率 | GP | Active Learning | 實驗成本高 | 25 次達標，節省 40% |
| 催化劑篩選 | Multi-objective | Pareto Opt | 多目標衝突 | 8 個最優解，節省 90% |

**共同特點**：
- 充分利用領域知識設計搜索空間
- 選擇合適的優化策略（成本 vs 性能）
- 嚴格驗證（時間序列 CV, 實驗驗證）

---

## 9. 總結與最佳實踐

### 9.1 方法選擇指南

```
開始
  │
  ├─ 超參數數量 ≤ 3？
  │   ├─ Yes → Grid Search
  │   └─ No → 繼續
  │
  ├─ 模型訓練時間 < 1 秒？
  │   ├─ Yes → Random Search (n_iter=100)
  │   └─ No → 繼續
  │
  ├─ 模型訓練時間 > 10 分鐘？
  │   ├─ Yes → Bayesian Optimization (n_trials=50)
  │   └─ No → 繼續
  │
  ├─ 預算 < 50 次？
  │   ├─ Yes → Bayesian Optimization
  │   └─ No → Random Search + Halving
  │
  └─ 實驗成本極高（化工實驗）？
      └─ Yes → Bayesian Optimization with GP
```

### 9.2 最佳實踐清單

#### ✅ 搜索空間設計

- [ ] 使用對數尺度於跨度大的超參數（learning_rate, alpha）
- [ ] 設定合理的範圍（避免過大或過小）
- [ ] 利用領域知識排除無效區域
- [ ] 處理條件超參數（只在特定模型有意義）

#### ✅ 驗證策略

- [ ] 使用交叉驗證（至少 5-fold）
- [ ] 時間序列數據必須用 TimeSeriesSplit
- [ ] 小數據集考慮 LOOCV 或 Stratified CV
- [ ] 監控訓練-驗證差距（過擬合檢測）

#### ✅ 計算效率

- [ ] 設定 `n_jobs=-1` 充分利用多核
- [ ] 考慮 Halving Search 加速
- [ ] 使用 Early Stopping（XGBoost 等）
- [ ] 多保真度評估（先用小數據篩選）

#### ✅ 結果分析

- [ ] 可視化搜索過程（收斂曲線、參數重要性）
- [ ] 分析最優解的穩定性（多次運行）
- [ ] 檢查是否觸及搜索邊界（需擴大範圍）
- [ ] 記錄所有嘗試（便於後續分析）

#### ✅ 化工特殊考量

- [ ] 平衡準確度與可解釋性
- [ ] 考慮推理速度（線上控制需求）
- [ ] 模型大小限制（嵌入式系統）
- [ ] 多目標優化（產率、能耗、安全）

### 9.3 常見錯誤

| 錯誤 | 後果 | 正確做法 |
|------|------|---------|
| 在訓練集上調參 | 過擬合 | 使用驗證集或交叉驗證 |
| 忽略隨機性 | 結果不穩定 | 多次運行取平均 |
| 過度調參 | 過擬合驗證集 | 保留測試集最終評估 |
| 搜索空間過大 | 浪費資源 | 利用先驗知識縮小 |
| 只看單一指標 | 忽略其他重要因素 | 多目標平衡 |

### 9.4 工具對比

| 工具 | 優點 | 缺點 | 推薦場景 |
|------|------|------|---------|
| GridSearchCV | 簡單、完整 | 組合爆炸 | ≤ 3 個超參數 |
| RandomizedSearchCV | 高效、靈活 | 無智能 | 4-6 個超參數 |
| Optuna | 智能、視覺化豐富 | 需額外安裝 | 複雜優化 |
| Hyperopt | 成熟、文獻多 | 語法複雜 | 學術研究 |
| Ray Tune | 分佈式、可擴展 | 學習曲線陡 | 大規模訓練 |

### 9.5 進階學習資源

#### 📚 推薦論文

1. **Bergstra & Bengio (2012)**: "Random Search for Hyper-Parameter Optimization"
   - 證明 Random Search 在高維空間優於 Grid Search

2. **Snoek et al. (2012)**: "Practical Bayesian Optimization of Machine Learning Algorithms"
   - Bayesian Optimization 的經典論文

3. **Li et al. (2017)**: "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization"
   - Halving Search 的理論基礎

#### 🔗 工具文檔

- Optuna: https://optuna.org/
- Hyperopt: http://hyperopt.github.io/hyperopt/
- Sklearn Model Selection: https://scikit-learn.org/stable/model_selection.html

#### 🎓 學習路徑

1. **初學者**：掌握 GridSearchCV 和 RandomizedSearchCV
2. **進階**：學習 Optuna，理解 Bayesian Optimization 原理
3. **專家**：實作自定義獲取函數，多保真度優化

### 9.6 課後練習

#### 練習 1：比較三種方法

使用 Titanic 數據集，比較 Grid Search, Random Search, Bayesian Optimization 在 Random Forest 上的性能。

**要求**：
- 設定相同的搜索空間
- 記錄每種方法找到最優解的時間
- 繪製收斂曲線

#### 練習 2：化工應用

使用課程提供的反應器數據，建立產率預測模型。

**要求**：
- 嘗試至少 3 種模型（Linear, Random Forest, XGBoost）
- 使用 Bayesian Optimization 調參
- 分析參數重要性
- 驗證模型的化學合理性

#### 練習 3：多目標優化

建立一個同時優化準確度和推理速度的模型。

**要求**：
- 使用 Optuna 多目標優化
- 繪製 Pareto Front
- 根據業務需求選擇最終模型

---

## 🎯 重點回顧

1. **超參數 vs 模型參數**：
   - 超參數：訓練前設定（如 learning_rate）
   - 模型參數：訓練中學習（如權重）

2. **三大搜索方法**：
   - Grid Search: 窮舉，適合低維
   - Random Search: 隨機，適合中維
   - Bayesian Optimization: 智能，適合高維/昂貴評估

3. **搜索空間設計**：
   - 跨度大的參數用對數尺度
   - 利用領域知識縮小範圍
   - 處理條件超參數

4. **化工應用要點**：
   - 平衡準確度與可解釋性
   - 考慮實驗成本
   - 多目標優化

5. **工具推薦**：
   - 入門：Sklearn GridSearchCV / RandomizedSearchCV
   - 進階：Optuna（最推薦）

---

**下一單元預告**：Unit15 將整合所有學過的技術，進行完整的化工案例實戰！

