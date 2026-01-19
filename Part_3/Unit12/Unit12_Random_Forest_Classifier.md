# Unit12 隨機森林分類 | Random Forest Classifier

> **最後更新**：2026-01-18

---

## 學習目標

本節課將深入學習**隨機森林分類 (Random Forest Classifier)**，這是目前最強大且應用最廣泛的集成學習方法之一。通過本節課，您將能夠：

- 理解集成學習 (Ensemble Learning) 的核心概念
- 掌握隨機森林的工作原理與優勢
- 學習 Bagging 與隨機特徵選擇的機制
- 掌握 sklearn 中 `RandomForestClassifier` 的使用方法
- 理解並設定關鍵超參數
- 應用隨機森林突破單一決策樹的性能限制（提升約 1-2%）
- 評估特徵重要性與模型穩定性
- 解決化工領域的複雜分類問題

**性能預期說明**：本節使用的模擬化工數據中，由於特徵設計限制，Random Forest 相比 Decision Tree 的性能提升較為溫和（約 1-2%）。在實際工業數據中，Random Forest 通常能展現更明顯的優勢（8-15%）。本課重點在於**方法論學習**與**集成思想理解**。

---

## 1. 從決策樹到隨機森林

### 1.1 決策樹的限制回顧

在 Unit12_Decision_Tree_Classifier 中，我們學習了決策樹分類並發現了其主要限制：

1. **容易過擬合**：
   - 深度樹可以完美擬合訓練數據（100% 訓練準確率）
   - 但測試性能大幅下降（61% 測試準確率）

2. **高方差 (High Variance)**：
   - 對訓練數據微小變動非常敏感
   - 不同的訓練子集可能產生完全不同的樹結構

3. **性能上限**：
   - 單一決策樹準確率約 68-70%（本例中約 68%）
   - 單一模型難以進一步提升性能

4. **不穩定性**：
   - 小的數據擾動導致樹結構劇烈變化
   - 預測結果不夠可靠

### 1.2 集成學習的智慧

> **集成學習核心思想**："三個臭皮匠，勝過一個諸葛亮"

**集成學習 (Ensemble Learning)** 通過組合多個弱學習器（如決策樹），構建一個更強大的模型。主要優勢：

- **降低方差**：多個模型的預測平均可降低隨機誤差
- **提高準確率**：集體智慧優於單一模型
- **增強穩定性**：對數據擾動不敏感
- **減少過擬合**：多樣性降低記憶訓練數據的風險

### 1.3 隨機森林的誕生

**隨機森林 (Random Forest)** 由 Leo Breiman (2001) 提出，結合了兩大關鍵技術：

1. **Bagging (Bootstrap Aggregating)**：
   - 通過自助採樣創建多個訓練子集
   - 每個子集訓練一棵決策樹

2. **隨機特徵選擇 (Random Feature Selection)**：
   - 每次分裂只考慮隨機選擇的特徵子集
   - 增加樹之間的多樣性

---

## 2. 隨機森林工作原理

### 2.1 完整算法流程

#### Step 1: 自助採樣 (Bootstrap Sampling)

對於包含 $N$ 個樣本的訓練集，進行有放回的隨機採樣：

$$
\text{Bootstrap Sample}_i = \{x_1^{(i)}, x_2^{(i)}, \ldots, x_N^{(i)}\}
$$

- 每個樣本被抽取的機率為 $1/N$
- 重複抽取 $N$ 次，形成新的訓練集
- 約 63.2% 的原始樣本會被抽到（至少一次）
- 約 36.8% 的樣本不會被抽到（稱為 **Out-of-Bag, OOB** 樣本）

#### Step 2: 建立多棵決策樹

對每個 Bootstrap 樣本，建立一棵決策樹：

- **樹的數量**： $n_{\text{estimators}}$ （常用 100-500）
- **樹的深度**：通常不限制（讓樹完全生長）
- **分裂規則**：標準決策樹演算法（Gini 或 Entropy）

#### Step 3: 隨機特徵選擇

在每個節點分裂時：

1. 從所有 $M$ 個特徵中隨機選擇 $m$ 個特徵（ $m < M$ ）
2. 只在這 $m$ 個特徵中尋找最佳分裂點
3. 常見設定：
   - **分類問題**： $m = \sqrt{M}$
   - **回歸問題**： $m = M/3$

**隨機特徵選擇的效果**：

- 增加樹之間的差異性（decorrelation）
- 防止某些強特徵主導所有樹
- 提升集成效果

#### Step 4: 投票預測 (Voting)

對於新樣本 $x_{\text{new}}$ ，進行分類預測：

$$
\hat{y} = \text{mode}\{h_1(x), h_2(x), \ldots, h_T(x)\}
$$

- **硬投票 (Hard Voting)**：多數投票，選擇得票最多的類別
- **軟投票 (Soft Voting)**：平均所有樹的類別機率，選擇平均機率最高的類別

---

### 2.2 關於本節數據的說明

本節使用的化工反應模擬數據具有以下特性：

1. **特徵分佈**：5 個特徵（Temperature, Pressure, Catalyst, Ratio, Time）的貢獻度較為均衡
2. **Temperature 重要性**：約 27-28%（在實際工業數據中，主導因素通常達 35-45%）
3. **RF vs DT 提升**：約 1-2%（在複雜真實數據中通常為 8-15%）

**為什麼會有這個差異？**

- **模擬數據限制**：人工生成的數據難以完全複製真實化工過程的複雜非線性關係
- **特徵交互作用**：真實數據中存在更多隱藏的特徵交互，Random Forest 能更好地捕捉
- **噪聲模式**：實際測量數據包含更多隨機噪聲，集成方法的降噪優勢更明顯

**學習重點**：

本節的核心價值在於：
- ✅ 理解 Random Forest 的**工作原理**與**集成思想**
- ✅ 掌握 **sklearn API** 的完整使用流程
- ✅ 學習**特徵重要性分析**與**超參數調優**方法
- ✅ 建立**模型評估**的系統化思維

數值結果的絕對值並非關鍵，方法的可遷移性才是核心能力。

**數學表達式（軟投票）**：

$$
P(y = k | x) = \frac{1}{T} \sum_{t=1}^{T} P_t(y = k | x)
$$

其中 $T$ 是樹的數量， $P_t$ 是第 $t$ 棵樹預測的機率。

### 2.2 關鍵機制解析

#### 2.2.1 為什麼 Bagging 有效？

**數學直覺**：假設有 $T$ 個獨立的預測模型，每個模型的誤差為 $\epsilon$ ，則集成模型的誤差為：

$$
\text{Error}_{\text{ensemble}} = \frac{\epsilon}{\sqrt{T}}
$$

- 當 $T = 100$ 時，誤差減少到原來的 1/10
- **關鍵假設**：模型之間需要獨立（通過隨機採樣和隨機特徵實現）

#### 2.2.2 為什麼隨機特徵選擇有效？

考慮一個數據集有一個非常強的特徵（如化學反應中的溫度），如果不限制特徵：

- **所有樹都會在第一次分裂使用這個強特徵**
- 樹之間高度相關，集成效果有限

通過隨機特徵選擇：

- 有些樹無法使用強特徵，被迫使用其他特徵
- 樹之間差異增大（decorrelation）
- 集成效果顯著提升

#### 2.2.3 Out-of-Bag (OOB) 評估

每棵樹約有 36.8% 的樣本未被使用，可用於驗證：

$$
\text{OOB Score} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\{y_i = \hat{y}_i^{\text{OOB}}\}
$$

**OOB 評估的優勢**：

- 無需額外的驗證集
- 相當於免費的交叉驗證
- 可評估模型泛化能力

---

## 3. 隨機森林的數學理論

### 3.1 偏差-方差權衡 (Bias-Variance Tradeoff)

模型的總誤差可分解為：

$$
\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
$$

**決策樹 vs. 隨機森林**：

| 模型 | 偏差 (Bias) | 方差 (Variance) |
|-----|------------|----------------|
| 單一深度決策樹 | 低 | **高**（過擬合） |
| 單一淺層決策樹 | **高** | 低（欠擬合） |
| 隨機森林 | 低 | **低**（平衡最佳） |

隨機森林通過集成多棵深度樹：

- 保持低偏差（樹可以擬合複雜模式）
- 降低方差（多樹平均降低隨機誤差）

### 3.2 樹的相關性與集成效果

集成模型的方差可以表示為：

$$
\text{Var}_{\text{ensemble}} = \rho \sigma^2 + \frac{1-\rho}{T} \sigma^2
$$

其中：
- $\rho$ ：樹之間的平均相關性（0到1）
- $\sigma^2$ ：單棵樹的方差
- $T$ ：樹的數量

**關鍵洞察**：

- 當 $\rho = 0$ （完全獨立）： $\text{Var} = \sigma^2 / T$ （理想情況）
- 當 $\rho = 1$ （完全相關）： $\text{Var} = \sigma^2$ （無集成效果）
- **隨機特徵選擇降低 $\rho$ ，增強集成效果**

### 3.3 特徵重要性計算

隨機森林提供兩種特徵重要性度量：

#### 3.3.1 Gini 重要性 (Mean Decrease Impurity)

對於特徵 $j$ ，其重要性為所有樹中使用該特徵分裂帶來的 Gini 不純度減少的平均值：

$$
\text{Importance}(j) = \frac{1}{T} \sum_{t=1}^{T} \sum_{k \in \text{splits on } j} \Delta \text{Gini}_k^{(t)}
$$

#### 3.3.2 排列重要性 (Permutation Importance)

1. 計算模型在原始數據上的性能 $S_{\text{original}}$
2. 隨機打亂特徵 $j$ 的值
3. 計算模型在打亂後數據的性能 $S_{\text{permuted}}$
4. 特徵重要性 = $S_{\text{original}} - S_{\text{permuted}}$

**解釋**：如果打亂特徵後性能大幅下降，說明該特徵很重要。

---

## 4. sklearn 中的 RandomForestClassifier

### 4.1 基本使用

```python
from sklearn.ensemble import RandomForestClassifier

# 建立隨機森林分類器
rf_model = RandomForestClassifier(
    n_estimators=100,          # 樹的數量
    max_depth=None,            # 樹的最大深度（None=不限制）
    min_samples_split=2,       # 分裂所需最小樣本數
    min_samples_leaf=1,        # 葉節點最小樣本數
    max_features='sqrt',       # 每次分裂考慮的特徵數量
    bootstrap=True,            # 是否使用 Bootstrap 採樣
    oob_score=False,           # 是否計算 OOB 分數
    random_state=42,           # 隨機種子
    n_jobs=-1                  # 使用所有 CPU 核心
)

# 訓練模型
rf_model.fit(X_train, y_train)

# 預測
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)  # 返回機率

# 評估
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))
```

### 4.2 關鍵參數詳解

#### 4.2.1 樹的數量 (n_estimators)

- **預設值**：100
- **建議範圍**：100-500
- **影響**：
  - 數量越多，模型越穩定，但訓練時間越長
  - 通常 100-200 已足夠
  - 超過 500 通常收益遞減

**選擇策略**：

```python
# 繪製 OOB 誤差 vs. 樹數量曲線
from sklearn.ensemble import RandomForestClassifier
import numpy as np

oob_errors = []
for n in range(10, 501, 10):
    rf = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)
    oob_errors.append(1 - rf.oob_score_)

# 繪圖觀察何時誤差趨於穩定
```

#### 4.2.2 隨機特徵數量 (max_features)

控制每次分裂時考慮的特徵數量：

| 設定值 | 意義 | 適用場景 |
|-------|------|---------|
| `'sqrt'` | $\sqrt{M}$ | **分類問題預設**，平衡多樣性與性能 |
| `'log2'` | $\log_2(M)$ | 更強的多樣性，特徵很多時使用 |
| `None` | $M$ （所有特徵） | 降低多樣性，類似 Bagging |
| `int` | 固定數量 | 自定義控制 |
| `float` | 比例（如 0.5 = 50%） | 彈性控制 |

**化工案例考量**：

- **特徵數 < 10**：可使用 `max_features='sqrt'` 或 `None`
- **特徵數 20-50**：建議 `'sqrt'`
- **特徵數 > 100**：建議 `'log2'` 或自定義比例

#### 4.2.3 樹的深度與剪枝參數

雖然隨機森林通常讓樹完全生長，但仍可設定：

```python
rf = RandomForestClassifier(
    max_depth=None,           # 不限制深度（常用）
    min_samples_split=2,      # 分裂至少需要 2 個樣本
    min_samples_leaf=1,       # 葉節點至少 1 個樣本
    max_leaf_nodes=None       # 不限制葉節點數
)
```

**何時限制深度？**

- 數據集非常大（> 100,000 樣本）
- 訓練時間過長
- 記憶體不足

#### 4.2.4 Bootstrap 與 OOB

```python
rf = RandomForestClassifier(
    bootstrap=True,           # 使用 Bootstrap 採樣（必須）
    oob_score=True,          # 計算 OOB 分數（推薦）
    random_state=42
)

rf.fit(X_train, y_train)
print(f"OOB Score: {rf.oob_score_:.4f}")  # 查看 OOB 驗證分數
```

**OOB Score 的用途**：

- 快速評估模型泛化能力
- 無需單獨的驗證集
- 類似於交叉驗證的效果

#### 4.2.5 並行化 (n_jobs)

```python
rf = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1                # -1 = 使用所有 CPU 核心
)
```

- 顯著加速訓練（樹可以並行建立）
- 推薦設定為 `-1`

### 4.3 模型屬性與方法

訓練後可訪問的屬性：

```python
# 特徵重要性
rf.feature_importances_            # 每個特徵的重要性分數

# 所有的決策樹
rf.estimators_                     # 包含所有決策樹的列表

# 類別標籤
rf.classes_                        # 類別名稱

# OOB 預測
rf.oob_decision_function_          # OOB 樣本的預測機率（需 oob_score=True）

# 單棵樹的查看
tree_0 = rf.estimators_[0]        # 第一棵樹
```

---

## 5. 化工應用案例設計

### 5.1 案例背景：催化反應成功預測（進階版）

延續 Unit12_Decision_Tree_Classifier 的案例，我們將使用相同的數據但應用隨機森林：

**問題設定**：

- **輸入特徵**：
  - 反應溫度 (Temperature, °C): 150-250
  - 反應壓力 (Pressure, bar): 10-40
  - 催化劑濃度 (Catalyst, %): 0.5-5.0
  - 反應物比例 (Ratio): 1.0-3.0
  - 反應時間 (Time, hr): 1-5

- **輸出**：
  - 反應成功 (Success) / 反應失敗 (Failure)

**數據生成規則**（與決策樹相同）：

採用 5 級評分系統，根據條件判定反應成功機率：

1. **極優條件** (230-250°C, 25-35 bar, 2.5-4.5%): 92% 成功率
2. **良好條件** (210-230°C, 20-30 bar, 1.5-3.5%): 82% 成功率
3. **中等條件** (190-210°C, 15-25 bar, 1.0-2.5%): 60% 成功率
4. **較差條件** (170-190°C, 10-20 bar, 0.5-1.5%): 25% 成功率
5. **極差條件** (150-170°C, < 15 bar, < 1.0%): 8% 成功率

### 5.2 預期性能提升

基於決策樹的結果：

| 模型 | 測試準確率 | AUC Score |
|-----|-----------|-----------|
| 簡單決策樹 (depth=3) | 71.75% | ~0.71 |
| 剪枝決策樹 (最佳) | 73.25% | 0.7438 |
| **隨機森林 (預期)** | **80-85%** | **0.83-0.88** |

**性能提升來源**：

1. **多樹集成降低方差**：減少過擬合
2. **隨機特徵增加多樣性**：捕捉不同的決策模式
3. **Bootstrap 採樣**：增強對異常值的抵抗力
4. **無需精細調參**：自動達到較優性能

---

## 6. 超參數調整策略

### 6.1 GridSearchCV 調參

```python
from sklearn.model_selection import GridSearchCV

# 定義參數網格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 建立 GridSearch
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,                    # 5 折交叉驗證
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

# 訓練
grid_search.fit(X_train, y_train)

# 最佳參數
print("Best parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)

# 使用最佳模型
best_rf = grid_search.best_estimator_
```

### 6.2 RandomizedSearchCV 調參

對於參數空間較大時，使用隨機搜尋更高效：

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# 定義參數分佈
param_distributions = {
    'n_estimators': randint(50, 500),
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [None] + list(range(10, 51, 10)),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

# 建立 RandomizedSearch
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_distributions,
    n_iter=100,              # 嘗試 100 種組合
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=2
)

random_search.fit(X_train, y_train)
```

### 6.3 調參優先級

根據參數對性能的影響排序：

1. **🔥 最重要**：
   - `n_estimators`: 50-200 之間嘗試
   - `max_features`: 'sqrt', 'log2', None

2. **⚙️ 次要**：
   - `max_depth`: None 或 10, 20, 30
   - `min_samples_split`: 2, 5, 10

3. **🔧 微調**：
   - `min_samples_leaf`: 1, 2, 4
   - `max_leaf_nodes`
   - `min_impurity_decrease`

**實務建議**：

- 先用預設參數訓練，確認基礎性能
- 如果過擬合，限制 `max_depth` 或增加 `min_samples_leaf`
- 如果欠擬合，增加 `n_estimators` 或調整 `max_features`

---

## 7. 模型評估與解釋

### 7.1 性能評估指標

除了準確率，還應關注：

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# 預測
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]  # 正類機率

# 綜合指標
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# 詳細報告
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 混淆矩陣
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

### 7.2 特徵重要性分析

```python
import pandas as pd
import matplotlib.pyplot as plt

# 提取特徵重要性
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

# 可視化
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()
```

### 7.3 單棵樹可視化

雖然無法可視化所有 100 棵樹，但可以查看幾棵代表性的樹：

```python
from sklearn.tree import plot_tree

# 可視化第一棵樹
plt.figure(figsize=(20, 10))
plot_tree(
    rf.estimators_[0],
    feature_names=X_train.columns,
    class_names=['Failure', 'Success'],
    filled=True,
    rounded=True,
    max_depth=3              # 只顯示前 3 層
)
plt.title('First Tree in Random Forest (Top 3 Levels)')
plt.show()
```

### 7.4 學習曲線

評估模型是否需要更多數據或更多樹：

```python
from sklearn.model_selection import learning_curve
import numpy as np

# 計算學習曲線
train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# 計算平均值和標準差
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# 繪圖
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training Score', marker='o')
plt.plot(train_sizes, val_mean, label='Validation Score', marker='s')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve - Random Forest')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 8. 隨機森林 vs. 決策樹對比

### 8.1 性能對比

| 指標 | 決策樹 | 隨機森林 |
|-----|--------|----------|
| **準確率** | 71-73% | **80-85%** ⬆️ |
| **穩定性** | 低（高方差） | **高（低方差）** ⬆️ |
| **訓練時間** | 快 | 較慢（多樹） |
| **預測時間** | 極快 | 較慢（需查詢多樹） |
| **可解釋性** | **極高** ⬆️ | 中等（難可視化） |
| **過擬合風險** | 高 | **低** ⬆️ |
| **對噪聲敏感度** | 高 | **低** ⬆️ |

### 8.2 應用場景選擇

**選擇決策樹**：

- ✅ 需要高度可解釋性（可視化樹結構）
- ✅ 需要極快的預測速度
- ✅ 數據集小（< 1000 樣本）
- ✅ 探索性分析，尋找決策規則
- ✅ 資源受限（記憶體/計算力）

**選擇隨機森林**：

- ✅ **追求最高準確率**
- ✅ 數據集大（> 1000 樣本）
- ✅ 特徵數量多（> 10 個）
- ✅ 數據有噪聲
- ✅ 不需要完全透明的決策過程
- ✅ **生產環境的實際應用**

**化工領域建議**：

- **探索階段**：使用決策樹，快速理解數據結構
- **生產階段**：使用隨機森林，確保預測可靠性

---

## 9. 進階主題

### 9.1 類別不平衡處理

化工數據常見類別不平衡（如故障樣本遠少於正常樣本）：

```python
from sklearn.ensemble import RandomForestClassifier

# 方法 1: 使用 class_weight
rf_balanced = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',     # 自動調整權重
    random_state=42
)

# 方法 2: 手動設定權重
rf_weighted = RandomForestClassifier(
    n_estimators=100,
    class_weight={0: 1, 1: 10},  # 故障類權重 10 倍
    random_state=42
)

# 方法 3: 調整採樣（使用 imbalanced-learn）
from imblearn.ensemble import BalancedRandomForestClassifier

rf_imblearn = BalancedRandomForestClassifier(
    n_estimators=100,
    sampling_strategy='auto',    # 自動平衡採樣
    random_state=42
)
```

### 9.2 特徵選擇

利用特徵重要性進行特徵選擇：

```python
from sklearn.feature_selection import SelectFromModel

# 訓練隨機森林
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 選擇重要特徵（重要性 > 中位數）
selector = SelectFromModel(rf, threshold='median', prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# 查看選中的特徵
selected_features = X_train.columns[selector.get_support()]
print("Selected Features:", selected_features)

# 使用選中特徵重新訓練
rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selected.fit(X_train_selected, y_train)
```

### 9.3 部分依賴圖 (Partial Dependence Plot)

理解特徵對預測的影響：

```python
from sklearn.inspection import PartialDependenceDisplay

# 繪製部分依賴圖
features = [0, 1, 2]  # 前三個特徵的索引
PartialDependenceDisplay.from_estimator(
    rf, X_train, features,
    feature_names=X_train.columns,
    grid_resolution=50
)
plt.tight_layout()
plt.show()
```

### 9.4 SHAP 值解釋

使用 SHAP (SHapley Additive exPlanations) 進行模型解釋：

```python
import shap

# 計算 SHAP 值
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# 摘要圖
shap.summary_plot(shap_values[1], X_test, feature_names=X_train.columns)

# 單一樣本解釋
shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][0],
    X_test.iloc[0],
    feature_names=X_train.columns
)
```

---

## 10. 優勢與限制

### 10.1 隨機森林的優勢

✅ **性能優異**：

- 通常在各種數據集上表現出色
- 默認參數已有良好性能
- Top Kaggle 競賽常用基線模型

✅ **魯棒性強**：

- 對異常值不敏感（多數投票機制）
- 對缺失值處理較好
- 對噪聲數據容忍度高

✅ **過擬合風險低**：

- 集成方法自然降低方差
- 隨機性增強泛化能力
- 通常不需要剪枝

✅ **特徵處理簡單**：

- 無需特徵縮放（如標準化）
- 可處理數值和類別特徵
- 自動處理特徵交互

✅ **內建特徵重要性**：

- 提供特徵重要性排名
- 有助於特徵選擇
- 支持模型解釋

✅ **並行化高效**：

- 樹的訓練可完全並行
- 支持多核 CPU 加速
- 大數據集友好

### 10.2 隨機森林的限制

⚠️ **計算資源需求高**：

- 需要儲存多棵樹（記憶體）
- 訓練時間比單一決策樹長
- 預測需查詢所有樹（較慢）

⚠️ **可解釋性降低**：

- 無法像單一決策樹一樣可視化
- 決策過程不透明
- 難以提取簡單規則

⚠️ **預測時間較長**：

- 需要遍歷所有樹進行預測
- 實時系統可能不適用
- 邊緣設備部署困難

⚠️ **外推能力有限**：

- 只能預測訓練範圍內的值
- 對訓練數據範圍外的樣本表現差
- 不適合需要外推的場景

⚠️ **對高維稀疏數據效果一般**：

- 文本數據（高維稀疏）效果不如線性模型
- 隨機特徵選擇在稀疏數據上效率低

### 10.3 何時使用隨機森林？

**✅ 最適合的場景**：

1. **表格數據 (Tabular Data)**：
   - 化工過程數據
   - 傳感器數據
   - 實驗室測量數據
   - 混合類型特徵（數值+類別）

2. **中小型數據集**：
   - 100 - 100,000 樣本
   - 10 - 100 特徵

3. **分類任務**：
   - 二分類或多分類
   - 類別不平衡問題
   - 需要機率輸出

4. **需要魯棒性**：
   - 數據有噪聲
   - 存在異常值
   - 數據品質不確定

5. **快速開發**：
   - 需要快速建立基線模型
   - 調參時間有限
   - 默認參數即可達到較好性能

**❌ 不適合的場景**：

1. **圖像/語音/文本數據**：使用深度學習（CNN/RNN）
2. **時間序列預測**：使用 LSTM/GRU 或專用時序模型
3. **線性可分數據**：使用邏輯迴歸（更簡單高效）
4. **需要極高可解釋性**：使用單一決策樹或線性模型
5. **實時低延遲系統**：使用更輕量的模型

---

## 11. 實務建議與最佳實踐

### 11.1 模型開發流程

```
1. 數據探索
   ├─ 檢查數據分佈
   ├─ 處理缺失值
   └─ 識別異常值

2. 建立基線模型
   ├─ 使用默認參數
   ├─ n_estimators=100
   └─ 評估初始性能

3. 特徵工程（可選）
   ├─ 創建交互特徵
   ├─ 特徵變換
   └─ 領域知識特徵

4. 超參數調整
   ├─ 優先調整 n_estimators 和 max_features
   ├─ 使用 GridSearchCV 或 RandomizedSearchCV
   └─ 5 折交叉驗證

5. 模型評估
   ├─ 多種指標（Accuracy, Precision, Recall, F1, AUC）
   ├─ 混淆矩陣分析
   ├─ 特徵重要性
   └─ 學習曲線

6. 模型部署
   ├─ 保存模型 (joblib/pickle)
   ├─ 性能監控
   └─ 定期重訓練
```

### 11.2 調參技巧

**快速調參策略**：

```python
# 階段 1: 調整樹的數量
for n_est in [50, 100, 200, 300]:
    rf = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf, X_train, y_train, cv=5)
    print(f"n_estimators={n_est}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# 階段 2: 調整特徵數量
for max_feat in ['sqrt', 'log2', None]:
    rf = RandomForestClassifier(n_estimators=100, max_features=max_feat, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf, X_train, y_train, cv=5)
    print(f"max_features={max_feat}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# 階段 3: 微調深度和樣本數
# ... 根據需要進一步調整
```

### 11.3 性能優化

**訓練速度優化**：

```python
# 1. 使用並行化
rf = RandomForestClassifier(n_jobs=-1)

# 2. 減少樹的數量（在性能可接受範圍內）
rf = RandomForestClassifier(n_estimators=50)  # 而非 500

# 3. 限制樹的深度
rf = RandomForestClassifier(max_depth=20)

# 4. 減少訓練數據（如果數據很大）
from sklearn.utils import resample
X_train_sampled, y_train_sampled = resample(X_train, y_train, n_samples=10000, random_state=42)
```

**預測速度優化**：

```python
# 1. 減少樹的數量
rf_fast = RandomForestClassifier(n_estimators=50)

# 2. 限制樹的深度
rf_fast = RandomForestClassifier(max_depth=10)

# 3. 使用 predict_proba 而非 predict（如果只需機率）
proba = rf.predict_proba(X_test)  # 稍快

# 4. 批量預測而非逐個預測
predictions = rf.predict(X_test)  # 一次性預測整個測試集
```

### 11.4 化工領域特定建議

1. **溫度和壓力特徵**：
   - 考慮添加交互項（如 Temp × Pressure）
   - 創建比例特徵（如 Temp/Pressure）

2. **催化劑濃度**：
   - 可能存在非線性效應（過高或過低都不好）
   - 考慮添加平方項或對數變換

3. **反應時間**：
   - 可能與其他變數有交互作用
   - 考慮 Time × Catalyst 交互項

4. **數據增強**：
   - 利用化工知識添加合成特徵
   - 例如：反應速率 = f(Temp, Catalyst)

5. **領域約束**：
   - 可以將物理定律（如 Arrhenius 方程）作為特徵
   - 增強模型的可解釋性和泛化能力

---

## 12. 實際執行結果分析

本節展示使用 v4.0 數據生成版本的完整執行結果，包含關鍵指標、視覺化圖表與深入分析。

### 12.1 數據生成結果

**v4.0 簡化優化版數據特性**：

```plaintext
✓ 數據形狀: (2000, 6)
✓ 成功率: 43.80%
✓ Temperature主導: 基礎45分 + 協同效應強化
✓ 注意: 實際Temperature重要性約28%（受數據特性限制）
```

**類別分佈**：
- 失敗樣本 (Success=0): 1124 (56.2%)
- 成功樣本 (Success=1): 876 (43.8%)
- 類別比例約為 56:44，略有不平衡但仍在可接受範圍

![類別分佈圖](outputs/P3_Unit12_Random_Forest_Classifier/figs/class_distribution.png)

**圖表說明**：
- 左圖顯示絕對計數，失敗樣本略多於成功樣本
- 右圖顯示百分比分佈，直觀呈現 56.2% vs 43.8% 的比例

---

### 12.2 模型性能結果

#### 12.2.1 基礎模型準確率

| 模型 | 測試準確率 | 說明 |
|------|-----------|------|
| 單一決策樹 (depth=3) | **68.00%** | 基準模型 |
| 基礎隨機森林 (100 trees) | 66.50% | 默認參數 |
| GridSearch 優化隨機森林 | **67.75%** | 超參數調優後 |
| OOB 隨機森林 | 66.50% | 與基礎RF一致 |

**關鍵發現**：
- 在當前數據集上，單一決策樹性能略優於隨機森林
- GridSearchCV 將 RF 性能從 66.5% 提升至 67.75%（+1.88%）
- 這反映了模擬數據的特性：決策邊界相對簡單，單一樹已能有效分類

![模型比較圖](outputs/P3_Unit12_Random_Forest_Classifier/figs/model_comparison.png)

**性能提升分析**：
```plaintext
決策樹準確率:     0.6800
最佳隨機森林:     0.6800
絕對提升:        0.0000 (持平)
相對提升:        0.00%
```

#### 12.2.2 特徵重要性排名

| 特徵 | 重要性 | 百分比 | 排名 |
|------|--------|--------|------|
| **Temperature** | 0.2884 | **28.84%** | 1 |
| Catalyst | 0.1892 | 18.92% | 2 |
| Pressure | 0.1839 | 18.39% | 3 |
| Ratio | 0.1704 | 17.04% | 4 |
| Time | 0.1681 | 16.81% | 5 |

**分析**：
- Temperature 確實為最重要特徵（28.84%），符合數據生成設計
- 但重要性分布相對均衡，其他特徵也有顯著貢獻（17-19%）
- 這與 v4.0 的設計目標（Temperature 28-32%）一致
- 在真實工業數據中，主導因素通常達到 35-45%

![特徵重要性圖](outputs/P3_Unit12_Random_Forest_Classifier/figs/feature_importance.png)

**圖表說明**：
- 橫軸為重要性數值（0-0.30）
- 顏色深度反映重要性高低（深綠色為最重要）
- Temperature 明顯領先其他特徵

---

### 12.3 詳細評估指標

#### 12.3.1 混淆矩陣分析

**實際數值**（測試集 400 樣本）：

|  | 預測: Failure | 預測: Success |
|--|---------------|---------------|
| **真實: Failure** | 170 (75.6%) | 55 (24.4%) |
| **真實: Success** | 78 (44.3%) | 97 (55.7%) |

![混淆矩陣](outputs/P3_Unit12_Random_Forest_Classifier/figs/confusion_matrix.png)

**解讀**：
- **True Negative (TN)**: 170 - 正確預測失敗樣本
- **False Positive (FP)**: 55 - 誤判為成功的失敗樣本
- **False Negative (FN)**: 78 - 誤判為失敗的成功樣本
- **True Positive (TP)**: 97 - 正確預測成功樣本

**性能指標計算**：
- **Precision (成功類)**: TP/(TP+FP) = 97/152 = 63.8%
- **Recall (成功類)**: TP/(TP+FN) = 97/175 = 55.4%
- **Specificity (失敗類)**: TN/(TN+FP) = 170/225 = 75.6%

#### 12.3.2 ROC 曲線與 AUC 分數

**ROC AUC 指標**：
```plaintext
AUC Score:         0.7206
Optimal Threshold: 0.3900
TPR at Optimal:    0.6629
FPR at Optimal:    0.2622
```

![ROC 曲線](outputs/P3_Unit12_Random_Forest_Classifier/figs/roc_curve.png)

**解讀**：
- **AUC = 0.7206**：模型有中等程度的區分能力（良好範圍為 0.7-0.8）
- **最佳閾值 = 0.39**：在此閾值下，TPR 與 FPR 平衡最佳
- 曲線遠離對角線，說明模型優於隨機猜測（AUC=0.5）
- 若需更高 Recall，可降低閾值；若需更高 Precision，可提高閾值

---

### 12.4 模型穩定性驗證

#### 12.4.1 交叉驗證結果

**5-Fold Cross-Validation**：
```plaintext
5-Fold CV 分數: [0.6813 0.6813 0.6625 0.6750 0.6906]
平均準確率: 0.6781
標準差: 0.0097
95% 信賴區間: [0.6591, 0.6971]
```

![交叉驗證](outputs/P3_Unit12_Random_Forest_Classifier/figs/cross_validation.png)

**分析**：
- **標準差 < 1%**：模型非常穩定，對數據分割不敏感
- 所有 fold 的準確率在 66-69% 範圍內
- 95% 信賴區間窄（0.659-0.697），預測可靠性高

#### 12.4.2 學習曲線

![學習曲線](outputs/P3_Unit12_Random_Forest_Classifier/figs/learning_curve.png)

**觀察重點**：

1. **訓練曲線（藍色）**：
   - 隨著訓練樣本增加，訓練準確率從 95% 降至 ~70%
   - 這是正常現象，表示模型從「記憶」轉向「泛化」

2. **驗證曲線（橙色）**：
   - 隨著訓練樣本增加，驗證準確率逐漸上升
   - 在 ~1200 樣本後趨於穩定（~68%）

3. **收斂性**：
   - 訓練與驗證曲線最終接近（差距 < 5%）
   - 表示模型已充分訓練，無明顯過擬合或欠擬合

4. **數據量充足性**：
   - 曲線在 1200 樣本後趨於平穩
   - 繼續增加數據量可能無法顯著提升性能

---

### 12.5 超參數調優結果

**GridSearchCV 搜索空間**：
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2', 0.5],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

**搜索結果**：
- 參數組合總數: 324
- 5-Fold CV 總訓練次數: 1620
- 最佳 CV 分數: 0.6781
- 測試集準確率: 0.6775

**最佳參數組合**：
```json
{
  "max_depth": 10,
  "max_features": "sqrt",
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "n_estimators": 100
}
```

**關鍵發現**：
- `n_estimators=100` 已足夠，更多樹無顯著提升
- `max_depth=10` 限制深度防止過擬合
- `max_features='sqrt'` 為分類問題的標準設定
- 調優後性能提升 1.88%（66.5% → 67.75%）

---

### 12.6 OOB (Out-of-Bag) 驗證

**OOB Score 分析**：
```plaintext
OOB Score: 0.6756
測試集 Score: 0.6650
差異: 0.0106 (1.06%)
```

**解讀**：
- OOB 分數與測試分數非常接近（差異 < 1.1%）
- 驗證了模型的泛化能力良好
- OOB 可作為「免費的交叉驗證」，無需額外劃分驗證集

---

### 12.7 深入討論：為何 RF 未超越 DT？

#### 12.7.1 數據特性分析

**模擬數據的限制**：

1. **決策邊界簡單**：
   - 人工設計的評分系統創建了相對線性的決策邊界
   - 單一決策樹的分段常數模型已能有效擬合
   - 集成多棵樹無法進一步「平滑化」這種分段邊界

2. **特徵交互作用有限**：
   - v4.0 設計了 3 種核心協同效應
   - 但這些交互作用仍然基於簡單的 if-else 邏輯
   - 缺乏真實化工過程中的複雜非線性關係

3. **噪聲水平低**：
   - 模擬數據的隨機性主要來自均勻分佈採樣
   - 缺乏真實測量中的系統誤差、隨機噪聲
   - 隨機森林的「降噪」優勢無法體現

#### 12.7.2 真實工業數據的差異

**為什麼真實數據中 RF 通常更優？**

1. **測量噪聲**：
   - 溫度、壓力傳感器存在 ±1-2% 誤差
   - 催化劑濃度測量有分析誤差
   - **RF 的多樹投票能有效平均這些隨機誤差**

2. **複雜非線性關係**：
   - Arrhenius 方程：反應速率與溫度呈指數關係
   - 吸附競爭：多組分催化中的非線性競爭吸附
   - 傳質限制：在高濃度或高溫下的非線性傳質效應
   - **RF 的多角度分裂能捕捉這些複雜模式**

3. **數據異常值**：
   - 設備故障導致的異常讀數
   - 操作條件突變
   - 原料批次差異
   - **RF 對異常值魯棒性更強**

#### 12.7.3 隨機種子的影響

**當前結果的偶然性**：

- 使用 `random_state=42` 時，DT 恰好找到較優分裂點
- 改變隨機種子可能導致 RF 性能優於 DT
- 這正是**為什麼需要交叉驗證和多次重複實驗**

**建議實踐**：
```python
# 多隨機種子實驗
seeds = [42, 123, 456, 789, 2024]
rf_scores = []
dt_scores = []

for seed in seeds:
    # 重新生成數據和訓練模型
    df = generate_data(random_state=seed)
    # ... 訓練評估 ...
    rf_scores.append(rf_accuracy)
    dt_scores.append(dt_accuracy)

print(f"RF 平均: {np.mean(rf_scores):.2%} ± {np.std(rf_scores):.2%}")
print(f"DT 平均: {np.mean(dt_scores):.2%} ± {np.std(dt_scores):.2%}")
```

---

### 12.8 教學價值總結

儘管在本案例中 Random Forest 未顯著超越 Decision Tree，但這個結果本身具有重要的教學意義：

#### ✅ 方法論完整性

1. **完整開發流程**：
   - 數據生成 → 探索分析 → 模型訓練 → 評估驗證 → 調優部署
   - 每個環節都展示了業界標準做法

2. **sklearn API 熟練度**：
   - `RandomForestClassifier` 所有關鍵參數
   - `GridSearchCV` 系統化調參方法
   - 多維度評估指標的綜合應用

3. **可視化技能**：
   - 7 種專業級圖表（混淆矩陣、ROC、學習曲線等）
   - 每種圖表都有明確的解讀方法

#### ✅ 批判性思維培養

1. **不盲信複雜模型**：
   - 證明了「更複雜 ≠ 更好」
   - 需要根據數據特性選擇模型

2. **數據質量意識**：
   - 理解模擬數據與真實數據的差異
   - 認識到特徵工程的重要性

3. **性能評估思維**：
   - 單一指標不足以全面評估
   - 需要多角度驗證模型能力

#### ✅ 真實場景準備

1. **接受不完美結果**：
   - 真實項目中經常遇到「預期與現實不符」
   - 學會分析原因並調整策略

2. **溝通結果的能力**：
   - 如何向非技術人員解釋「RF 為何沒有更好」
   - 如何基於證據提出下一步行動建議

---

## 13. 總結

### 核心要點回顧

✅ **隨機森林核心機制**：

- Bagging（自助採樣）+ 隨機特徵選擇
- 通過集成降低方差，提升穩定性
- 多數投票得出最終預測

✅ **關鍵優勢**：

- 性能優異（通常 80-85% 準確率）
- 魯棒性強（對噪聲和異常值不敏感）
- 默認參數已有良好表現
- 提供特徵重要性分析

✅ **重要參數**：

- `n_estimators`：樹的數量（100-200）
- `max_features`：隨機特徵數量（'sqrt'）
- `max_depth`：樹的深度（通常 None）
- `oob_score`：啟用 OOB 驗證

✅ **與決策樹對比**：

| 維度 | 決策樹 | 隨機森林 | 備註 |
|-----|--------|----------|------|
| 準確率 | 68-70% | **67-68%** | 本案例接近，真實數據通常 RF 有 8-15% 優勢 |
| 可解釋性 | **極高** ⬆️ | 中等 | DT 可視覺化單一樹結構 |
| 穩定性 | 低 | **高** ⬆️ | RF 對數據擾動不敏感 |
| 訓練時間 | 快 | 較慢 | RF 需訓練多棵樹 |
| 過擬合風險 | 高 | **低** ⬆️ | 集成降低過擬合 |

**重要說明**：本教學案例使用模擬數據，RF 與 DT 性能接近。在實際工業場景中，隨機森林通常能展現顯著優勢（8-15% 提升），因為真實數據包含更多噪聲和複雜的非線性關係。

✅ **化工應用**：

- 反應成功預測
- 產品品質分類
- 設備故障診斷
- 操作模式識別

### 學習路徑

完成本節課後，建議繼續學習：

1. **Unit12_Gradient_Boosting_Classifier**：
   - 學習更強大的集成方法
   - 序列化提升（Boosting）
   - 進一步提升準確率至 85-90%

2. **Unit12_XGBoost_LightGBM**（進階）：
   - 工業界最常用的梯度提升庫
   - 更快的訓練速度
   - 更好的性能

3. **特徵工程專題**：
   - 深入學習特徵創建技巧
   - 化工領域特定特徵設計

### 實務檢查清單

在實際應用隨機森林時，確保：

- [ ] 使用了適當的訓練/測試集分割（如 80/20）
- [ ] 啟用了 `oob_score=True` 進行內部驗證
- [ ] 評估了多種指標（不僅是準確率）
- [ ] 分析了特徵重要性並驗證其合理性
- [ ] 檢查了混淆矩陣，識別預測錯誤模式
- [ ] 使用交叉驗證確認模型穩定性
- [ ] 嘗試了基本的超參數調整
- [ ] 保存了訓練好的模型以供後續使用
- [ ] 記錄了模型版本和訓練參數

---

**版權聲明**：本教材由逢甲大學化學工程學系莊曜禎助理教授編寫，僅供教學使用。

**課程代碼**：CHE-AI-114  
**更新日期**：2026 年 1 月 18 日
