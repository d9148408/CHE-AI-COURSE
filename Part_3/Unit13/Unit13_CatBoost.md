# Unit 13: CatBoost 模型 (Categorical Boosting)

**課程名稱**：AI 在化工上之應用  
**課程代碼**：CHE-AI-114  
**授課教師**：莊曜禎 助理教授  
**單元主題**：CatBoost 模型  
**適用對象**：化學工程學系學生  

---

## 學習目標

完成本單元後，學生將能夠：

1. 理解 CatBoost 的核心原理與創新技術
2. 掌握 CatBoost 相較於 XGBoost 與 LightGBM 的優勢
3. 安裝並使用 CatBoost 套件進行回歸與分類任務
4. 理解 Ordered Boosting 與 Target Statistics 技術
5. 掌握 CatBoost 的類別特徵自動處理機制
6. 應用 CatBoost 處理不平衡資料問題
7. 理解並設定 CatBoost 的超參數
8. 使用 CatBoost 的可視化與模型分析工具
9. 應用 CatBoost 於化工製程預測問題
10. 比較 GBDT 三巨頭（XGBoost、LightGBM、CatBoost）的異同

---

## 1. CatBoost 簡介

### 1.1 什麼是 CatBoost？

**CatBoost (Categorical Boosting)** 是由俄羅斯科技公司 Yandex 於 2017 年開發的梯度提升框架，專注於**類別特徵處理**與**預測準確性**。

**核心特色**：
- **類別特徵處理專家**：無需預處理，自動最佳化編碼
- **高準確度**：Ordered Boosting 技術減少過擬合
- **易於使用**：預設參數即有優異表現
- **魯棒性強**：對超參數不敏感，訓練穩定
- **內建交叉驗證**：自動偵測過擬合
- **豐富可視化**：訓練過程與特徵分析工具完整
- **支援 GPU 加速**：大幅提升訓練速度

**CatBoost 的歷史與影響**：
- 2017 年由 Yandex 發表並開源
- 論文："CatBoost: unbiased boosting with categorical features"
- 在處理高基數類別特徵上表現卓越
- 在 Kaggle 等競賽中與 XGBoost、LightGBM 並列三強
- 特別適合工業界真實場景（含大量類別變數）

### 1.2 為什麼需要 CatBoost？

**XGBoost 與 LightGBM 的侷限**（類別特徵處理）：
- **需要手動編碼**：One-Hot Encoding 導致高維稀疏
- **Target Leakage 風險**：Target Encoding 容易過擬合
- **高基數問題**：類別數量多時效果不佳
- **順序敏感**：訓練資料順序影響結果

**CatBoost 的突破**：
- **Ordered Target Statistics**：解決 Target Leakage 問題
- **Ordered Boosting**：減少預測偏移（Prediction Shift）
- **原生類別特徵支援**：自動最佳化處理
- **對稱樹結構**：提升推理速度與模型穩定性
- **魯棒性設計**：預設參數即可達到優異效果

### 1.3 CatBoost vs XGBoost vs LightGBM

| 特性 | XGBoost | LightGBM | CatBoost |
|------|---------|----------|----------|
| **訓練速度** | 快 | 非常快 | 中等 |
| **準確度** | 非常高 | 非常高 | 最高 |
| **類別特徵** | 需編碼 | 原生支援（基本） | 原生支援（最佳） |
| **高基數類別** | 弱 | 中 | 強 |
| **Target Leakage** | 有風險 | 有風險 | 已解決 |
| **預設參數** | 需調整 | 需調整 | 開箱即用 |
| **過擬合風險** | 中 | 較高 | 低 |
| **魯棒性** | 中 | 低 | 高 |
| **樹結構** | 不對稱 | 不對稱 | 對稱 |
| **可視化** | 基本 | 基本 | 豐富 |
| **GPU 支援** | 有 | 有 | 有（最佳） |
| **不平衡資料** | 基本 | 基本 | 進階支援 |
| **調參難度** | 中 | 高 | 低 |

### 1.4 CatBoost 的應用場景

**最適合的情境**：
- **大量類別特徵**：用戶 ID、產品編號、位置代碼等
- **高基數類別**：類別數量 > 100
- **需要高準確度**：對精度要求高的場景
- **快速原型開發**：希望用預設參數快速建模
- **不平衡資料**：正負樣本比例懸殊
- **穩定性要求高**：生產環境需要魯棒模型

**化工領域應用案例**：

| 應用領域 | 預測目標 | CatBoost 優勢 |
|---------|---------|--------------|
| **配方優化** | 產品性質預測 | 自動處理原料編號、供應商等類別 |
| **批次追蹤** | 品質分類 | 處理批次號、設備編號等高基數特徵 |
| **製程監控** | 異常檢測 | 魯棒性強，誤報率低 |
| **供應鏈優化** | 交貨時間預測 | 處理供應商、物流路線等類別 |
| **設備診斷** | 故障類型分類 | 不平衡資料處理能力強 |
| **多廠區建模** | 統一產率模型 | 自動處理廠區、產線等類別差異 |

---

## 2. CatBoost 核心技術原理

### 2.1 類別特徵處理技術

#### 2.1.1 傳統方法的問題

**One-Hot Encoding 的問題**：
- 高基數類別導致特徵爆炸（例如：1000 個類別 → 1000 個特徵）
- 稀疏矩陣消耗大量記憶體
- 樹模型分裂效率低

**Target Encoding 的問題**：
- **Target Leakage**：使用目標值統計會導致過擬合
- 訓練集與測試集分佈不一致

**範例**：假設我們有類別特徵「供應商」與目標「產率」

| 樣本 | 供應商 | 產率 |
|------|--------|------|
| 1 | A | 85 |
| 2 | A | 90 |
| 3 | B | 75 |
| 4 | A | 88 |

傳統 Target Encoding：供應商 A → (85+90+88)/3 = 87.67

**問題**：計算樣本 1 的特徵時，使用了包含樣本 1 自己的目標值！

#### 2.1.2 CatBoost 的解決方案：Ordered Target Statistics

**核心思想**：計算某樣本的類別編碼時，只使用**該樣本之前**的資料。

**演算法步驟**：
1. 對訓練資料進行隨機排列
2. 對每個樣本 $i$ 與類別 $c$ ，計算：

$$
\text{TargetStat}_i(c) = \frac{\sum_{j<i, \text{cat}_j=c} y_j + \alpha \cdot P}{\sum_{j<i, \text{cat}_j=c} 1 + \alpha}
$$

其中：
- $y_j$ ：樣本 $j$ 的目標值
- $\alpha$ ：平滑係數（Prior 權重）
- $P$ ：先驗值（通常為目標的全局平均）

**範例**（接續上表）：

| 順序 | 樣本 | 供應商 | 真實產率 | Ordered Encoding |
|------|------|--------|---------|-----------------|
| 1 | 1 | A | 85 | P (先驗值，如 80) |
| 2 | 4 | A | 88 | 85 (只用樣本1) |
| 3 | 2 | A | 90 | (85+88)/2 = 86.5 |
| 4 | 3 | B | 75 | P (B 首次出現) |

**優勢**：
- 完全避免 Target Leakage
- 保留類別特徵的預測能力
- 自動處理新類別（使用先驗值）

#### 2.1.3 多重隨機排列

CatBoost 使用**多組隨機排列**，進一步提升魯棒性：
- 每棵樹使用不同的隨機排列
- 減少排列順序對結果的影響
- 類似 Bagging 的集成效果

### 2.2 Ordered Boosting

#### 2.2.1 傳統 Boosting 的 Prediction Shift 問題

**Prediction Shift**：訓練時使用的模型預測值，與測試時的預測值分佈不同。

**原因**：傳統 GBDT 訓練殘差時：
1. 使用當前模型預測訓練集
2. 計算殘差
3. 訓練下一棵樹

**問題**：預測訓練集時，每個樣本都**參與了**模型訓練，導致過於"樂觀"的預測。

#### 2.2.2 CatBoost 的 Ordered Boosting

**核心思想**：訓練第 $t$ 棵樹時，對每個樣本 $i$ ：
- 只使用**該樣本之前的樣本**訓練的模型來預測
- 計算殘差
- 確保預測值與測試時一致

**演算法**（簡化版）：
1. 對資料進行隨機排列
2. 對每個樣本 $i$ ：
   - 使用樣本 $1$ 到 $i-1$ 訓練一個模型 $M_{<i}$
   - 用 $M_{<i}$ 預測樣本 $i$ ，計算殘差
3. 使用所有殘差訓練新樹

**數學表示**：

$$
g_i = y_i - M_{<i}(x_i)
$$

其中 $M_{<i}$ 只在樣本 $1, \ldots, i-1$ 上訓練。

**實際實現**：完全實現開銷太大，CatBoost 使用**多個排列**的近似方法。

### 2.3 對稱樹結構 (Oblivious Trees)

#### 2.3.1 傳統樹 vs 對稱樹

**傳統決策樹（不對稱樹）**：
- 每個節點可以有不同的分裂特徵與閾值
- 靈活但複雜

**對稱樹（Oblivious Tree）**：
- 同一層的所有節點使用**相同的分裂條件**
- 樹的深度決定葉節點數量： $2^{\text{depth}}$

**示意圖**：

```
對稱樹 (深度=2)：
         [Feature A < 5?]
        /                \
   [Feature B < 10?]   [Feature B < 10?]
   /      \            /      \
 Leaf1  Leaf2      Leaf3  Leaf4
```

所有第二層節點都用 "Feature B < 10?" 分裂。

#### 2.3.2 對稱樹的優勢

**優勢**：
1. **快速推理**：預測時只需計算 depth 次比較
2. **減少過擬合**：結構限制降低模型複雜度
3. **高效實現**：便於並行化與 GPU 加速
4. **模型穩定**：對資料擾動不敏感

**劣勢**：
- 單棵樹的表達能力略低（需更多樹）

### 2.4 CatBoost 演算法流程總結

**完整訓練流程**：

1. **初始化**：
   - 設定初始預測值（通常為目標平均值）
   - 生成多組隨機排列

2. **For each tree** $t = 1$ **to** $T$ ：
   - **類別特徵編碼**：使用 Ordered Target Statistics
   - **Ordered Boosting**：計算每個樣本的梯度（基於有序預測）
   - **樹生長**：
     - 貪婪選擇最佳分裂特徵與閾值
     - 構建對稱樹
   - **更新模型**：
     - $F_t(x) = F_{t-1}(x) + \eta \cdot \text{Tree}_t(x)$

3. **輸出最終模型**： $F(x) = F_T(x)$

---

## 3. CatBoost 套件安裝與基本使用

### 3.1 安裝 CatBoost

#### 3.1.1 使用 pip 安裝

```bash
# 基本安裝
pip install catboost

# 指定版本
pip install catboost==1.2

# 升級到最新版
pip install --upgrade catboost
```

#### 3.1.2 驗證安裝

```python
import catboost
print(f"CatBoost version: {catboost.__version__}")
```

#### 3.1.3 GPU 支援

CatBoost 預設支援 GPU，無需額外安裝。使用時設定 `task_type='GPU'` 即可。

**要求**：
- NVIDIA GPU with CUDA support
- CUDA Toolkit (通常 TensorFlow 已安裝)

### 3.2 CatBoost 的主要類別

CatBoost 提供兩種主要 API：

#### 3.2.1 Scikit-learn 風格 API（推薦新手）

```python
from catboost import CatBoostRegressor, CatBoostClassifier

# 回歸
model = CatBoostRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 分類
model = CatBoostClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

#### 3.2.2 原生 API（進階功能）

```python
from catboost import CatBoost, Pool

# 創建數據池
train_pool = Pool(X_train, y_train, cat_features=[0, 2])
test_pool = Pool(X_test, cat_features=[0, 2])

# 訓練
model = CatBoost({'iterations': 100, 'learning_rate': 0.1})
model.fit(train_pool)

# 預測
y_pred = model.predict(test_pool)
```

### 3.3 數據準備：Pool 類別

**Pool** 是 CatBoost 的專屬數據結構，提供高效處理。

#### 3.3.1 基本用法

```python
from catboost import Pool

# 創建訓練集 Pool
train_pool = Pool(
    data=X_train,           # 特徵
    label=y_train,          # 目標
    cat_features=[0, 2, 5], # 類別特徵索引
    feature_names=['feature1', 'feature2', ...],
    weight=sample_weights   # 樣本權重（選填）
)

# 創建測試集 Pool（無標籤）
test_pool = Pool(
    data=X_test,
    cat_features=[0, 2, 5]
)
```

#### 3.3.2 指定類別特徵的方法

**方法 1：使用索引**
```python
cat_features = [0, 2, 5]  # 第 0、2、5 列是類別特徵
```

**方法 2：使用特徵名稱**
```python
cat_features = ['supplier', 'reactor_type', 'shift']
```

**方法 3：自動偵測（不推薦）**
```python
# CatBoost 會自動將 object 或 category 類型視為類別特徵
# 但明確指定更安全
```

#### 3.3.3 Pool 的優勢

- **高效記憶體使用**：內部優化資料結構
- **類別特徵管理**：自動追蹤類別特徵
- **缺失值處理**：自動處理 NaN
- **快速訓練**：避免重複資料轉換

### 3.4 回歸任務基本範例

#### 3.4.1 簡單回歸

```python
from catboost import CatBoostRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. 生成模擬數據
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 建立模型
model = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    verbose=False  # 關閉訓練輸出
)

# 3. 訓練
model.fit(X_train, y_train)

# 4. 預測與評估
y_pred = model.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

#### 3.4.2 處理類別特徵的回歸

```python
import pandas as pd
from catboost import CatBoostRegressor, Pool

# 1. 創建含類別特徵的數據
data = pd.DataFrame({
    'temperature': np.random.uniform(300, 400, 1000),
    'pressure': np.random.uniform(1, 10, 1000),
    'catalyst': np.random.choice(['A', 'B', 'C'], 1000),
    'reactor': np.random.choice(['R1', 'R2', 'R3', 'R4'], 1000),
    'yield': np.random.uniform(60, 95, 1000)
})

# 2. 分割數據
X = data.drop('yield', axis=1)
y = data['yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 指定類別特徵
cat_features = ['catalyst', 'reactor']

# 4. 建立模型
model = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    cat_features=cat_features,  # 方法1：直接指定
    verbose=False
)

# 5. 訓練
model.fit(X_train, y_train)

# 6. 預測
y_pred = model.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
```

**或使用 Pool**：

```python
# 使用 Pool 指定類別特徵
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, cat_features=cat_features)

model = CatBoostRegressor(iterations=100, learning_rate=0.1, verbose=False)
model.fit(train_pool)
y_pred = model.predict(test_pool)
```

### 3.5 分類任務基本範例

#### 3.5.1 二元分類

```python
from catboost import CatBoostClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score

# 1. 生成數據
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 建立模型
model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    verbose=False
)

# 3. 訓練
model.fit(X_train, y_train)

# 4. 預測
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
```

#### 3.5.2 多元分類

```python
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report

# 1. 生成多類別數據
X, y = make_classification(
    n_samples=1000, 
    n_features=10, 
    n_classes=3,      # 3 類
    n_informative=8,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 建立模型
model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    loss_function='MultiClass',  # 明確指定多類別
    verbose=False
)

# 3. 訓練與預測
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```

---

## 4. CatBoost 重要超參數

### 4.1 基本超參數

#### 4.1.1 iterations（迭代次數）

- **意義**：樹的數量
- **範圍**：通常 100-10000
- **預設**：1000
- **調整建議**：
  - 搭配早停使用，設定較大值
  - 數據量大時可增加

```python
model = CatBoostRegressor(iterations=500)
```

#### 4.1.2 learning_rate（學習率）

- **意義**：每棵樹的權重
- **範圍**：0.01-0.3
- **預設**：自動（根據 iterations 調整）
- **調整建議**：
  - 與 iterations 呈反比關係
  - 較小學習率需更多迭代

```python
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05  # 較小的學習率
)
```

#### 4.1.3 depth（樹深度）

- **意義**：對稱樹的深度
- **範圍**：1-16
- **預設**：6
- **調整建議**：
  - 深度越大，模型越複雜
  - CatBoost 對深度不敏感（對稱樹特性）
  - 通常 4-10 足夠

```python
model = CatBoostRegressor(depth=8)
```

#### 4.1.4 l2_leaf_reg（L2 正則化）

- **意義**：葉節點輸出的 L2 正則化係數
- **範圍**：0-∞
- **預設**：3.0
- **調整建議**：
  - 增加可減少過擬合
  - CatBoost 預設值已很好

```python
model = CatBoostRegressor(l2_leaf_reg=5.0)
```

### 4.2 類別特徵相關參數

#### 4.2.1 cat_features

```python
# 指定類別特徵
model = CatBoostRegressor(
    cat_features=['catalyst', 'reactor', 'shift']
)
```

#### 4.2.2 one_hot_max_size

- **意義**：當類別數量 ≤ 此值時，使用 One-Hot Encoding
- **預設**：2
- **調整建議**：
  - 低基數類別（≤ 10 類）可提高到 10
  - 高基數類別保持預設

```python
model = CatBoostRegressor(
    cat_features=[0, 1],
    one_hot_max_size=10  # ≤10 類用 One-Hot
)
```

### 4.3 處理不平衡資料

#### 4.3.1 class_weights

```python
# 自動平衡
model = CatBoostClassifier(
    class_weights='Balanced'
)

# 手動設定權重
model = CatBoostClassifier(
    class_weights={0: 1.0, 1: 5.0}  # 類別 1 的權重為 5
)
```

#### 4.3.2 auto_class_weights

```python
model = CatBoostClassifier(
    auto_class_weights='Balanced'  # 自動計算類別權重
)
```

### 4.4 早停與過擬合控制

#### 4.4.1 early_stopping_rounds

```python
model = CatBoostRegressor(
    iterations=1000,
    early_stopping_rounds=50,  # 驗證集 50 輪無改善則停止
    verbose=False
)

model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val)  # 必須提供驗證集
)
```

#### 4.4.2 use_best_model

```python
model = CatBoostRegressor(
    iterations=1000,
    use_best_model=True,  # 使用驗證集上最佳的模型
    verbose=False
)

model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val)
)

print(f"Best iteration: {model.best_iteration_}")
```

### 4.5 加速與性能參數

#### 4.5.1 task_type（CPU / GPU）

```python
# 使用 GPU（大幅加速）
model = CatBoostRegressor(
    task_type='GPU',
    devices='0'  # 使用第 0 張 GPU
)

# 使用 CPU
model = CatBoostRegressor(
    task_type='CPU',
    thread_count=-1  # 使用所有 CPU 核心
)
```

#### 4.5.2 thread_count

```python
model = CatBoostRegressor(
    thread_count=4  # 使用 4 個 CPU 核心
)
```

### 4.6 常用超參數組合

#### 4.6.1 快速原型（預設參數）

```python
model = CatBoostRegressor(verbose=False)
```

#### 4.6.2 高準確度配置

```python
model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=5,
    random_strength=2,
    bagging_temperature=0.5,
    early_stopping_rounds=100,
    verbose=False
)
```

#### 4.6.3 快速訓練配置

```python
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.1,
    depth=4,
    task_type='GPU',
    verbose=False
)
```

---

## 5. 模型訓練與評估

### 5.1 訓練過程監控

#### 5.1.1 顯示訓練日誌

```python
model = CatBoostRegressor(
    iterations=100,
    verbose=10  # 每 10 輪顯示一次
)

model.fit(X_train, y_train)
```

輸出示例：
```
0:      learn: 10.5432  total: 50ms     remaining: 4.95s
10:     learn: 8.2341   total: 520ms    remaining: 4.21s
...
```

#### 5.1.2 使用驗證集監控

```python
model = CatBoostRegressor(
    iterations=500,
    verbose=50
)

model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),  # 驗證集
    plot=True  # 顯示訓練曲線（Jupyter Notebook）
)
```

### 5.2 交叉驗證

#### 5.2.1 使用 cv() 函數

```python
from catboost import cv, Pool

# 準備數據
train_pool = Pool(X_train, y_train, cat_features=['catalyst', 'reactor'])

# 交叉驗證
cv_results = cv(
    pool=train_pool,
    params={
        'iterations': 500,
        'learning_rate': 0.1,
        'depth': 6,
        'loss_function': 'RMSE'
    },
    fold_count=5,
    shuffle=True,
    verbose=False
)

print(cv_results.head())
print(f"\nBest RMSE: {cv_results['test-RMSE-mean'].min():.4f}")
```

#### 5.2.2 使用 sklearn cross_val_score

```python
from sklearn.model_selection import cross_val_score

model = CatBoostRegressor(iterations=100, verbose=False)

scores = cross_val_score(
    model, X_train, y_train,
    cv=5,
    scoring='neg_mean_squared_error'
)

print(f"CV RMSE: {np.sqrt(-scores.mean()):.4f} (+/- {np.sqrt(scores.std()):.4f})")
```

### 5.3 特徵重要性分析

#### 5.3.1 基本特徵重要性

```python
import matplotlib.pyplot as plt

# 訓練模型
model = CatBoostRegressor(iterations=100, verbose=False)
model.fit(X_train, y_train)

# 獲取特徵重要性
feature_importance = model.get_feature_importance()
feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'Feature_{i}' for i in range(X_train.shape[1])]

# 視覺化
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance)
plt.xlabel('Feature Importance')
plt.title('CatBoost Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### 5.3.2 多種重要性類型

```python
# PredictionValuesChange（預設）
importance_pvc = model.get_feature_importance(type='PredictionValuesChange')

# LossFunctionChange
importance_lfc = model.get_feature_importance(type='LossFunctionChange')

# FeatureImportance
importance_fi = model.get_feature_importance(type='FeatureImportance')

print("PredictionValuesChange:", importance_pvc[:3])
print("LossFunctionChange:", importance_lfc[:3])
```

### 5.4 模型保存與載入

#### 5.4.1 保存模型

```python
# 方法 1：CatBoost 原生格式（推薦）
model.save_model('catboost_model.cbm')

# 方法 2：JSON 格式（可讀性高）
model.save_model('catboost_model.json', format='json')

# 方法 3：ONNX 格式（跨平台）
model.save_model('catboost_model.onnx', format='onnx')
```

#### 5.4.2 載入模型

```python
from catboost import CatBoostRegressor

# 載入模型
model_loaded = CatBoostRegressor()
model_loaded.load_model('catboost_model.cbm')

# 使用載入的模型預測
y_pred = model_loaded.predict(X_test)
```

---

## 6. 超參數調整策略

### 6.1 使用 Grid Search

```python
from sklearn.model_selection import GridSearchCV

# 定義參數網格
param_grid = {
    'iterations': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8]
}

# 建立模型
model = CatBoostRegressor(verbose=False)

# Grid Search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
```

### 6.2 使用 Randomized Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# 定義參數分佈
param_distributions = {
    'iterations': randint(100, 1000),
    'learning_rate': uniform(0.01, 0.3),
    'depth': randint(4, 10),
    'l2_leaf_reg': uniform(1, 10)
}

# Randomized Search
random_search = RandomizedSearchCV(
    estimator=CatBoostRegressor(verbose=False),
    param_distributions=param_distributions,
    n_iter=20,  # 嘗試 20 組參數
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
```

### 6.3 調參建議

#### 6.3.1 調參優先級

1. **iterations + learning_rate**（最重要）
2. **depth**（次要）
3. **l2_leaf_reg**（防止過擬合）
4. **其他參數**（微調）

#### 6.3.2 調參策略

**階段 1：快速探索**
```python
model = CatBoostRegressor(
    iterations=100,    # 少量迭代快速測試
    depth=6,
    verbose=False
)
```

**階段 2：精細調整**
```python
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=5,
    early_stopping_rounds=100,
    verbose=False
)
```

**階段 3：最終優化**
```python
model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=5,
    random_strength=2,
    bagging_temperature=0.5,
    early_stopping_rounds=200,
    verbose=False
)
```

---

## 7. CatBoost 化工領域應用案例

### 7.1 案例一：化學反應產率預測（含類別特徵）

#### 7.1.1 問題背景

某化工廠有多個反應器，使用不同催化劑與操作條件，需預測產率。

**特徵**：
- 數值特徵：溫度、壓力、流量、濃度
- 類別特徵：催化劑類型、反應器編號、操作班次

**挑戰**：
- 催化劑類型有 20 種（高基數）
- 反應器編號有 15 個
- 傳統 One-Hot Encoding 會產生 35 個稀疏特徵

#### 7.1.2 CatBoost 解決方案

```python
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 模擬數據
np.random.seed(42)
n_samples = 5000

data = pd.DataFrame({
    'temperature': np.random.uniform(300, 450, n_samples),
    'pressure': np.random.uniform(1, 15, n_samples),
    'flow_rate': np.random.uniform(10, 100, n_samples),
    'concentration': np.random.uniform(0.1, 0.9, n_samples),
    'catalyst': np.random.choice([f'CAT_{i}' for i in range(1, 21)], n_samples),
    'reactor': np.random.choice([f'R{i}' for i in range(1, 16)], n_samples),
    'shift': np.random.choice(['Morning', 'Afternoon', 'Night'], n_samples)
})

# 生成目標（產率）：受多個因素影響
data['yield'] = (
    0.5 * data['temperature'] +
    2.0 * data['pressure'] +
    0.1 * data['flow_rate'] +
    50 * data['concentration'] +
    np.random.normal(0, 5, n_samples)  # 噪音
)

# 分割數據
X = data.drop('yield', axis=1)
y = data['yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 指定類別特徵
cat_features = ['catalyst', 'reactor', 'shift']

# 建立 CatBoost 模型
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features,
    early_stopping_rounds=50,
    verbose=False
)

# 訓練
model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    plot=False
)

# 預測與評估
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# 特徵重要性
feature_importance = model.get_feature_importance()
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance)
plt.xlabel('Feature Importance')
plt.title('Chemical Reaction Yield Prediction - Feature Importance')
plt.tight_layout()
plt.show()
```

### 7.2 案例二：設備故障分類（不平衡資料）

#### 7.2.1 問題背景

化工廠設備監控系統，需預測設備是否會發生故障。

**挑戰**：
- 正常樣本遠多於故障樣本（不平衡：正常 95%、故障 5%）
- 類別特徵：設備類型、製造商、維護記錄

#### 7.2.2 CatBoost 解決方案

```python
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns

# 模擬不平衡數據
np.random.seed(42)
n_samples = 10000
n_fault = int(n_samples * 0.05)  # 5% 故障

data = pd.DataFrame({
    'vibration': np.concatenate([
        np.random.normal(50, 10, n_samples - n_fault),  # 正常
        np.random.normal(80, 15, n_fault)  # 故障
    ]),
    'temperature': np.concatenate([
        np.random.normal(70, 5, n_samples - n_fault),
        np.random.normal(90, 10, n_fault)
    ]),
    'pressure': np.random.uniform(5, 20, n_samples),
    'equipment_type': np.random.choice(['Type_A', 'Type_B', 'Type_C'], n_samples),
    'manufacturer': np.random.choice(['Mfr_1', 'Mfr_2', 'Mfr_3', 'Mfr_4'], n_samples),
    'fault': np.concatenate([
        np.zeros(n_samples - n_fault),
        np.ones(n_fault)
    ])
})

# 打亂數據
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# 分割數據
X = data.drop('fault', axis=1)
y = data['fault']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 檢查類別分佈
print(f"Training set fault ratio: {y_train.mean():.4f}")
print(f"Test set fault ratio: {y_test.mean():.4f}")

# 建立 CatBoost 模型（處理不平衡）
model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.1,
    depth=6,
    cat_features=['equipment_type', 'manufacturer'],
    auto_class_weights='Balanced',  # 自動平衡類別權重
    eval_metric='AUC',
    verbose=False
)

# 訓練
model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test)
)

# 預測
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 評估
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fault']))
print(f"\nAUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# 混淆矩陣
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Fault'], yticklabels=['Normal', 'Fault'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Equipment Fault Detection - Confusion Matrix')
plt.tight_layout()
plt.show()
```

---

## 8. GBDT 三巨頭比較總結

### 8.1 特性對比表

| 特性 | XGBoost | LightGBM | CatBoost |
|------|---------|----------|----------|
| **開發者** | 陳天奇 | Microsoft | Yandex |
| **發表年份** | 2014 | 2017 | 2017 |
| **核心優勢** | 通用性強 | 速度快 | 準確度高 |
| **訓練速度** | ★★★☆☆ | ★★★★★ | ★★★☆☆ |
| **準確度** | ★★★★☆ | ★★★★☆ | ★★★★★ |
| **易用性** | ★★★☆☆ | ★★☆☆☆ | ★★★★★ |
| **類別特徵** | 需編碼 | 基本支援 | 最佳支援 |
| **高基數類別** | ★☆☆☆☆ | ★★★☆☆ | ★★★★★ |
| **大數據** | ★★★☆☆ | ★★★★★ | ★★★☆☆ |
| **魯棒性** | ★★★☆☆ | ★★☆☆☆ | ★★★★★ |
| **過擬合控制** | ★★★☆☆ | ★★☆☆☆ | ★★★★★ |
| **調參難度** | 中 | 高 | 低 |
| **社群成熟度** | 非常成熟 | 成熟 | 成熟 |

### 8.2 選擇建議

**選擇 XGBoost**：
- 需要通用、穩定的解決方案
- 團隊已有 XGBoost 經驗
- 不需要處理大量類別特徵
- 數據量適中（< 1M）

**選擇 LightGBM**：
- 追求極致訓練速度
- 大規模數據（> 1M）
- 高維特徵（> 100）
- 願意投入較多調參時間

**選擇 CatBoost**：
- 大量類別特徵或高基數類別
- 追求最高準確度
- 快速原型開發（預設參數即可）
- 不平衡資料問題
- 需要穩定、魯棒的模型

### 8.3 化工領域推薦

**推薦 CatBoost 的場景**：
- 配方優化（原料編號、供應商等類別）
- 批次追蹤（批次號、設備編號等高基數）
- 多廠區建模（廠區、產線等類別）
- 設備診斷（設備型號、製造商等）

**推薦 LightGBM 的場景**：
- 大規模製程監控（百萬級時間序列數據）
- 實時異常檢測（需要快速訓練與推理）

**推薦 XGBoost 的場景**：
- 通用預測任務（無特殊需求）
- 團隊已有豐富經驗

---

## 9. 學習資源與參考文獻

### 9.1 官方資源

- **CatBoost 官網**：https://catboost.ai/
- **GitHub**：https://github.com/catboost/catboost
- **官方文檔**：https://catboost.ai/docs/
- **Tutorial**：https://github.com/catboost/tutorials

### 9.2 論文

1. **CatBoost 原始論文**：
   - Prokhorenkova, L., et al. (2018). "CatBoost: unbiased boosting with categorical features." *NeurIPS 2018*.

2. **相關論文**：
   - Dorogush, A. V., et al. (2018). "CatBoost: gradient boosting with categorical features support."

### 9.3 學習建議

1. **初學者**：
   - 先使用預設參數建立基線模型
   - 理解類別特徵的自動處理機制
   - 練習使用 Pool 與特徵重要性分析

2. **進階學習**：
   - 深入理解 Ordered Boosting 原理
   - 學習超參數調整策略
   - 比較 GBDT 三巨頭的性能差異

3. **實戰應用**：
   - 應用於真實化工數據
   - 處理不平衡與高基數類別問題
   - 結合領域知識進行特徵工程

---

## 10. 實戰演練與結果分析

本章節將詳細展示 CatBoost 在兩個化工領域實際案例中的應用，並對執行結果進行深入分析。完整程式碼請參考：
- `Unit13_CatBoost_Regression.ipynb`
- `Unit13_CatBoost_Classification.ipynb`

### 10.1 案例一：能源消耗預測（回歸任務）

#### 10.1.1 問題背景與數據探索

某化工廠需要預測設備的能源消耗量，以優化能源管理與成本控制。數據集包含：
- **樣本數量**：100,000 筆
- **特徵數量**：27 個（10 個基礎特徵 + 2 個類別特徵 + 4 個時間特徵 + 11 個衍生特徵）
- **預測目標**：Energy_Consumption (kWh)

**特徵類型**：
- **數值特徵**：進料流量、進料溫度、壓力、濃度、環境溫度、濕度等
- **類別特徵**：設備狀態 (5 種)、操作模式 (4 種)
- **時間特徵**：月份、季節、星期、小時
- **衍生特徵**：多項式特徵、交互特徵、滾動統計量

**數據分布分析**：

![Energy Consumption Distribution](./Unit13_Results/energy_distribution.png)

**圖 10.1：能源消耗分布與統計特性**

**關鍵觀察**：

1. **目標變數分布（左上）**：
   - 呈現近似常態分布，中心約在 490 kWh
   - 範圍從 300 kWh 到 800 kWh，跨度約 500 kWh
   - 無明顯偏態，有利於回歸模型訓練

2. **箱型圖分析（右上）**：
   - 中位數 (487.83 kWh) 與平均值 (490.59 kWh) 接近，分布對稱
   - 四分位距 (IQR) 約 115 kWh，數據集中度良好
   - 存在少量極端值（離群點），但比例很小

3. **統計摘要**：
   - **平均值**：490.59 kWh
   - **中位數**：487.83 kWh
   - **標準差**：84.48 kWh（約為平均值的 17%，變異度適中）
   - 數據品質良好，適合建模

**類別特徵分布**：

![Categorical Features Distribution](./Unit13_Results/categorical_features.png)

**圖 10.2：類別特徵分布與能源消耗關係**

**類別特徵分析**：

1. **Equipment_Status（設備狀態）分布**：
   - **Status A**：70,000 筆（70%）- 主要運行狀態
   - **Status B**：20,000 筆（20%）- 次要狀態
   - **Status C**：10,000 筆（10%）- 特殊狀態
   - 不同狀態下能源消耗差異明顯：
     * Status A: 中位數約 480 kWh，分布集中
     * Status B: 中位數約 500 kWh，變異較大
     * Status C: 中位數約 520 kWh，高能耗模式

2. **Operation_Mode（操作模式）分布**：
   - **Mode 1**：60,000 筆（60%）- 標準模式
   - **Mode 2**：30,000 筆（30%）- 高負載模式
   - **Mode 3**：10,000 筆（10%）- 低負載模式
   - 操作模式對能耗影響顯著：
     * Mode 1: 中位數約 475 kWh
     * Mode 2: 中位數約 495 kWh
     * Mode 3: 中位數約 510 kWh

**化工意義**：
- 類別特徵（設備狀態、操作模式）對能耗有明顯影響
- CatBoost 能自動處理這些類別特徵，無需手動 One-Hot Encoding
- 不同類別間的能耗差異為特徵工程提供了重要線索

#### 10.1.2 模型訓練過程

訓練配置：
```python
model = CatBoostRegressor(
    iterations=1200,
    learning_rate=0.05,
    depth=6,
    cat_features=['Equipment_Status', 'Operation_Mode'],
    early_stopping_rounds=50,
    verbose=100
)
```

訓練過程摘要（顯示關鍵迭代）：
```
0:      learn: 80.9453   test: 80.9843   best: 80.9843 (0)   total: 15.7ms
100:    learn: 13.8658   test: 13.9477   best: 13.9477 (100) total: 1.52s
200:    learn: 9.9661    test: 10.1078   best: 10.1078 (200) total: 3.01s
300:    learn: 9.1643    test: 9.3467    best: 9.3467 (300)  total: 4.57s
400:    learn: 8.8676    test: 9.1219    best: 9.1219 (400)  total: 6.08s
500:    learn: 8.7291    test: 9.0444    best: 9.0444 (500)  total: 7.62s
600:    learn: 8.6280    test: 9.0140    best: 9.0140 (600)  total: 9.46s
700:    learn: 8.5371    test: 8.9948    best: 8.9947 (691)  total: 11.3s
800:    learn: 8.4552    test: 8.9884    best: 8.9884 (772)  total: 13.1s
900:    learn: 8.3767    test: 8.9839    best: 8.9834 (897)  total: 14.9s
1000:   learn: 8.3033    test: 8.9799    best: 8.9794 (998)  total: 16.8s
1100:   learn: 8.2375    test: 8.9761    best: 8.9760 (1095) total: 18.6s
bestTest = 8.972529
bestIteration = 1120

✓ 模型訓練完成 (耗時: 21.27秒)
✓ 最佳迭代次數: 1120
✓ 最佳驗證集 RMSE: 8.9725
```

**訓練過程分析**：
1. **快速收斂**：前 200 輪從 RMSE 80.98 降至 10.11（收斂 87%）
2. **穩定優化**：200-1120 輪持續緩慢優化（收斂 11%）
3. **早停機制**：第 1120 輪達到最佳，後續 50 輪無改善自動停止
4. **訓練效率**：平均每輪 18.9ms，1120 輪僅需 21.27 秒

#### 10.1.3 模型評估結果

```
============================================================
模型評估結果
============================================================
訓練集:
  RMSE: 8.2935
  MAE:  6.5564
  R²:   0.9903

驗證集:
  RMSE: 8.9794
  MAE:  7.0064
  R²:   0.9887

測試集:
  RMSE: 8.8707
  MAE:  6.9605
  R²:   0.9891
```

**結果分析**：

1. **優異的預測準確度**：
   - 測試集 R² = 0.9891，解釋了 98.91% 的變異
   - RMSE = 8.87 kWh，相對於平均能耗約 5% 誤差
   - MAE = 6.96 kWh，絕對誤差小

2. **良好的泛化能力**：
   - 訓練集 R² (0.9903) 與測試集 R² (0.9891) 差距僅 0.12%
   - 驗證集與測試集性能一致，無過擬合
   - RMSE 在三個數據集上接近（8.29 → 8.98 → 8.87）

3. **模型穩定性**：
   - 驗證集與測試集 RMSE 差距僅 0.11 kWh（1.2%）
   - 表明模型對新數據具有魯棒性

#### 10.1.4 特徵重要性分析

```
Feature Importance Ranking:
                         Feature  Importance
                      Flow_Cubed   45.458246
                 Operating_Hours   19.761283
Pressure_Composition_Interaction   16.074803
                      Steam_Flow    5.263160
                Equipment_Status    4.349656
                  Operation_Mode    2.524987
           Temp_Flow_Interaction    1.670148
                     Load_Factor    1.341824
                    Ambient_Temp    0.614422
                    Temp_Squared    0.545667
```

![Feature Importance - Regression](./Unit13_Results/regression_feature_importance.png)

**圖 10.3：CatBoost 回歸模型特徵重要性排名**

**特徵重要性分析**：

1. **多項式特徵主導**：
   - **Flow_Cubed (45.46%)**：流量的三次方是最重要特徵
   - 能源消耗與流量呈非線性關係（流體力學中壓降 ∝ 流量³）
   - 驗證了領域知識與物理規律

2. **累積效應顯著**：
   - **Operating_Hours (19.76%)**：運轉時數排名第二
   - 反映設備老化、熱慣性等累積效應

3. **交互特徵有效**：
   - **Pressure_Composition_Interaction (16.07%)**：壓力與濃度交互作用
   - 符合化工過程中的熱力學與動力學耦合

4. **類別特徵貢獻**：
   - **Equipment_Status (4.35%)** + **Operation_Mode (2.52%)** = 6.87%
   - CatBoost 自動處理類別特徵，無需預編碼
   - 兩個類別特徵合計貢獻接近 7%，說明設備狀態與操作模式對能耗有重要影響

5. **基礎特徵相對較弱**：
   - Feed_Flow、Feed_Temp 等原始特徵重要性 < 0.02%
   - 衍生特徵（多項式、交互）大幅提升模型性能

#### 10.1.5 預測結果可視化

**Parity Plot（預測值 vs 實際值）**：

![Parity Plot](./Unit13_Results/regression_parity_plot.png)

**圖 10.4：三數據集 Parity Plot 與 R² 分析**

**Parity Plot 詳細分析**：

1. **完美對齊的紅色虛線**（y = x）：
   - 代表理想預測情況（預測值 = 實際值）
   - 數據點越接近此線，預測越準確

2. **訓練集（左圖，R² = 0.9903）**：
   - 數據點緊密分佈在對角線周圍
   - 全範圍（250-800 kWh）預測一致性優異
   - 幾乎無系統性偏差，模型學習充分

3. **驗證集（中圖，R² = 0.9887）**：
   - 與訓練集性能接近，泛化能力良好
   - 高值區域（>700 kWh）略有分散，但在可接受範圍
   - 無過擬合跡象

4. **測試集（右圖，R² = 0.9891）**：
   - **最終性能驗證**：R² = 0.9891 表示模型解釋了 98.91% 的變異
   - 預測品質與驗證集一致，模型穩定
   - 低值區域（<400 kWh）和高值區域（>600 kWh）皆預測準確

**化工意義**：
- 能源消耗範圍跨越 500+ kWh，模型在全範圍內皆準確
- 可用於實時能源監控與異常檢測
- 預測誤差小，適合用於成本估算與排程優化

**殘差分析**：

![Residual Analysis](./Unit13_Results/regression_residuals.png)

**圖 10.5：測試集殘差分布與趨勢分析**

**殘差分析詳解**：

1. **殘差分布直方圖（左圖）**：
   - **近似常態分布**：以零為中心對稱分布
   - **均值接近零**：-0.016 kWh ≈ 0，無系統性偏差
   - **標準差 = 8.87 kWh**：與 RMSE 一致
   - **68% 數據在 ±8.87 kWh 內**：符合常態分布特性
   - **95% 數據在 ±17.7 kWh 內**：極少極端誤差
   - 紅色虛線標示零誤差線，分布完美對稱

2. **殘差 vs 預測值散點圖（右圖）**：
   - **水平帶狀分布**：殘差隨機分散，無明顯模式
   - **零線兩側對稱**：無系統性高估或低估
   - **變異數同質性**：全預測範圍內殘差標準差穩定
   - **無漏斗效應**：高值區域殘差未放大
   - **極端值分析**：
     * 最大正殘差：+41.80 kWh（高估）
     * 最大負殘差：-42.87 kWh（低估）
     * 極端值比例 < 0.1%，可忽略

**統計檢驗結果**：

| 指標 | 數值 | 意義 |
|-----|------|------|
| 殘差均值 | -0.016 kWh | 接近 0，無偏估計 |
| 殘差標準差 | 8.87 kWh | 預測不確定性 |
| 最大殘差 | 41.80 kWh | < 平均值的 9% |
| 殘差偏態 | 0.02 | 近似對稱 |
| 殘差峰態 | 2.98 | 接近常態 (3.0) |

**化工應用啟示**：

1. **誤差範圍可接受**：
   - 95% 預測誤差 < ±17.7 kWh
   - 相對誤差約 ±3.6%（相對於平均能耗 490 kWh）
   - 適用於日常能源管理與成本控制

2. **無系統性偏差**：
   - 模型在全範圍內公正預測
   - 不會持續高估或低估特定區間
   - 可信賴用於長期規劃

3. **殘差模式診斷**：
   - 無明顯非線性模式 → 特徵工程充分
   - 無變異數異質性 → 模型假設合理
   - 常態分布假設成立 → 統計推論有效

#### 10.1.6 化工應用啟示

1. **能源管理優化**：
   - 準確預測能耗，可提前調整操作參數
   - 識別異常能耗模式，及時維護設備
   - 基於 R² = 0.9891 的高準確度，可用於實時決策

2. **操作建議**：
   - 流量控制最關鍵（貢獻 45%），應精確調節
   - 關注設備運轉時數，定期保養可降低能耗

3. **成本節約**：
   - 基於預測結果優化排程，可節省 3-5% 能源成本
   - RMSE 8.87 kWh 對應約 $1-2/批次預測誤差成本

---

### 10.2 案例二：多類別故障診斷（分類任務）

#### 10.2.1 問題背景與類別分布分析

某化工設備需要進行故障診斷，共有 7 種狀態，存在嚴重的類別不平衡問題。

**數據集特性**：
- **樣本數量**：150,000 筆
- **特徵數量**：30 個（溫度、壓力、流量、振動、電流等感測器數據）
- **類別數量**：7 類
- **類別分佈**（極度不平衡）：
  - Class 0: Normal (正常) - 70%
  - Class 1: Minor Wear (輕微磨損) - 15%
  - Class 2: Temp Abnormal (溫度異常) - 5%
  - Class 3: Pressure Fluct (壓力波動) - 4%
  - Class 4: Leak Warning (洩漏警告) - 3%
  - Class 5: Severe Fault (嚴重故障) - 2%
  - Class 6: Emergency Stop (緊急停機) - 1%

**挑戰**：
- 最大類與最小類樣本比例為 70:1
- 緊急停機（Class 6）樣本僅 1%，但誤判代價最高
- 需要平衡精確度與召回率

**類別不平衡視覺化**：

![Class Distribution](./Unit13_Results/classification_class_distribution.png)

**圖 10.6：多類別故障診斷的極度不平衡分布**

**類別分布詳細分析**：

1. **線性尺度圖（左）**：
   - **Normal（正常）**：104,965 筆（70.0%）- 主導類別
   - **Minor Wear（輕微磨損）**：22,522 筆（15.0%）
   - **Temp Abnormal（溫度異常）**：7,576 筆（5.1%）
   - **Pressure Fluct（壓力波動）**：5,929 筆（4.0%）
   - **Leak Warning（洩漏警告）**：4,446 筆（3.0%）
   - **Severe Fault（嚴重故障）**：2,983 筆（2.0%）
   - **Emergency Stop（緊急停機）**：1,479 筆（1.0%）- 極少數類別

2. **對數尺度圖（右）**：
   - 使用對數刻度更清楚顯示少數類別
   - 最多類（Normal）與最少類（Emergency Stop）樣本數差異達 **71 倍**
   - 這是典型的極度不平衡問題，對傳統機器學習模型是重大挑戰

**不平衡問題的化工意義**：

1. **符合真實場景**：
   - 正常運行時間遠多於故障時間（70% vs 30%）
   - 緊急停機事件罕見但關鍵（僅 1%）
   - 這種分布反映了實際工業運行狀態

2. **診斷挑戰**：
   - **Class 6（Emergency Stop）**：僅 1,479 筆訓練樣本
     * 模型學習困難，容易被多數類別主導
     * 誤判代價最高（停機損失巨大）
     * 需要特殊處理策略
   
   - **Class 5（Severe Fault）**：2,983 筆
     * 樣本數是 Emergency Stop 的 2 倍
     * 仍屬少數類別，但相對較易學習

3. **CatBoost 的優勢**：
   - `auto_class_weights='Balanced'`：自動計算類別權重
   - 給予少數類別更高權重，平衡學習
   - 無需手動調整權重或過採樣

**不平衡比例計算**：
```
最多類（Class 0）：104,965 筆
最少類（Class 6）：1,479 筆
不平衡比例：71.0 : 1
```

這種極度不平衡在化工安全監控中很常見，CatBoost 的自動類別權重平衡機制能有效處理。

#### 10.2.2 模型訓練過程

訓練配置（使用自動類別權重平衡）：
```python
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',
    auto_class_weights='Balanced',  # 自動平衡類別權重
    cat_features=['Equipment_Model', 'Shift', 'Operator_ID'],
    early_stopping_rounds=50,
    verbose=100
)
```

訓練過程摘要：
```
0:      learn: 1.7747   test: 1.7758   best: 1.7758 (0)   total: 25.7ms
100:    learn: 0.4189   test: 0.4398   best: 0.4398 (100) total: 1.34s
200:    learn: 0.3527   test: 0.4127   best: 0.4127 (200) total: 2.66s
300:    learn: 0.3142   test: 0.4040   best: 0.4040 (300) total: 3.98s
400:    learn: 0.2861   test: 0.4011   best: 0.4011 (385) total: 5.35s
bestTest = 0.4009628
bestIteration = 409

✓ 模型訓練完成 (耗時: 7.81秒)
✓ 最佳迭代次數: 409
✓ 最佳驗證集 MultiClass Loss: 0.4010
```

**訓練過程分析**：
1. **極快收斂**：100 輪內從 1.77 降至 0.44（降低 75%）
2. **訓練速度**：409 輪僅需 7.81 秒，平均每輪 19.1ms
3. **早停生效**：第 409 輪達到最佳，避免過擬合
4. **Loss 穩定**：訓練集 0.2861，測試集 0.4010，差距合理

#### 10.2.3 模型評估結果

```
============================================================
模型評估結果（多類別分類）
============================================================
訓練集:
  Accuracy:      0.8337
  F1 (Macro):    0.8934
  F1 (Weighted): 0.8437

驗證集:
  Accuracy:      0.8201
  F1 (Macro):    0.7513
  F1 (Weighted): 0.8299

測試集:
  Accuracy:      0.8195
  F1 (Macro):    0.7455
  F1 (Weighted): 0.8290
```

**整體性能分析**：

1. **良好的準確度**：
   - 測試集準確度 81.95%，考慮到 7 類別分類問題表現優異
   - 隨機猜測準確度僅 14.3%（1/7），模型提升 5.7 倍

2. **泛化能力穩定**：
   - 訓練集與測試集準確度差距僅 1.42%（83.37% → 81.95%）
   - 驗證集與測試集幾乎一致（82.01% vs 81.95%）

3. **類別平衡處理有效**：
   - Macro F1 (0.7455)：所有類別平等加權，顯示少數類也有合理性能
   - Weighted F1 (0.8290)：按樣本數加權，整體性能更高
   - Macro 與 Weighted F1 差距 8.35%，說明少數類確實較難預測但仍可接受

#### 10.2.4 詳細分類報告

```
測試集詳細分類報告
============================================================
                precision    recall  f1-score   support

        Normal     0.9332    0.8255    0.8760     20993
    Minor Wear     0.4940    0.7257    0.5879      4509
 Temp Abnormal     0.8894    0.9830    0.9339      1530
Pressure Fluct     0.8417    0.9328    0.8849      1191
  Leak Warning     0.9426    0.9606    0.9515       889
  Severe Fault     0.6637    0.7466    0.7027       592
Emergency Stop     0.3274    0.2466    0.2813       296

      accuracy                         0.8195     30000
     macro avg     0.7274    0.7744    0.7455     30000
  weighted avg     0.8503    0.8195    0.8290     30000
```

**各類別性能分析**：

1. **Class 0 (Normal - 正常狀態)**：
   - Precision: 93.32%，Recall: 82.55%，F1: 87.60%
   - 性能優異，但召回率 82.55% 意味著 17.45% 正常狀態被誤判為異常
   - 誤判主要流向 Class 1 (Minor Wear)，可能是輕微磨損前兆

2. **Class 1 (Minor Wear - 輕微磨損)**：
   - Precision: 49.40%，Recall: 72.57%，F1: 58.79%
   - 召回率高但精確度低，模型傾向於將其他類別誤判為輕微磨損
   - 這是合理的安全策略（寧可誤報，及早發現問題）

3. **Class 2 (Temp Abnormal - 溫度異常)**：
   - Precision: 88.94%，Recall: 98.30%，F1: 93.39%
   - **性能最佳**，溫度特徵區分度高
   - 高召回率（98.30%）確保溫度異常幾乎不會遺漏

4. **Class 3 (Pressure Fluct - 壓力波動)**：
   - Precision: 84.17%，Recall: 93.28%，F1: 88.49%
   - 性能良好，壓力特徵也具有良好區分能力

5. **Class 4 (Leak Warning - 洩漏警告)**：
   - Precision: 94.26%，Recall: 96.06%，F1: 95.15%
   - **性能優異**，洩漏特徵明顯（流量、壓力異常組合）
   - 高精確度與召回率兼顧，適合安全關鍵應用

6. **Class 5 (Severe Fault - 嚴重故障)**：
   - Precision: 66.37%，Recall: 74.66%，F1: 70.27%
   - 性能中等，部分被誤判為 Class 6 (Emergency Stop)
   - 兩者症狀相似，誤判在可接受範圍

7. **Class 6 (Emergency Stop - 緊急停機)**：
   - Precision: 32.74%，Recall: 24.66%，F1: 28.13%
   - **性能最差**，樣本最少（296 筆測試樣本）
   - 75% 被誤判為 Class 5 (Severe Fault)，但仍觸發警報
   - 精確度低意味著部分其他類別被誤判為緊急停機，屬於保守策略

**關鍵洞察**：
- 溫度異常、洩漏警告檢測優異（F1 > 93%）
- 輕微磨損、緊急停機較難區分（F1 < 60%）
- 模型整體傾向保守（寧可誤報，不願漏報），符合安全要求

**各類別性能視覺化**：

![Per-Class Performance](./Unit13_Results/classification_per_class_metrics.png)

**圖 10.7：七類別 Precision、Recall、F1-Score 對比**

**視覺化分析**：

1. **優異性能類別**（三指標皆 > 0.88）：
   - **Temp Abnormal（溫度異常）**：
     * Precision = 0.889，Recall = 0.983，F1 = 0.934
     * 三指標最均衡，幾乎完美的檢測性能
     * 溫度特徵區分度極高
   
   - **Leak Warning（洩漏警告）**：
     * Precision = 0.943，Recall = 0.961，F1 = 0.952
     * **最高 F1 分數**，精確度與召回率皆優
     * 流量與壓力組合特徵明顯
   
   - **Pressure Fluct（壓力波動）**：
     * Precision = 0.842，Recall = 0.933，F1 = 0.885
     * 召回率高，適合安全關鍵應用

2. **良好性能類別**（F1 > 0.70）：
   - **Normal（正常）**：
     * Precision = 0.933（最高精確度），Recall = 0.826
     * 樣本數最多（70%），學習充分
     * 召回率 82.6% 意味著 17.4% 正常被誤判為異常（保守策略）
   
   - **Severe Fault（嚴重故障）**：
     * Precision = 0.664，Recall = 0.747，F1 = 0.703
     * 性能中等，部分與 Emergency Stop 混淆

3. **挑戰性類別**（F1 < 0.60）：
   - **Minor Wear（輕微磨損）**：
     * Precision = 0.494，Recall = 0.726，F1 = 0.588
     * 召回率高但精確度低 → 過度預測輕微磨損
     * 這是**保守策略**：寧可誤報，提前維護
   
   - **Emergency Stop（緊急停機）**：
     * Precision = 0.327，Recall = 0.247，F1 = 0.281
     * **最低性能**，但可接受：
       - 樣本極少（僅 1%）導致學習不足
       - 大部分被誤判為 Severe Fault（仍觸發警報）
       - 精確度低意味著部分其他類被誤判為緊急停機（保守）

**柱狀圖解讀**：
- **藍色（Precision）**：預測為該類的正確比例
- **橙色（Recall）**：實際該類被正確識別的比例
- **綠色（F1-Score）**：精確度與召回率的調和平均

**化工安全考量**：
- 高風險類別（Leak Warning, Temp Abnormal）召回率 > 96%，極少漏報
- Normal 召回率 82.6%，17.4% 被保守判為異常（可接受的誤報）
- Emergency Stop 召回率雖低，但誤判多為 Severe Fault（皆高風險）

#### 10.2.5 混淆矩陣分析

![Confusion Matrix](./Unit13_Results/classification_confusion_matrix.png)

**圖 10.8：7×7 混淆矩陣（絕對計數 vs 列標準化百分比）**

**混淆矩陣關鍵觀察**：

1. **對角線主導**：
   - 大部分預測集中在對角線（正確分類）
   - Normal: 82.55%，Minor Wear: 72.57%，Temp Abnormal: 98.30%

2. **Class 0 → Class 1 混淆**：
   - 3,339 筆正常狀態被誤判為輕微磨損（15.91%）
   - 可能原因：設備處於磨損初期，特徵介於兩者之間

3. **Class 6 → Class 5 混淆**：
   - 223 筆緊急停機被誤判為嚴重故障（75.34%）
   - 兩者皆為高風險狀態，誤判仍會觸發警報，影響可控

4. **Class 1 誤判分散**：
   - 1,186 筆被誤判為 Normal
   - 輕微磨損與正常狀態界線模糊，需要更多特徵或更長時間序列

#### 10.2.6 各類別召回率分析

```
各類別召回率 (Recall):
  Class 0 (Normal         ): 0.8255
  Class 1 (Minor Wear     ): 0.7257
  Class 2 (Temp Abnormal  ): 0.9830
  Class 3 (Pressure Fluct ): 0.9328
  Class 4 (Leak Warning   ): 0.9606
  Class 5 (Severe Fault   ): 0.7466
  Class 6 (Emergency Stop ): 0.2466
```

**召回率分析**：
- **高召回率類別**（> 90%）：Temp Abnormal (98.30%)、Leak Warning (96.06%)、Pressure Fluct (93.28%)
  - 這些異常有明顯特徵，模型能有效識別
  
- **中等召回率類別**（70-85%）：Normal (82.55%)、Severe Fault (74.66%)、Minor Wear (72.57%)
  - 需要進一步優化，可能需要更多特徵或更深模型
  
- **低召回率類別**（< 30%）：Emergency Stop (24.66%)
  - 樣本極少（測試集僅 296 筆）導致學習不足
  - 大部分誤判為相鄰高風險類別（Severe Fault），仍能觸發警報

#### 10.2.7 特徵重要性分析

```
Top 15 Feature Importance:
               Feature  Importance
           Noise_Level   21.746276
               Current    7.884756
             Temp_Wall    7.144263
     Temp_Diff_Ambient    6.534490
Days_Since_Maintenance    6.470499
   Vibration_Magnitude    5.806899
        Abnormal_Count    5.243777
            Temp_Inlet    4.608504
          Flow_Product    3.277016
           Temp_Outlet    2.921897
```

![Feature Importance - Classification](./Unit13_Results/classification_feature_importance.png)

**圖 10.9：CatBoost 分類模型 Top 20 特徵重要性**

**特徵重要性分析**：

1. **Noise_Level (21.75%) 主導**：
   - 噪音水平是最重要特徵，異常狀態通常伴隨噪音增加
   - 振動、磨損、洩漏都會產生異常聲響

2. **電流與溫度關鍵**：
   - Current (7.88%)：設備負載異常會反映在電流變化
   - Temp_Wall (7.14%)、Temp_Diff_Ambient (6.53%)、Temp_Inlet (4.61%)
   - 溫度相關特徵合計 > 20%，驗證溫度異常檢測性能優異

3. **維護歷史重要**：
   - Days_Since_Maintenance (6.47%)：距上次維護天數
   - 設備老化與故障風險正相關

4. **振動與異常計數**：
   - Vibration_Magnitude (5.81%)：振動幅度反映機械狀態
   - Abnormal_Count (5.24%)：累積異常次數，預測未來故障

5. **流量與壓力**：
   - Flow_Product (3.28%)、Pressure_Inlet (1.90%)
   - 洩漏警告依賴流量與壓力特徵

**與回歸任務對比**：
- 回歸任務：多項式特徵主導（Flow_Cubed 45%）
- 分類任務：原始感測器特徵主導（Noise_Level 22%）
- 不同任務的特徵重要性模式完全不同，驗證 CatBoost 能自動學習任務特定模式

#### 10.2.8 化工應用啟示

1. **預測性維護策略**：
   - 溫度異常、洩漏警告檢測準確（F1 > 93%），可作為自動觸發維護的依據
   - 輕微磨損召回率 72.57%，可設置人工複核機制

2. **安全風險管理**：
   - 緊急停機召回率僅 24.66%，建議結合其他監控手段
   - 75% 緊急停機被誤判為嚴重故障，仍在可控範圍（皆為高風險狀態）

3. **感測器布局建議**：
   - 噪音感測器最重要（21.75%），應確保覆蓋關鍵設備
   - 溫度感測器合計貢獻 > 20%，建議增加溫度測點
   - 振動與電流感測器也應優先配置

4. **類別不平衡處理**：
   - `auto_class_weights='Balanced'` 有效提升少數類性能
   - Macro F1 達到 74.55%，證明極度不平衡（70:1）下仍能有效學習

5. **模型部署建議**：
   - 設置不同類別的動作閾值（高風險類別降低閾值）
   - 結合時間序列模式（連續異常更可信）
   - 建立人工複核機制處理不確定預測

---

### 10.3 兩案例對比總結

| 維度 | 回歸任務（能源預測） | 分類任務（故障診斷） |
|------|---------------------|---------------------|
| **數據規模** | 100K 樣本，27 特徵 | 150K 樣本，30 特徵 |
| **訓練時間** | 21.27 秒（1120 輪） | 7.81 秒（409 輪） |
| **性能指標** | R² = 0.9891, RMSE = 8.87 | Accuracy = 81.95%, F1 = 74.55% |
| **關鍵特徵** | Flow_Cubed (45%) | Noise_Level (22%) |
| **特徵類型** | 衍生特徵主導 | 原始感測器主導 |
| **類別特徵** | 2 個（貢獻 6.87%） | 3 個（自動處理） |
| **主要挑戰** | 非線性關係 | 極度不平衡（70:1） |
| **CatBoost 優勢** | 自動處理類別，快速訓練 | auto_class_weights 處理不平衡 |
| **應用價值** | 能源成本節省 3-5% | 預測性維護，降低停機時間 |

**共同特點**：
- 兩個案例都展示了 CatBoost 的魯棒性與高準確度
- 類別特徵自動處理無需預編碼
- 早停機制有效防止過擬合
- 特徵重要性分析提供可解釋性

**關鍵收穫**：
- CatBoost 適合化工領域的各種預測任務
- 預設參數即可獲得優異結果，調參成本低
- 對類別特徵與不平衡資料處理能力強
- 訓練速度快，適合工業界快速迭代

---

## 11. 總結

### 11.1 核心要點回顧

1. **CatBoost 是什麼**：
   - Yandex 開發的 GBDT 框架
   - 專注於類別特徵處理與高準確度

2. **核心技術**：
   - **Ordered Target Statistics**：解決 Target Leakage
   - **Ordered Boosting**：減少預測偏移
   - **對稱樹**：提升速度與穩定性

3. **主要優勢**：
   - 類別特徵處理最佳
   - 魯棒性強，預設參數優異
   - 不平衡資料處理能力強
   - 豐富的可視化工具

4. **適用場景**：
   - 大量類別特徵
   - 高基數類別
   - 追求高準確度
   - 快速原型開發

### 11.2 與其他模型的關係

- **vs sklearn GBDT**：更快、更準確、功能更豐富
- **vs XGBoost**：類別處理更好、更易用
- **vs LightGBM**：更準確、更魯棒，但稍慢

### 11.3 實踐建議

1. **從簡單開始**：
   ```python
   model = CatBoostRegressor(cat_features=cat_cols, verbose=False)
   model.fit(X_train, y_train)
   ```

2. **使用驗證集與早停**：
   ```python
   model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
   ```

3. **分析特徵重要性**：
   ```python
   importance = model.get_feature_importance()
   ```

4. **根據需求選擇 GBDT**：
   - CatBoost：準確度優先
   - LightGBM：速度優先
   - XGBoost：通用穩定

5. **實戰案例驗證**：
   - 能源消耗預測：R² = 0.9891，訓練時間 21.27 秒
   - 故障診斷分類：準確度 81.95%，處理 70:1 不平衡
   - 證明 CatBoost 在化工領域的實用性與優異性能

---

## 12. 課後練習

### 12.1 基礎練習

1. 使用 CatBoost 建立一個回歸模型，預測化學反應產率
2. 嘗試處理含有類別特徵的數據集
3. 比較有無指定 `cat_features` 的性能差異
4. 使用 `plot=True` 觀察訓練過程

### 12.2 進階練習

1. 使用 Grid Search 調整 CatBoost 超參數
2. 處理不平衡分類問題，比較不同 `class_weights` 設定
3. 比較 CatBoost、XGBoost、LightGBM 在同一數據集上的表現
4. 使用 Pool 物件與 cv() 進行交叉驗證

### 12.3 專案練習

完成 `Unit13_CatBoost_Regression.ipynb` 與 `Unit13_CatBoost_Classification.ipynb`：
- 回歸任務：能源消耗預測（含類別特徵，R² > 0.98）
- 分類任務：設備故障診斷（7 類不平衡資料，Accuracy > 80%）
- 完整流程：數據探索、特徵工程、模型訓練、超參數調整、模型評估
- 深入分析：特徵重要性、混淆矩陣、預測結果可視化

### 12.4 挑戰練習

1. **能源預測優化**：
   - 嘗試添加更多時間序列特徵（lag、rolling）
   - 探索不同的多項式次數對性能的影響
   - 將 R² 提升至 0.995 以上

2. **故障診斷改進**：
   - 針對 Emergency Stop 類別（召回率僅 24.66%），設計提升策略
   - 嘗試使用 SMOTE 或其他過採樣技術
   - 探索集成多個 CatBoost 模型的效果

3. **模型比較分析**：
   - 在同一數據集上訓練 XGBoost、LightGBM、CatBoost
   - 比較訓練時間、預測準確度、特徵重要性
   - 撰寫技術報告說明三者的優劣勢

---

**恭喜完成 CatBoost 單元！**

您已經掌握：
✓ CatBoost 的核心原理與技術  
✓ 類別特徵的最佳處理方法  
✓ 如何建立、訓練、評估 CatBoost 模型  
✓ 超參數調整與模型優化策略  
✓ GBDT 三巨頭的比較與選擇建議  
✓ 化工領域實戰案例的完整分析  

**實戰成果**：
✓ 能源預測模型：R² = 0.9891，RMSE = 8.87 kWh  
✓ 故障診斷模型：Accuracy = 81.95%，處理 70:1 類別不平衡  
✓ 理解特徵重要性分析與模型可解釋性  
✓ 掌握類別特徵自動處理與不平衡資料處理  

**下一步**：
- 複習並執行兩個 notebook，理解完整建模流程
- 嘗試在自己的化工數據上應用 CatBoost
- 探索 CatBoost 的進階功能（GPU 加速、自訂損失函數）
- 準備進入深度學習單元！

