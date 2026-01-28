# Unit12 分類模型總覽 | Classification Models Overview

---

## 課程目標

本單元將深入介紹分類模型 (Classification Models) 的理論基礎與實務應用，專注於 scikit-learn 模組提供的各種分類方法。學生將學習：

- 理解分類問題的本質與數學原理
- 掌握從回歸到分類的轉換邏輯
- 學習使用 sklearn 建立、訓練與評估分類模型
- 了解分類模型在化工領域的實際應用案例
- 掌握分類模型的評估指標與選擇策略

---

## 1. 分類模型基礎理論

### 1.1 什麼是分類問題？

分類 (Classification) 是監督式學習中的核心任務之一，目標是根據特徵變數預測**離散的類別標籤**。與回歸預測連續數值不同，分類預測的是樣本屬於哪一個類別。

#### 1.1.1 分類問題的類型

**二元分類 (Binary Classification)**：
- 目標變數只有兩個類別（正類/負類，0/1）
- 範例：良品/不良品、安全/危險、反應成功/失敗

**多元分類 (Multi-class Classification)**：
- 目標變數有三個或以上的類別
- 範例：產品等級（A/B/C/D）、故障類型識別

**多標籤分類 (Multi-label Classification)**：
- 一個樣本可同時屬於多個類別
- 範例：化學品危害分類（可燃、腐蝕、毒性等）

### 1.2 從回歸到分類：為什麼回歸模型也可以做分類？

許多分類模型實際上是回歸模型的延伸，其核心思想是：

1. **先預測機率**：使用回歸模型預測樣本屬於某類別的機率值（0到1之間）
2. **再進行決策**：根據機率值與閾值比較，決定最終類別

數學上，分類可以視為：

$$
P(y=1|\mathbf{x}) = f(\mathbf{x})
$$

其中 $f(\mathbf{x})$ 是將特徵 $\mathbf{x}$ 映射到 [0,1] 區間的函數。

**關鍵轉換：激活函數 (Activation Function)**

為了將回歸輸出轉換為機率，需要使用激活函數：

- **Sigmoid 函數** (邏輯函數)：

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

將 $(-\infty, +\infty)$ 映射到 $(0, 1)$，適用於二元分類。

- **Softmax 函數**：

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

將多個輸出轉換為機率分佈，適用於多元分類。

### 1.3 分類模型的優點

分類模型在化工領域具有以下優勢：

1. **決策明確**：直接給出類別判斷，便於執行操作
2. **風險管理**：可設定不同閾值平衡誤判成本
3. **異常檢測**：識別製程異常、產品不良等問題
4. **品質控制**：自動化分級與檢驗流程

### 1.4 分類模型的基本假設

與回歸模型類似，分類模型也有一些基本要求：

1. **特徵相關性**：特徵與目標類別存在關聯
2. **類別可分性**：不同類別在特徵空間中具有一定的區隔度
3. **訓練資料充足**：每個類別都有足夠的樣本數
4. **類別平衡性**：各類別樣本數量不宜差異過大（若不平衡需特殊處理）

---

## 2. sklearn 中的分類模型方法

scikit-learn 提供了多種分類模型，從線性到非線性、從簡單到複雜，適用於不同的應用場景。

### 2.1 邏輯迴歸 (Logistic Regression)

**模型**：`sklearn.linear_model.LogisticRegression`

**核心概念**：
- 雖然名稱為「回歸」，實際上是**分類模型**
- 使用 Sigmoid 函數將線性組合轉換為機率
- 適用於線性可分或近似線性可分的問題

**數學原理**：

對於二元分類，邏輯迴歸預測正類的機率為：

$$
P(y=1|\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}
$$

訓練目標是最小化**對數損失 (Log Loss)**：

$$
L(\mathbf{w}) = -\frac{1}{m} \sum_{i=1}^{m} [y_i \log(\hat{p}_i) + (1-y_i) \log(1-\hat{p}_i)]
$$

**主要參數**：
- `penalty`: 正則化類型 ('l1', 'l2', 'elasticnet', None)
- `C`: 正則化強度的倒數（越小正則化越強）
- `solver`: 優化演算法 ('lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga')
- `multi_class`: 多元分類策略 ('ovr', 'multinomial')
- `class_weight`: 類別權重 ('balanced' 或自定義字典)

**適用場景**：
- 二元或多元分類問題
- 需要機率輸出
- 特徵與對數勝算 (log-odds) 呈線性關係
- 需要模型可解釋性

### 2.2 支持向量分類 (Support Vector Classification, SVC)

**模型**：`sklearn.svm.SVC`

**核心概念**：
- 尋找最佳**決策邊界 (Decision Boundary)** 使類別間隔最大化
- 透過**核函數 (Kernel Function)** 處理非線性問題
- 只依賴**支持向量 (Support Vectors)** 進行決策

**數學原理**：

目標是找到最大間隔超平面：

$$
\max_{\mathbf{w}, b} \frac{2}{\|\mathbf{w}\|} \quad \text{subject to} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1
$$

對於非線性問題，透過核函數映射到高維空間：

$$
K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)
$$

**主要參數**：
- `C`: 懲罰參數（控制誤分類的容忍度）
- `kernel`: 核函數類型 ('linear', 'poly', 'rbf', 'sigmoid')
- `gamma`: RBF、poly、sigmoid 核的係數 ('scale', 'auto' 或數值)
- `degree`: 多項式核的次數
- `class_weight`: 類別權重

**適用場景**：
- 高維度特徵空間
- 樣本數量中等（大數據集計算成本高）
- 需要處理非線性邊界
- 對異常值有一定容忍度

### 2.3 決策樹分類 (Decision Tree Classifier)

**模型**：`sklearn.tree.DecisionTreeClassifier`

**核心概念**：
- 透過樹狀結構進行分層決策
- 每個節點根據特徵進行分割
- 葉節點代表最終類別

**分割準則**：

**基尼不純度 (Gini Impurity)**：

$$
\text{Gini} = 1 - \sum_{i=1}^{K} p_i^2
$$

**信息熵 (Entropy)**：

$$
\text{Entropy} = -\sum_{i=1}^{K} p_i \log_2(p_i)
$$

其中 $p_i$ 是類別 $i$ 的比例。

**主要參數**：
- `criterion`: 分割準則 ('gini', 'entropy')
- `max_depth`: 樹的最大深度
- `min_samples_split`: 分割節點所需的最小樣本數
- `min_samples_leaf`: 葉節點所需的最小樣本數
- `max_features`: 尋找最佳分割時考慮的最大特徵數
- `class_weight`: 類別權重

**適用場景**：
- 需要模型可解釋性（可視化決策路徑）
- 特徵包含類別變數
- 資料有非線性關係
- 不需要特徵標準化

---

### 2.4 梯度提升分類 (Gradient Boosting Classifier)

**模型**：`sklearn.ensemble.GradientBoostingClassifier`

**核心概念**：
- **Boosting 策略**：依序建立決策樹，每棵樹修正前一棵的誤差
- **梯度下降**：沿著損失函數的負梯度方向更新模型
- **加性模型**：最終預測是所有樹的加權和

**數學原理**：

模型以加性形式建構：

$$
F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \gamma_m h_m(\mathbf{x})
$$

其中 $h_m(\mathbf{x})$ 是第 $m$ 棵樹， $\gamma_m$ 是學習率。

**主要參數**：
- `n_estimators`: 提升迭代次數（樹的數量）
- `learning_rate`: 學習率（縮減每棵樹的貢獻）
- `max_depth`: 每棵樹的最大深度（通常較淺，3-5）
- `min_samples_split`: 分割節點所需的最小樣本數
- `min_samples_leaf`: 葉節點所需的最小樣本數
- `subsample`: 訓練每棵樹的樣本子集比例
- `loss`: 損失函數 ('log_loss', 'exponential')

**適用場景**：
- 追求最高預測準確度
- 特徵與目標有複雜非線性關係
- 需要特徵重要性分析
- 有足夠的計算資源和時間

**注意事項**：
- 訓練時間較長
- 容易過擬合（需謹慎調參）
- 對異常值較敏感

### 2.5 貝氏分類器 (Gaussian Naive Bayes)

**模型**：`sklearn.naive_bayes.GaussianNB`

**核心概念**：
- 基於**貝氏定理 (Bayes' Theorem)** 
- 假設特徵之間**條件獨立**（Naive 假設）
- 假設特徵服從**高斯分佈**

**數學原理**：

貝氏定理：

$$
P(y|\mathbf{x}) = \frac{P(\mathbf{x}|y) P(y)}{P(\mathbf{x})}
$$

在獨立性假設下：

$$
P(\mathbf{x}|y) = \prod_{i=1}^{n} P(x_i|y)
$$

對於高斯分佈：

$$
P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_{y,i}^2}} \exp\left(-\frac{(x_i - \mu_{y,i})^2}{2\sigma_{y,i}^2}\right)
$$

**主要參數**：
- `priors`: 各類別的先驗機率
- `var_smoothing`: 為數值穩定性加入的變異數平滑項

**適用場景**：
- 訓練資料較少
- 需要快速訓練和預測
- 特徵相對獨立
- 多元分類問題
- 文字分類、垃圾郵件過濾

**優點**：
- 訓練和預測速度極快
- 參數少，不易過擬合
- 對缺失資料有較好的處理能力

**限制**：
- 特徵獨立假設在實務中常不成立
- 特徵分佈假設可能不符實際

### 2.6 模型比較總結

| 模型 | 線性/非線性 | 可解釋性 | 訓練速度 | 預測速度 | 適合資料量 | 特徵要求 |
|------|-------------|----------|----------|----------|------------|----------|
| Logistic Regression | 線性 | 高 | 快 | 快 | 中到大 | 需標準化 |
| SVC | 非線性 | 低 | 慢 | 中 | 小到中 | 需標準化 |
| Decision Tree | 非線性 | 高 | 快 | 快 | 中 | 無要求 |
| Gradient Boosting | 非線性 | 中 | 慢 | 中 | 中到大 | 無要求 |
| Naive Bayes | 線性 | 高 | 極快 | 極快 | 小到中 | 假設獨立 |

---

## 3. 資料前處理技術

分類模型的資料前處理與回歸模型類似，但有一些分類特有的考量。

### 3.1 特徵縮放

**標準化 (Standardization)**：

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**何時需要**：
- 邏輯迴歸、SVC：強烈建議
- 決策樹系列：不需要
- 貝氏分類器：視情況而定

### 3.2 類別變數編碼

**獨熱編碼 (One-Hot Encoding)**：

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, drop='first')
X_encoded = encoder.fit_transform(X_categorical)
```

**標籤編碼 (Label Encoding)**：

```python
from sklearn.preprocessing import LabelEncoder

# 用於目標變數
le = LabelEncoder()
y_encoded = le.fit_transform(y)
```

### 3.3 處理類別不平衡問題

類別不平衡是分類問題中的常見挑戰，當某個類別樣本數遠少於其他類別時，模型容易偏向多數類。

#### 3.3.1 類別權重調整

**在模型中設置 class_weight**：

```python
from sklearn.linear_model import LogisticRegression

# 自動平衡
model = LogisticRegression(class_weight='balanced')

# 或手動設定
model = LogisticRegression(class_weight={0: 1, 1: 10})
```

權重計算方式：

$$
w_i = \frac{n_{\text{samples}}}{n_{\text{classes}} \times n_{\text{samples}, i}}
$$

#### 3.3.2 重採樣技術

**過採樣 (Oversampling)**：增加少數類樣本

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**欠採樣 (Undersampling)**：減少多數類樣本

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
```

**組合策略**：

```python
from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X_train, y_train)
```

#### 3.3.3 閾值調整

預測時調整決策閾值：

```python
# 預測機率
y_proba = model.predict_proba(X_test)[:, 1]

# 自定義閾值
threshold = 0.3
y_pred_adjusted = (y_proba >= threshold).astype(int)
```

### 3.4 Pipeline 整合

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

---

## 4. 模型評估方法

分類模型的評估指標與回歸模型完全不同，需要考慮混淆矩陣及衍生指標。

### 4.1 混淆矩陣 (Confusion Matrix)

混淆矩陣是分類評估的基礎，顯示預測與實際的對應關係。

**二元分類混淆矩陣**：

|  | 預測為正類 | 預測為負類 |
|---|-----------|-----------|
| **實際為正類** | TP (真陽性) | FN (假陰性) |
| **實際為負類** | FP (假陽性) | TN (真陰性) |

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
```

### 4.2 準確率 (Accuracy)

最直觀的評估指標，表示預測正確的比例。

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

```python
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)
```

**注意**：在類別不平衡時，準確率可能誤導（多數類偏差）。

### 4.3 精確率 (Precision)

預測為正類的樣本中，實際為正類的比例。

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

```python
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred)
```

**意義**：衡量**假陽性**的控制能力。高精確率表示預測為正類時較可靠。

### 4.4 召回率 / 靈敏度 (Recall / Sensitivity)

實際為正類的樣本中，被正確預測的比例。

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

```python
from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred)
```

**意義**：衡量**假陰性**的控制能力。高召回率表示能找出大部分正類樣本。

### 4.5 F1 分數 (F1-Score)

精確率和召回率的調和平均數。

$$
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)
```

**意義**：在精確率和召回率之間取得平衡，適合類別不平衡問題。

### 4.6 ROC 曲線與 AUC

**ROC 曲線 (Receiver Operating Characteristic Curve)**：

- X 軸：假陽性率 (FPR) = $\frac{FP}{FP + TN}$ 
- Y 軸：真陽性率 (TPR) = Recall = $\frac{TP}{TP + FN}$ 

**AUC (Area Under Curve)**：ROC 曲線下方的面積
- AUC = 1：完美分類器
- AUC = 0.5：隨機猜測
- AUC > 0.8：通常認為是良好的模型

```python
from sklearn.metrics import roc_curve, roc_auc_score, auc

# 計算 ROC 曲線
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# 計算 AUC
roc_auc = auc(fpr, tpr)
# 或直接使用
roc_auc = roc_auc_score(y_test, y_proba)

# 繪製 ROC 曲線
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

### 4.7 分類報告 (Classification Report)

綜合顯示主要評估指標：

```python
from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred)
print(report)
```

輸出範例：
```
              precision    recall  f1-score   support

           0       0.88      0.92      0.90       100
           1       0.91      0.86      0.88        90

    accuracy                           0.89       190
   macro avg       0.89      0.89      0.89       190
weighted avg       0.89      0.89      0.89       190
```

### 4.8 多元分類評估

對於多元分類，可採用不同的平均策略：

- **macro average**：每個類別同等重要，計算各類別指標的算術平均
- **weighted average**：依各類別樣本數加權平均
- **micro average**：將所有類別的 TP、FP、FN 合併計算

```python
# 多元分類的 F1 分數
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
```

---

## 5. 交叉驗證與超參數調整

### 5.1 K-折交叉驗證 (K-Fold Cross-Validation)

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f'Cross-validation scores: {scores}')
print(f'Mean accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})')
```

**分層 K-折交叉驗證 (Stratified K-Fold)**：

保持每折中各類別的比例與原始資料集相同，特別適合類別不平衡問題。

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
```

### 5.2 網格搜索 (Grid Search)

窮舉式搜索最佳超參數組合。

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f'Best parameters: {grid_search.best_params_}')
print(f'Best F1 score: {grid_search.best_score_:.3f}')

# 使用最佳模型
best_model = grid_search.best_estimator_
```

### 5.3 隨機搜索 (Randomized Search)

隨機採樣參數組合，比網格搜索更有效率。

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 300),
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': randint(3, 10),
    'min_samples_split': randint(2, 20),
    'subsample': uniform(0.6, 0.4)
}

random_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_distributions,
    n_iter=50,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
```

### 5.4 評估指標選擇

針對不同問題選擇合適的評估指標：

| 問題類型 | 推薦指標 | 原因 |
|---------|---------|------|
| 類別平衡 | Accuracy | 直觀且有效 |
| 類別不平衡 | F1-score, AUC | 避免多數類偏差 |
| 重視假陽性控制 | Precision | 如：垃圾郵件過濾 |
| 重視假陰性控制 | Recall | 如：疾病檢測、安全監控 |
| 需要機率輸出 | Log Loss, AUC | 評估機率預測品質 |

```python
# 使用多個評估指標
from sklearn.model_selection import cross_validate

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

scores = cross_validate(model, X, y, cv=5, scoring=scoring)

for metric, values in scores.items():
    if metric.startswith('test_'):
        print(f'{metric[5:]}: {values.mean():.3f} (+/- {values.std():.3f})')
```

---

## 6. 化工領域應用案例

分類模型在化學工程領域有廣泛的應用，以下列舉幾個典型案例：

### 6.1 製程異常檢測

**問題描述**：
根據製程參數（溫度、壓力、流量等）即時判斷設備運行狀態是否正常。

**應用模型**：
- **梯度提升分類器**：可捕捉參數間複雜的非線性關係，處理多個感測器資料
- **決策樹分類器**：提供可解釋的決策規則

**特徵範例**：
- 操作溫度、壓力、流量
- 統計特徵（移動平均、標準差）
- 時序特徵（變化率、趨勢）

**目標變數**：
- 正常運行 vs. 異常狀態
- 或細分為：正常 / 輕微異常 / 嚴重異常

**評估重點**：
- 高召回率（避免漏檢異常）
- 可接受的假陽性率（避免過多誤報）

### 6.2 產品品質分級

**問題描述**：
根據生產參數和檢測數據，將產品分為不同品質等級（A/B/C 級或合格/不合格）。

**應用模型**：
- **邏輯迴歸**：簡單問題，需要可解釋性
- **支持向量分類**：處理中等規模、高維度數據
- **決策樹**：可視化決策規則，便於製程優化

**特徵範例**：
- 原料性質
- 操作條件（溫度、時間、催化劑用量）
- 產品物理/化學性質

**目標變數**：
- 二元分類：合格/不合格
- 多元分類：A級/B級/C級/不合格

**評估重點**：
- 準確率與混淆矩陣（了解誤判類型）
- 精確率（避免錯誤分級）

### 6.3 化學反應成功預測

**問題描述**：
預測在給定條件下，化學反應是否能成功進行或達到預期轉化率。

**應用模型**：
- **梯度提升**：追求最高預測準確度，處理複雜的非線性關係
- **支持向量分類**：處理中等規模數據的非線性問題
- **貝氏分類器**：資料量少時的快速建模

**特徵範例**：
- 反應物濃度、比例
- 反應溫度、壓力、時間
- 催化劑類型和用量
- 溶劑性質

**目標變數**：
- 反應成功/失敗
- 轉化率等級（高/中/低）

**評估重點**：
- F1-score（平衡成功和失敗類別的預測）
- ROC-AUC（評估整體分類能力）

### 6.4 設備故障診斷

**問題描述**：
根據設備運行數據和振動/聲音信號，診斷設備故障類型。

**應用模型**：
- **支持向量分類**：處理高維特徵（如頻譜分析結果）
- **梯度提升分類**：多傳感器融合，特徵重要性分析
- **決策樹**：簡單故障樹診斷邏輯

**特徵範例**：
- 振動信號頻譜特徵
- 溫度、壓力異常指標
- 運轉時間、負載情況
- 歷史維護記錄

**目標變數**：
- 正常/軸承故障/齒輪故障/不平衡等

**評估重點**：
- 多元分類的混淆矩陣
- 各故障類型的召回率（避免漏診）

### 6.5 安全風險等級評估

**問題描述**：
評估化學品處理或製程操作的安全風險等級。

**應用模型**：
- **邏輯迴歸**：可解釋性強，符合法規要求
- **決策樹**：可視化風險決策路徑
- **梯度提升分類**：綜合多因素的風險評估

**特徵範例**：
- 化學品性質（毒性、易燃性、反應性）
- 操作條件（溫度、壓力、濃度）
- 安全措施（防護設備、監控系統）
- 環境因素（通風、溫度、濕度）

**目標變數**：
- 低風險/中風險/高風險
- 安全/需警戒/危險

**評估重點**：
- 高召回率（不能漏判高風險情況）
- 加權 F1-score（高風險類別更重要）

### 6.6 實際應用考量

在化工領域應用分類模型時，需要特別注意：

1. **安全性優先**：寧可誤報也不能漏檢危險情況
2. **可解釋性**：模型決策需要能向工程師和管理層解釋
3. **實時性**：某些應用需要快速預測（如異常檢測）
4. **穩健性**：模型需對感測器雜訊和異常值有容忍度
5. **持續更新**：隨著製程變化，模型需要定期重新訓練

---

## 7. 實用技巧與最佳實踐

### 7.1 模型選擇策略

**初步建模階段**：
1. 從簡單模型開始（邏輯迴歸、決策樹）
2. 建立基準線 (Baseline) 性能
3. 理解資料特性與問題複雜度

**模型改進階段**：
1. 嘗試集成學習模型（梯度提升）或核方法（支持向量機）
2. 進行超參數調整
3. 處理類別不平衡問題

**最終選擇考量**：
- 預測性能
- 訓練和預測時間
- 可解釋性需求
- 維護成本

### 7.2 特徵工程技巧

**領域知識融入**：
```python
# 範例：化工製程特徵
df['temp_pressure_ratio'] = df['temperature'] / df['pressure']
df['flow_rate_normalized'] = df['flow_rate'] / df['reactor_volume']
df['conversion_efficiency'] = df['product_yield'] / df['reactant_input']
```

**時序特徵**：
```python
# 移動平均
df['temp_ma_5'] = df['temperature'].rolling(window=5).mean()

# 變化率
df['temp_change'] = df['temperature'].diff()

# 標準差（波動性）
df['temp_std_10'] = df['temperature'].rolling(window=10).std()
```

**交互特徵**：
```python
from sklearn.preprocessing import PolynomialFeatures

# 創建交互項
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interactions = poly.fit_transform(X)
```

### 7.3 模型診斷與調試

**學習曲線 (Learning Curve)**：

```python
from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='f1'
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, val_mean, label='Validation score')
plt.xlabel('Training Set Size')
plt.ylabel('F1 Score')
plt.legend()
plt.title('Learning Curve')
plt.show()
```

**特徵重要性分析**：

```python
# 樹模型的特徵重要性
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()
```

**錯誤分析**：

```python
# 找出預測錯誤的樣本
errors = X_test[y_test != y_pred]
print(f'Number of errors: {len(errors)}')

# 分析錯誤樣本的特徵分佈
errors_analysis = pd.DataFrame(errors)
print(errors_analysis.describe())
```

### 7.4 模型保存與部署

**保存模型**：

```python
import joblib

# 保存模型
joblib.dump(model, 'classification_model.pkl')

# 保存 Pipeline（含前處理）
joblib.dump(pipeline, 'classification_pipeline.pkl')

# 載入模型
loaded_model = joblib.load('classification_model.pkl')
```

**模型版本管理**：

```python
import datetime

model_version = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f'model_{model_version}.pkl'
joblib.dump(model, model_filename)

# 保存模型元數據
metadata = {
    'version': model_version,
    'model_type': type(model).__name__,
    'features': feature_names,
    'performance': {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba)
    }
}
```

### 7.5 常見錯誤與解決方案

| 問題 | 可能原因 | 解決方案 |
|------|---------|---------|
| 訓練集準確但測試集差 | 過擬合 | 增加正則化、減少模型複雜度、增加訓練數據 |
| 兩者準確率都低 | 欠擬合 | 增加特徵、使用更複雜模型、調整超參數 |
| 某類別完全預測錯誤 | 類別不平衡 | 使用 class_weight、重採樣、調整閾值 |
| 預測機率不準確 | 模型未校準 | 使用 CalibratedClassifierCV |
| 訓練時間過長 | 資料量大或模型複雜 | 使用 SGD 系列、減少特徵、調整 n_jobs |

**模型校準**：

```python
from sklearn.calibration import CalibratedClassifierCV

# 校準模型的機率輸出
calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
calibrated_model.fit(X_train, y_train)
```

---

## 8. 分類 vs. 回歸：何時使用哪種方法

### 8.1 選擇決策樹

| 考量因素 | 分類 | 回歸 |
|---------|------|------|
| **目標變數** | 離散類別標籤 | 連續數值 |
| **預測輸出** | 類別或機率 | 具體數值 |
| **評估指標** | 準確率、F1、AUC | MSE、RMSE、R² |
| **應用範例** | 良品/不良品判定 | 產量預測 |
| **決策邊界** | 明確的類別分界 | 連續的函數關係 |

### 8.2 可以轉換的情況

**回歸轉分類**：
- 將連續目標變數離散化（如：溫度 → 低/中/高溫）
- 適用於只需要類別判斷的場景

```python
# 範例：將連續溫度轉為類別
df['temp_category'] = pd.cut(df['temperature'], 
                               bins=[0, 50, 100, 150], 
                               labels=['Low', 'Medium', 'High'])
```

**分類轉回歸**：
- 使用機率輸出作為連續預測值
- 適用於需要置信度量化的場景

```python
# 使用預測機率作為"風險分數"
risk_score = model.predict_proba(X)[:, 1]
```

### 8.3 實務建議

- 若業務需求是**做決策**（如：是否停機檢修），使用**分類**
- 若業務需求是**預測數值**（如：預測產量），使用**回歸**
- 若兩者皆可，考慮**評估指標的可解釋性**和**模型選擇的靈活性**

---

## 9. 本單元學習路徑

完成本單元後，您將依序學習以下主題：

1. **Unit12_Logistic_Regression**：邏輯迴歸的詳細理論與實作
2. **Unit12_Support_Vector_Classification**：SVC 的核函數與應用
3. **Unit12_Decision_Tree_Classifier**：決策樹分類的原理與可視化
4. **Unit12_Gradient_Boosting_Classifier**：梯度提升的進階技術
5. **Unit12_Gaussian_Naive_Bayes**：貝氏分類器的機率推論
6. **Unit12_Classification_Models_Homework**：綜合練習與模型比較

每個子主題都包含詳細的數學推導、程式碼實作和化工領域的應用案例，請按順序學習以獲得最佳效果。

---

## 10. 進階主題預覽

本單元之後，您還可以深入學習：

1. **集成學習進階**：
   - Stacking (堆疊)
   - Voting Classifiers (投票分類器)
   - XGBoost, LightGBM (進階梯度提升)

2. **深度學習分類**：
   - 多層感知器 (MLP Classifier)
   - 卷積神經網路 (CNN) 用於影像分類
   - 序列模型用於時序分類

3. **不平衡學習**：
   - 進階重採樣技術 (ADASYN, Borderline-SMOTE)
   - 成本敏感學習 (Cost-sensitive Learning)
   - 異常檢測 (Anomaly Detection)

4. **多標籤與多輸出分類**：
   - Binary Relevance
   - Classifier Chains
   - Label Powerset

---

## 11. 課前準備

在開始本單元之前，請確保您已經：

1. ✅ 完成 Unit10 (線性模型回歸) 和 Unit11 (非線性模型回歸)
2. ✅ 理解基本統計概念：機率、條件機率、貝氏定理
3. ✅ 熟悉 sklearn 基本操作流程
4. ✅ 安裝必要套件：
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
   ```
5. ✅ 準備 Jupyter Notebook 或 Python IDE 環境

---

## 12. 總結

本單元介紹了分類模型的核心概念：

- **分類問題**是預測離散類別標籤，與回歸預測連續數值不同
- **從回歸到分類**的關鍵是使用激活函數將輸出轉換為機率
- **sklearn 提供多種分類模型**，從邏輯迴歸到集成學習，各有優缺點
- **評估指標**包括準確率、精確率、召回率、F1、AUC 等，需根據問題選擇
- **類別不平衡**是常見挑戰，可透過權重調整、重採樣、閾值調整解決
- **化工應用**涵蓋異常檢測、品質分級、故障診斷、安全評估等領域

掌握分類模型後，您將能夠解決化工領域中的各種決策問題，從製程監控到產品品質控制，大幅提升生產效率與安全性。

接下來的各子單元將深入探討每種分類模型的細節與實作，請繼續學習！

---

**課程資訊**
- 課程名稱：AI在化工上之應用
- 課程單元：Unit12 Classification Models Overview 分類模型總覽
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-01-28

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---

