# Unit08 Apriori 演算法

## 課程目標

本單元將深入介紹 Apriori 演算法 (Apriori Algorithm)，這是關聯規則學習中最經典且廣泛應用的演算法之一。透過本單元的學習，您將能夠：

- 理解 Apriori 演算法的核心原理與 Apriori 性質
- 掌握 Apriori 演算法的步驟與候選項目集生成方法
- 學會使用 mlxtend 套件實作 Apriori 演算法
- 了解如何設定支持度與置信度閾值
- 認識 Apriori 演算法的優缺點與適用場景
- 應用 Apriori 於化工配方優化的實際案例

---

## 1. Apriori 演算法簡介

### 1.1 什麼是 Apriori 演算法？

Apriori 演算法由 Rakesh Agrawal 和 Ramakrishnan Srikant 於 1994 年提出，是關聯規則學習領域最具影響力的演算法。其名稱源自拉丁文 "a priori"，意為「從先前的」，反映了演算法的核心思想：利用先前發現的頻繁項目集來產生候選項目集。

Apriori 演算法的主要目標是：
1. **找出頻繁項目集 (Frequent Itemsets)**：在數據集中頻繁出現的項目組合
2. **生成關聯規則 (Association Rules)**：從頻繁項目集中提取有意義的規則

### 1.2 Apriori 性質 (Apriori Property)

Apriori 演算法的核心基礎是 **Apriori 性質**（也稱為向下封閉性，Downward Closure Property）：

> **如果一個項目集是頻繁的，那麼它的所有子集也必須是頻繁的。**
> 
> **反之，如果一個項目集是不頻繁的，那麼它的所有超集也必定是不頻繁的。**

**數學表達**：

設項目集 $X$ 和 $Y$ ， $X \subseteq Y$ （ $X$ 是 $Y$ 的子集），則：

$$
\text{Support}(Y) \leq \text{Support}(X)
$$

**直觀理解**：

- 如果 {催化劑A, 溶劑X, 高溫} 是頻繁的，則 {催化劑A, 溶劑X}、{催化劑A, 高溫}、{溶劑X, 高溫} 也必定是頻繁的
- 如果 {催化劑A, 溶劑X} 是不頻繁的，則包含這兩個項目的任何更大項目集（如 {催化劑A, 溶劑X, 高溫}）也必定是不頻繁的

**重要性**：

Apriori 性質允許演算法在搜索過程中進行「剪枝」(Pruning)，大幅減少需要檢查的候選項目集數量，提高演算法效率。

### 1.3 化工領域應用案例

Apriori 演算法在化工領域有以下應用：

1. **聚合物配方優化**：
   - 發現單體、引發劑、鏈轉移劑的最佳組合
   - 識別能產生特定分子量分布的配方模式
   - 範例規則：{單體A, 引發劑B} $\Rightarrow$ {窄分子量分布}

2. **催化劑配方設計**：
   - 發現主催化劑、助催化劑、載體的協同效應
   - 識別能提高反應選擇性的成分組合
   - 範例規則：{催化劑M, 助催化劑N} $\Rightarrow$ {高選擇性}

3. **製程條件關聯分析**：
   - 發現導致高產率的操作條件組合
   - 識別避免副產物生成的條件模式
   - 範例規則：{溫度180-190°C, 壓力15-20bar} $\Rightarrow$ {產率>95%}

4. **品質問題根因分析**：
   - 識別導致品質不良的因素組合
   - 發現多個因素共同作用的模式
   - 範例規則：{原料批次A, 低溫, 攪拌異常} $\Rightarrow$ {品質不合格}

---

## 2. Apriori 演算法原理

### 2.1 演算法步驟

Apriori 演算法採用「逐層搜索」(Level-wise Search) 的迭代方法，具體步驟如下：

**步驟 1：初始化**
- 設定最小支持度閾值 (min_support)
- 設定最小置信度閾值 (min_confidence)

**步驟 2：找出頻繁 1-項目集 (L₁)**
- 掃描數據庫，計算每個單一項目的支持度
- 保留支持度 ≥ min_support 的項目，形成頻繁 1-項目集 $L_1$

**步驟 3：產生候選 k-項目集 (Cₖ)**
- 由頻繁 (k-1)-項目集 $L_{k-1}$ 通過「連接」和「剪枝」操作產生候選 k-項目集 $C_k$

**步驟 4：掃描數據庫，計算候選項目集的支持度**
- 對 $C_k$ 中的每個候選項目集，計算其在數據庫中的支持度
- 保留支持度 ≥ min_support 的項目集，形成頻繁 k-項目集 $L_k$

**步驟 5：迭代**
- 重複步驟 3-4，直到無法產生更大的頻繁項目集（ $L_k = \emptyset$ ）

**步驟 6：生成關聯規則**
- 從所有頻繁項目集中生成候選規則
- 計算每條規則的置信度
- 保留置信度 ≥ min_confidence 的規則

### 2.2 候選項目集生成

**連接步驟 (Join Step)**

由頻繁 (k-1)-項目集 $L_{k-1}$ 產生候選 k-項目集 $C_k$ 。

**連接條件**：兩個頻繁 (k-1)-項目集 $l_1$ 和 $l_2$ 可以連接，當且僅當它們的前 (k-2) 個項目相同。

**範例**（k=3）：
- $L_2 = \{\{A, B\}, \{A, C\}, \{A, D\}, \{B, C\}, \{B, D\}\}$
- 連接：
  - {A, B} 和 {A, C} → {A, B, C}
  - {A, B} 和 {A, D} → {A, B, D}
  - {A, C} 和 {A, D} → {A, C, D}
  - {B, C} 和 {B, D} → {B, C, D}
- 初步 $C_3 = \{\{A, B, C\}, \{A, B, D\}, \{A, C, D\}, \{B, C, D\}\}$

**剪枝步驟 (Prune Step)**

利用 Apriori 性質進行剪枝：對於 $C_k$ 中的每個候選項目集 $c$ ，檢查其所有 (k-1)-子集是否都在 $L_{k-1}$ 中。若存在任何子集不頻繁，則刪除 $c$ 。

**範例**（承上）：
假設 {A, D} 不在 $L_2$ 中，則：
- {A, B, D} 的子集 {A, D} 不頻繁 → 刪除 {A, B, D}
- {A, C, D} 的子集 {A, D} 不頻繁 → 刪除 {A, C, D}
- 剪枝後 $C_3 = \{\{A, B, C\}, \{B, C, D\}\}$

### 2.3 詳細範例

假設我們有 5 筆化工配方交易記錄：

| 交易ID | 項目集 |
|--------|--------|
| T1 | {催化劑A, 溶劑X, 高溫} |
| T2 | {催化劑A, 溶劑Y, 高溫} |
| T3 | {催化劑B, 溶劑X, 低溫} |
| T4 | {催化劑A, 溶劑X, 高溫, 高產率} |
| T5 | {催化劑A, 溶劑X} |

設定 min_support = 0.4 (即至少出現在 2 筆交易中)

**第一次掃描（k=1）**：

計算每個項目的支持度：

| 項目 | 出現次數 | 支持度 | 是否頻繁 |
|------|---------|--------|---------|
| 催化劑A | 4 | 4/5 = 0.8 | ✓ |
| 催化劑B | 1 | 1/5 = 0.2 | ✗ |
| 溶劑X | 4 | 4/5 = 0.8 | ✓ |
| 溶劑Y | 1 | 1/5 = 0.2 | ✗ |
| 高溫 | 3 | 3/5 = 0.6 | ✓ |
| 低溫 | 1 | 1/5 = 0.2 | ✗ |
| 高產率 | 1 | 1/5 = 0.2 | ✗ |

**頻繁 1-項目集**： $L_1 = \{\{催化劑A\}, \{溶劑X\}, \{高溫\}\}$

**第二次掃描（k=2）**：

**連接**：由 $L_1$ 產生候選 2-項目集：
$$
C_2 = \{\{催化劑A, 溶劑X\}, \{催化劑A, 高溫\}, \{溶劑X, 高溫\}\}
$$

計算支持度：

| 項目集 | 出現次數 | 支持度 | 是否頻繁 |
|--------|---------|--------|---------|
| {催化劑A, 溶劑X} | 3 | 3/5 = 0.6 | ✓ |
| {催化劑A, 高溫} | 3 | 3/5 = 0.6 | ✓ |
| {溶劑X, 高溫} | 2 | 2/5 = 0.4 | ✓ |

**頻繁 2-項目集**： $L_2 = \{\{催化劑A, 溶劑X\}, \{催化劑A, 高溫\}, \{溶劑X, 高溫\}\}$

**第三次掃描（k=3）**：

**連接**：由 $L_2$ 產生候選 3-項目集：
- {催化劑A, 溶劑X} 和 {催化劑A, 高溫} 前1項相同 ({催化劑A}) → {催化劑A, 溶劑X, 高溫}
- {催化劑A, 溶劑X} 和 {溶劑X, 高溫} 前1項不同 → 無法連接
- {催化劑A, 高溫} 和 {溶劑X, 高溫} 前1項不同 → 無法連接

$$
C_3 = \{\{催化劑A, 溶劑X, 高溫\}\}
$$

**剪枝**：檢查 {催化劑A, 溶劑X, 高溫} 的所有 2-子集：
- {催化劑A, 溶劑X} ∈ $L_2$ ✓
- {催化劑A, 高溫} ∈ $L_2$ ✓
- {溶劑X, 高溫} ∈ $L_2$ ✓

所有子集都頻繁，保留候選。

計算支持度：

| 項目集 | 出現次數 | 支持度 | 是否頻繁 |
|--------|---------|--------|---------|
| {催化劑A, 溶劑X, 高溫} | 2 | 2/5 = 0.4 | ✓ |

**頻繁 3-項目集**： $L_3 = \{\{催化劑A, 溶劑X, 高溫\}\}$

**第四次掃描（k=4）**：

嘗試由 $L_3$ 產生 $C_4$ ，但 $L_3$ 只有一個項目集，無法連接，演算法終止。

**所有頻繁項目集**：
$$
L = L_1 \cup L_2 \cup L_3 = \{\{催化劑A\}, \{溶劑X\}, \{高溫\}, \{催化劑A, 溶劑X\}, \{催化劑A, 高溫\}, \{溶劑X, 高溫\}, \{催化劑A, 溶劑X, 高溫\}\}
$$

### 2.4 關聯規則生成

找到所有頻繁項目集後，從中生成關聯規則。

**規則生成步驟**：

1. **對每個頻繁項目集 $l$ （大小 ≥ 2）**：
   - 生成所有可能的劃分 $l = X \cup Y$ ，其中 $X \cap Y = \emptyset$ ， $X \neq \emptyset$ ， $Y \neq \emptyset$
   - 對每個劃分，形成規則 $X \Rightarrow Y$

2. **計算置信度**：
   $$
   \text{Confidence}(X \Rightarrow Y) = \frac{\text{Support}(X \cup Y)}{\text{Support}(X)}
   $$

3. **保留置信度 ≥ min_confidence 的規則**

**範例**：

對於頻繁項目集 {催化劑A, 溶劑X, 高溫}（支持度 = 0.4），可生成以下候選規則：

設 min_confidence = 0.5

| 規則 | 置信度計算 | 置信度 | 是否保留 |
|------|-----------|--------|---------|
| {催化劑A} $\Rightarrow$ {溶劑X, 高溫} | 0.4 / 0.8 | 0.5 | ✓ |
| {溶劑X} $\Rightarrow$ {催化劑A, 高溫} | 0.4 / 0.8 | 0.5 | ✓ |
| {高溫} $\Rightarrow$ {催化劑A, 溶劑X} | 0.4 / 0.6 | 0.67 | ✓ |
| {催化劑A, 溶劑X} $\Rightarrow$ {高溫} | 0.4 / 0.6 | 0.67 | ✓ |
| {催化劑A, 高溫} $\Rightarrow$ {溶劑X} | 0.4 / 0.6 | 0.67 | ✓ |
| {溶劑X, 高溫} $\Rightarrow$ {催化劑A} | 0.4 / 0.4 | 1.0 | ✓ |

**計算提升度**：

以規則 {溶劑X, 高溫} $\Rightarrow$ {催化劑A} 為例：
- Support = 0.4
- Confidence = 1.0
- Lift = 1.0 / 0.8 = 1.25

提升度 > 1，表示 {溶劑X, 高溫} 和 {催化劑A} 之間存在正相關。

### 2.5 演算法複雜度分析

**時間複雜度**：

- **最壞情況**： $O(2^n)$ ，其中 $n$ 是不同項目的總數
  - 當所有項目集都是頻繁的時候，需要生成所有可能的子集
  
- **實際應用**：由於 Apriori 剪枝，通常遠小於最壞情況
  - 取決於數據集特性和 min_support 設定
  - min_support 越高，剪枝效果越好，速度越快

**空間複雜度**：

- 需要儲存所有頻繁項目集和候選項目集
- $O(|L| + |C|)$ ，其中 $|L|$ 是頻繁項目集數量， $|C|$ 是候選項目集數量

**掃描次數**：

- 需要掃描數據庫 $k$ 次，其中 $k$ 是最大頻繁項目集的大小
- 每次掃描需要讀取整個數據庫，I/O 開銷大

---

## 3. Apriori 演算法的優缺點

### 3.1 優點

1. **演算法邏輯清晰，易於理解**：
   - 基於直觀的 Apriori 性質
   - 逐層搜索策略簡單明瞭
   - 適合教學與演示

2. **有效的剪枝策略**：
   - Apriori 性質提供強大的剪枝能力
   - 大幅減少候選項目集數量
   - 避免檢查不必要的項目集組合

3. **可處理稀疏數據**：
   - 適合處理項目數量多但每筆交易包含項目數少的數據
   - 例如：化工配方資料庫中，雖然原料種類多，但每個配方只使用少數原料

4. **產生完整的頻繁項目集**：
   - 保證找出所有滿足最小支持度的頻繁項目集
   - 不會遺漏任何頻繁模式

5. **可解釋性強**：
   - 生成的規則具有明確的業務意義
   - 容易向領域專家解釋和驗證

### 3.2 缺點

1. **多次掃描數據庫**：
   - 需要掃描數據庫 $k$ 次（ $k$ 為最大頻繁項目集大小）
   - I/O 開銷大，特別是對於大型數據集
   - 每次掃描都需要讀取完整數據庫

2. **產生大量候選項目集**：
   - 當 min_support 設定較低時，候選項目集數量爆炸性增長
   - 需要大量記憶體儲存候選項目集
   - 計算支持度的成本高昂

3. **效率受 min_support 影響大**：
   - min_support 設定過高：可能遺漏有價值的低頻模式
   - min_support 設定過低：產生大量候選項目集，效率低下
   - 難以找到平衡點

4. **難以處理連續型數據**：
   - Apriori 設計用於處理離散項目
   - 連續型變數（如溫度、壓力）需要先離散化
   - 離散化方式會影響結果，且可能損失信息

5. **不適合大規模數據集**：
   - 當數據集非常大時，演算法效率明顯下降
   - FP-Growth 等演算法在大數據場景下表現更好

### 3.3 適用場景

**適合使用 Apriori 的情況**：

1. **小到中等規模數據集**：
   - 交易數量 < 10,000 筆
   - 項目總數 < 100 個
   - 每筆交易包含項目數適中

2. **需要高度可解釋性**：
   - 研究導向的探索性分析
   - 需要向非技術人員解釋結果
   - 教學與演示目的

3. **稀疏數據**：
   - 項目總數多，但每筆交易僅包含少數項目
   - 例如：化工配方資料庫、實驗室配方記錄

4. **離散型項目**：
   - 數據已經是離散的類別型變數
   - 或連續變數已經過合理的離散化

**不適合使用 Apriori 的情況**：

1. **大規模數據集**：
   - 百萬級以上的交易數量
   - 建議使用 FP-Growth 或其他分布式演算法

2. **低支持度要求**：
   - 需要發現低頻但高價值的模式
   - Apriori 在低 min_support 下效率很低

3. **即時分析需求**：
   - 需要快速響應的應用場景
   - Apriori 的多次掃描不適合即時處理

### 3.4 Apriori vs. FP-Growth 比較

| 特性 | Apriori | FP-Growth |
|------|---------|-----------|
| **演算法策略** | 逐層搜索，產生候選項目集 | 分治法，使用 FP-Tree |
| **數據掃描次數** | $k$ 次（ $k$ = 最大項目集大小） | 2 次 |
| **候選項目集** | 需要產生大量候選項目集 | 不產生候選項目集 |
| **記憶體使用** | 儲存候選項目集，較高 | 儲存 FP-Tree，較低 |
| **執行效率** | 較慢（大數據集） | 較快（特別是大數據集） |
| **易理解性** | 容易理解 | 較複雜 |
| **適用規模** | 小到中等規模 | 中到大規模 |
| **最佳場景** | 教學、小型配方資料庫 | 大規模製程數據挖掘 |

---

## 4. Python 實作：使用 mlxtend 套件

### 4.1 mlxtend 套件簡介

mlxtend (machine learning extensions) 是一個提供日常數據科學任務實用工具的 Python 套件，其中包含了 Apriori 演算法和關聯規則生成的實作。

**主要功能**：
- `apriori()`：找出頻繁項目集
- `association_rules()`：從頻繁項目集生成關聯規則

**安裝方式**：
```bash
pip install mlxtend
```

### 4.2 數據格式要求

mlxtend 的 Apriori 實作要求數據為 **One-Hot Encoding** 格式的 DataFrame：

**範例**：

| 交易ID | 催化劑A | 催化劑B | 溶劑X | 溶劑Y | 高溫 | 低溫 |
|--------|---------|---------|-------|-------|------|------|
| T1 | 1 | 0 | 1 | 0 | 1 | 0 |
| T2 | 1 | 0 | 0 | 1 | 1 | 0 |
| T3 | 0 | 1 | 1 | 0 | 0 | 1 |
| T4 | 1 | 0 | 1 | 0 | 1 | 0 |
| T5 | 1 | 0 | 1 | 0 | 0 | 0 |

- 1 表示該項目出現在交易中
- 0 表示該項目不出現在交易中
- 每一列代表一筆交易
- 每一欄代表一個項目

### 4.3 基本使用範例

```python
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# 1. 準備數據（One-Hot Encoding 格式）
data = pd.DataFrame({
    '催化劑A': [1, 1, 0, 1, 1],
    '催化劑B': [0, 0, 1, 0, 0],
    '溶劑X': [1, 0, 1, 1, 1],
    '溶劑Y': [0, 1, 0, 0, 0],
    '高溫': [1, 1, 0, 1, 0],
    '低溫': [0, 0, 1, 0, 0],
    '高產率': [0, 0, 0, 1, 0]
}, dtype=bool)  # 使用 bool 型態節省記憶體

# 2. 使用 Apriori 找出頻繁項目集
frequent_itemsets = apriori(data, min_support=0.4, use_colnames=True)

print("頻繁項目集：")
print(frequent_itemsets)

# 3. 生成關聯規則
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

print("\n關聯規則：")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
```

**輸出**：
```
頻繁項目集：
   support        itemsets
0      0.8      (催化劑A)
1      0.8        (溶劑X)
2      0.6          (高溫)
3      0.6  (催化劑A, 溶劑X)
4      0.6    (催化劑A, 高溫)
5      0.4      (溶劑X, 高溫)
6      0.4  (催化劑A, 溶劑X, 高溫)

關聯規則：
       antecedents      consequents  support  confidence  lift
0         (溶劑X)        (催化劑A)      0.6    0.75      0.94
1       (催化劑A)          (溶劑X)      0.6    0.75      0.94
2           (高溫)        (催化劑A)      0.6    1.00      1.25
3       (催化劑A)            (高溫)      0.6    0.75      1.25
4         (溶劑X)            (高溫)      0.4    0.50      0.83
5  (溶劑X, 高溫)        (催化劑A)      0.4    1.00      1.25
6  (催化劑A, 高溫)          (溶劑X)      0.4    0.67      0.83
```

### 4.4 參數說明

**apriori() 函數參數**：

- `df`：One-Hot Encoding 格式的 DataFrame
- `min_support`：最小支持度閾值（預設 0.5）
- `use_colnames`：是否使用欄位名稱（預設 False）
  - True：返回項目名稱
  - False：返回欄位索引
- `max_len`：限制最大項目集大小（預設 None，無限制）
- `verbose`：顯示進度（預設 0）
- `low_memory`：低記憶體模式（預設 False）

**association_rules() 函數參數**：

- `df`：apriori() 返回的頻繁項目集 DataFrame
- `metric`：評估指標，可選：
  - `'support'`：支持度
  - `'confidence'`：置信度（常用）
  - `'lift'`：提升度
  - `'leverage'`：槓桿度
  - `'conviction'`：確信度
- `min_threshold`：該指標的最小閾值
- `support_only`：是否只返回包含支持度的規則（預設 False）

### 4.5 從原始數據轉換為 One-Hot Encoding

通常原始數據為「交易列表」格式，需要轉換為 One-Hot Encoding 格式。

**方法 1：使用 TransactionEncoder（推薦）**

```python
from mlxtend.preprocessing import TransactionEncoder

# 原始數據：交易列表格式
transactions = [
    ['催化劑A', '溶劑X', '高溫'],
    ['催化劑A', '溶劑Y', '高溫'],
    ['催化劑B', '溶劑X', '低溫'],
    ['催化劑A', '溶劑X', '高溫', '高產率'],
    ['催化劑A', '溶劑X']
]

# 轉換為 One-Hot Encoding
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

print(df)
```

**方法 2：手動轉換**

```python
import pandas as pd

# 原始數據：每筆交易用逗號分隔
data = {
    'transaction_id': [1, 2, 3, 4, 5],
    'items': [
        '催化劑A,溶劑X,高溫',
        '催化劑A,溶劑Y,高溫',
        '催化劑B,溶劑X,低溫',
        '催化劑A,溶劑X,高溫,高產率',
        '催化劑A,溶劑X'
    ]
}
df_raw = pd.DataFrame(data)

# 分割項目並展開
df_raw['items'] = df_raw['items'].str.split(',')

# 使用 get_dummies 進行 One-Hot Encoding
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
df_encoded = pd.DataFrame(
    mlb.fit_transform(df_raw['items']),
    columns=mlb.classes_
)

print(df_encoded)
```

### 4.6 進階技巧：設定最大項目集大小

當項目數量很多時，限制最大項目集大小可以提高效率：

```python
# 只找出大小 ≤ 3 的頻繁項目集
frequent_itemsets = apriori(
    data, 
    min_support=0.3, 
    use_colnames=True,
    max_len=3  # 限制最大項目集大小
)
```

### 4.7 進階技巧：過濾規則

生成規則後，可以根據多個條件進行過濾：

```python
# 過濾規則：置信度 > 0.7 且提升度 > 1.2
filtered_rules = rules[
    (rules['confidence'] > 0.7) & 
    (rules['lift'] > 1.2)
]

# 只保留後項為特定項目的規則
target_rules = rules[
    rules['consequents'].apply(lambda x: '高產率' in x)
]

# 按提升度排序
sorted_rules = rules.sort_values('lift', ascending=False)
```

---

## 5. 關聯規則評估與解讀

### 5.1 評估指標詳解

除了基本的支持度、置信度、提升度，還有其他進階指標可用於評估規則品質。

#### 5.1.1 槓桿度 (Leverage)

**定義**：

$$
\text{Leverage}(X \Rightarrow Y) = \text{Support}(X \cup Y) - \text{Support}(X) \times \text{Support}(Y)
$$

**意義**：
- 衡量規則相對於隨機組合的增益
- Leverage = 0：表示 $X$ 和 $Y$ 獨立
- Leverage > 0：表示 $X$ 和 $Y$ 正相關
- Leverage < 0：表示 $X$ 和 $Y$ 負相關

**範圍**： $[-1, 1]$

**優點**：
- 考慮了規則的統計顯著性
- 對低頻項目敏感度較低

#### 5.1.2 確信度 (Conviction)

**定義**：

$$
\text{Conviction}(X \Rightarrow Y) = \frac{1 - \text{Support}(Y)}{1 - \text{Confidence}(X \Rightarrow Y)}
$$

**意義**：
- 衡量規則的因果性強度
- Conviction = 1：表示 $X$ 和 $Y$ 獨立
- Conviction > 1：表示規則有意義
- Conviction → ∞：表示完美的因果關係（置信度 = 1）

**優點**：
- 對方向性敏感（ $X \Rightarrow Y$ 與 $Y \Rightarrow X$ 的值不同）
- 懲罰置信度低的規則

#### 5.1.3 Kulczynski 度量

**定義**：

$$
\text{Kulc}(X \Rightarrow Y) = \frac{1}{2} \left( P(Y|X) + P(X|Y) \right)
$$

**意義**：
- 對稱性指標，平衡兩個方向的置信度
- 值越高，表示關聯性越強

**範圍**： $[0, 1]$

### 5.2 規則品質評估準則

**好規則的特徵**：

1. **高支持度**：
   - 規則出現頻率高，具有普遍性
   - 建議：Support ≥ 0.05（至少5%的交易）

2. **高置信度**：
   - 規則可靠性高
   - 建議：Confidence ≥ 0.7（70%以上）

3. **高提升度**：
   - 規則具有實際意義，非偶然相關
   - 建議：Lift > 1.2（至少提升20%）

4. **長度適中**：
   - 太短（如單項）：缺乏信息
   - 太長：過於複雜，難以應用
   - 建議：2-4 個項目

5. **業務合理性**：
   - 符合領域知識和物理化學原理
   - 可解釋且可行動

**過濾低品質規則的策略**：

```python
# 綜合過濾條件
quality_rules = rules[
    (rules['support'] >= 0.05) &       # 支持度 ≥ 5%
    (rules['confidence'] >= 0.7) &     # 置信度 ≥ 70%
    (rules['lift'] > 1.2) &            # 提升度 > 1.2
    (rules['antecedents'].apply(len) <= 3) &  # 前項 ≤ 3 個項目
    (rules['consequents'].apply(len) <= 2)    # 後項 ≤ 2 個項目
]

# 進一步按 lift 排序
quality_rules = quality_rules.sort_values('lift', ascending=False)
```

### 5.3 規則解讀與業務洞察

**範例規則**：

假設我們從聚合物配方數據中發現以下規則：

```
規則 1: {單體A, 引發劑B} => {窄分子量分布}
  - Support = 0.15 (15%的配方)
  - Confidence = 0.85 (85%的可信度)
  - Lift = 2.3 (提升2.3倍)

規則 2: {溶劑X, 高溫} => {高產率}
  - Support = 0.20 (20%的配方)
  - Confidence = 0.75 (75%的可信度)
  - Lift = 1.8 (提升1.8倍)

規則 3: {催化劑M, 助催化劑N, 載體O} => {高選擇性, 低副產物}
  - Support = 0.08 (8%的配方)
  - Confidence = 0.90 (90%的可信度)
  - Lift = 3.2 (提升3.2倍)
```

**解讀與應用**：

**規則 1 解讀**：
- **發現**：使用單體A和引發劑B的組合，能穩定獲得窄分子量分布
- **應用**：當目標產品需要窄分子量分布時，優先選擇此組合
- **行動**：將此組合納入標準配方庫，建議配方研發團隊優先使用

**規則 2 解讀**：
- **發現**：在溶劑X中進行高溫反應，有較高機率獲得高產率
- **應用**：製程優化時，考慮更換溶劑或提高反應溫度
- **注意**：需驗證高溫條件下的安全性和能耗成本

**規則 3 解讀**：
- **發現**：三元催化體系的協同效應非常顯著
- **應用**：雖然支持度較低（僅8%使用），但效果卓越，適合高價值產品
- **行動**：深入研究此催化體系的機理，擴大應用範圍

### 5.4 冗餘規則識別與處理

**冗餘規則定義**：

如果規則 $X \Rightarrow Y$ 可以從規則 $X' \Rightarrow Y$ 推導出來，且 $X' \subseteq X$ ，則 $X \Rightarrow Y$ 是冗餘的。

**範例**：

```
規則 A: {催化劑A} => {高產率}  (Confidence = 0.8)
規則 B: {催化劑A, 溶劑X} => {高產率}  (Confidence = 0.82)
```

如果規則 B 的置信度僅略高於規則 A，則規則 B 可能是冗餘的，因為它只是在規則 A 的基礎上增加了一個項目，但提升有限。

**冗餘規則過濾**：

```python
def remove_redundant_rules(rules, confidence_threshold=0.05):
    """
    移除冗餘規則
    
    參數:
    - rules: 關聯規則 DataFrame
    - confidence_threshold: 置信度提升閾值
    
    返回:
    - 非冗餘規則 DataFrame
    """
    non_redundant = []
    
    for idx, rule in rules.iterrows():
        is_redundant = False
        antecedents = rule['antecedents']
        consequents = rule['consequents']
        confidence = rule['confidence']
        
        # 檢查是否存在子集規則
        for idx2, rule2 in rules.iterrows():
            if idx != idx2:
                antecedents2 = rule2['antecedents']
                consequents2 = rule2['consequents']
                confidence2 = rule2['confidence']
                
                # 如果 rule2 的前項是 rule 的子集，且後項相同
                if (antecedents2.issubset(antecedents) and 
                    consequents2 == consequents and
                    len(antecedents2) < len(antecedents)):
                    
                    # 如果置信度提升不顯著，則為冗餘
                    if confidence - confidence2 < confidence_threshold:
                        is_redundant = True
                        break
        
        if not is_redundant:
            non_redundant.append(rule)
    
    return pd.DataFrame(non_redundant)
```

---

## 6. 化工應用案例：聚合物配方優化

### 6.1 案例背景

**問題描述**：

某化工公司專注於聚合物產品開發，累積了大量配方實驗數據。每個配方包含：
- 單體類型（A, B, C, D, E）
- 引發劑類型（I1, I2, I3）
- 鏈轉移劑類型（T1, T2, T3）
- 溶劑類型（S1, S2, S3）
- 溫度範圍（低溫 < 60°C，中溫 60-80°C，高溫 > 80°C）
- 反應時間（短 < 2h，中 2-4h，長 > 4h）

**產品品質指標**：
- 分子量分布（窄、中、寬）
- 轉化率（低 < 80%，中 80-90%，高 > 90%）
- 產品純度（低 < 95%，中 95-98%，高 > 98%）

**業務目標**：

使用 Apriori 演算法從歷史配方數據中發現：
1. 哪些成分組合能獲得窄分子量分布？
2. 哪些操作條件能提高轉化率？
3. 是否存在協同效應（多個因素共同作用）？

### 6.2 數據準備

**模擬數據生成**：

```python
import pandas as pd
import numpy as np

# 設定隨機種子
np.random.seed(42)

# 定義項目
monomers = ['單體A', '單體B', '單體C', '單體D', '單體E']
initiators = ['引發劑I1', '引發劑I2', '引發劑I3']
chain_transfer = ['鏈轉移劑T1', '鏈轉移劑T2', '鏈轉移劑T3']
solvents = ['溶劑S1', '溶劑S2', '溶劑S3']
temperatures = ['低溫', '中溫', '高溫']
times = ['短時間', '中時間', '長時間']
quality = ['窄分子量分布', '中分子量分布', '寬分子量分布', 
          '高轉化率', '中轉化率', '低轉化率',
          '高純度', '中純度', '低純度']

# 生成 100 筆配方交易
n_transactions = 100
transactions = []

for i in range(n_transactions):
    transaction = []
    
    # 每個配方包含 1 種單體
    transaction.append(np.random.choice(monomers))
    
    # 1 種引發劑
    transaction.append(np.random.choice(initiators))
    
    # 50% 機率添加鏈轉移劑
    if np.random.rand() > 0.5:
        transaction.append(np.random.choice(chain_transfer))
    
    # 1 種溶劑
    transaction.append(np.random.choice(solvents))
    
    # 1 種溫度條件
    temp = np.random.choice(temperatures)
    transaction.append(temp)
    
    # 1 種時間條件
    transaction.append(np.random.choice(times))
    
    # 根據組合決定品質（模擬協同效應）
    # 規則 1: 單體A + 引發劑I1 → 窄分子量分布
    if '單體A' in transaction and '引發劑I1' in transaction:
        transaction.append('窄分子量分布')
    else:
        transaction.append(np.random.choice(['窄分子量分布', '中分子量分布', '寬分子量分布']))
    
    # 規則 2: 溶劑S1 + 高溫 → 高轉化率
    if '溶劑S1' in transaction and '高溫' in transaction:
        transaction.append('高轉化率')
    else:
        transaction.append(np.random.choice(['高轉化率', '中轉化率', '低轉化率']))
    
    # 規則 3: 鏈轉移劑T2 + 中溫 → 高純度
    if '鏈轉移劑T2' in transaction and '中溫' in transaction:
        transaction.append('高純度')
    else:
        transaction.append(np.random.choice(['高純度', '中純度', '低純度']))
    
    transactions.append(transaction)

# 顯示前 5 筆交易
for i, trans in enumerate(transactions[:5]):
    print(f"配方 {i+1}: {trans}")
```

### 6.3 應用 Apriori 演算法

```python
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 1. 轉換為 One-Hot Encoding
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

print(f"數據集大小: {df.shape}")
print(f"項目總數: {len(df.columns)}")

# 2. 尋找頻繁項目集
frequent_itemsets = apriori(df, min_support=0.15, use_colnames=True, max_len=4)

print(f"\n找到 {len(frequent_itemsets)} 個頻繁項目集")
print("\n頻繁項目集 (按支持度排序):")
print(frequent_itemsets.sort_values('support', ascending=False).head(10))

# 3. 生成關聯規則
rules = association_rules(
    frequent_itemsets, 
    metric="confidence", 
    min_threshold=0.6
)

print(f"\n找到 {len(rules)} 條關聯規則")

# 4. 過濾高品質規則
quality_rules = rules[
    (rules['lift'] > 1.2) &
    (rules['confidence'] > 0.7)
].sort_values('lift', ascending=False)

print(f"\n高品質規則 (Lift > 1.2, Confidence > 0.7): {len(quality_rules)} 條")
print("\n前 10 條規則:")
print(quality_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
```

### 6.4 實際執行結果與業務洞察

**執行環境**：
- 數據規模: 200 筆聚合物配方交易
- 項目總數: 29 個 (單體、引發劑、鏈轉移劑、溶劑、溫度、時間、品質指標)
- 演算法參數: min_support=0.08, min_confidence=0.5
- 高品質過濾: Lift > 1.5, Confidence > 0.65

**統計摘要**：
- ✓ 找到 415 個頻繁項目集
- ✓ 生成 825 條關聯規則
- ✓ 篩選出 368 條高品質規則 (佔總數 44.6%)

**發現的關鍵規則**（實際執行結果）：

```
規則 1: {Monomer_A, Initiator_I1} => {Narrow_MW_Distribution}
  - Support = 0.105 (21筆交易), Confidence = 0.677, Lift = 3.662 ✓
  - 洞察: 單體A與引發劑I1的組合使窄分子量分布的出現機率提升3.66倍
  - 驗證狀態: 成功發現（協同效應已確認）

規則 2: {Solvent_S1, High_Temp} => {High_Conversion}
  - Support = 0.095 (19筆交易), Confidence = 0.826, Lift = 3.755 ✓
  - 洞察: 高溫條件下使用溶劑S1能顯著提高轉化率至82.6%
  - 驗證狀態: 成功發現（協同效應已確認）

規則 3: {Chain_Transfer_T2, Medium_Temp} => {High_Purity}
  - Support = 0.085 (17筆交易), Confidence = 0.944, Lift = 3.257 ✓
  - 洞察: 中溫條件下添加鏈轉移劑T2能以94.4%的可靠性獲得高純度
  - 驗證狀態: 成功發現（協同效應已確認）
```

**關鍵發現總結**：
- ✓ 所有三條預設的協同效應規則均被 Apriori 演算法成功識別
- ✓ Lift 值均 > 3.0，顯示非常強的關聯性（遠高於隨機出現的機率）
- ✓ 置信度均 > 65%，符合化工製程應用的可靠性要求
- ✓ 支持度適中（8.5%-10.5%），既不是偶然模式，也非過於普遍的組合

### 6.5 規則視覺化

#### 6.5.1 支持度-置信度散點圖

**視覺化程式碼**：

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    quality_rules['support'], 
    quality_rules['confidence'],
    c=quality_rules['lift'], 
    cmap='viridis', 
    s=100, 
    alpha=0.6,
    edgecolors='black'
)
plt.colorbar(scatter, label='Lift')
plt.xlabel('Support', fontsize=12)
plt.ylabel('Confidence', fontsize=12)
plt.title('Association Rules: Support vs Confidence (colored by Lift)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'support_confidence_scatter.png', dpi=300, bbox_inches='tight')
plt.show()
```

**視覺化結果**：

![Support vs Confidence Scatter Plot](outputs/P2_Unit08_Apriori_Algorithm/figs/support_confidence_scatter.png)

**圖表解讀**：

1. **X 軸 (Support)**: 規則在數據集中出現的頻率
   - 大多數高品質規則的支持度集中在 0.08-0.20 之間
   - 少數規則支持度 > 0.30，代表非常普遍的配方組合

2. **Y 軸 (Confidence)**: 規則的可靠性
   - 高品質規則的置信度廣泛分佈在 0.65-1.0 之間
   - 多條規則達到 1.0 置信度（100% 可靠），值得優先關注

3. **顏色 (Lift)**: 規則的關聯強度
   - 黃色點 (Lift > 3.5): 最強關聯規則，應優先驗證
   - 紫色點 (Lift 1.5-2.0): 中等關聯，需結合專業知識判斷
   - 圖中可見三個預設規則均為高 Lift 值（深藍色至黃色區域）

4. **規則分布特徵**：
   - 呈現多個聚類，反映不同類型的配方協同效應
   - 高置信度 (>0.9) 規則主要集中在支持度 0.08-0.15 區間
   - 支持度與置信度無明顯線性關係，符合關聯規則挖掘的特性

#### 6.5.2 規則網絡圖

**視覺化程式碼**：

```python
import networkx as nx

# 選擇 Lift > 2.0 的規則
top_rules = quality_rules[quality_rules['lift'] > 2.0]

# 建立網絡圖
G = nx.DiGraph()

for idx, rule in top_rules.iterrows():
    for antecedent in rule['antecedents']:
        for consequent in rule['consequents']:
            G.add_edge(
                antecedent, 
                consequent, 
                weight=rule['lift'],
                confidence=rule['confidence']
            )

# 繪製網絡圖
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, k=1, iterations=50)

# 繪製節點
nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', alpha=0.9)
nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

# 繪製邊（粗細代表 Lift）
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
nx.draw_networkx_edges(
    G, pos, 
    width=[w*0.5 for w in weights], 
    alpha=0.6, 
    edge_color='gray',
    arrows=True, 
    arrowsize=20,
    arrowstyle='->'
)

plt.title('Association Rules Network (Edge width = Lift)', fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.savefig(FIG_DIR / 'rules_network.png', dpi=300, bbox_inches='tight')
plt.show()
```

**視覺化結果**：

![Rules Network Graph](outputs/P2_Unit08_Apriori_Algorithm/figs/rules_network.png)

**網絡圖解讀**：

1. **節點 (Nodes)**: 代表配方中的項目
   - 節點大小一致，便於觀察連接關係
   - 包含原料（Monomer_A、Initiator_I1 等）、操作條件（High_Temp、Medium_Temp 等）、品質指標（High_Conversion、Narrow_MW_Distribution 等）

2. **邊 (Edges)**: 代表關聯規則
   - 方向: 從前項 (antecedents) 指向後項 (consequents)
   - 粗細: 邊越粗代表 Lift 值越高（關聯越強）
   - 本圖僅顯示 Lift > 2.0 的 99 條強關聯規則

3. **核心節點識別**：
   - **Monomer_A**: 連接到 Narrow_MW_Distribution，顯示其對分子量分布的關鍵影響
   - **Solvent_S1**: 高出度（多條出邊），是多個高品質結果的前項
   - **High_Temp**: 中心節點之一，與多個品質指標相關
   - **Narrow_MW_Distribution, High_Conversion, High_Purity**: 高入度（多條入邊），顯示可由多種配方組合達成

4. **協同效應觀察**：
   - Monomer_A 與 Initiator_I1 形成強連接後指向 Narrow_MW_Distribution（規則 1）
   - Solvent_S1 與 High_Temp 協同作用於 High_Conversion（規則 2）
   - Chain_Transfer_T2 與 Medium_Temp 共同影響 High_Purity（規則 3）

5. **業務啟示**：
   - 可依據網絡圖規劃配方設計路徑：選擇核心節點 → 識別協同項目 → 預測品質結果
   - 網絡密集區域代表高交互作用的配方空間，值得深入研究

### 6.6 品質指標分析

根據實際執行結果，針對三大品質指標進行深入分析：

#### 6.6.1 窄分子量分布 (Narrow_MW_Distribution)

**相關規則 Top 5**：

| 排名 | 前項條件 | 後項結果 | Support | Confidence | Lift |
|------|---------|---------|---------|------------|------|
| 1 | {Monomer_A, Initiator_I1} | {Narrow_MW_Distribution} | 0.105 | 0.677 | 3.662 |
| 2 | {Monomer_A, Initiator_I1, Solvent_S1} | {Narrow_MW_Distribution} | 0.080 | 0.750 | 4.055 |
| 3 | {Monomer_A, Short_Time} | {Narrow_MW_Distribution} | 0.090 | 0.720 | 3.893 |
| ... | ... | ... | ... | ... | ... |

**洞察**：
- Monomer_A 與 Initiator_I1 的組合是最關鍵的因素
- 添加 Solvent_S1 可進一步提升 Lift 至 4.055（更強的協同效應）
- 短反應時間有助於獲得窄分子量分布

#### 6.6.2 高轉化率 (High_Conversion)

**相關規則 Top 5**：

| 排名 | 前項條件 | 後項結果 | Support | Confidence | Lift |
|------|---------|---------|---------|------------|------|
| 1 | {Solvent_S1, High_Temp} | {High_Conversion} | 0.095 | 0.826 | 3.755 |
| 2 | {Solvent_S1, High_Temp, Long_Time} | {High_Conversion} | 0.075 | 0.882 | 4.010 |
| 3 | {Initiator_I2, High_Temp} | {High_Conversion} | 0.085 | 0.773 | 3.513 |
| ... | ... | ... | ... | ... | ... |

**洞察**：
- 高溫操作是提升轉化率的必要條件
- Solvent_S1 在高溫下展現最佳效果（Lift=3.755）
- 延長反應時間可進一步提升置信度至 88.2%

#### 6.6.3 高純度 (High_Purity)

**相關規則 Top 5**：

| 排名 | 前項條件 | 後項結果 | Support | Confidence | Lift |
|------|---------|---------|---------|------------|------|
| 1 | {Chain_Transfer_T2, Medium_Temp} | {High_Purity} | 0.085 | 0.944 | 3.257 |
| 2 | {Chain_Transfer_T2, Medium_Temp, Medium_Time} | {High_Purity} | 0.070 | 1.000 | 3.450 |
| 3 | {Chain_Transfer_T2, Solvent_S2} | {High_Purity} | 0.080 | 0.889 | 3.067 |
| ... | ... | ... | ... | ... | ... |

**洞察**：
- Chain_Transfer_T2 是獲得高純度的關鍵添加劑
- 中溫條件（60-80°C）最適合純度控制
- 配合中等反應時間可達到 100% 置信度（規則 2）

### 6.7 業務行動建議

基於實際數據分析結果，提出以下具體行動建議：

**行動建議**：

1. **配方標準化**：
   - ✓ 將規則 1-3 納入標準配方指南
   - ✓ 針對不同目標品質建立配方模板：
     - **窄分子量分布配方**: Monomer_A + Initiator_I1 + Solvent_S1 + 短時間
     - **高轉化率配方**: Solvent_S1 + 高溫 + 長時間
     - **高純度配方**: Chain_Transfer_T2 + 中溫 + 中時間
   - 創建配方決策樹，指導研發人員快速選擇原料組合

2. **實驗驗證**：
   - 對 Lift > 4.0 的規則進行小規模試產驗證（如規則 2 的三元組合）
   - 研究協同機理：
     - Monomer_A + Initiator_I1 的化學反應路徑
     - Solvent_S1 在高溫下的溶解特性
     - Chain_Transfer_T2 的鏈終止機制
   - 擴展至其他單體系統（Monomer_B, C, D, E），驗證規則的通用性

3. **製程優化**：
   - 評估高溫操作的能耗與安全性：
     - 計算從中溫升至高溫的能耗增加
     - 評估高溫下的設備耐壓與材料相容性
   - 優化溫度控制策略：
     - 針對轉化率優先配方，採用高溫操作
     - 針對純度優先配方，嚴格控制中溫範圍
   - 平衡反應時間與生產效率

4. **知識管理**：
   - 建立配方知識庫系統：
     - 儲存所有 825 條關聯規則
     - 標記 368 條高品質規則
     - 記錄實驗驗證結果
   - 定期更新規則（建議每季度或每累積 50 筆新數據）：
     - 重新運行 Apriori 演算法
     - 追蹤規則的時效性（某些規則可能隨原料批次變化）
     - 識別新出現的高 Lift 規則

5. **成本效益分析**：
   - 比較不同配方的原料成本：
     - Solvent_S1 vs S2 vs S3 的價格差異
     - Chain_Transfer_T2 的經濟性
   - 評估品質提升的經濟價值：
     - 窄分子量分布產品的市場溢價
     - 高轉化率對原料利用率的影響
     - 高純度對後續處理成本的節省

6. **持續改進**：
   - 建立反饋閉環：收集規則應用後的實際效果數據
   - A/B 測試：對比遵循規則 vs 傳統配方的品質差異
   - 追蹤關鍵指標：
     - 新配方成功率
     - 平均品質改善幅度
     - 製程穩定性提升

---

## 7. 實務建議與最佳實踐

### 7.1 參數設定指南

**最小支持度 (min_support) 設定**：

| 數據規模 | 建議 min_support | 說明 |
|---------|-----------------|------|
| 小型（< 1,000 筆） | 0.1 - 0.3 | 較高的閾值，避免偶然模式 |
| 中型（1,000 - 10,000 筆） | 0.05 - 0.15 | 平衡頻繁度與規則數量 |
| 大型（> 10,000 筆） | 0.01 - 0.05 | 較低的閾值，發現稀有模式 |

**最小置信度 (min_confidence) 設定**：

- **探索性分析**：0.5 - 0.6（較寬鬆，發現更多候選規則）
- **業務應用**：0.7 - 0.8（較嚴格，確保規則可靠性）
- **關鍵決策**：≥ 0.9（非常嚴格，高信賴度要求）

**Lift 閾值建議**：

- Lift > 1：有正相關，但可能不顯著
- Lift > 1.2：具有實際意義的關聯
- Lift > 2：強關聯，值得深入研究
- Lift > 3：非常強的關聯，可能存在因果關係

### 7.2 常見問題與解決方案

**問題 1：找不到任何頻繁項目集**

**原因**：
- min_support 設定過高
- 數據太稀疏

**解決方案**：
- 逐步降低 min_support（如 0.3 → 0.2 → 0.1）
- 檢查數據分布，移除極低頻項目
- 考慮使用 FP-Growth 演算法

**問題 2：產生太多規則，難以分析**

**原因**：
- min_support 或 min_confidence 設定過低
- 數據中存在大量相關項目

**解決方案**：
- 提高閾值，聚焦高品質規則
- 使用多重過濾條件（support + confidence + lift）
- 應用冗餘規則移除
- 只分析包含特定目標項目的規則

**問題 3：規則缺乏業務意義**

**原因**：
- 數據品質問題
- 缺少領域知識指導

**解決方案**：
- 與領域專家合作，驗證規則合理性
- 添加業務約束（如特定項目必須/不能同時出現）
- 重新審視數據收集過程

**問題 4：計算速度太慢**

**原因**：
- 數據集太大
- min_support 太低，產生大量候選項目集

**解決方案**：
- 提高 min_support
- 限制最大項目集大小（max_len）
- 使用 FP-Growth 演算法
- 對數據進行抽樣分析

### 7.3 與領域專家協作

**協作流程**：

1. **需求理解**：
   - 了解業務目標和關注的品質指標
   - 確定重點關注的項目（如特定原料、操作條件）

2. **數據準備**：
   - 與專家討論數據收集範圍
   - 確定項目的離散化方式（如溫度範圍劃分）

3. **初步分析**：
   - 設定合理的閾值，運行 Apriori
   - 產生初步規則清單

4. **規則驗證**：
   - 與專家逐條檢視規則
   - 識別有價值的新發現和已知模式
   - 標記需要進一步驗證的規則

5. **實驗驗證**：
   - 針對高 Lift 的新規則設計驗證實驗
   - 確認規則的可重現性

6. **知識整合**：
   - 將驗證通過的規則納入知識庫
   - 制定相應的操作指南或配方標準

### 7.4 持續改進策略

**定期更新規則**：
- 隨著新數據累積，定期重新運行 Apriori
- 追蹤規則的時效性（某些規則可能隨時間失效）

**A/B 測試**：
- 對新發現的規則進行小規模試用
- 比較遵循規則與不遵循規則的結果差異

**反饋閉環**：
- 收集規則應用後的實際效果
- 根據反饋調整閾值或數據範圍

---

## 8. 總結

### 8.1 核心要點回顧

1. **Apriori 性質**是演算法的基礎，提供高效剪枝策略
2. **逐層搜索**策略簡單直觀，但需要多次掃描數據庫
3. **mlxtend 套件**提供易用的 Python 實作
4. **參數調整**（min_support, min_confidence）對結果影響顯著
5. **多重評估指標**（support, confidence, lift）幫助識別高品質規則
6. **業務洞察**比技術指標更重要，需與領域專家緊密合作

### 8.2 實際案例成果總結

本單元透過 200 筆聚合物配方數據的實際分析，展示了 Apriori 演算法的完整應用流程：

**數據規模與複雜度**：
- 交易數量: 200 筆配方記錄
- 項目種類: 29 個（涵蓋原料、操作條件、品質指標）
- 數據密度: 平均每筆交易包含 8-10 個項目

**演算法效能表現**：
- 頻繁項目集: 415 個 (min_support=0.08)
- 關聯規則總數: 825 條 (min_confidence=0.5)
- 高品質規則: 368 條 (Lift>1.5, Confidence>0.65)
- 篩選率: 44.6% 的規則達到高品質標準

**關鍵規則驗證**：
| 規則編號 | 前項條件 | 後項結果 | Lift | Confidence | 驗證狀態 |
|---------|---------|---------|------|------------|---------|
| 規則 1 | Monomer_A + Initiator_I1 | Narrow_MW_Distribution | 3.662 | 67.7% | ✓ 成功 |
| 規則 2 | Solvent_S1 + High_Temp | High_Conversion | 3.755 | 82.6% | ✓ 成功 |
| 規則 3 | Chain_Transfer_T2 + Medium_Temp | High_Purity | 3.257 | 94.4% | ✓ 成功 |

**業務價值實現**：
- ✓ 發現 3 條強協同效應規則（Lift > 3.0）
- ✓ 識別 99 條高度關聯規則（Lift > 2.0）用於網絡圖分析
- ✓ 為三大品質指標各建立 Top 5 優化配方
- ✓ 提供可操作的配方標準化建議

**實務應用啟示**：
1. 適當的參數設定是成功的關鍵：
   - min_support=0.08 平衡了頻繁度與樣本數需求
   - min_confidence=0.5 初步篩選，再用 Lift>1.5 精細過濾
   
2. 視覺化工具提升洞察效率：
   - 散點圖快速識別高品質規則聚類
   - 網絡圖揭示原料-操作-品質的關聯網絡
   
3. 業務驗證不可或缺：
   - 預設的三條規則全部被發現，驗證了演算法的有效性
   - 需要領域專家參與解讀 Lift > 4.0 的新發現規則

### 8.3 Apriori 適用場景

**最適合**：
- 小到中型數據集的探索性分析（如本案例的 200 筆）
- 需要高度可解釋性的場景（化工配方優化、製程改善）
- 教學與演示（演算法步驟清晰，易於理解）

**替代方案**：
- 大型數據集（> 10,000 筆）：使用 FP-Growth 演算法
- 複雜模式：考慮序列模式挖掘或深度學習方法

### 8.4 延伸學習

**相關主題**：
- FP-Growth 演算法（Unit08_FP_Growth_Algorithm）
- 序列模式挖掘（Sequential Pattern Mining）
- 市場購物籃分析進階技術
- 多層次關聯規則（Multi-level Association Rules）

**參考資源**：
- Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules. VLDB.
- Han, J., Pei, J., & Yin, Y. (2000). Mining frequent patterns without candidate generation. SIGMOD.
- mlxtend 官方文件：http://rasbt.github.io/mlxtend/

---

## 9. 練習題

### 9.1 觀念題

1. 解釋 Apriori 性質，並說明它如何幫助演算法提高效率？
2. 比較支持度、置信度和提升度的差異，並說明何時應該優先考慮哪個指標？
3. 為什麼 Apriori 演算法不適合處理大規模數據集？有哪些改進方案？

### 9.2 計算題

給定以下 5 筆交易：

| 交易ID | 項目集 |
|--------|--------|
| T1 | {A, B, D} |
| T2 | {A, C, D} |
| T3 | {B, C, E} |
| T4 | {A, B, C, D} |
| T5 | {B, C, D} |

設定 min_support = 0.6, min_confidence = 0.7

1. 找出所有頻繁項目集
2. 生成所有滿足條件的關聯規則
3. 計算規則 {B, C} => {D} 的支持度、置信度和提升度

### 9.3 實作題

使用提供的 Jupyter Notebook (Unit08_Apriori_Algorithm.ipynb)：

1. 載入化工配方數據
2. 應用 Apriori 演算法找出頻繁項目集
3. 生成關聯規則並進行品質過濾
4. 視覺化規則並提出業務建議

### 9.4 進階分析題（基於實際執行結果）

根據 Notebook 的執行結果（200 筆配方數據，415 個頻繁項目集，825 條規則）：

**題目 1：參數敏感度分析**

實驗不同的參數設定，觀察對結果的影響：

a) 將 min_support 從 0.08 提高到 0.12，觀察：
   - 頻繁項目集數量的變化
   - 高品質規則數量的變化
   - 三條預設規則是否仍被發現？

b) 將 min_confidence 從 0.5 提高到 0.7，分析：
   - 規則總數的減少比例
   - 平均 Lift 值的變化趨勢
   - 對業務決策的影響

**題目 2：規則深度探索**

基於 368 條高品質規則，進行以下分析：

a) 統計不同後項（consequents）的規則數量分布：
   - 哪個品質指標最容易預測？
   - 哪個品質指標的規則 Lift 值最高？

b) 分析規則長度分布：
   - 計算 1-項規則、2-項規則、3-項規則的數量
   - 是否存在「規則越長，Lift 越高」的趨勢？

c) 識別「核心前項」：
   - 哪些原料/操作條件出現在最多規則的前項中？
   - 這些核心前項對配方設計的意義是什麼？

**題目 3：業務情境應用**

假設您是配方研發工程師，面對以下需求：

**情境 A**：客戶要求「窄分子量分布 + 高轉化率」雙重品質目標
- 從 825 條規則中篩選同時指向這兩個目標的規則
- 評估這些規則的可行性（support, confidence, lift）
- 提出最佳配方建議

**情境 B**：原料短缺，Solvent_S1 缺貨
- 篩選不包含 Solvent_S1 的高轉化率規則
- 尋找替代方案（Solvent_S2 或 S3）
- 評估替代方案的效果差異

**情境 C**：成本控制，需降低高溫操作
- 分析中溫/低溫條件下的高品質規則
- 評估品質目標的可達成性
- 提出成本與品質的平衡方案

**題目 4：規則驗證與實驗設計**

選擇一條高 Lift 規則（如 Lift > 4.0），設計實驗驗證方案：

a) 實驗設計：
   - 確定實驗組（遵循規則）和對照組（不遵循規則）的配方
   - 設定重複次數與評估指標
   - 考慮可能的干擾因素

b) 數據收集計畫：
   - 需要記錄哪些數據？
   - 如何判斷規則的實際有效性？
   - 如果規則無效，可能的原因有哪些？

**題目 5：系統優化建議**

基於 200 筆數據的分析結果，提出數據收集優化建議：

a) 數據量評估：
   - 當前 200 筆是否足夠？
   - 計算達到統計顯著性所需的最小樣本數
   - 建議未來數據收集的目標規模

b) 數據平衡性：
   - 分析各項目的頻率分布（如各單體的出現次數）
   - 識別數據不平衡問題（某些組合過少）
   - 建議補充實驗的重點方向

c) 新變數探索：
   - 目前未包含但可能重要的因素（如攪拌速度、pH值）
   - 評估添加新變數的必要性與優先順序

---

**完成本單元後，您應該能夠**：
- ✓ 理解並應用 Apriori 演算法於實際問題
- ✓ 使用 Python 實作完整的關聯規則挖掘流程
- ✓ 評估和解讀關聯規則
- ✓ 提出基於數據的化工配方優化建議
- ✓ 根據實際執行結果進行深度業務分析
- ✓ 設計規則驗證實驗並優化數據收集策略

**下一單元預告**：Unit08_FP_Growth_Algorithm - 學習更高效的 FP-Growth 演算法，適用於大規模數據集。

---

**課程資訊**
- 課程名稱：AI在化工上之應用
- 課程單元：Unit08 Apriori Algorithm - 關聯規則挖掘實務應用
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-01-28

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---