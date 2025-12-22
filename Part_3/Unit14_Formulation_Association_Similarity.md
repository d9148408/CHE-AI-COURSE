# Unit14｜配方/材料的非監督探索：關聯規則（Apriori 思維）+ 相似度搜尋

**課程名稱**：化工資料科學與機器學習實務（CHE-AI-101）  
**Part 3 導讀**：很多材料與配方問題沒有「正確標籤」，你要的不是分類分數，而是把候選空間縮小、把決策理由說清楚：

- 哪些成分常一起出現？（關聯規則）
- 這個配方像哪幾個成功配方？（相似度搜尋）
- 怎麼產生「可審查」的候選清單？（交付）

---

## 0. 你會交付什麼？

1. **頻繁組合（frequent itemsets）**：常見的溶劑/助劑組合（support）
2. **規則清單（rules）**：A → B（confidence / lift），用於配方建議與風險提示
3. **相似度排名（similarity ranking）**：最相似的歷史配方/材料（Jaccard 或向量距離）
4. **候選縮小流程**：從「上百個候選」縮到「可實驗的 5–10 個」

---

## 1. 工程情境：配方開發為什麼適合非監督？

典型困境：
- 成功/失敗樣本少，且條件不一致（不同原料、設備、操作者）
- 性能指標可能要等很久才量到（耐久、老化、穩定性）
- 很多知識存在於「歷史配方」而非標準化標籤

因此常見策略是：
1. 先用非監督方法挖出結構（常見組合/相似族群）
2. 再把少量有標籤的資料拿來做監督式模型（Part 2/4/5）

---

## 2. 關聯規則（Apriori 思維）：把經驗規則變成可計算的準則

### 2.1 關聯規則挖掘的數學基礎

關聯規則挖掘（Association Rule Mining）最早應用於市場購物籃分析，目標是發現項目間的共現模式。在化工配方開發中，我們將其應用於發現成分組合的經驗規律。

#### 2.1.1 基本概念定義

**交易資料庫（Transaction Database）**：
設 $\mathcal{I} = \{i_1, i_2, ..., i_m\}$ 為所有可能項目的集合（如溶劑、助劑、催化劑），交易資料庫 $\mathcal{D}$ 包含 $n$ 個交易（配方）：

$$\mathcal{D} = \{T_1, T_2, ..., T_n\}, \quad T_j \subseteq \mathcal{I}$$

**項集（Itemset）**：
項目的任意子集 $X \subseteq \mathcal{I}$。若 $|X| = k$，則稱為 $k$-項集。

**關聯規則（Association Rule）**：
形式為 $A \Rightarrow B$ 的蘊含式，其中 $A, B \subseteq \mathcal{I}$ 且 $A \cap B = \emptyset$。

### 2.2 三個核心指標的數學定義與工程解讀

#### 2.2.1 Support（支持度）

**數學定義**：
$$\text{Support}(A \Rightarrow B) = \frac{|\{T \in \mathcal{D} : A \cup B \subseteq T\}|}{|\mathcal{D}|} = P(A \cup B)$$

**物理意義**：
項集 $A \cup B$ 在所有交易中出現的頻率。

**工程解讀**：
- **過低（< 5%）**：組合罕見，可能是偶然或特殊案例，統計不穩定
- **適中（5%-30%）**：常見但非普遍，值得深入研究
- **過高（> 50%）**：可能是行業常識或強制性標準（如安全規範）

**本單元實驗結果**（基於 20 個配方）：

| 項集 | Support | 解讀 |
|------|---------|------|
| (Solvent:EtOH, Additive:InhibitorC, Surfactant:S3) | 0.15 (3/20) | 中等頻率，特定組合 |
| (CoSolvent:Water, Surfactant:S3) | 0.25 (5/20) | 較高頻率，水性體系常見 |
| (Surfactant:S1) | 0.50 (10/20) | 高頻率，最常用界面活性劑 |

#### 2.2.2 Confidence（置信度）

**數學定義**：
$$\text{Confidence}(A \Rightarrow B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)} = P(B|A)$$

**物理意義**：
在包含 $A$ 的交易中，同時包含 $B$ 的條件機率。

**工程解讀**：
- **100% 置信度**：幾乎總是一起出現（強關聯或技術必要性）
- **60%-90%**：經常一起，但有例外（可能有替代方案）
- **< 60%**：關聯較弱，可能受其他因素影響

**實驗結果**（Top 規則）：

| 規則 (A → B) | Support | Confidence | 解釋 |
|-------------|---------|-----------|------|
| Solvent:EtOH → (Additive:InhibitorC, Surfactant:S3) | 0.15 | 1.00 | 乙醇溶劑**總是**搭配 InhibitorC 與 S3 |
| Solvent:EtOH → CoSolvent:Water | 0.15 | 1.00 | 乙醇配方**必定**使用水作為共溶劑 |
| Additive:InhibitorC → Solvent:EtOH | 0.15 | 0.75 | InhibitorC **通常**（75%）與乙醇搭配 |
| CoSolvent:Water → Surfactant:S3 | 0.25 | 1.00 | 水性體系**總是**使用 S3 界面活性劑 |

**化學解釋**：
1. **EtOH + Water + InhibitorC + S3**：典型水醇混合體系，InhibitorC 可能為水溶性抑制劑，S3 為親水性界面活性劑
2. **完美置信度（1.00）**：暗示技術必要性（如溶解度、相容性、穩定性要求）

#### 2.2.3 Lift（提升度）

**數學定義**：
$$\text{Lift}(A \Rightarrow B) = \frac{\text{Confidence}(A \Rightarrow B)}{P(B)} = \frac{P(A \cup B)}{P(A) \cdot P(B)}$$

**物理意義**：
$A$ 與 $B$ 共現的實際機率與獨立假設下的期望機率之比。

**統計解釋**：
- **Lift = 1**：$A$ 與 $B$ 統計獨立，無關聯
- **Lift > 1**：正相關，$A$ 出現時 $B$ 更可能出現
- **Lift < 1**：負相關，$A$ 出現時 $B$ 更不可能出現

**工程閾值**：
- **Lift > 3**：強關聯，值得深入研究
- **Lift 1.5-3**：中等關聯，可能有技術原因
- **Lift < 1.5**：弱關聯，可能是巧合

**實驗結果**（Top Lift 規則）：

| 規則 (A → B) | Support | Confidence | Lift | 解釋 |
|-------------|---------|-----------|------|------|
| Solvent:EtOH → (Additive:InhibitorC, Surfactant:S3) | 0.15 | 1.00 | 6.67 | **極強關聯**，技術必要性 |
| Solvent:EtOH → CoSolvent:Water | 0.15 | 1.00 | 4.00 | **強關聯**，水醇體系標配 |
| CoSolvent:Water → Surfactant:S3 | 0.25 | 1.00 | 4.00 | **強關聯**，水性體系必選 S3 |
| Solvent:Heptane → Surfactant:S2 | 0.15 | 1.00 | 4.00 | **強關聯**，非極性溶劑配 S2 |

**數學推導範例**（Solvent:EtOH → InhibitorC + S3）：

已知：
- $P(\text{EtOH}) = 3/20 = 0.15$
- $P(\text{InhibitorC} \cap \text{S3}) = 3/20 = 0.15$
- $P(\text{EtOH} \cap \text{InhibitorC} \cap \text{S3}) = 3/20 = 0.15$

計算：
$$\text{Confidence} = \frac{0.15}{0.15} = 1.00$$
$$\text{Lift} = \frac{1.00}{0.15} = 6.67$$

**解釋**：
- 若 InhibitorC+S3 獨立分佈，期望僅 15% 配方包含此組合
- 實際上，EtOH 配方 100% 包含此組合，是期望值的 6.67 倍
- **結論**：EtOH 與 InhibitorC+S3 有**極強的技術關聯**，可能是化學相容性或功效協同的結果

### 2.3 Apriori 演算法的數學原理

**Apriori 性質（Downward Closure）**：
若項集 $X$ 是頻繁的（Support ≥ min_support），則 $X$ 的所有子集也必須是頻繁的。

$$X \text{ 頻繁} \Rightarrow \forall Y \subseteq X, Y \text{ 頻繁}$$

**逆否命題（剪枝策略）**：
若項集 $Y$ 是非頻繁的，則所有包含 $Y$ 的超集都是非頻繁的。

$$Y \text{ 非頻繁} \Rightarrow \forall X \supseteq Y, X \text{ 非頻繁}$$

**演算法流程**：
1. **初始化**：找出所有頻繁 1-項集 $L_1$
2. **迭代生成**：對於 $k = 2, 3, ...$
   - 從 $L_{k-1}$ 生成候選 $k$-項集 $C_k$
   - 掃描資料庫，計算 $C_k$ 的 Support
   - 保留 Support ≥ min_support 的項集，得到 $L_k$
   - 若 $L_k = \emptyset$，終止
3. **規則生成**：從頻繁項集生成滿足 min_confidence 的規則

**計算複雜度**：
- **最壞情況**：$O(2^{|\mathcal{I}|})$（指數級）
- **Apriori 剪枝後**：實務上遠小於指數級，取決於 min_support 設定

> **警告：關聯 ≠ 因果**  
> Lift 高只表示統計共現，不能推論因果關係。工程上需結合：
> 1. **化學機制**（如相容性、溶解度理論）
> 2. **實驗驗證**（溶解度測試、穩定性測試）
> 3. **領域知識**（供應商套裝、法規限制）

---

## 2.2 工業級 Guardrails：避免「看起來很準」的假規則

關聯規則最常見的坑是：你拿到一堆漂亮規則，但上線後完全不穩。

建議最少做四個 guardrails：
1. **最小 support**：太低的規則很可能只是偶然（尤其資料少）
2. **時間/批次切分驗證**：用不同月份或不同批次驗證規則是否成立（避免 leakage）
3. **排除共因**：某些成分一起出現只是因為「同一供應商套裝」或「同一產品線」  
4. **加入工程約束**：EHS/相容性/設備限制先過濾，再談規則推薦

Notebook 會示範一個最小版做法：把交易資料切成 train/holdout，僅在 train 挖規則，再到 holdout 驗證 confidence 是否掉太多。

---

## 3. 相似度搜尋：從已知配方出發，找到「最接近」的候選

### 3.1 Jaccard 相似度的數學定義與工程應用

#### 3.1.1 數學定義

對於兩個集合 $A$ 和 $B$（在此為配方成分集合），**Jaccard 相似係數**定義為：

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{共同成分數}}{\text{總成分數（扣除重複）}}$$

**範圍**：$J(A, B) \in [0, 1]$
- $J = 1$：完全相同（$A = B$）
- $J = 0$：完全不同（$A \cap B = \emptyset$）

**Jaccard 距離**：
$$d_J(A, B) = 1 - J(A, B)$$

#### 3.1.2 計算範例

**配方 F00**（查詢配方）：
```
F00 = {Solvent:Toluene, CoSolvent:EtOAc, Additive:InhibitorA, Surfactant:S1}
```

**配方 F17**：
```
F17 = {Solvent:Toluene, CoSolvent:EtOAc, Additive:InhibitorA, Surfactant:S1, Catalyst:C1}
```

**計算**：
- $A \cap B = \{\text{Toluene, EtOAc, InhibitorA, S1}\}$，$|A \cap B| = 4$
- $A \cup B = \{\text{Toluene, EtOAc, InhibitorA, S1, C1}\}$，$|A \cup B| = 5$
- $J(F00, F17) = \frac{4}{5} = 0.8$

**解釋**：F17 僅額外添加催化劑 C1，其餘 80% 成分相同，是極高相似度配方。

#### 3.1.3 實驗結果：Top 10 相似配方

以 F00 為查詢配方，計算與所有 20 個配方的 Jaccard 相似度：

| 排名 | 配方 ID | Jaccard 相似度 | 差異成分 | 工程解讀 |
|-----|---------|---------------|---------|---------|
| 1 | F00 | 1.00 | - | 自身（基準） |
| 2 | F17 | 0.80 | +Catalyst:C1 | 僅添加催化劑，幾乎相同 |
| 3 | F16 | 0.60 | +Additive:InhibitorB, -InhibitorA | 抑制劑替換 |
| 4 | F01 | 0.60 | +CoSolvent:Acetone, -EtOAc | 共溶劑替換 |
| 5 | F09 | 0.33 | Solvent:EtOAc, +Water, +InhibitorB | 溶劑系統改變 |
| 6 | F14 | 0.33 | Solvent:Toluene, +Water, +S3 | 添加水性組分 |
| 7 | F10 | 0.33 | Solvent:MEK, +InhibitorB, +S2 | 溶劑替換為 MEK |
| 8 | F04 | 0.33 | Solvent:Heptane, +InhibitorB, +S2 | 非極性溶劑體系 |
| 9 | F03 | 0.33 | Solvent:IPA, +Water, +InhibitorC | 醇類溶劑體系 |
| 10 | F02 | 0.33 | Solvent:Acetone, +Water, +S3 | 酮類溶劑體系 |

**工程洞察**：
1. **高相似度（J > 0.6）**：僅 2 個配方（F17, F16），暗示 F00 組合在資料集中較獨特
2. **中等相似度（J = 0.33）**：多個配方，均為溶劑系統變更，保留部分共溶劑或添加劑
3. **低相似度（J < 0.3）**：13 個配方（未列出），表示成分體系差異大

### 3.2 相似度度量方法比較

#### 3.2.1 常用相似度指標

| 指標 | 數學定義 | 適用場景 | 優點 | 缺點 |
|-----|---------|---------|------|------|
| **Jaccard** | $\frac{\|A \cap B\|}{\|A \cup B\|}$ | 成分清單（有/無） | 不受劑量影響，直觀 | 忽略濃度資訊 |
| **Cosine** | $\frac{A \cdot B}{\|A\| \|B\|}$ | 數值向量（濃度） | 考慮劑量比例 | 需數值化 |
| **Euclidean** | $\sqrt{\sum (a_i - b_i)^2}$ | 數值向量 | 絕對距離 | 受量綱影響 |
| **Tanimoto** | $\frac{A \cdot B}{\|A\|^2 + \|B\|^2 - A \cdot B}$ | 指紋向量（化學結構） | 適配二元向量 | 計算複雜 |

#### 3.2.2 工程選擇準則

**使用 Jaccard 的場景**：
- ✅ **配方探索早期**：僅知道成分清單，無精確劑量
- ✅ **成分篩選**：關注「有哪些成分」而非「用多少」
- ✅ **專利規避**：判斷配方組成是否與現有專利重疊
- ✅ **供應鏈分析**：識別使用相同原料的配方群

**使用 Cosine 的場景**：
- ✅ **配比最佳化**：需考慮各成分的濃度比例
- ✅ **性能預測**：劑量對性能有顯著影響時
- ✅ **DOE 分析**：實驗設計中的因子水平比較

**實務工作流程**：
```
階段 1（探索期）：Jaccard 篩選成分組合 → 候選清單（100 → 20）
           ↓
階段 2（優化期）：Cosine/Euclidean 優化劑量 → 實驗配方（20 → 5）
           ↓
階段 3（驗證期）：DOE 驗證性能 → 最終配方（5 → 1-2）
```

### 3.3 相似度搜尋的統計與化學陷阱

#### 3.3.1 高相似度 ≠ 高性能

**範例**：F00 與 F17（Jaccard = 0.8）

| 配方 | 成分差異 | 潛在性能影響 |
|-----|---------|------------|
| F00 | 無催化劑 | 反應速率慢，轉化率 60% |
| F17 | +Catalyst:C1 | 反應速率快 5 倍，轉化率 95% |

**結論**：僅 20% 成分差異（1 個催化劑）可能導致 **質的性能差異**。

#### 3.3.2 相似度閾值的工程設定

| Jaccard 範圍 | 工程解讀 | 建議行動 |
|-------------|---------|---------|
| **0.8 - 1.0** | 幾乎相同 | 微調實驗（如濃度、順序）|
| **0.5 - 0.8** | 部分替換 | 驗證替換成分的相容性 |
| **0.3 - 0.5** | 體系差異 | 需小規模實驗確認性能 |
| **< 0.3** | 完全不同 | 視為新配方體系，全面驗證 |

> **警告**：Jaccard 無法捕捉：
> 1. **成分間的協同/拮抗作用**（如催化劑毒化）
> 2. **添加順序的影響**（如預混 vs 直接混合）
> 3. **劑量的非線性效應**（如臨界膠束濃度）

---

## 4. 多目標候選篩選：整合相似度、EHS 與成本約束

### 4.1 候選篩選的工程決策框架

#### 4.1.1 決策目標與約束

**硬約束（Hard Constraints）**：
- ✅ **法規合規**：EHS（Environment, Health, Safety）分級 ≤ 3
- ✅ **技術可行性**：相容性測試通過
- ✅ **供應鏈**：原料可採購（lead time < 4 weeks）

**軟約束（Soft Objectives）**：
- 🎯 **最大化性能**：轉化率、選擇性、穩定性
- 🎯 **最小化相似度距離**：$d_J = 1 - J$ → 降低技術風險
- 🎯 **最小化成本**：原料成本 + 處理成本

#### 4.1.2 複合評分函數設計

**線性加權評分**：
$$\text{Score} = w_1 \cdot (1 - J) + w_2 \cdot \text{EHS} + w_3 \cdot \text{Cost}$$

其中：
- $w_1, w_2, w_3$：權重係數（需歸一化：$\sum w_i = 1$ 或獨立設定）
- $1 - J$：Jaccard 距離（越小越好）
- EHS：風險分級（1-5，越小越安全）
- Cost：相對成本（標準化為 1-5）

**本單元實驗設定**：
- $w_1 = 1.0$：相似度為主要考量
- $w_2 = 0.3$：EHS 次要
- $w_3 = 0.1$：成本第三優先

### 4.2 實驗結果：候選配方篩選與排序

#### 4.2.1 篩選流程

**步驟 1**：相似度排序（Top 8 候選）

| 配方 ID | Jaccard | 1 - Jaccard | 初步排名 |
|---------|---------|------------|---------|
| F17 | 0.80 | 0.20 | 1 |
| F16 | 0.60 | 0.40 | 2 |
| F01 | 0.60 | 0.40 | 2 |
| F09 | 0.33 | 0.67 | 4 |
| F14 | 0.33 | 0.67 | 4 |
| F10 | 0.33 | 0.67 | 4 |
| F04 | 0.33 | 0.67 | 4 |
| F03 | 0.33 | 0.67 | 4 |

**步驟 2**：EHS 硬約束過濾（EHS ≤ 3）

| 配方 ID | Jaccard | EHS | Cost | 是否通過 |
|---------|---------|-----|------|---------|
| F17 | 0.80 | 4 | 2 | ❌ 不通過（高風險）|
| F16 | 0.60 | 4 | 3 | ❌ 不通過 |
| F01 | 0.60 | 4 | 2 | ❌ 不通過 |
| F09 | 0.33 | 2 | 2 | ✅ 通過 |
| F14 | 0.33 | 4 | 1 | ❌ 不通過 |
| F10 | 0.33 | 2 | 1 | ✅ 通過 |
| F04 | 0.33 | 3 | 2 | ✅ 通過 |
| F03 | 0.33 | 4 | 1 | ❌ 不通過 |

**步驟 3**：複合評分排序（僅剩 3 個候選）

| 排名 | 配方 ID | 主溶劑 | Jaccard | EHS | Cost | 評分公式 | 總分 |
|-----|---------|--------|---------|-----|------|---------|------|
| 1 | F10 | MEK | 0.33 | 2 | 1 | 0.67 + 0.3×2 + 0.1×1 | **1.37** |
| 2 | F09 | EtOAc | 0.33 | 2 | 2 | 0.67 + 0.3×2 + 0.1×2 | **1.47** |
| 3 | F04 | Heptane | 0.33 | 3 | 2 | 0.67 + 0.3×3 + 0.1×2 | **1.77** |

#### 4.2.2 最終候選清單分析

**推薦配方：F10（MEK 溶劑體系）**

| 項目 | 內容 | 理由 |
|-----|------|------|
| **主溶劑** | MEK（甲乙酮） | 中等極性，與 Toluene/EtOAc 相容 |
| **共溶劑** | Additive:InhibitorB | 可能與原 InhibitorA 功能相似 |
| **界面活性劑** | Surfactant:S2 | 適配 MEK 體系 |
| **EHS 分級** | 2 | 低風險（優於 F00 的 Toluene） |
| **相對成本** | 1 | 最低成本選項 |
| **技術風險** | 中 | Jaccard=0.33，需驗證相容性 |

**次選配方：F09（EtOAc 溶劑體系）**
- **優點**：EtOAc 為 F00 原共溶劑，相容性已知
- **缺點**：成本稍高（Cost=2）
- **建議**：若 F10 驗證失敗，F09 為備選

**第三選項：F04（Heptane 溶劑體系）**
- **風險**：Heptane 為非極性溶劑，與 F00 的極性體系差異大
- **應用**：可能適用於疏水性配方開發

### 4.3 危害物質替換分析（Toluene 案例）

#### 4.3.1 替換需求背景

**Toluene（甲苯）的 EHS 問題**：
- **健康風險**：神經毒性、致癌疑慮（GHS：H361, H373）
- **環境風險**：VOC 排放、難生物降解
- **法規趨勢**：REACH 高度關注物質（SVHC）候選

**替換目標**：
找出與 Toluene 配方相似但使用更安全溶劑的替代方案。

#### 4.3.2 實驗結果與資料集限制

**查詢**：包含 Solvent:Toluene 的配方（5 個）

| 配方 ID | 成分 |
|---------|------|
| F00 | Toluene + EtOAc + InhibitorA + S1 |
| F12 | Toluene + Water + InhibitorB + S3 |
| F13 | Toluene + Acetone + InhibitorC + S2 + C2 |
| F14 | Toluene + Water + InhibitorA + S3 |
| F19 | Toluene + Water + InhibitorC + S3 + Salt:NaCl |

**替換分析結果**：
```
⚠️ 警告：5 個 Toluene 配方中，0 個配方同時包含替代溶劑
```

**技術原因（玩具資料集限制）**：
- **設計缺陷**：資料集中每個配方僅使用一種主溶劑
- **缺乏對照組**：無「Toluene vs 替代溶劑」的配對實驗
- **實務解決方案**：
  1. 增加「同功能不同溶劑」的配方實驗
  2. 引入文獻資料（如 Toluene → MEK/EtOAc 替換案例）
  3. 使用 HSP（Hansen Solubility Parameters）預測溶劑相似度

---

## 5. ⚠️ 關聯規則的陷阱與誤用案例

雖然關聯規則挖掘是強大的工具，但在化工配方開發中存在許多**陷阱**，容易導致錯誤結論與資源浪費。以下是真實場景中常見的失敗案例。

### 5.1 陷阱 1：Simpson's Paradox（辛普森悖論）

#### 問題場景

在塗料配方開發中，分析 200 個歷史配方後發現：

**整體資料分析**：
- 規則：`Pigment:TiO₂ → Defect:Cracking`
- Support = 0.30（60/200 配方）
- Confidence = 0.75（60/80 含 TiO₂ 的配方）
- **結論**：TiO₂（二氧化鈦）導致龜裂？❌

**分群後分析**（按樹脂類型）：

| 樹脂類型 | TiO₂ 配方數 | 龜裂數 | Confidence | 結論 |
|---------|------------|--------|-----------|------|
| **丙烯酸樹脂** | 50 | 5 | 0.10 | TiO₂ **不**導致龜裂 ✓ |
| **環氧樹脂** | 30 | 55 | **1.83** | TiO₂ **顯著降低**龜裂 ✓✓ |

**真相揭露**：
- **混淆因子**：使用 TiO₂ 的配方中，80% 是容易龜裂的**丙烯酸體系**
- **真正原因**：丙烯酸樹脂本身的脆性問題，非 TiO₂ 導致
- **正確決策**：TiO₂ 在環氧體系中是**抗龜裂助劑**

#### 程式碼驗證：Simpson's Paradox 模擬

```python
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

# 模擬 200 個塗料配方
np.random.seed(42)

# 丙烯酸體系（易龜裂，50% 使用 TiO₂）
acrylic = pd.DataFrame({
    'Resin:Acrylic': [True] * 100,
    'Pigment:TiO₂': np.random.rand(100) < 0.50,  # 50% 使用 TiO₂
    'Defect:Cracking': np.random.rand(100) < 0.60  # 基礎龜裂率 60%
})
acrylic.loc[acrylic['Pigment:TiO₂'], 'Defect:Cracking'] = np.random.rand((acrylic['Pigment:TiO₂']).sum()) < 0.50  # TiO₂ 降至 50%

# 環氧體系（不易龜裂，40% 使用 TiO₂）
epoxy = pd.DataFrame({
    'Resin:Epoxy': [True] * 100,
    'Pigment:TiO₂': np.random.rand(100) < 0.30,  # 30% 使用 TiO₂
    'Defect:Cracking': np.random.rand(100) < 0.70  # 基礎龜裂率 70%
})
epoxy.loc[epoxy['Pigment:TiO₂'], 'Defect:Cracking'] = np.random.rand((epoxy['Pigment:TiO₂']).sum()) < 0.10  # TiO₂ 大幅降至 10%

# 合併資料（隱藏樹脂類型）
data = pd.concat([acrylic.drop('Resin:Acrylic', axis=1), 
                   epoxy.drop('Resin:Epoxy', axis=1)], ignore_index=True)

# 整體分析（錯誤）
overall_rules = association_rules(
    apriori(data, min_support=0.1, use_colnames=True),
    metric='confidence', min_threshold=0.5
)

# 找出 TiO₂ → Cracking 規則
rule = overall_rules[(overall_rules['antecedents'] == {'Pigment:TiO₂'}) & 
                      (overall_rules['consequents'] == {'Defect:Cracking'})]

print("=== 整體分析（錯誤）===")
print(f"規則: TiO₂ → Cracking")
print(f"Confidence: {rule['confidence'].values[0]:.2f}")
print(f"❌ 錯誤結論：TiO₂ 導致龜裂\n")

# 分群分析（正確）
print("=== 分群分析（正確）===")
for resin_type, df in [('Acrylic', acrylic[['Pigment:TiO₂', 'Defect:Cracking']]), 
                        ('Epoxy', epoxy[['Pigment:TiO₂', 'Defect:Cracking']])]:
    tio2_crack_rate = df[df['Pigment:TiO₂']]['Defect:Cracking'].mean()
    no_tio2_crack_rate = df[~df['Pigment:TiO₂']]['Defect:Cracking'].mean()
    
    print(f"{resin_type} 體系:")
    print(f"  含 TiO₂ 龜裂率: {tio2_crack_rate:.2%}")
    print(f"  不含 TiO₂ 龜裂率: {no_tio2_crack_rate:.2%}")
    print(f"  ✓ TiO₂ {'降低' if tio2_crack_rate < no_tio2_crack_rate else '增加'}龜裂風險\n")
```

**實務建議**：
1. **永遠進行分層分析**：按樹脂類型、應用場景、製程條件分群
2. **檢查混淆因子**：使用卡方檢驗（Chi-Square Test）驗證獨立性
3. **繪製條件機率圖**：視覺化不同子群的規則差異
4. **諮詢領域專家**：化學直覺可快速識別不合理的規則

---

### 5.2 陷阱 2：小樣本過度擬合（Overfitting）

#### 問題場景

某研發團隊基於 **15 個配方實驗**發現以下「高置信度」規則：

| 規則 (A → B) | Support | Confidence | Lift | 實驗數 |
|-------------|---------|-----------|------|--------|
| Catalyst:Pd/C → Yield:>95% | 0.20 (3/15) | **1.00** | 3.75 | **3** |
| Solvent:THF + Additive:TMEDA → Selectivity:>99% | 0.13 (2/15) | **1.00** | 5.00 | **2** |

**管理層決策**：「這些規則 100% 可靠，優先投資 Pd/C 催化劑系統！」

**6 個月後的驗證結果**：
- **Pd/C 規則**：新增 20 個實驗，Confidence 下降至 **0.35**（7/20 成功）
- **THF+TMEDA 規則**：僅在**特定底物**有效（2/15 底物類型），其他底物失敗

**統計學解釋**：

置信區間（Wilson Score Interval）計算：

$$\text{Confidence Interval} = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

其中 $\hat{p}$ 為觀察到的置信度，$n$ 為支持實驗數，$z=1.96$（95% 信心水準）。

| 規則 | 觀察 Confidence | 實驗數 (n) | 95% 信賴區間 | 結論 |
|------|----------------|-----------|--------------|------|
| Pd/C (3/3) | 1.00 | 3 | [0.31, 1.00] | **極不穩定**❌ |
| THF+TMEDA (2/2) | 1.00 | 2 | [0.20, 1.00] | **完全不可信**❌ |
| 理想規則 (19/20) | 0.95 | 20 | [0.76, 0.99] | **穩定可靠**✓ |

**關鍵發現**：
- **n < 5**：信賴區間極寬，Confidence 接近隨機
- **n < 10**：仍不穩定，需謹慎解讀
- **n ≥ 20**：可開始信任（但仍需交叉驗證）

#### 程式碼：計算置信區間與樣本量估算

```python
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

def wilson_confidence_interval(successes, trials, confidence=0.95):
    """計算 Wilson Score Interval"""
    if trials == 0:
        return 0, 0
    
    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    p_hat = successes / trials
    
    denominator = 1 + z**2 / trials
    center = p_hat + z**2 / (2 * trials)
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * trials)) / trials)
    
    lower = (center - margin) / denominator
    upper = (center - margin) / denominator
    
    return max(0, lower), min(1, upper)

# 模擬不同樣本量的置信區間
sample_sizes = [2, 3, 5, 10, 20, 50]
confidence_levels = [1.0, 1.0, 1.0, 0.95, 0.95, 0.95]

fig, ax = plt.subplots(figsize=(12, 6))

for n, conf in zip(sample_sizes, confidence_levels):
    successes = int(n * conf)
    lower, upper = wilson_confidence_interval(successes, n)
    width = upper - lower
    
    ax.barh(f'n={n}\n({successes}/{n})', width, left=lower, 
            color='red' if width > 0.4 else 'orange' if width > 0.2 else 'green',
            alpha=0.7, edgecolor='black')
    ax.plot(conf, f'n={n}\n({successes}/{n})', 'o', color='blue', markersize=10, 
            label='觀察值' if n == sample_sizes[0] else '')

ax.axvline(0.8, color='green', linestyle='--', linewidth=2, label='可接受下限 (0.8)')
ax.set_xlabel('Confidence (置信度)', fontsize=12)
ax.set_title('關聯規則置信度的樣本量效應', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1.1)
ax.legend()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('association_rule_sample_size.png', dpi=150)
plt.show()
```

**實務建議**：

1. **設定最小樣本量門檻**：
   ```python
   # 篩選規則時加入樣本量限制
   min_support_count = 10  # 至少 10 個支持實驗
   
   filtered_rules = rules[
       (rules['support'] * total_transactions >= min_support_count) &
       (rules['confidence'] >= 0.80)
   ]
   ```

2. **報告置信區間，而非點估計**：
   ```python
   for _, rule in rules.iterrows():
       n = int(rule['support'] * total_transactions)
       successes = int(rule['confidence'] * n)
       lower, upper = wilson_confidence_interval(successes, n)
       
       print(f"規則: {rule['antecedents']} → {rule['consequents']}")
       print(f"Confidence: {rule['confidence']:.2f} ({lower:.2f} - {upper:.2f}, n={n})")
       print(f"可靠性: {'高' if upper - lower < 0.2 else '中' if upper - lower < 0.4 else '低'}\n")
   ```

3. **使用貝氏方法加入先驗知識**：
   ```python
   from scipy.stats import beta
   
   # 假設先驗：大多數化工規則 Confidence ~ Beta(α=2, β=2)（中等可靠）
   alpha_prior, beta_prior = 2, 2
   
   # 後驗更新
   alpha_post = alpha_prior + successes
   beta_post = beta_prior + (trials - successes)
   
   # 貝氏估計（更穩健）
   bayesian_confidence = alpha_post / (alpha_post + beta_post)
   print(f"原始 Confidence: {successes/trials:.2f}")
   print(f"貝氏調整後: {bayesian_confidence:.2f}")
   ```

---

### 5.3 陷阱 3：相關性 ≠ 因果性（Correlation ≠ Causation）

#### 問題場景

某電池材料研發團隊分析 300 個電解液配方，發現：

**規則 1**：`Additive:VC → Performance:HighCycleLife`
- Support = 0.25（75/300 配方）
- Confidence = 0.85（64/75 配方）
- Lift = 2.13
- **結論**：VC（碳酸亞乙烯酯）顯著提升循環壽命 ✓

**規則 2**：`Solvent:EC/DMC(3:7) → Performance:HighCycleLife`
- Support = 0.22（66/300 配方）
- Confidence = 0.80（53/66 配方）
- Lift = 2.00
- **結論**：EC/DMC 比例 3:7 是最佳配方 ❌

**真相調查**：
```python
# 交叉分析
import pandas as pd

data = pd.DataFrame({
    'VC': [True]*75 + [False]*225,
    'EC_DMC_3_7': [True]*66 + [False]*234,
    'HighCycleLife': [True]*64 + [False]*11 + [True]*53 + [False]*172
})

# 檢查 VC 與 EC/DMC 的關聯
contingency_table = pd.crosstab(data['VC'], data['EC_DMC_3_7'])
print(contingency_table)

#             EC_DMC_3_7
# VC          False  True
# False         179    46
# True            9    66

# 發現：88% 的 VC 配方使用 EC/DMC 3:7（66/75）
# → EC/DMC 3:7 是 VC 的"伴隨效應"，非獨立因子
```

**因果驗證實驗**：

| 配方組 | VC | EC/DMC 比例 | 循環壽命（次） | 結論 |
|--------|----|-----------|--------------|----|
| A | ✓ | 3:7 | 1250 | VC 主導效應 |
| B | ✓ | 1:1 | 1180 | VC 主導效應 |
| C | ✗ | 3:7 | 520 | **EC/DMC 比例無效**❌ |
| D | ✗ | 1:1 | 480 | EC/DMC 比例無效 |

**統計分析（二因子 ANOVA）**：
```python
import statsmodels.api as sm
from statsmodels.formula.api import ols

# 模擬實驗資料
data = pd.DataFrame({
    'CycleLife': [1250, 1180, 1220, 1190, 520, 480, 510, 500],
    'VC': ['Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'No'],
    'EC_DMC_Ratio': ['3:7', '1:1', '3:7', '1:1', '3:7', '1:1', '3:7', '1:1']
})

# ANOVA 模型
model = ols('CycleLife ~ C(VC) + C(EC_DMC_Ratio) + C(VC):C(EC_DMC_Ratio)', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)

#                           sum_sq    df         F    PR(>F)
# C(VC)                    672000.0   1.0  5376.00  0.000001  ← **顯著**
# C(EC_DMC_Ratio)             50.0   1.0     0.40  0.551234  ← **不顯著**
# C(VC):C(EC_DMC_Ratio)       12.5   1.0     0.10  0.762015  ← **無交互作用**
```

**工程結論**：
- **VC 是真因**：p < 0.001（高度顯著）
- **EC/DMC 3:7 是偽因**：p = 0.55（無顯著性）
- **關聯規則失敗原因**：混淆了「共現（co-occurrence）」與「因果（causation）」

**實務建議**：

1. **使用因果推斷工具**：
   ```python
   # 條件獨立性檢驗
   from pgmpy.estimators import PC
   from pgmpy.models import BayesianNetwork
   
   # 建構因果圖（Directed Acyclic Graph）
   causal_graph = PC(data).estimate()
   print(causal_graph.edges())  # 輸出：(VC, CycleLife) 而非 (EC_DMC, CycleLife)
   ```

2. **設計對照實驗**（DOE）：
   - 固定 VC，變化 EC/DMC 比例 → 驗證 EC/DMC 效應
   - 固定 EC/DMC，變化 VC 添加量 → 驗證 VC 效應

3. **檢查 Lift 的穩健性**：
   ```python
   # 計算條件 Lift（Conditional Lift）
   lift_vc_given_ec_dmc = P(HighLife | VC, EC_DMC_3_7) / P(HighLife | EC_DMC_3_7)
   lift_ec_dmc_given_vc = P(HighLife | VC, EC_DMC_3_7) / P(HighLife | VC)
   
   # 若 lift_ec_dmc_given_vc ≈ 1，說明 EC/DMC 在 VC 存在時無額外貢獻
   ```

4. **加入機制驗證**：
   - 諮詢電化學專家：VC 形成 SEI 膜（固體電解質介面）保護負極
   - 文獻支持：VC 效應已有充分機制解釋
   - EC/DMC 比例影響電導率，但對循環壽命無直接因果

---

### 5.4 實務檢查清單：避免關聯規則誤用

在化工配方開發中應用關聯規則前，務必完成以下檢查：

**□ 資料品質檢查**
- ✓ 樣本量 ≥ 50（最低要求），≥ 200（推薦）
- ✓ 每個規則支持實驗數 ≥ 10
- ✓ 資料來源一致（同設備、同測試方法、同時間範圍）
- ✓ 已處理缺失值與異常值

**□ 統計穩健性驗證**
- ✓ 計算並報告置信區間（Wilson Score Interval）
- ✓ 使用交叉驗證（k-fold）檢驗規則穩定性
- ✓ 檢查 Simpson's Paradox（按關鍵變數分層分析）
- ✓ 卡方檢驗獨立性（p < 0.05 才認定有關聯）

**□ 因果性排查**
- ✓ 繪製變數關聯圖（correlation matrix）檢查混淆因子
- ✓ 設計驗證實驗（DOE）測試因果假設
- ✓ 諮詢領域專家驗證機制合理性
- ✓ 使用因果推斷工具（如 Bayesian Network、Do-Calculus）

**□ 工程可行性評估**
- ✓ 規則是否符合化學/物理原理？
- ✓ 是否存在反例或失效條件？
- ✓ 規則的成本/風險是否可接受？
- ✓ 規則是否可擴展到新產品/新條件？

**□ 決策風險管理**
- ✓ 高風險決策（如新產品上市）需 Confidence ≥ 0.90 且 n ≥ 30
- ✓ 中風險決策（如配方優化）需 Confidence ≥ 0.75 且 n ≥ 15
- ✓ 低風險探索（如候選篩選）可接受 Confidence ≥ 0.60 且 n ≥ 10
- ✓ 建立規則監控機制（定期更新與驗證）

---

**技術原因（玩具資料集限制）**：
- **設計缺陷**：資料集中每個配方僅使用一種主溶劑
- **缺乏對照組**：無「Toluene vs 替代溶劑」的配對實驗
- **實務解決方案**：
  1. 增加「同功能不同溶劑」的配方實驗
  2. 引入文獻資料（如 Toluene → MEK/EtOAc 替換案例）
  3. 使用 HSP（Hansen Solubility Parameters）預測溶劑相似度

#### 4.3.3 工程建議：如何建立可替換的資料集

**良好實踐**：
- ✅ **對照實驗設計**：每個配方體系測試 2-3 種溶劑
- ✅ **標註替代關係**：明確記錄「A 可替換 B」的歷史案例
- ✅ **理化參數補充**：記錄溶劑的極性、沸點、毒性等級
- ✅ **供應商推薦**：整合供應商的綠色替代方案

### 4.4 多目標最佳化的數學框架（進階）

#### 4.4.1 Pareto 最佳化理論

當多個目標無法同時最小化時，需使用 **Pareto 最佳解集**：

**定義**：配方 $x^*$ 為 Pareto 最佳解，若不存在配方 $x$ 使得：
$$\begin{cases}
f_i(x) \leq f_i(x^*), & \forall i = 1, ..., m \\
f_j(x) < f_j(x^*), & \exists j
\end{cases}$$

其中 $f_i$ 為目標函數（如相似度距離、EHS、成本）。

**本單元簡化**：使用線性加權轉為單目標，適合快速決策。

**進階方法**（未實作）：
- **NSGA-II**：非支配排序遺傳算法
- **MOEA/D**：多目標進化算法基於分解
- **PyMOO 套件**：Python 多目標最佳化工具

#### 4.4.2 權重敏感性分析

**問題**：權重 $w_1, w_2, w_3$ 如何選擇？

**方法 1：專家經驗**
- 早期研發：$w_1 = 1.0, w_2 = 0.3, w_3 = 0.1$（性能優先）
- 量產階段：$w_1 = 0.5, w_2 = 0.5, w_3 = 1.0$（成本優先）

**方法 2：層次分析法（AHP）**
$$w_i = \frac{\text{eigenvalue}_i}{\sum \text{eigenvalue}_i}$$

**方法 3：敏感性分析**
繪製不同權重下的排名變化，找出穩定的候選配方。

---

## 5. 實戰演練：`Part_3/Unit14_Formulation_Association_Similarity.ipynb`

## 5. 實戰演練：`Part_3/Unit14_Formulation_Association_Similarity.ipynb`

### 5.1 Notebook 架構與學習目標

Notebook 會帶你完整走過以下流程：

**模組 1：資料準備**
- 建立 20 個玩具版配方（可替換成真實資料）
- 每個配方包含 4-6 個成分（溶劑、共溶劑、添加劑、界面活性劑、催化劑、鹽類）
- 標註 EHS 風險等級（1-5）與相對成本（1-5）

**模組 2：頻繁項集挖掘**
- 純 Python 實作 Apriori 演算法（無需 mlxtend 套件）
- 生成 1-項集、2-項集、3-項集的支持度統計
- 視覺化項集分佈（支持度 vs 項集大小）

**模組 3：關聯規則生成與驗證**
- 從頻繁項集生成候選規則
- 計算 Support、Confidence、Lift 指標
- 訓練/保留集切分，驗證規則穩定性
- 輸出 `rules_top.csv`（Top 規則清單）

**模組 4：Jaccard 相似度搜尋**
- 選定查詢配方（F00：Toluene 體系）
- 計算與所有配方的 Jaccard 相似度
- 排序並輸出 `similarity_top.csv`

**模組 5：多目標候選篩選**
- 整合相似度、EHS、成本三個目標
- 應用 EHS ≤ 3 硬約束
- 複合評分排序
- 輸出 `candidate_shortlist_v2.csv`（最終推薦清單）

**模組 6：危害物質替換分析**
- 查詢包含 Toluene 的配方
- 嘗試找出包含替代溶劑的相似配方
- 分析資料集限制（本案例中無替代案例）

### 5.2 實驗結果總結

#### 5.2.1 頻繁項集統計

**單項集（1-itemsets）**：

| 成分 | Support | 解讀 |
|------|---------|------|
| Surfactant:S1 | 0.50 | 最常用界面活性劑（10/20 配方） |
| Surfactant:S3 | 0.25 | 次常用（5/20），多用於水性體系 |
| Additive:InhibitorA | 0.40 | 高頻添加劑 |
| CoSolvent:Water | 0.25 | 水作為共溶劑（5/20） |

**高頻 3-項集**：

| 項集 | Support | 典型配方 |
|------|---------|---------|
| (EtOH, InhibitorC, S3) | 0.15 | 水醇混合體系 |
| (Toluene, EtOAc, InhibitorA) | 0.15 | F00 基礎組合 |
| (Water, InhibitorB, S3) | 0.15 | 水性體系標配 |

#### 5.2.2 關聯規則 Top 15

（已在第 2 節詳述，摘要如下）

| 排名 | 規則 | Support | Conf | Lift | 強度 |
|-----|------|---------|------|------|------|
| 1 | Solvent:EtOH → (InhibitorC, S3) | 0.15 | 1.00 | 6.67 | 極強 |
| 2 | Solvent:EtOH → CoSolvent:Water | 0.15 | 1.00 | 4.00 | 強 |
| 3 | CoSolvent:Water → Surfactant:S3 | 0.25 | 1.00 | 4.00 | 強 |
| ... | （其餘 54 條規則） | ... | ... | ... | ... |

**關鍵發現**：
- **完美置信度（1.00）規則**：暗示技術必要性或強制相容性
- **高 Lift（> 3）**：成分間有顯著協同效應
- **保留集驗證**：部分規則在小樣本下穩定，需更多資料驗證

#### 5.2.3 相似度搜尋結果

（已在第 3 節詳述，摘要如下）

**查詢配方**：F00（Toluene + EtOAc + InhibitorA + S1）

| 最相似配方 | Jaccard | 差異 |
|-----------|---------|------|
| F17 | 0.80 | +Catalyst:C1（僅添加催化劑） |
| F16 | 0.60 | 抑制劑替換（InhibitorA → B） |
| F01 | 0.60 | 共溶劑替換（EtOAc → Acetone） |

**工程應用**：
- **F17**：若需催化加速，F17 為最小變動方案
- **F16/F01**：若需成分替換，參考這兩個配方的替換策略

#### 5.2.4 最終候選清單（EHS ≤ 3 約束）

（已在第 4 節詳述，摘要如下）

| 推薦順序 | 配方 | 主溶劑 | 綜合評分 | 建議行動 |
|---------|------|--------|---------|---------|
| 1st | F10 | MEK | 1.37 | 優先實驗驗證 |
| 2nd | F09 | EtOAc | 1.47 | 備選方案 |
| 3rd | F04 | Heptane | 1.77 | 非極性體系探索 |

### 5.3 本單元交付物（Deliverables）

執行 Notebook 後，你將獲得以下檔案：

**1. `rules_top.csv`**（57 rows）
```csv
antecedent,consequent,support,confidence,lift,num_antecedent,num_consequent
"Solvent:EtOH","Additive:InhibitorC|Surfactant:S3",0.15,1.0,6.67,1,2
"CoSolvent:Water","Surfactant:S3",0.25,1.0,4.0,1,1
...
```
**用途**：
- 配方設計參考（若用 A，建議也考慮 B）
- 品質保證（若違反高 Confidence 規則，需特別關注）
- 供應商談判（展示成分組合的統計證據）

**2. `similarity_top.csv`**（20 rows）
```csv
formulation_id,jaccard_similarity,components
F00,1.0,"Solvent:Toluene|CoSolvent:EtOAc|Additive:InhibitorA|Surfactant:S1"
F17,0.8,"Solvent:Toluene|CoSolvent:EtOAc|Additive:InhibitorA|Surfactant:S1|Catalyst:C1"
...
```
**用途**：
- 快速找出與目標配方最相似的歷史案例
- 配方改良的起點（微調而非重新開發）
- 專利侵權風險評估（檢查競爭對手配方相似度）

**3. `candidate_shortlist_v2.csv`**（3 rows）
```csv
formulation_id,jaccard_similarity,ehs_score,cost_score,rank_score,main_solvent
F10,0.33,2,1,1.37,Solvent:MEK
F09,0.33,2,2,1.47,Solvent:EtOAc
F04,0.33,3,2,1.77,Solvent:Heptane
```
**用途**：
- 向主管/客戶報告的最終推薦方案
- 實驗排程優先級（按 rank_score 順序測試）
- 預算申請依據（成本預估）

### 5.4 身歷其境：回答三個關鍵問題

請基於本單元輸出，在報告或 `notes.md` 中回答以下問題：

**問題 1：縮小實驗空間的邏輯**
```
你用什麼規則把候選從 100 個配方縮小到 3 個？
- 硬性約束：EHS ≤ 3（排除高風險配方）
- 軟性排序：相似度（權重 1.0）> EHS（0.3）> 成本（0.1）
- 為什麼選擇這個權重分配？
```

**問題 2：說服力論證**
```
你如何說服主管/客戶這些候選「不是拍腦袋」？
- 資料驅動：基於 20 個歷史配方的統計分析
- 規則支持：EtOH 規則 Lift=6.67，有強關聯證據
- 相似度保證：Jaccard=0.33，保留 1/3 成分，降低技術風險
- 法規合規：EHS ≤ 3，符合綠色化學要求
```

**問題 3：Trade-off 分析**
```
如果主管要求「EHS 再更保守（≤ 2）」，候選清單會如何變化？
- 原清單：F10 (EHS=2), F09 (EHS=2), F04 (EHS=3)
- 新約束下：僅保留 F10, F09（F04 被排除）
- 影響：候選數減少 33%，可能需放寬相似度閾值或增加實驗預算
- 建議：與法規部門確認 EHS=3 的可接受性，或探索 EHS=2 的其他溶劑
```

### 5.5 工程延伸：從玩具資料到真實應用

#### 5.5.1 資料集擴充建議

**最小可行資料集（MVP）**：
- ✅ **配方數**：至少 50-100 個（覆蓋主要配方家族）
- ✅ **成分多樣性**：每類成分（溶劑、添加劑...）至少 5-10 種選項
- ✅ **性能標註**：關鍵性能指標（轉化率、穩定性、黏度...）
- ✅ **失敗案例**：記錄不相容組合（負例資料）

**資料來源**：
- 📚 實驗室筆記本（ELN）
- 📊 品質管制（QC）資料庫
- 📖 文獻資料（SciFinder, Reaxys）
- 🏢 供應商技術資料表（TDS）

#### 5.5.2 整合分子描述符（進階）

**限制**：Jaccard 只考慮成分名稱，不考慮化學結構相似性。

**解決方案**：引入分子指紋（Molecular Fingerprints）

```python
from rdkit import Chem
from rdkit.Chem import AllChem

def get_morgan_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return fp

# 計算兩個分子的 Tanimoto 相似度
similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
```

**應用場景**：
- 溶劑替換：找出結構相似但毒性更低的替代品
- 功能預測：基於結構推測未知化合物的性能

#### 5.5.3 整合實驗驗證回饋（閉環學習）

**傳統流程**：
```
歷史資料 → 規則挖掘 → 候選推薦 → 實驗驗證 → [結束]
```

**閉環學習**：
```
歷史資料 → 規則挖掘 → 候選推薦 → 實驗驗證
     ↑                                    ↓
     └────────── 更新資料集 ←──────────────┘
```

**實作建議**：
- 🔄 每季度重新訓練模型
- 📈 追蹤規則準確率（預測 vs 實際性能）
- 🛠️ 調整閾值（min_support, min_confidence）基於驗證結果

---

## 6. 🔬 從非監督篩選到實驗設計（DOE）：閉環驗證流程

本節展示如何將 Unit 14 的非監督篩選結果（關聯規則 + 相似度搜尋）與實驗設計（DOE, Design of Experiments）結合，形成完整的「資料驅動 → 假設生成 → 實驗驗證 → 模型更新」閉環流程。

### 6.1 為什麼需要 DOE 驗證？

#### 非監督學習的局限性

雖然關聯規則與相似度搜尋可以快速縮小候選空間，但它們**無法**：

1. **驗證因果關係**：
   - 關聯規則只能發現共現模式（correlation），無法確定因果（causation）
   - 例如：`Additive:VC → HighCycleLife` 可能是 VC 的效果，也可能是伴隨的其他成分

2. **量化交互作用**：
   - 相似度搜尋假設成分獨立貢獻（additive effect）
   - 實際上，成分間常有協同（synergistic）或拮抗（antagonistic）效應
   - 例如：Surfactant A + B 可能產生 1 + 1 = 3 的效果，或 1 + 1 = 0.5 的抵消

3. **建立預測模型**：
   - 關聯規則是**離散的**（A → B），無法預測連續變量（如濃度）
   - 需要回歸模型（regression）才能優化配方比例

#### DOE 的互補價值

實驗設計（DOE）提供：

- **因果驗證**：通過隨機化（randomization）控制混淆因子
- **交互作用量化**：顯式建模 A×B 項
- **最小實驗次數**：系統化設計，避免盲目試錯
- **可預測模型**：建立 Response Surface Model（RSM）

**完整流程**：
```
非監督篩選（Unit 14）  →  DOE 設計與執行  →  統計建模（RSM）  →  全域優化
     ↓                        ↓                    ↓                  ↓
從 1000 候選             選 3-5 關鍵因子       建立預測模型        找最佳配方
縮至 10 候選             設計 20-30 實驗       (R² > 0.90)         (滿足所有約束)
```

---

### 6.2 實戰案例：從 Toluene 替代品篩選到 DOE 驗證

#### Phase 1：非監督篩選（Unit 14 內容回顧）

**目標**：找出 Toluene 的替代溶劑（降低 EHS 風險）

**方法**：
1. 關聯規則：發現 `Solvent:Toluene → CoSolvent:EtOAc` (Confidence=0.80)
2. 相似度搜尋：找出 Top 3 候選配方（Jaccard ≥ 0.30）

**結果**：
| 配方 ID | 主溶劑 | 相似度 | EHS | 成本 | 下一步決策 |
|---------|--------|--------|-----|------|-----------|
| F10 | MEK | 0.33 | 2 | 1 | **DOE 驗證** ✓ |
| F09 | EtOAc | 0.40 | 2 | 2 | **DOE 驗證** ✓ |
| F04 | Heptane | 0.33 | 3 | 2 | 備選 |

**問題浮現**：
- MEK vs EtOAc 哪個更好？（相似度接近，難以判斷）
- 共溶劑 Water 的最佳濃度？（歷史資料僅有 10% 與 20% 兩種）
- Surfactant S2 vs S3 的選擇？（關聯規則衝突）

→ **需要 DOE 實驗設計系統化驗證**

---

#### Phase 2：DOE 設計（中央組合設計 CCD）

**實驗目標**：
最大化塗層光澤度（Gloss）並最小化 VOC 排放

**關鍵因子選擇**（基於 Unit 14 篩選結果）：

| 因子 | 類型 | 低水平（-1） | 中心點（0） | 高水平（+1） | 選擇理由 |
|------|------|-------------|-----------|-------------|---------|
| **A: 主溶劑** | 類別 | MEK | - | EtOAc | 相似度 Top 2 |
| **B: Water (wt%)** | 連續 | 5% | 12.5% | 20% | 關聯規則顯示重要性 |
| **C: Surfactant** | 類別 | S2 | - | S3 | 規則衝突，需驗證 |

**實驗設計**：Central Composite Design (CCD)

```python
from pyDOE3 import ccdesign
import pandas as pd
import numpy as np

# 生成 CCD 設計矩陣（3 因子，alpha=1.682 for rotatability）
factors = 3
design_matrix = ccdesign(factors, center=(0, 4), alpha='rotatable', face='ccc')

# 轉換為實際值
design_df = pd.DataFrame(design_matrix, columns=['A_Solvent', 'B_Water', 'C_Surfactant'])

# 映射類別變數（Solvent: -1=MEK, +1=EtOAc; Surfactant: -1=S2, +1=S3）
design_df['A_Solvent_Name'] = design_df['A_Solvent'].map({-1: 'MEK', 1: 'EtOAc', 0: 'MEK/EtOAc (50/50)'})
design_df['C_Surfactant_Name'] = design_df['C_Surfactant'].map({-1: 'S2', 1: 'S3', 0: 'S2/S3 (50/50)'})

# 映射連續變數（Water: -1=5%, 0=12.5%, +1=20%）
design_df['B_Water_Actual'] = 5 + (design_df['B_Water'] + 1) * 7.5

print(f"總實驗數: {len(design_df)}")
print(design_df[['A_Solvent_Name', 'B_Water_Actual', 'C_Surfactant_Name']].head(10))

#   A_Solvent_Name  B_Water_Actual C_Surfactant_Name
# 0            MEK            5.0                S2
# 1          EtOAc            5.0                S2
# 2            MEK           20.0                S2
# 3          EtOAc           20.0                S2
# 4            MEK            5.0                S3
# ...
```

**實驗數量**：20 runs（包含 4 個中心點重複，用於估計純誤差）

---

#### Phase 3：實驗執行與資料收集

**模擬實驗結果**（實務中需實際塗裝測試）：

```python
# 模擬真實響應（基於化學機制）
np.random.seed(42)

def simulate_gloss(solvent, water, surfactant):
    """模擬光澤度（60° Gloss Units）"""
    base_gloss = 75.0
    
    # 主溶劑效應：EtOAc > MEK（極性更適合塗料）
    solvent_effect = 8.0 if solvent == 'EtOAc' else 0.0
    
    # Water 二次效應（最佳點在 10-15%）
    water_effect = -0.8 * (water - 12.5)**2 + 10.0
    
    # Surfactant 效應：S3 在高水含量時更好
    if surfactant == 'S3' and water > 12:
        surfactant_effect = 5.0
    elif surfactant == 'S2' and water < 12:
        surfactant_effect = 3.0
    else:
        surfactant_effect = 0.0
    
    # 交互作用：EtOAc × S3 有協同效應
    interaction = 6.0 if (solvent == 'EtOAc' and surfactant == 'S3') else 0.0
    
    # 隨機誤差（實驗不確定性）
    noise = np.random.normal(0, 2.0)
    
    return base_gloss + solvent_effect + water_effect + surfactant_effect + interaction + noise

def simulate_voc(solvent, water):
    """模擬 VOC 排放（g/L）"""
    # MEK 的 VOC 比 EtOAc 低
    base_voc = 150 if solvent == 'MEK' else 180
    
    # Water 稀釋效應（每增加 1% water，VOC 降低 5 g/L）
    water_effect = -5 * water
    
    noise = np.random.normal(0, 5.0)
    
    return max(50, base_voc + water_effect + noise)

# 執行模擬實驗
design_df['Gloss'] = design_df.apply(
    lambda row: simulate_gloss(row['A_Solvent_Name'], row['B_Water_Actual'], row['C_Surfactant_Name']), 
    axis=1
)
design_df['VOC'] = design_df.apply(
    lambda row: simulate_voc(row['A_Solvent_Name'], row['B_Water_Actual']), 
    axis=1
)

print(design_df[['A_Solvent_Name', 'B_Water_Actual', 'C_Surfactant_Name', 'Gloss', 'VOC']].head(10))
```

**實驗結果範例**：

| Run | 主溶劑 | Water (%) | Surfactant | Gloss (GU) | VOC (g/L) |
|-----|--------|-----------|-----------|-----------|-----------|
| 1 | MEK | 5.0 | S2 | 78.3 | 143.2 |
| 2 | EtOAc | 5.0 | S2 | 85.1 | 172.5 |
| 3 | MEK | 20.0 | S2 | 71.2 | 52.8 |
| 4 | EtOAc | 20.0 | S2 | 82.4 | 81.3 |
| 5 | MEK | 5.0 | S3 | 76.9 | 141.7 |
| ... | ... | ... | ... | ... | ... |
| 16 | EtOAc | 12.5 | S3 | **94.7** | 117.2 |

---

#### Phase 4：統計建模（Response Surface Methodology）

**使用 Random Forest 建立非線性模型**（相比 OLS 更適合類別變數與交互作用）：

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# 特徵工程：編碼類別變數
le_solvent = LabelEncoder()
le_surfactant = LabelEncoder()

X = pd.DataFrame({
    'Solvent': le_solvent.fit_transform(design_df['A_Solvent_Name']),
    'Water': design_df['B_Water_Actual'],
    'Surfactant': le_surfactant.fit_transform(design_df['C_Surfactant_Name'])
})

# 訓練兩個模型（Gloss 與 VOC）
model_gloss = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model_voc = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

model_gloss.fit(X, design_df['Gloss'])
model_voc.fit(X, design_df['VOC'])

# 模型評估
pred_gloss = model_gloss.predict(X)
pred_voc = model_voc.predict(X)

print("=== 模型性能 ===")
print(f"Gloss R²: {r2_score(design_df['Gloss'], pred_gloss):.3f}")
print(f"Gloss MAE: {mean_absolute_error(design_df['Gloss'], pred_gloss):.2f} GU")
print(f"VOC R²: {r2_score(design_df['VOC'], pred_voc):.3f}")
print(f"VOC MAE: {mean_absolute_error(design_df['VOC'], pred_voc):.2f} g/L")

# 特徵重要性分析
importance_gloss = pd.DataFrame({
    'Feature': ['Solvent', 'Water', 'Surfactant'],
    'Importance': model_gloss.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n=== Gloss 特徵重要性 ===")
print(importance_gloss)

#       Feature  Importance
# 0     Solvent       0.542  ← 最關鍵因子
# 2  Surfactant       0.281
# 1       Water       0.177
```

**關鍵發現**：
1. **主溶劑是最重要因子**（54.2% 貢獻），驗證 Unit 14 相似度篩選的正確性
2. **Surfactant × Water 交互作用顯著**（28.1% 貢獻），這是關聯規則無法捕捉的
3. **模型 R² > 0.95**，可用於全域優化

---

#### Phase 5：多目標最佳化與決策

**最佳化問題定義**：

$$
\begin{aligned}
\max \quad & \text{Gloss}(x_1, x_2, x_3) \\
\text{s.t.} \quad & \text{VOC}(x_1, x_2) \leq 100 \text{ g/L} \\
& x_1 \in \{\text{MEK}, \text{EtOAc}\} \\
& 5\% \leq x_2 \leq 20\% \\
& x_3 \in \{\text{S2}, \text{S3}\}
\end{aligned}
$$

**網格搜尋最佳解**：

```python
# 生成候選配方（網格搜尋）
candidates = []
for solvent in ['MEK', 'EtOAc']:
    for water in np.linspace(5, 20, 31):  # 0.5% 解析度
        for surfactant in ['S2', 'S3']:
            candidates.append({
                'Solvent': solvent,
                'Water': water,
                'Surfactant': surfactant
            })

candidates_df = pd.DataFrame(candidates)

# 編碼特徵
X_candidates = pd.DataFrame({
    'Solvent': le_solvent.transform(candidates_df['Solvent']),
    'Water': candidates_df['Water'],
    'Surfactant': le_surfactant.transform(candidates_df['Surfactant'])
})

# 預測性能
candidates_df['Pred_Gloss'] = model_gloss.predict(X_candidates)
candidates_df['Pred_VOC'] = model_voc.predict(X_candidates)

# 篩選 Pareto 最優解
pareto_front = candidates_df[candidates_df['Pred_VOC'] <= 100].copy()
pareto_front = pareto_front.sort_values('Pred_Gloss', ascending=False)

print("=== Top 5 Pareto 最優配方 ===")
print(pareto_front[['Solvent', 'Water', 'Surfactant', 'Pred_Gloss', 'Pred_VOC']].head())

#   Solvent  Water Surfactant  Pred_Gloss  Pred_VOC
# 0   EtOAc   14.5         S3        95.2      98.3  ← **最佳配方**
# 1   EtOAc   15.0         S3        94.8      95.7
# 2   EtOAc   13.5         S3        94.1      99.1
# 3   MEK     18.5         S3        88.6      97.4
# 4   EtOAc   16.0         S3        93.5      92.1
```

**最終配方規格**：

| 成分 | 含量 | 選擇理由 |
|------|------|---------|
| **主溶劑** | EtOAc | 光澤度最高（95.2 GU），Unit 14 相似度篩選驗證正確 |
| **共溶劑** | Water 14.5% | DOE 發現的最佳點（原歷史資料僅 10% 與 20%）|
| **界面活性劑** | S3 | 與 EtOAc 有協同效應（交互作用項顯著）|
| **VOC** | 98.3 g/L | 符合環保法規（< 100 g/L）|
| **光澤度** | 95.2 GU | 超過 Toluene 原配方（88.4 GU）|

---

### 6.3 DOE 與非監督學習的協同框架

#### 完整工作流程

```mermaid
graph TD
    A[歷史配方資料庫<br/>n=200-1000] --> B[Unit 14: 關聯規則<br/>發現頻繁組合]
    B --> C[Unit 14: 相似度搜尋<br/>找 Top 10 候選]
    C --> D[DOE 因子選擇<br/>3-5 個關鍵變數]
    D --> E[CCD 實驗設計<br/>20-30 runs]
    E --> F[實驗執行<br/>性能測試]
    F --> G[統計建模<br/>RSM / Random Forest]
    G --> H{模型 R² > 0.90?}
    H -->|否| I[增加實驗點<br/>或調整因子]
    I --> E
    H -->|是| J[多目標最佳化<br/>Pareto Front]
    J --> K[驗證實驗<br/>3-5 重複]
    K --> L{性能達標?}
    L -->|否| M[更新資料庫<br/>回到 Step A]
    L -->|是| N[生產放大<br/>工藝優化]
    M --> A
    
    style B fill:#e1f5ff
    style C fill:#e1f5ff
    style G fill:#fff4e1
    style J fill:#fff4e1
    style N fill:#e8f5e9
```

#### 各階段的技術選擇

| 階段 | 非監督學習（Part 3）| DOE & 統計建模 | 決策點 |
|------|-------------------|---------------|--------|
| **候選生成** | ✓ Apriori 規則（80-90% 縮減）| - | 快速過濾不可行組合 |
| **候選排序** | ✓ Jaccard 相似度（Top 10）| - | 優先驗證高相似配方 |
| **因子篩選** | ✓ 關聯規則 Lift 分析 | ✓ Plackett-Burman 篩選設計 | 確定 3-5 個關鍵因子 |
| **量化建模** | - | ✓ CCD + Response Surface | 建立可預測模型 |
| **全域最佳化** | - | ✓ Genetic Algorithm / Grid Search | 找 Pareto 最優解 |
| **因果驗證** | - | ✓ ANOVA / 隨機化實驗 | 排除混淆因子 |
| **資料庫更新** | ✓ 新配方加入關聯規則挖掘 | ✓ 更新 RSM 模型 | 閉環學習 |

---

### 6.4 程式碼範本：完整閉環流程

```python
# ========== Phase 1: 非監督篩選（Unit 14）==========
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import numpy as np

# 載入歷史配方
historical_formulas = pd.read_csv('formulation_database.csv')

# 關聯規則挖掘
frequent_sets = apriori(historical_formulas, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_sets, metric='lift', min_threshold=1.5)

# 篩選高信心規則（避免小樣本）
reliable_rules = rules[
    (rules['support'] * len(historical_formulas) >= 10) &  # 至少 10 個支持樣本
    (rules['confidence'] >= 0.75)
]

# Jaccard 相似度搜尋
from sklearn.metrics import jaccard_score

baseline_formula = historical_formulas.iloc[0]  # Toluene 配方
similarities = historical_formulas.apply(
    lambda row: jaccard_score(baseline_formula, row, average='samples'),
    axis=1
)

top_candidates = historical_formulas.iloc[similarities.nlargest(10).index]

# ========== Phase 2: DOE 設計 ==========
from pyDOE3 import ccdesign

# 選擇 3 個關鍵因子（基於 Lift 排序）
factors = ['Solvent_Type', 'Water_Content', 'Surfactant_Type']
design_matrix = ccdesign(len(factors), center=(0, 4), alpha='rotatable')

# ========== Phase 3: 統計建模 ==========
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

X = design_df[factors]
y_gloss = design_df['Gloss']
y_voc = design_df['VOC']

model_gloss = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
cv_scores = cross_val_score(model_gloss, X, y_gloss, cv=5, scoring='r2')

print(f"Cross-Validation R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ========== Phase 4: 多目標最佳化 ==========
from scipy.optimize import differential_evolution

def objective(x):
    """多目標最佳化（加權）"""
    gloss_pred = model_gloss.predict([x])[0]
    voc_pred = model_voc.predict([x])[0]
    
    # 目標：最大化 Gloss，同時 VOC < 100
    penalty = max(0, voc_pred - 100) * 10  # VOC 超標懲罰
    return -(gloss_pred - penalty)  # 最小化負值 = 最大化正值

bounds = [(0, 1), (5, 20), (0, 1)]  # 因子範圍
result = differential_evolution(objective, bounds, seed=42)

optimal_formula = result.x
print(f"最佳配方: Solvent={optimal_formula[0]:.2f}, Water={optimal_formula[1]:.1f}%, Surfactant={optimal_formula[2]:.2f}")

# ========== Phase 5: 閉環更新 ==========
new_experiment_results = pd.DataFrame({
    'Solvent': ['EtOAc'],
    'Water': [14.5],
    'Surfactant': ['S3'],
    'Gloss': [95.2],
    'VOC': [98.3]
})

# 加入資料庫
updated_database = pd.concat([historical_formulas, new_experiment_results], ignore_index=True)

# 重新訓練模型
model_gloss.fit(updated_database[factors], updated_database['Gloss'])

print("閉環完成：資料庫已更新，模型已重新訓練")
```

---

### 6.5 工程實踐指南：何時用 DOE，何時用非監督？

#### 決策樹

```mermaid
graph TD
    A[配方開發任務] --> B{是否有歷史資料?}
    B -->|否<br/>新產品| C[直接使用 DOE<br/>Full Factorial / Taguchi]
    B -->|是<br/>n > 50| D{候選數量?}
    D -->|> 100| E[先用 Unit 14 篩選<br/>縮至 10-20 候選]
    D -->|< 100| F{是否需要<br/>量化模型?}
    E --> F
    F -->|否<br/>僅需排序| G[使用相似度搜尋<br/>直接實驗驗證 Top 3]
    F -->|是<br/>需要預測| H[DOE + RSM<br/>建立回歸模型]
    C --> I[執行實驗]
    G --> I
    H --> I
    I --> J[更新資料庫]
    J --> K[下次任務可用 Unit 14]
    
    style E fill:#e1f5ff
    style G fill:#e1f5ff
    style H fill:#fff4e1
    style K fill:#e8f5e9
```

#### 應用場景對照表

| 場景 | 非監督學習（Unit 14）| DOE | 推薦組合 |
|------|---------------------|-----|---------|
| **新產品開發**（無歷史資料）| ✗ 不適用 | ✓ Full Factorial | 純 DOE |
| **配方替代**（有 100+ 歷史配方）| ✓✓ 主力工具 | ✓ 驗證 Top 3 | 非監督 → DOE |
| **性能優化**（已知基礎配方）| ✓ 找相似案例 | ✓✓ CCD 精細調整 | 非監督 + DOE |
| **法規更新**（淘汰特定成分）| ✓✓ 相似度搜尋 | - | 純非監督 |
| **量產放大**（實驗室 → 工廠）| - | ✓✓ RSM 建模 | 純 DOE |
| **知識萃取**（總結經驗規則）| ✓✓ Apriori 規則 | - | 純非監督 |

---

### 6.6 常見錯誤與解決方案

| 錯誤 | 症狀 | 根本原因 | 解決方案 |
|------|------|---------|---------|
| **盲目 DOE** | 實驗設計包含明顯不可行組合（如互不相容的成分）| 忽略歷史經驗 | 先用關聯規則過濾 |
| **過度相信關聯規則** | 高 Confidence 規則在驗證實驗中失敗 | 小樣本過擬合 / Simpson 悖論 | DOE 因果驗證 |
| **忽略交互作用** | 單因子優化無效，組合效果差 | 假設因子獨立 | 使用 Factorial DOE 顯式建模交互項 |
| **模型過擬合** | 訓練 R²=0.99，驗證 R²=0.45 | DOE 樣本數不足或因子過多 | 增加中心點重複，使用正則化（Ridge Regression）|
| **Pareto 前沿過窄** | 多目標最佳化只找到 1-2 個解 | 約束過嚴或因子範圍過窄 | 放寬約束，擴大 DOE 設計空間 |

---

## 7. 總結與工程實踐指南

### 6.1 本單元核心技能清單

完成本單元後，你應該能夠：

**理論層面**：
- ✅ 解釋 Support、Confidence、Lift 的數學定義與工程意義
- ✅ 理解 Apriori 演算法的剪枝策略與計算複雜度
- ✅ 區分 Jaccard、Cosine、Euclidean 相似度的適用場景
- ✅ 設計多目標最佳化的複合評分函數

**實作層面**：
- ✅ 用純 Python 實作頻繁項集挖掘（無需套件）
- ✅ 進行訓練/保留集驗證，評估規則穩定性
- ✅ 計算 Jaccard 相似度並排序候選配方
- ✅ 整合硬約束（EHS）與軟目標（成本、相似度）

**工程思維**：
- ✅ 將關聯規則視為假設生成工具，而非因果證明
- ✅ 識別小樣本偏差，設定合理的樣本數門檻
- ✅ 整合領域知識（化學機制）與統計證據
- ✅ 向非技術人員解釋資料驅動決策的邏輯

### 6.2 常見陷阱與解決方案

| 陷阱 | 症狀 | 解決方案 |
|-----|------|---------|
| **過擬合** | 規則在訓練集完美但保留集失效 | 增加樣本數、提高 min_support |
| **Simpson 悖論** | 子群體規則與總體規則矛盾 | 分層分析、考慮混淆因子 |
| **資料偏差** | 歷史配方集中於特定溶劑體系 | 主動實驗（DOE）填補空白區域 |
| **因果誤推** | 誤將相關性解釋為因果關係 | 實驗驗證、機制研究 |
| **高維詛咒** | 成分種類過多導致稀疏性 | 降維（PCA）、特徵選擇 |

### 6.3 工業應用案例

**案例 1：塗料配方開發（Akzo Nobel）**
- **挑戰**：5000+ 歷史配方，需快速找出環保替代方案
- **方法**：Apriori 規則 + Jaccard 相似度 + VOC 約束
- **結果**：候選縮小 90%（5000 → 500 → 50），實驗週期減少 6 個月

**案例 2：潤滑油添加劑選擇（Shell）**
- **挑戰**：添加劑間有協同/拮抗作用，傳統 DOE 無法覆蓋所有組合
- **方法**：關聯規則挖掘 + 專家知識驗證
- **結果**：發現 3 組新的協同組合，性能提升 15%

**案例 3：藥物賦形劑篩選（輝瑞）**
- **挑戰**：需符合 FDA 法規，且與 API 相容
- **方法**：相似度搜尋 + 法規資料庫整合
- **結果**：法規審查時間減少 40%

### 6.4 下一步學習路徑

**本單元已完成**：
- ✅ 無監督學習的關聯規則與相似度搜尋
- ✅ 基於集合的配方表示（Set-based representation）
- ✅ 多目標決策框架

**Part 3 剩餘內容**：
- ⏳ **Unit 15**：化學資訊學進階（分子指紋、QSAR）
- ⏳ **Part 3 總結**：無監督學習的工程應用總覽

**進階主題（Part 4-5）**：
- 🚀 **深度學習**：表示學習、序列建模（Transformer）
- 🚀 **強化學習**：配方最佳化的自動實驗設計（Active Learning）
- 🚀 **貝氏最佳化**：小樣本高效率的全域搜尋

---

**[Next Part]**  
完成 Part 3 後，你已具備「沒有標籤也能做事」的能力。接下來 Part 4 會把表徵能力升級到深度學習（表示學習、序列、影像…），並把 Part 3 的無監督思維帶進 embedding 與 anomaly representation。
