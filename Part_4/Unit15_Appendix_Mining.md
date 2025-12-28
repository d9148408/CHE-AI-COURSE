# Unit15_Appendix | Mining Flotation Process：以 DNN 預測矽石濃度（% Silica Concentrate）

本附錄提供一個「礦業浮選過程品質預測」的真實工業資料案例，讓學生透過完整流程練習以 **DNN（MLP）模型**預測 **矽石濃度（% Silica Concentrate）**。資料來源與背景可參閱 Kaggle：`https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process`。

---

## 1. 案例背景：為什麼要預測矽石濃度？

### 1.1 浮選過程基礎理論（Froth Flotation）

浮選法（Froth Flotation）是礦物加工中最重要的分離技術之一，廣泛應用於鐵礦、銅礦、金礦等金屬礦物的富集過程。其基本原理是利用礦物表面物理化學性質的差異，特別是**疏水性（hydrophobicity）**的不同，在氣泡作用下實現有價礦物與脈石的分離。

#### 1.1.1 浮選基本原理

浮選過程涉及三相系統：固相（礦物顆粒）、液相（礦漿）、氣相（氣泡）。

**Young-Dupré 方程**描述了固-液-氣三相接觸角： 

$$
\gamma_{SG} = \gamma_{SL} + \gamma_{LG} \cos\theta
$$

其中：
- $\gamma_{SG}$ ：固-氣界面張力
- $\gamma_{SL}$ ：固-液界面張力
- $\gamma_{LG}$ ：液-氣界面張力
- $\theta$ ：接觸角

**浮選判據**：
- $\theta < 90^\circ$ ：親水性礦物，不易浮選
- $\theta > 90^\circ$ ：疏水性礦物，易於浮選

#### 1.1.2 浮選藥劑系統

浮選過程需要添加多種藥劑來調控礦物表面性質：

1. **捕收劑（Collector）**：
   - 作用：增強目標礦物的疏水性
   - 本案例使用：**胺類（Amina）**
   - 機制：選擇性吸附在礦物表面，形成疏水膜

2. **抑制劑（Depressant）**：
   - 作用：抑制脈石礦物浮選
   - 本案例使用：**澱粉（Starch）**
   - 機制：在脈石表面形成親水膜，防止其上浮

3. **pH 調整劑**：
   - 作用：控制礦漿 pH 值，影響藥劑效果
   - 影響：礦物表面電荷、藥劑解離度

#### 1.1.3 鐵礦反浮選工藝

本案例採用**反浮選（Reverse Flotation）**工藝：

- **目標**：浮選出脈石（SiO₂），鐵礦物留在底流
- **優勢**：適合處理高矽低品位鐵礦石
- **關鍵指標**：
  - **鐵精礦品位**：% Iron Concentrate（越高越好）
  - **矽石含量**：% Silica Concentrate（越低越好，本案例預測目標）

### 1.2 工業軟測器的重要性

在實際生產中，矽石濃度的測量面臨以下挑戰：

1. **分析延遲**：實驗室分析通常需要 1-2 小時
2. **採樣頻率低**：通常每小時採樣一次
3. **成本高昂**：需要專業分析設備與人員
4. **滯後控制**：無法即時調整操作參數

因此，建立 **資料驅動的軟測器（Soft Sensor）**可以：

- **即時預測**：每分鐘甚至每 20 秒更新預測值
- **提前預警**：在品質惡化前調整藥劑用量、pH 值等
- **降低成本**：減少實驗室分析頻率
- **環境效益**：降低矽石含量 → 減少尾礦排放 → 提高資源利用率

### 1.3 預測問題定義

本案例為 **監督式回歸問題** ：

$$
\hat{y} = f(\mathbf{x}), \quad \mathbf{x} \in \mathbb{R}^{22}, \ y \in \mathbb{R}
$$

其中：
- $y$ ：% Silica Concentrate（矽石濃度，目標變數）
- $\mathbf{x}$ ：22 維特徵向量（進料品質、藥劑流量、過程變數等）
- $f(\cdot)$ ：待學習的映射函數（本案例使用 DNN）

---

## 2. 資料集與欄位

### 2.1 資料來源

資料檔案位於本課程資料夾：

- 資料檔：`Part_4/data/mining/MiningProcess_Flotation_Plant_Database.csv`

**資料特性**：
- 時間範圍：2017 年 3 月至 9 月
- 總筆數：737,453 筆
- 採樣頻率：部分變數每 20 秒採樣，部分每小時採樣
- 格式：CSV 檔案，使用**逗號作為小數點**（歐洲格式）

### 2.2 欄位說明

資料共有 24 個欄位（1 個時間戳記 + 22 個特徵 + 1 個目標）：

#### 2.2.1 時間戳記

| 欄位 | 說明 |
|------|------|
| `date` | 日期時間（格式：YYYY-MM-DD HH:MM:SS）|

#### 2.2.2 進料品質（Feed Quality）

| 欄位 | 說明 | 單位 | 物理意義 |
|------|------|------|----------|
| `% Iron Feed` | 進料鐵品位 | % | 原礦中鐵的含量 |
| `% Silica Feed` | 進料矽石含量 | % | 原礦中 SiO₂ 的含量 |

這兩個變數是**最重要的前置指標**，直接反映原礦品質。

#### 2.2.3 藥劑流量（Reagent Flows）

| 欄位 | 說明 | 單位 | 作用 |
|------|------|------|------|
| `Starch Flow` | 澱粉流量 | m³/h | 抑制劑，防止脈石上浮 |
| `Amina Flow` | 胺類流量 | m³/h | 捕收劑，促進脈石上浮（反浮選）|

藥劑用量直接影響浮選效果，是**可控操作變數**。

#### 2.2.4 礦漿性質（Ore Pulp Properties）

| 欄位 | 說明 | 單位 | 物理意義 |
|------|------|------|----------|
| `Ore Pulp Flow` | 礦漿流量 | t/h | 處理量 |
| `Ore Pulp pH` | 礦漿 pH 值 | - | 影響藥劑效果與礦物表面電荷 |
| `Ore Pulp Density` | 礦漿濃度 | kg/cm³ | 固液比，影響碰撞機率 |

#### 2.2.5 浮選柱操作變數（Flotation Column Variables）

浮選柱共有 7 個（Column 01 ~ Column 07），每個柱有兩個監測變數：

| 變數類型 | 欄位範例 | 說明 | 單位 |
|---------|---------|------|------|
| **液位** | `Flotation Column 01 Level` | 浮選柱液位 | % |
| **氣流** | `Flotation Column 01 Air Flow` | 充氣量 | Nm³/h |

共 14 個欄位（7 柱 × 2 變數）。

**物理意義**：
- **液位**：影響礦漿停留時間與分離效率
- **氣流**：提供氣泡，影響礦化效果

#### 2.2.6 目標變數（Target）

| 欄位 | 角色 | 說明 | 單位 | 目標 |
|------|------|------|------|------|
| `% Iron Concentrate` | 輔助輸出 | 鐵精礦品位 | % | 越高越好 |
| `% Silica Concentrate` | **主要目標** | 矽石濃度（雜質） | % | **越低越好** |

**注意**：
- `% Iron Concentrate` 與 `% Silica Concentrate` 高度負相關
- 本案例以 `% Silica Concentrate` 為預測目標
- 實務上也可建立多目標模型同時預測兩者

### 2.3 資料格式注意事項

> [!IMPORTANT]
> 資料格式的特殊性
> 
> - CSV 使用**逗號 `,`** 作為小數點（歐洲格式）
> - 讀取時需指定 `decimal=','` 參數
> - 部分欄位可能包含缺失值

**資料讀取程式碼片段**：
```python
import pandas as pd

df = pd.read_csv('data/mining/MiningProcess_Flotation_Plant_Database.csv', 
                 decimal=',',  # 歐洲格式：逗號為小數點
                 parse_dates=['date'])
```

---

## 3. EDA 與特徵直覺

### 3.1 目標變數分佈分析

在進行建模前，先了解目標變數 `% Silica Concentrate` 的分佈特性。

#### 3.1.1 統計特性（Notebook 輸出）

![Target distribution](outputs/P4_Unit15_Example_Mining/figs/target_hist.png)

**基本統計量**：
- 平均值：約 2.0%
- 標準差：約 1.2%
- 最小值：約 0.5%
- 最大值：約 8.0%
- 分佈特徵：右偏（right-skewed），大部分樣本集中在 1-3% 範圍

**物理解釋**：
- 正常操作下，矽石濃度應控制在 2% 以下
- 高於 3% 表示品質異常，需要調整操作
- 極端值（> 5%）可能來自設備故障或原礦品質劇變

### 3.2 相關性分析（Correlation Analysis）

#### 3.2.1 Pearson 相關係數

使用 Pearson 相關係數快速檢查各特徵與目標的線性關聯：

$$
\rho_{X,Y} = \frac{\mathrm{cov}(X,Y)}{\sigma_X \sigma_Y}
= \frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}}
$$

其中 $\rho \in [-1,1]$ ；越接近 $1$ 表示正相關越強，越接近 $-1$ 表示負相關越強。

> [!WARNING]
> 相關性的限制
> 
> - Pearson 相關係數僅能捕捉**線性關係**
> - 相關不等於因果（correlation ≠ causation）
> - 可能存在非線性關係但相關係數為零
> - DNN 的優勢正是能捕捉這些非線性與交互作用

#### 3.2.2 執行結果分析（Notebook 輸出）

![Correlation with target](outputs/P4_Unit15_Example_Mining/figs/corr_bar.png)

**關鍵發現**（按相關性強度排序）：

從相關性分析中，我們觀察到以下重要模式：

**強相關特徵（ $|\rho| > 0.5$ ）**：

| 特徵 | 相關係數 | 物理意義與解釋 |
|------|---------|---------------|
| `% Silica Feed` | +0.90 | **原礦品質主導效應**：進料矽石含量直接決定精礦矽石含量。這反映了浮選過程的基本限制：即使工藝再優化，也無法完全消除原礦中的矽石。此強正相關驗證了「垃圾進、垃圾出」（GIGO）原則在礦物加工中的適用性。 |
| `% Iron Feed` | -0.55 | **品位互補關係**：高鐵品位原礦通常意味著低矽石含量（地質成因相關）。此負相關反映了鐵礦石的礦物學特性：磁鐵礦（Fe₃O₄）與石英（SiO₂）在成礦過程中的分離程度。 |

**中度相關特徵（ $0.2 < |\rho| < 0.5$ ）**：

| 特徵 | 相關係數 | 物理意義與解釋 |
|------|---------|---------------|
| `Ore Pulp Flow` | +0.35 | **處理量與效率的權衡**：礦漿流量增加會縮短停留時間，降低分離效率。這是典型的生產率-品質權衡（throughput-quality trade-off）。從浮選動力學角度，停留時間 $\tau$ 與回收率 $R$ 的關係可近似為： $R = 1 - e^{-k\tau}$ ，其中 k 為浮選速率常數。 |
| `Ore Pulp Density` | +0.28 | **固液比效應**：高礦漿濃度增加顆粒碰撞機率，但也可能導致藥劑分散不均。最佳濃度需在碰撞效率與流體動力學之間平衡。 |

**弱相關特徵（ $|\rho| < 0.2$ ）**：

| 特徵類型 | 代表特徵 | 相關係數範圍 | 重要性說明 |
|---------|---------|-------------|-----------|
| 藥劑流量 | `Starch Flow`, `Amina Flow` | ±0.10 ~ 0.15 | **非線性與閾值效應**：弱線性相關 ≠ 不重要。藥劑效果通常呈現 S 型曲線（Sigmoid），存在最佳用量區間。過低無效，過高飽和甚至反效果。此外，澱粉與胺類存在協同作用（synergistic effect），需透過非線性模型（如 DNN）捕捉。 |
| 浮選柱變數 | `Column XX Level/Air Flow` | < 0.15 | **局部優化變數**：這些變數對精細調控重要，但對整體品質的線性貢獻有限。其效果可能透過與其他變數的交互作用體現（例如：液位 × 氣流 → 氣泡尺寸分佈）。 |

#### 3.2.3 相關性熱力圖分析

![Correlation heatmap](outputs/P4_Unit15_Example_Mining/figs/corr_heatmap.png)

**熱力圖深度解讀**：

此圖展示了前 15 個最相關特徵之間的相關性矩陣。關鍵觀察：

1. **進料品質變數的強相關區塊**：
   - `% Silica Feed` 與 `% Iron Feed` 呈現強負相關（ $\rho \approx -0.85$ ）
   - 這反映了鐵礦石的礦物學特性：鐵品位與矽石含量通常呈反比關係
   - 地質意義：磁鐵礦富集區域通常脈石（石英）含量較低

2. **浮選柱變數的聚類現象**：
   - 不同浮選柱的液位（Level）之間呈現中度正相關（ $\rho \approx 0.4-0.6$ ）
   - 氣流（Air Flow）變數之間也有類似模式
   - 這表示浮選柱之間存在操作協調性，可能受到統一的控制策略影響

3. **藥劑流量的獨立性**：
   - `Starch Flow` 與 `Amina Flow` 之間相關性較低（ $\rho \approx 0.2$ ）
   - 這是合理的，因為兩種藥劑在反浮選中扮演不同角色
   - 但實際效果可能存在非線性交互（需要 DNN 捕捉）

4. **多重共線性（Multicollinearity）考量**：
   - 部分特徵間存在較高相關性（如不同柱的同類變數）
   - 對於線性模型（如 Ridge），這會導致係數不穩定
   - 但對於 DNN 和 Random Forest，影響較小（內建正則化機制）

**深入解讀**：

**為何藥劑流量線性相關弱，但實際很重要？**

這是浮選化學的典型特徵。藥劑效果遵循 **Langmuir 吸附等溫線** ：

$$
\Gamma = \Gamma_{\max} \frac{KC}{1 + KC}
$$

其中：
- $\Gamma$ ：表面吸附量
- $\Gamma_{\max}$ ：飽和吸附量
- $K$ ：吸附平衡常數
- $C$ ：藥劑濃度

此非線性關係導致：
- **低濃度區**：吸附量與濃度近似線性（ $\Gamma \approx \Gamma_{\max} KC$ ）
- **高濃度區**：吸附飽和，增加用量無效（ $\Gamma \approx \Gamma_{\max}$ ）
- **最佳區間**：通常在曲線拐點附近

因此，Pearson 相關係數（線性指標）無法充分反映藥劑的實際重要性。

**為何 DNN 可能優於線性模型？**

從相關性分析可預期：
1. **非線性關係豐富**：藥劑效果、流量-效率權衡等
2. **交互作用複雜**：pH × 藥劑、液位 × 氣流等
3. **閾值效應明顯**：藥劑用量、處理量等存在最佳區間

這些特性正是 DNN 的優勢所在，我們將在後續建模中驗證。

### 3.3 缺失值分析

> [!NOTE]
> 缺失值處理策略
> 
> 本資料集存在缺失值，主要集中在：
> - 實驗室分析結果（`% Iron Concentrate`, `% Silica Concentrate`）
> - 部分浮選柱變數
> 
> **處理方法**：
> 1. **刪除法**：刪除目標變數缺失的樣本（無法用於訓練）
> 2. **插補法**：對特徵缺失值進行插補（前向填充、均值插補等）
> 3. **時序對齊**：將高頻變數（20秒）與低頻變數（1小時）對齊

---

## 4. 建模流程：切分、清理、標準化

### 4.1 資料預處理策略

#### 4.1.1 缺失值處理

**步驟**：
1. 刪除目標變數 `% Silica Concentrate` 缺失的樣本
2. 對特徵變數的缺失值進行前向填充（Forward Fill）
3. 若仍有缺失，使用訓練集均值插補

#### 4.1.2 資料切分

**切分比例**：70% / 15% / 15%（Train / Valid / Test）

```python
from sklearn.model_selection import train_test_split

# 第一次切分：分離 test set (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# 第二次切分：從剩餘 85% 中分離 valid set (約 15%)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42
)
```

> [!CAUTION]
> Data Leakage 的風險
> 
> **絕對不可**在切分前進行以下操作：
> - 標準化（會洩漏 test set 的統計資訊）
> - 特徵選擇（會根據全部資料選特徵）
> - 任何基於全資料集的轉換

#### 4.1.3 標準化（Standardization）

神經網路對特徵尺度敏感，標準化可加速收斂並提升性能。

**Z-score 標準化**：

$$
x' = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}
$$

**重要原則**：
$$
\mu_{\text{train}}, \sigma_{\text{train}} \text{ 必須只由訓練集估計}
$$

**X 特徵標準化**：

```python
from sklearn.preprocessing import StandardScaler

# 只在訓練集上 fit
scaler_X = StandardScaler()
scaler_X.fit(X_train)

# Transform 所有資料集
X_train_scaled = scaler_X.transform(X_train)
X_valid_scaled = scaler_X.transform(X_valid)
X_test_scaled = scaler_X.transform(X_test)
```

**y 目標變數標準化**：

對於回歸問題，標準化目標變數 y 也是常見做法，特別是當：
- 目標變數的尺度與特徵差異較大
- 使用 MSE 作為損失函數時，標準化可以讓梯度更穩定
- 有助於神經網路的數值穩定性

```python
# y 標準化
scaler_y = StandardScaler()
scaler_y.fit(y_train.values.reshape(-1, 1))

y_train_scaled = scaler_y.transform(y_train.values.reshape(-1, 1)).flatten()
y_valid_scaled = scaler_y.transform(y_valid.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
```

> [!IMPORTANT]
> 預測後需要反標準化
> 
> 使用標準化的 y 訓練模型後，預測結果也會是標準化的值，需要使用 `scaler_y.inverse_transform()` 轉換回原始尺度：
> 
> ```python
> # 預測（標準化尺度）
> y_pred_scaled = model.predict(X_test_scaled)
> 
> # 反標準化（轉回原始尺度）
> y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
> ```

---

## 5. Baseline 模型：Random Forest Regressor

### 5.1 為什麼選擇 Random Forest 作為 Baseline？

相較於線性模型（如 Ridge Regression），Random Forest 具有以下優勢：

1. **非線性建模能力**：可捕捉特徵間的非線性關係
2. **特徵交互作用**：自動學習特徵組合（例如 pH × 藥劑流量）
3. **魯棒性強**：對異常值與缺失值不敏感
4. **無需標準化**：基於樹的模型對特徵尺度不敏感
5. **特徵重要性**：可直接輸出特徵重要性排序

**為何不用 Ridge？**
- 本案例特徵間存在明顯的非線性與交互作用（藥劑配比、pH 效應等）
- Ridge 的線性假設過於簡化，可能無法充分反映 DNN 的優勢
- Random Forest 作為「強 baseline」更能凸顯 DNN 的價值

### 5.2 Random Forest 理論

#### 5.2.1 決策樹回歸（Decision Tree Regression）

單棵決策樹透過遞迴分割特徵空間，最小化節點內的變異數：

**分割準則（MSE Reduction）**：

$$
\text{Gain} = \text{MSE}_{\text{parent}} - \left(\frac{n_{\text{left}}}{n}\text{MSE}_{\text{left}} + \frac{n_{\text{right}}}{n}\text{MSE}_{\text{right}}\right)
$$

**預測值**：葉節點內樣本的平均值

$$
\hat{y}_{\text{leaf}} = \frac{1}{n_{\text{leaf}}}\sum_{i \in \text{leaf}} y_i
$$

#### 5.2.2 Random Forest 集成（Ensemble）

Random Forest 透過 **Bagging（Bootstrap Aggregating）** 與 **隨機特徵選擇** 構建多棵不相關的樹：

**演算法流程**：
1. 從訓練集中有放回抽樣（Bootstrap）生成 $B$ 個子集
2. 對每個子集訓練一棵決策樹
3. 每次分割時，隨機選擇 $m$ 個特徵（$m \ll d$）
4. 最終預測為所有樹的平均值：

$$
\hat{y}_{\text{RF}} = \frac{1}{B}\sum_{b=1}^{B} \hat{y}_b(\mathbf{x})
$$

**優勢**：
- **降低變異數**：多棵樹的平均降低過擬合風險
- **提升泛化**：隨機性使樹之間不相關
- **穩定性**：對資料擾動不敏感

### 5.3 執行結果（Notebook 輸出）

**模型配置**：
- `n_estimators=100`（樹的數量）
- `max_depth=20`（最大深度，防止過擬合）
- `min_samples_split=10`（最小分割樣本數）
- `random_state=42`（可重現性）

**訓練集大小**：
- Training: 515,417 樣本
- Validation: 110,518 樣本  
- Test: 110,518 樣本

#### 5.3.1 性能指標詳細分析

**Test Set 性能**：

| 資料集 | MAE | RMSE | R² |
|--------|-----|------|----|
| **Training** | 0.088 | 0.148 | 0.966 |
| **Validation** | 0.220 | 0.348 | 0.905 |
| **Test** | 0.221 | 0.349 | 0.904 |

**關鍵觀察**：

1. **訓練集與測試集的性能差距**：
   - Training MAE (0.088) vs Test MAE (0.221)：差距約 2.5 倍
   - Training R² (0.966) vs Test R² (0.904)：下降 6.2%
   - **解釋**：Random Forest 在訓練集上有輕微過擬合傾向，這是樹模型的常見特性
   - 但 Validation 與 Test 性能幾乎一致，表示模型泛化能力穩定

2. **R² = 0.904 的意義**：
   - 模型可解釋目標變異的 **90.4%**
   - 這是相當優秀的表現，表示 Random Forest 成功捕捉了主要的非線性關係
   - 剩餘 9.6% 的未解釋變異可能來自：
     - 測量誤差（實驗室分析精度限制）
     - 未觀測變數（如礦石礦物學特性、設備磨損狀態等）
     - 隨機擾動（操作波動、環境因素等）

3. **MAE = 0.221% 的實務意義**：
   - 平均預測誤差約 0.22 個百分點
   - 考慮到目標變數範圍（0.5% ~ 8.0%），相對誤差約為 **11%** （以平均值 2.0% 計算）
   - 在工業應用中，這個精度足以支持：
     - 品質監控與預警（趨勢判斷）
     - 操作優化建議（藥劑調整方向）
     - 但可能不足以完全取代實驗室分析（需要更高精度）

4. **RMSE vs MAE 的比較**：
   - RMSE (0.349) / MAE (0.221) ≈ 1.58
   - 理想情況下（誤差呈常態分佈），此比值應接近 $\sqrt{\pi/2} \approx 1.25$ 
   - 實際比值較高，表示存在一些 **較大的預測誤差** （離群值）
   - 這可能對應於：
     - 原礦品質劇變時的過渡期
     - 設備異常或操作失誤
     - 極端操作條件（超出訓練集範圍）

#### 5.3.2 Random Forest 的優勢與限制

**在本案例中的優勢**：

1. **自動捕捉非線性**：
   - 成功建模藥劑效果的非線性（Langmuir 吸附曲線）
   - 捕捉流量-效率的權衡關係

2. **特徵交互作用**：
   - 自動學習 pH × 藥劑濃度的協同效應
   - 發現液位 × 氣流對氣泡尺寸的影響

3. **魯棒性**：
   - 對異常值不敏感（基於分位數分割）
   - 對缺失值有一定容忍度

**限制**：

1. **外推能力有限**：
   - 樹模型無法預測超出訓練集範圍的值
   - 對於新的操作條件（如新的藥劑配比），預測可能不可靠

2. **高維交互的複雜度**：
   - 雖然可以學習特徵交互，但對於高階交互（三個以上特徵的組合）效率較低
   - DNN 在這方面可能有優勢

3. **平滑性不足**：
   - 預測函數是分段常數（piecewise constant）
   - 在決策邊界處可能出現突變
   - DNN 的連續激活函數提供更平滑的預測

**為何選擇 Random Forest 作為 Baseline？**

相較於簡單的線性模型（如 Ridge Regression），Random Forest 提供了更具挑戰性的基準：

- **如果 DNN 無法超越 Random Forest**：表示資料的非線性已被樹模型充分捕捉，DNN 的額外複雜度不值得
- **如果 DNN 顯著超越 Random Forest**：表示存在 Random Forest 難以捕捉的模式（如高階交互、平滑非線性等），證明 DNN 的價值

這種「強 baseline」策略在工業 AI 中非常重要，避免過度複雜化模型。

---

## 6. DNN（MLP）回歸：理論與實作

### 6.1 MLP 的數學形式（前向傳播）

對於 $L$ 層的 MLP（最後一層為線性輸出），可寫成：

$$
\begin{aligned}
\mathbf{h}^{(0)} &= \mathbf{x} \\
\mathbf{z}^{(l)} &= \mathbf{W}^{(l)}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)} \\
\mathbf{h}^{(l)} &= \phi(\mathbf{z}^{(l)}) \quad \text{for } l=1,\ldots,L-1 \\
\hat{y} &= \mathbf{W}^{(L)}\mathbf{h}^{(L-1)} + \mathbf{b}^{(L)}
\end{aligned}
$$

其中 $\phi(\cdot)$ 常用 **ReLU**：

$$
\mathrm{ReLU}(z)=\max(0,z)
$$

### 6.2 損失函數與評估指標

**訓練損失**：均方誤差（MSE）

$$
\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

**評估指標**：

**MAE（平均絕對誤差）**

$$
\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

**RMSE（均方根誤差）**

$$
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

**$R^2$（決定係數）**

$$
R^2 = 1-\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}
$$

### 6.3 反向傳播與優化

#### 6.3.1 反向傳播（Backpropagation）

透過鏈式法則計算梯度：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \delta^{(l)} (\mathbf{h}^{(l-1)})^{\top}
$$

其中 $\delta^{(l)}$ 為第 $l$ 層的誤差項。

#### 6.3.2 Adam 優化器

Adam（Adaptive Moment Estimation）結合 Momentum 與 RMSprop：

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t \\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2 \\
\hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1-\beta_1^t} \\
\hat{\mathbf{v}}_t &= \frac{\mathbf{v}_t}{1-\beta_2^t} \\
\mathbf{w}_{t+1} &= \mathbf{w}_t - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
\end{aligned}
$$

**超參數**：
- $\beta_1 = 0.9$（一階動量）
- $\beta_2 = 0.999$（二階動量）
- $\eta$：學習率（learning rate）
- $\epsilon = 10^{-8}$（數值穩定項）

### 6.4 正則化技術

#### 6.4.1 Dropout

訓練時隨機丟棄神經元：

$$
\mathbf{h}_{\text{dropout}} = \mathbf{h} \odot \mathbf{m}, \quad m_i \sim \text{Bernoulli}(1-p)
$$

直覺：迫使網路不要過度依賴少數神經元，提高泛化能力。

#### 6.4.2 Early Stopping

監控驗證集損失，防止過擬合：

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    save_model()
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= patience:
        restore_best_model()
        break
```

#### 6.4.3 Learning Rate Scheduling

**ReduceLROnPlateau**：當驗證損失停止改善時降低學習率

```python
if val_loss_improvement < threshold:
    patience_counter += 1
    if patience_counter >= patience:
        lr = lr * factor
```

### 6.5 模型架構設計

**基礎架構**（Baseline DNN）：

```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(n_features,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)  # 輸出層（線性激活）
])
```

**訓練配置**：
- Optimizer: Adam (lr=0.001)
- Loss: MSE
- Batch size: 256
- Epochs: 200 (with early stopping)
- Callbacks: EarlyStopping, ReduceLROnPlateau

### 6.6 執行結果（Notebook 輸出）

#### 6.6.1 訓練曲線分析

![Training curves](outputs/P4_Unit15_Example_Mining/figs/loss_curve.png)

**訓練配置與過程**：
- 總訓練 epochs：實際執行約 80-100 epochs（Early Stopping 觸發）
- Batch size：2048（大批次以加速訓練）
- 學習率：初始 0.0005，透過 ReduceLROnPlateau 動態調整
- 損失函數：Huber Loss（對離群值更穩健）

**訓練曲線深度解讀**：

1. **收斂行為**：
   - **初期（0-20 epochs）**：
     - Train 和 Valid loss 快速下降
     - 這是梯度最陡峭的階段，模型快速學習主要模式
     - 學習率保持在初始值 0.0005
   
   - **中期（20-60 epochs）**：
     - 下降速度放緩，進入精細調整階段
     - Valid loss 開始出現小幅波動
     - ReduceLROnPlateau 可能觸發，學習率降低至 0.00025
   
   - **後期（60+ epochs）**：
     - Train loss 持續緩慢下降
     - Valid loss 趨於平穩或輕微上升
     - Early Stopping 監控機制準備觸發

2. **過擬合診斷**：
   - **Train vs Valid gap**：
     - 訓練後期，Train loss 持續下降但 Valid loss 不再改善
     - 這是輕微過擬合的跡象，但 Early Stopping 有效控制
   - **最佳 epoch 選擇**：
     - Early Stopping 會恢復到 Valid loss 最低的 epoch
     - 通常在第 60-80 epoch 之間

3. **MAE 曲線特性**：
   - MAE 曲線比 Loss 曲線更直觀（單位為 %）
   - 最終 Valid MAE 約 0.22%，與 Random Forest 相當
   - Train MAE 約 0.15%，顯示模型在訓練集上有更好的擬合

**與 Random Forest 訓練過程的對比**：

| 特性 | Random Forest | DNN |
|------|--------------|-----|
| 訓練時間 | ~5 分鐘（100 棵樹） | ~15 分鐘（80 epochs）|
| 收斂曲線 | 無（一次性訓練） | 漸進式收斂 |
| 過擬合風險 | 中等（透過樹深度控制） | 較高（需 Early Stopping）|
| 可解釋性 | 高（特徵重要性直觀） | 低（黑盒模型） |

#### 6.6.2 性能比較：Random Forest vs DNN（全數據集分析）

**完整性能比較表**：

為了全面評估兩個模型的表現，我們在所有數據集上進行了完整的性能評估：

| 資料集 | 模型 | MAE | RMSE | R² |
|--------|------|-----|------|----|
| **Training** | Random Forest | 0.171 | 0.304 | 0.927 |
| **Training** | DNN | 0.180 | 0.273 | 0.941 |
| **Validation** | Random Forest | 0.197 | 0.348 | 0.904 |
| **Validation** | DNN | 0.222 | 0.350 | 0.903 |
| **Test** | Random Forest | **0.198** | **0.349** | **0.904** |
| **Test** | DNN | 0.221 | 0.348 | 0.904 |
| **All Data** | Random Forest | 0.179 | 0.318 | 0.920 |
| **All Data** | DNN | 0.192 | 0.298 | 0.930 |

![Model Comparison - All Datasets](outputs/P4_Unit15_Example_Mining/figs/model_comparison_all_datasets.png)

**Test Set 詳細比較**： 

$$
\text{MAE Improvement} = \frac{0.198 - 0.221}{0.198} \times 100\% = -11.8\%
$$

$$
\text{RMSE Improvement} = \frac{0.349 - 0.348}{0.349} \times 100\% = +0.2\%
$$

$$
\Delta R^2 = 0.904 - 0.904 = 0.000
$$

**關鍵發現與深度分析**：

1. **Random Forest 在大部分數據集上優於 DNN**：
   
   從上表可以清楚看到：
   - **Training Set**: DNN 的 MAE (0.180) 略優於 RF (0.171)，差距約 5%
     - 但 DNN 的 R² (0.941) 顯著優於 RF (0.927)
     - 這表示 DNN 在訓練集上有更好的擬合能力
   - **Validation Set**: RF 的 MAE (0.197) 優於 DNN (0.222)，差距約 13%
   - **Test Set**: RF 的 MAE (0.198) 優於 DNN (0.221)，差距約 12%
   - **All Data**: DNN 的 MAE (0.192) 略優於 RF (0.179)，差距約 7%
     - 但 DNN 的 R² (0.930) 優於 RF (0.920)
   
   **重要觀察**：
   - 在 **Test Set**（最重要的評估指標）上，RF 明顯優於 DNN
   - DNN 在 Training 和 All Data 上表現較好，但這可能包含了訓練集的過擬合效應
   - **結論**：從泛化能力角度（Validation 和 Test），RF 更勝一籌

2. **為何 Random Forest 在 Test Set 表現更好？**

   從數據分析可以得出以下結論：
   
   a) **數據特性適合樹模型**：
      - 本案例的特徵間交互作用主要是**低階的**（二階、三階）
      - Random Forest 透過遞迴分割已經能夠充分捕捉這些模式
      - DNN 的深層非線性能力在此案例中優勢不明顯
   
   b) **特徵工程已經非常充分**：
      - 原始特徵（% Silica Feed, % Iron Feed, Ore Pulp Flow 等）已經高度相關於目標
      - 這些特徵是基於領域知識選擇的，具有明確的物理意義
      - 不需要 DNN 的自動特徵提取能力
      - **對比**：在圖像識別中，原始像素與目標的關係不明顯，需要 DNN 提取高階特徵
   
   c) **模型複雜度與數據維度的平衡**：
      - 雖然有 70 萬筆數據，但**有效特徵只有 22 個**
      - DNN 的大量參數（數千個）相對於特徵數過多
      - 容易導致過擬合到訓練集的特定模式
      - Random Forest 的參數量（樹的數量 × 樹的深度）更適合此數據規模
   
   d) **訓練集與測試集性能的對比**：
      - DNN 在訓練集上的 R² (0.941) 優於 RF (0.927)
      - 但在測試集上，RF 的 MAE (0.198) 優於 DNN (0.221)
      - 這表示 DNN 可能過度擬合了訓練集的某些模式
      - RF 的泛化能力更穩健

3. **不同評估指標的矛盾現象：MAE vs RMSE vs R²**：
   
   這是本案例最有趣的發現之一：**不同指標給出不同的結論**！
   
   **Test Set 上的指標對比**：
   - MAE: RF (0.198) **優於** DNN (0.221) ← 差距 11.8%
   - RMSE: RF (0.349) vs DNN (0.348) ← 幾乎相同
   - R²: RF (0.904) vs DNN (0.904) ← 完全相同
   
   **All Data 上的指標對比**：
   - MAE: RF (0.179) **優於** DNN (0.192) ← 差距 7.3%
   - RMSE: RF (0.318) vs DNN (0.298) ← **DNN 優於 RF**（差距 6.3%）
   - R²: RF (0.920) vs DNN (0.930) ← **DNN 優於 RF**（差距 1.1%）
   
   **為何會出現這種矛盾？**
   
   這涉及三個指標的數學特性：
   
    a) **MAE（Mean Absolute Error）** ：

    $$
    \text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
    $$

    - **特性**：對所有誤差一視同仁（線性懲罰）
    - **敏感於**：中位數誤差、整體誤差分佈
    - **物理意義**：平均預測偏離真值多少（單位：%）
   
    b) **RMSE（Root Mean Squared Error）** ：

    $$
    \text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
    $$

    - **特性**：對大誤差更敏感（平方懲罰）
    - **敏感於**：離群值、極端誤差
    - **物理意義**：誤差的標準差（單位：%）
   
    c) **R²（決定係數）** ：

    $$
    R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
    $$

    - **特性**：基於平方誤差，與 RMSE 高度相關
    - **敏感於**：變異的解釋比例
    - **物理意義**：模型解釋了多少目標變數的變異
   
   **數學關係**：
   - RMSE 和 R² 都基於**平方誤差**，因此高度相關
   - MAE 基於**絕對誤差**，與前兩者可能不一致
   - 當誤差分佈不均勻時（有離群值），三者會給出不同結論
   
   **本案例的解釋**：
   
   | 指標 | RF 表現 | DNN 表現 | 勝者 | 原因分析 |
   |------|---------|----------|------|----------|
   | **MAE** | 0.198 | 0.221 | **RF** | RF 在**大部分樣本**上預測更準確 |
   | **RMSE** | 0.349 | 0.348 | **DNN** | DNN 在**極端值/離群值**上控制更好 |
   | **R²** | 0.904 | 0.904 | 平手 | 兩者解釋變異的能力相同 |
   
   **深入分析**：
   
   1. **RF 的優勢（MAE 更低）**：
      - RF 在常見操作範圍（1-3% 矽石濃度）預測更準確
      - 這對日常監控更重要
      - 中位數誤差更小
   
   2. **DNN 的優勢（RMSE 更低）**：
      - DNN 在異常情況（高矽石濃度 > 4%）的預測誤差更小
      - 避免了極端的預測失誤
      - 這對異常預警可能更重要
   
   3. **RMSE/MAE 比值分析**：
      - RF: RMSE/MAE = 0.349/0.198 = **1.76**
      - DNN: RMSE/MAE = 0.348/0.221 = **1.57**
      - 理想常態分佈：RMSE/MAE ≈ 1.25
      - **解釋**：
        - RF 的比值更高 → 存在更多大誤差（離群值）
        - DNN 的比值較低 → 誤差分佈更均勻
        - **結論**：DNN 的預測更穩定，RF 偶爾會有較大失誤
   
   **視覺化理解**：
   
   想像誤差分佈：
   - **RF**：大部分誤差很小（0.1-0.2%），但偶爾有大誤差（0.5-1.0%）
     - → MAE 低（受小誤差主導）
     - → RMSE 高（被大誤差拉高）
   
   - **DNN**：誤差分佈更均勻（0.15-0.35%），極少極端誤差
     - → MAE 略高（整體誤差稍大）
     - → RMSE 低（沒有極端值）
   
   **實務選擇建議**：
   
   | 應用場景 | 推薦模型 | 原因 |
   |---------|---------|------|
   | **日常監控** | RF | MAE 更低，常見情況預測更準 |
   | **異常預警** | DNN | RMSE 更低，極端情況控制更好 |
   | **綜合應用** | **兩者結合** | 利用各自優勢（ensemble） |
   
   **教學啟示**：
   
   > [!IMPORTANT]
   > **評估指標的選擇至關重要**
   > 
   > 1. **沒有絕對的"最佳模型"**：取決於評估指標和應用場景
   > 2. **MAE vs RMSE 的權衡**：
   >    - 重視常見情況 → 選 MAE
   >    - 重視極端情況 → 選 RMSE
   > 3. **多指標綜合評估**：不要只看單一指標
   > 4. **領域知識指導**：
   >    - 在礦業中，避免極端失誤（高矽石）可能更重要 → DNN 有價值
   >    - 但日常操作的準確性也很重要 → RF 也有優勢
   
   **修正後的結論**：
   
   - **RF 優勢**：日常預測更準確（MAE 低）
   - **DNN 優勢**：極端情況控制更好（RMSE 低）、誤差分佈更均勻
   - **最佳策略**：根據實際需求選擇，或使用 **Ensemble（集成）** 結合兩者優勢

4. **R² 相同但 MAE/RMSE 不同的現象**：
   
   有趣的是，在 Test Set 上：
   - 兩個模型的 R² 都是 0.904（完全相同）
   - 但 MAE 有明顯差異（0.198 vs 0.221）
   
   **解釋**：
   - R² 衡量的是**變異的解釋比例**，對大誤差更敏感（因為是平方項）
   - MAE 衡量的是**平均絕對誤差**，對所有誤差一視同仁
   - 這表示兩個模型在**大誤差**上表現相似，但 DNN 在**小誤差**上表現較差
   - 換句話說：DNN 能捕捉主要趨勢，但細節預測不如 RF

5. **過擬合與泛化能力分析**：
   
   | 模型 | Train MAE | Test MAE | Gap | 泛化能力評估 |
   |------|-----------|----------|-----|-------------|
   | Random Forest | 0.171 | 0.198 | 0.027 | 優秀（極小過擬合）|
   | DNN | 0.180 | 0.221 | 0.041 | 良好（輕微過擬合）|
   
   **觀察**：
   - RF 的 Train-Test gap (0.027) 小於 DNN (0.041)
   - 這表示 RF 的泛化能力實際上**優於** DNN
   - DNN 雖然在訓練集上的 R² 較高，但測試集 MAE 較差
   - **結論**：RF 在訓練集和測試集上都保持了一致且優秀的性能

6. **實務意義與模型選擇**：
   
   **多指標綜合評估**：
   
   | 考量因素 | Random Forest | DNN | 分析 |
   |---------|--------------|-----|------|
   | 預測精度 (Test MAE) | ★★★★★ (0.198) | ★★★☆☆ (0.221) | **RF 勝** |
   | 極端值控制 (Test RMSE) | ★★★★☆ (0.349) | ★★★★★ (0.348) | **DNN 略勝** |
   | 變異解釋 (Test R²) | ★★★★★ (0.904) | ★★★★★ (0.904) | 平手 |
   | 訓練時間 | ★★★★★ (~5 min) | ★★★☆☆ (~15 min) | RF 勝 |
   | 推論速度 | ★★★★☆ | ★★★★★ | DNN 略勝 |
   | 可解釋性 | ★★★★★ | ★★☆☆☆ | RF 大勝 |
   | 維護成本 | ★★★★☆ | ★★★☆☆ | RF 勝 |
   | 泛化能力 (Train-Test gap) | ★★★★★ (0.027) | ★★★★☆ (0.041) | RF 勝 |
   | **綜合評分** | **★★★★★** | **★★★★☆** | **RF 略勝** |
   
   **建議策略**：
   
   1. **單一模型部署**：
      - 若重視日常監控準確性 → 選擇 **Random Forest**
      - 若重視異常情況控制 → 選擇 **DNN**
      - 若資源有限、需要可解釋性 → 選擇 **Random Forest**
   
   2. **Ensemble（集成）策略**（推薦）：
      ```python
      # 結合兩者優勢
      y_pred_ensemble = 0.6 * y_pred_rf + 0.4 * y_pred_dnn
      ```
      - 利用 RF 的日常準確性
      - 利用 DNN 的極端值控制
      - 可能獲得最佳的綜合性能
   
   3. **情境切換策略**：
      - 正常操作（矽石 < 3%）：使用 RF 預測
      - 異常情況（矽石 > 3%）：使用 DNN 預測
      - 根據實時數據動態選擇模型

7. **何時應選擇 DNN？**
   
   本案例清楚展示了 DNN 的**不適用場景**。DNN 更適合：
   
   - **高維原始數據**：圖像（百萬像素）、文本（詞嵌入）、音頻（波形）
   - **複雜非線性關係**：需要深層抽象的任務（如物體識別、語言理解）
   - **端到端學習**：從原始感測器數據直接預測，無需手工特徵工程
   - **大規模數據**：數百萬筆以上，且特徵維度也很高
   - **遷移學習**：可利用預訓練模型（在礦業中較少見）
   
   **本案例的特點**：
   - ✗ 低維特徵（22 個）
   - ✗ 已有良好的特徵工程
   - ✗ 低階交互為主
   - ✓ 大量數據（70 萬筆）
   - **結論**：3 個不利因素 vs 1 個有利因素 → 不適合 DNN

7. **教學價值與批判性思維**：
   
   本案例的**最大價值**在於：
   
   > [!IMPORTANT]
   > **模型選擇的科學方法論**
   > 
   > 1. **不盲目追求複雜模型**：DNN 不是萬能的，在表格數據上常常不如 RF/XGBoost
   > 2. **實證比較是關鍵**：透過實驗驗證假設，而非基於流行趨勢選擇模型
   > 3. **理解數據特性**：低維、低階交互、良好特徵工程 → 選樹模型
   > 4. **綜合考量多個因素**：精度、訓練成本、可解釋性、維護性
   > 5. **領域知識的重要性**：理解浮選化學幫助解釋為何 RF 更好
   
   這正是工業 AI 的核心理念：**實用主義優於技術炫耀，簡單有效優於複雜花俏**。

8. **如果一定要使用 DNN，如何改進？**
   
   雖然不建議在此案例使用 DNN，但若有特殊需求（如需要端到端學習），可嘗試：
   
   a) **架構調整**：
      - 使用更淺的網路（2-3 層）
      - 減少每層神經元數（32-64）
      - 降低 Dropout 比例（0.1-0.2）
   
   b) **訓練策略**：
      - 增加訓練 epochs
      - 使用更小的學習率（0.0001）
      - 嘗試不同的優化器（SGD with momentum）
   
   c) **特徵工程**：
      - 加入特徵交互項（如 pH × Starch Flow）
      - 使用多項式特徵
      - 標準化後再進行 PCA 降維
   
   d) **集成方法**：
      - 訓練多個 DNN 並平均預測
      - 或結合 RF 和 DNN 的預測（stacking）
   
   但即使如此，也很難超越 Random Forest 的性能。

5. **教學價值**：
   
   本案例的重要教訓：
   
   > [!IMPORTANT]
   > **模型選擇的科學方法**
   > 
   > 1. **先建立強 baseline**：Random Forest 已經很強，避免盲目追求複雜模型
   > 2. **實證比較**：透過實驗驗證 DNN 是否真的更好，而非假設
   > 3. **綜合考量**：不只看精度，還要考慮訓練成本、可解釋性、維護性
   > 4. **領域知識**：理解數據特性（低階交互為主）幫助選擇合適模型
   
   這正是工業 AI 的核心理念：**實用主義優於技術炫耀**。

---

## 7. 模型診斷與視覺化分析

### 7.1 Parity Plot（真值 vs 預測值）

![Parity plot](outputs/P4_Unit15_Example_Mining/figs/parity_plot.png)

**圖表解讀**：

Parity plot（也稱為 scatter plot of predicted vs actual）是回歸模型最直觀的診斷工具。理想情況下，所有點應落在 45° 對角線（ $y = \hat{y}$ ）上。

**Random Forest（左圖）觀察**：

1. **整體分佈**：
   - 點雲緊密圍繞對角線分佈
   - R² = 0.904 的視覺體現：大部分點接近完美預測線
   
2. **低濃度區（< 2%）**：
   - 預測非常準確，點幾乎完全貼合對角線
   - 這是數據最密集的區域（正常操作範圍）
   - 模型在此區間的訓練樣本最多，學習最充分
   
3. **高濃度區（> 4%）**：
   - 出現更大的散布
   - 部分點偏離對角線較遠
   - **原因**：
     - 樣本稀少（異常操作條件）
     - 測量誤差可能更大（高濃度時分析難度增加）
     - 模型外推能力有限（超出主要訓練範圍）

4. **系統性偏差檢查**：
   - 無明顯的「整體偏高」或「整體偏低」趨勢
   - 點雲對稱分佈於對角線兩側
   - 表示模型無系統性偏差（unbiased）

**DNN（右圖）觀察**：

1. **與 Random Forest 的對比**：
   - 視覺上幾乎無法區分兩個模型的 parity plot
   - 這進一步證實了兩者性能相當的結論
   
2. **細微差異**：
   - DNN 的點雲可能略微更平滑（連續激活函數的效果）
   - 但差異極小，不具統計顯著性

**實務應用建議**：

基於 parity plot 的觀察，建議：

- **正常操作範圍（< 3%）**：模型預測可信度高，可用於即時監控
- **異常範圍（> 4%）**：預測僅供參考，應結合實驗室分析
- **預警閾值設定**：建議設在 3%，超過此值應觸發人工檢查

### 7.2 Residual Plot（殘差分析）

![Residual plot](outputs/P4_Unit15_Example_Mining/figs/residual_plot.png)

**殘差定義與意義**：

$$
r_i = y_i - \hat{y}_i
$$

殘差圖用於診斷模型假設的違反情況。理想的殘差圖應呈現：
- **隨機性**：無明顯模式或趨勢
- **零中心**：殘差圍繞 0 線對稱分佈
- **同方差性**：殘差的散布程度在不同預測值下保持一致

**Random Forest（左圖）診斷**：

1. **隨機性檢驗**：
   - ✓ 殘差無明顯的曲線型或分段型模式
   - ✓ 表示模型已充分捕捉非線性關係
   - 若出現曲線型，表示缺少某種非線性項

2. **零中心性**：
   - ✓ 殘差大致對稱分佈於 y=0 線兩側
   - ✓ 無系統性高估或低估
   - 紅色虛線（y=0）穿過點雲中心

3. **同方差性（Homoscedasticity）**：
   - ✓ 殘差的垂直散布在不同預測值下大致相同
   - 無明顯的「漏斗形」（heteroscedasticity）
   - **意義**：模型的預測不確定性在不同濃度範圍下一致

4. **離群值識別**：
   - 存在少數殘差 > ±1.5% 的點
   - 這些可能對應於：
     - 設備故障時刻
     - 原礦品質突變
     - 測量誤差
   - **建議**：在實際部署時，可設定離群值檢測機制

**DNN（右圖）診斷**：

1. **與 Random Forest 的對比**：
   - 殘差分佈模式幾乎一致
   - 再次證實兩模型捕捉到相同的數據結構
   
2. **細微差異**：
   - DNN 的殘差可能略微更均勻分佈
   - 但差異不足以構成選擇依據

**統計檢驗（理論補充）** ：

可進一步進行以下統計檢驗：

1. **Shapiro-Wilk 常態性檢驗** ：

$$
H_0: \text{殘差服從常態分佈}
$$

若 p-value > 0.05，則殘差近似常態，模型假設合理。

2. **Breusch-Pagan 同方差性檢驗** ：

$$
H_0: \text{殘差變異數為常數}
$$

用於正式檢驗是否存在異方差性。

3. **Durbin-Watson 自相關檢驗** （若為時序數據）：

$$
DW = \frac{\sum_{t=2}^{n}(r_t - r_{t-1})^2}{\sum_{t=1}^{n}r_t^2}
$$

檢驗殘差是否存在時間相關性。

### 7.3 預測分佈比較

![Distribution comparison](outputs/P4_Unit15_Example_Mining/figs/test_true_vs_pred_dist.png)

**分佈匹配度分析**：

此圖比較真實值（藍色）與預測值（橙色）的機率分佈。良好的模型應使兩個分佈高度重疊。

**Random Forest（左圖）**：

1. **峰值位置（Mode）**：
   - 真實值與預測值的峰值都在 1.5-2.0% 範圍
   - ✓ 峰值高度相近，表示模型正確捕捉了最常見的操作狀態

2. **分佈形狀**：
   - 兩個分佈都呈現右偏（right-skewed）
   - ✓ 模型成功複製了目標變數的偏態特性
   - 這對於生成式任務（如模擬）很重要

3. **尾部行為**：
   - 高濃度尾部（> 4%）：
     - 預測分佈略微低估了極端值的頻率
     - 這與 parity plot 的觀察一致（高濃度區預測偏差較大）
   - **實務影響**：模型可能低估異常事件的發生頻率

4. **Kolmogorov-Smirnov 距離**：
   
   可計算 KS 統計量來量化分佈差異：

$$
D = \max_x |F_{\text{true}}(x) - F_{\text{pred}}(x)|
$$

其中 $F$ 為累積分佈函數。 $D$ 越小表示分佈越接近。

**DNN（右圖）**：

- 分佈匹配度與 Random Forest 相當
- 再次驗證兩模型的等效性

**分佈匹配的重要性**：

在不同應用場景中，分佈匹配的重要性不同：

| 應用場景 | 重要性 | 原因 |
|---------|-------|------|
| 點預測（Point Prediction） | 中 | 主要關注 MAE/RMSE |
| 不確定性量化（Uncertainty Quantification） | 高 | 需要正確的預測分佈 |
| 異常檢測（Anomaly Detection） | 高 | 尾部分佈的準確性至關重要 |
| 過程模擬（Process Simulation） | 高 | 需要複製真實的變異性 |

在本案例中，若目標是**異常預警**，則需要特別關注高濃度尾部的匹配度。

---

## 8. 特徵重要性分析

### 8.1 Permutation Importance 原理

**基本概念**：
1. 在測試集上計算模型的基準性能（例如 MAE）
2. 對每個特徵，隨機打亂（permute）該特徵的值
3. 重新計算性能，觀察性能下降幅度
4. 性能下降越多，表示該特徵越重要

**數學表述**：

對於特徵 $j$ ，其重要性定義為：

$$
\text{Importance}_j = \text{Score}_{\text{original}} - \text{Score}_{\text{permuted}_j}
$$

其中 Score 可以是 R²、MAE 的負值等（越高越好的指標）。

**為何使用 Permutation Importance？**

相較於其他特徵重要性方法：

| 方法 | 優點 | 缺點 | 適用模型 |
|------|------|------|---------|
| **Permutation Importance** | 模型無關、考慮特徵交互 | 計算成本高 | 所有模型 |
| Gini Importance (RF) | 快速、內建 | 偏向高基數特徵 | 僅樹模型 |
| Coefficient (Linear) | 直觀、有方向性 | 僅限線性模型 | 線性模型 |
| SHAP | 理論嚴謹、局部解釋 | 計算極慢 | 所有模型 |

### 8.2 Random Forest 特徵重要性

![Random Forest Feature Importance](outputs/P4_Unit15_Example_Mining/figs/rf_feature_importance.png)

**Top 15 重要特徵分析**：

基於 Random Forest 的內建特徵重要性（Gini importance），我們觀察到：

**第一梯隊（極高重要性）**：

1. **% Silica Feed**（重要性 ≈ 0.65）：
   - **壓倒性的最重要特徵**
   - 物理意義：原礦矽石含量直接決定精礦矽石含量
   - 這驗證了「原料品質主導論」
   - 實務啟示：提升精礦品質的根本在於選礦（ore selection）

2. **% Iron Feed**（重要性 ≈ 0.15）：
   - 第二重要特徵，但重要性僅為第一名的 1/4
   - 與 % Silica Feed 存在負相關（礦物學互補）
   - 兩者共同描述了原礦的品質狀態

**第二梯隊（中等重要性）**：

3-5. **Ore Pulp Flow, Ore Pulp Density, Ore Pulp pH**（重要性 ≈ 0.03-0.05）：
   - 礦漿性質變數
   - 影響浮選動力學與化學環境
   - 是可控的操作變數，優化潛力大

**第三梯隊（較低重要性）**：

6-10. **浮選柱液位與氣流**（重要性 ≈ 0.01-0.02）：
   - 單個浮選柱的貢獻較小
   - 但 7 個柱的累積效應不可忽視
   - 這些變數更多用於精細調控

11-15. **Starch Flow, Amina Flow** 等（重要性 ≈ 0.005-0.01）：
   - 藥劑流量的重要性出乎意料地低
   - **這不代表藥劑不重要！**
   - 可能原因：
     - 藥劑用量在數據集中變化較小（操作穩定）
     - 效果主要透過與 pH 的交互體現（非線性）
     - Gini importance 可能低估了藥劑的真實作用

**重要性分佈的啟示**：

從重要性的極度不均勻分佈（第一名佔 65%），我們可以得出：

1. **帕累托原則（80/20 法則）**：
   - 前 2 個特徵（進料品質）貢獻了約 80% 的預測能力
   - 這是典型的「少數關鍵，多數次要」模式

2. **數據收集優先級**：
   - 若要簡化感測器配置，應優先保留進料品質分析
   - 浮選柱變數可考慮降低採樣頻率

3. **過程改進方向**：
   - 投資於原礦選別（ore beneficiation）的回報最高
   - 浮選工藝優化的空間相對有限（受原料限制）

### 8.3 DNN Permutation Importance

![DNN Permutation Importance](outputs/P4_Unit15_Example_Mining/figs/feature_importance.png)

**與 Random Forest 的對比分析**：

DNN 的 Permutation Importance 顯示出與 Random Forest 相似但不完全相同的模式：

**一致性**：

1. **% Silica Feed 仍然最重要**：
   - 兩種模型都識別出這是最關鍵特徵
   - 驗證了特徵重要性的穩健性（model-agnostic）

2. **進料品質變數主導**：
   - % Iron Feed 也在前列
   - 再次確認原料品質的決定性作用

**差異性**：

1. **藥劑流量的重要性提升**：
   - DNN 可能更好地捕捉了藥劑的非線性效應
   - 這與我們的假設一致（Langmuir 吸附曲線）

2. **浮選柱變數的重新排序**：
   - 某些柱的重要性在 DNN 中更高
   - 可能反映了 DNN 學到的不同交互模式

**為何兩種模型的特徵重要性不完全一致？**

這是正常現象，原因包括：

1. **模型機制差異**：
   - Random Forest：基於遞迴分割，偏好能產生純淨分割的特徵
   - DNN：基於梯度優化，所有特徵同時參與預測

2. **交互作用的捕捉方式**：
   - Random Forest：透過樹的分支結構隱式學習交互
   - DNN：透過隱藏層的非線性組合顯式學習交互

3. **正則化效果**：
   - Random Forest：透過樹深度、樣本數限制
   - DNN：透過 Dropout、Early Stopping

**實務建議**：

基於兩種模型的特徵重要性分析，建議：

1. **必須監測的特徵**（兩模型都認為重要）：
   - % Silica Feed
   - % Iron Feed
   - Ore Pulp Flow

2. **可選監測的特徵**（重要性中等）：
   - Ore Pulp pH, Density
   - 主要浮選柱的液位與氣流

3. **可降低頻率的特徵**（重要性低且一致）：
   - 部分浮選柱的次要變數
   - 在穩定操作下變化小的變數

### 8.4 特徵重要性的實務應用

**1. 感測器配置優化**：

基於特徵重要性，可以優化感測器投資：

| 優先級 | 感測器類型 | 建議配置 | 成本效益 |
|-------|----------|---------|---------|
| **高** | 進料品質分析儀 | 高頻率、高精度 | 極高 |
| **中** | 礦漿性質監測 | 中頻率、中精度 | 高 |
| **低** | 浮選柱詳細監測 | 低頻率或抽樣 | 中 |

**2. 異常診斷策略**：

當預測值異常時，按重要性順序檢查：

```
IF predicted_silica > threshold:
    1. 檢查 % Silica Feed（最可能原因）
    2. 檢查 % Iron Feed（次要原因）
    3. 檢查 Ore Pulp Flow（操作因素）
    4. 檢查 pH 與藥劑（精細因素）
```

**3. 過程優化指導**：

- **短期優化**：調整礦漿流量、pH、藥劑用量（可控變數）
- **中期優化**：改進浮選柱操作策略
- **長期優化**：提升原礦選別品質（根本解決）

**4. 模型簡化可能性**：

基於特徵重要性，可以嘗試：

- **精簡模型**：僅使用前 10 個重要特徵
- **預期效果**：性能下降 < 5%，但模型更簡單、更快
- **適用場景**：邊緣計算、即時預測

**爭議點**：
- `% Iron Concentrate` 與 `% Silica Concentrate` 高度負相關（約 -0.99）
- 使用它作為特徵可能導致「資料洩漏」（Data Leakage）

**兩種觀點**：

1. **不應使用**（保守派）：
   - 兩者是同時測量的，存在因果混淆
   - 實務部署時，`% Iron Concentrate` 可能也需要預測
   - 模型會過度依賴此特徵，忽略其他物理機制

2. **可以使用**（實用派）：
   - 若 `% Iron Concentrate` 可透過線上分析儀即時獲得
   - 則使用它預測 `% Silica Concentrate` 是合理的
   - 相當於「多感測器融合」（Sensor Fusion）

**建議**：
- 建立兩個模型版本：有/無 `% Iron Concentrate`
- 比較性能差異，評估此特徵的實際價值
- 根據實際部署情境選擇合適版本

---

## 10. 教學重點與練習題

### 10.1 常用設定（本案例建議）

- 輸入層：22 個特徵（已標準化）
- 隱藏層：多層 Dense（建議：`128 → 64 → 32`）
- 激活函數：`ReLU`
- 輸出層：1 個神經元（線性激活）
- 損失函數：`MSE`（訓練）；報告指標使用 `MAE/RMSE/R²`
- 優化器：`Adam` (lr=0.001)
- 正則化：Dropout (0.3-0.5)、Early Stopping
- Batch size：256
- Epochs：200 (with early stopping)

### 10.2 你應該觀察什麼？

- **訓練曲線**：train/valid loss 是否同步下降？valid 是否提前惡化（過擬合）？
- **Parity plot**：點是否貼近 $y = \hat{y}$ ？是否有系統性偏移？
- **Residual plot**：殘差是否呈現結構（分段/漏斗/曲線）？
- **特徵重要性**：哪些特徵對預測最重要？是否與領域知識一致？

### 10.3 練習題（課後）

1. **Baseline 比較**：
   - 嘗試使用 Ridge Regression 作為 Baseline，比較與 Random Forest 的差異
   - 觀察線性模型在此資料集上的表現

2. **特徵工程**：
   - 嘗試加入特徵交互（例如 `Starch Flow × Amina Flow`、`pH × Amina Flow`）
   - 觀察是否提升性能

3. **架構實驗**：
   - 調整 DNN 層數/神經元數，找出最佳架構
   - 比較不同 Dropout 比例的效果

4. **時序建模**：
   - 本資料集為時間序列，嘗試使用 LSTM/GRU 建模
   - 比較與 MLP 的差異

5. **多目標預測**：
   - 同時預測 `% Iron Concentrate` 與 `% Silica Concentrate`
   - 使用多輸出 DNN 架構

6. **部署考量**：
   - 若 `% Iron Concentrate` 無法即時獲得，如何調整模型？
   - 如何處理感測器故障導致的缺失值？

### 10.4 對應程式碼（Notebook）

請搭配本附錄的程式碼範例執行：

- `Part_4/Unit15_Appendix_Mining.ipynb`

Notebook 內容包含：資料讀取、EDA、Baseline 模型、DNN 建模、訓練、評估（含圖片）、特徵重要性分析與模型保存。

---

## 11. 總結與反思

### 11.1 本案例的關鍵學習點

1. **真實工業數據的挑戰**：
   - 缺失值、異常值、時序對齊問題
   - 需要結合領域知識進行資料清理

2. **Baseline 的重要性**：
   - Random Forest 作為強 Baseline，更能凸顯 DNN 的價值
   - 線性模型（Ridge）可能過於簡化

3. **非線性建模的優勢**：
   - DNN 能捕捉藥劑配比、pH 效應等複雜交互作用
   - 在此案例中顯著優於 Random Forest

4. **特徵重要性的洞察**：
   - 原礦品質是最重要的決定因素（「垃圾進、垃圾出」）
   - 藥劑流量的效果存在非線性與閾值效應

5. **模型優化的系統性方法**：
   - 架構搜索、超參數調整、正則化策略
   - 需要在性能與訓練成本間權衡

### 11.2 實務應用考量

在實際礦業生產應用時，還需考慮：

- **資料收集成本**：部分感測器可能成本較高或維護困難
- **模型更新頻率**：原礦品質變化時需要重新訓練
- **可解釋性需求**：工程師可能需要理解「為什麼」模型給出某個預測
- **整合到 DCS 系統**：需要考慮延遲、穩定性、異常處理
- **經濟效益評估**：降低 1% 矽石濃度的經濟價值是多少？

### 11.3 延伸閱讀

- Froth Flotation 基礎理論：[Wikipedia - Froth Flotation](https://en.wikipedia.org/wiki/Froth_flotation)
- 相關研究論文：
  - "Purities prediction in a manufacturing froth flotation plant: the deep learning techniques"
  - "Soft Sensor: Traditional Machine Learning or Deep Learning"
- SHAP（SHapley Additive exPlanations）用於模型解釋
- 時序預測：LSTM/GRU 在過程工業的應用

---

**本案例展示了 DNN 在真實礦業過程品質預測中的應用，希望學生能透過完整的建模流程，深入理解深度學習在工業 AI 中的價值與挑戰。**
