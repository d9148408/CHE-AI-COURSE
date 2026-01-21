# Unit13 集成學習方法 (Ensemble Learning Methods)

**逢甲大學 化學工程學系 AI 課程**  
**課程名稱：AI在化工上之應用**  
**授課教師：莊曜禎 助理教授**

---

## 📚 課程目標

本單元將深入探討集成學習 (Ensemble Learning) 方法，這是機器學習中最強大且廣泛應用的技術之一。透過結合多個基礎模型的預測結果，集成學習能夠顯著提升模型的準確性和穩定性。

**學習目標：**
- 理解集成學習的核心概念與理論基礎
- 掌握三大類集成學習方法：Bagging、Boosting、Stacking
- 學習使用 Random Forest、XGBoost、LightGBM、CatBoost 等主流集成模型
- 了解集成學習在化工領域的實際應用
- 能夠根據問題特性選擇合適的集成學習方法

---

## 1️⃣ 集成學習概述

### 1.1 什麼是集成學習？

**集成學習 (Ensemble Learning)** 是一種機器學習範式，透過組合多個基礎學習器 (Base Learner) 的預測結果來獲得更好的預測性能。其核心思想是「三個臭皮匠，勝過一個諸葛亮」，即使個別模型的性能一般，但透過適當的組合方式，可以產生更強大且穩定的預測模型。

**核心優勢：**
- **提高準確性**：組合多個模型可以降低單一模型的預測誤差
- **降低過擬合風險**：透過平均或投票機制減少模型方差
- **增強穩健性**：對異常值和噪音數據更加穩健
- **捕捉多樣性**：不同模型可以從不同角度學習數據特徵

### 1.2 集成學習的理論基礎

#### 1.2.1 偏差-方差分解 (Bias-Variance Decomposition)

在機器學習中，模型的預測誤差可以分解為三個部分：

$$
\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
$$

- **偏差 (Bias)**：模型預測值與真實值之間的差異，反映模型的擬合能力
- **方差 (Variance)**：模型在不同訓練集上預測值的變異程度，反映模型的穩定性
- **不可約誤差 (Irreducible Error)**：數據本身的噪音，無法消除

**集成學習的作用機制：**
- **Bagging 方法**：主要降低方差，透過在不同數據子集上訓練多個模型並平均預測結果
- **Boosting 方法**：主要降低偏差，透過順序訓練多個模型，每個模型專注於前一個模型的錯誤
- **Stacking 方法**：同時降低偏差和方差，透過元學習器學習如何最佳組合基礎模型

#### 1.2.2 多樣性原則

集成學習的成功關鍵在於基礎學習器的**多樣性 (Diversity)**。如果所有基礎模型都做出相同的錯誤，集成並不會改善性能。多樣性來源包括：

1. **數據多樣性**：使用不同的訓練數據子集
2. **特徵多樣性**：使用不同的特徵子集
3. **算法多樣性**：使用不同類型的學習算法
4. **參數多樣性**：使用不同的超參數設定

### 1.3 集成學習的分類

根據基礎學習器的生成方式和組合策略，集成學習可分為三大類：

| 方法類別 | 代表算法 | 訓練方式 | 主要優勢 | 適用場景 |
|---------|---------|---------|---------|---------|
| **Bagging** | Random Forest | 並行訓練 | 降低方差 | 高方差模型（如決策樹） |
| **Boosting** | XGBoost, LightGBM, CatBoost | 序列訓練 | 降低偏差 | 需要高準確度的任務 |
| **Stacking** | Stacking Regressor/Classifier | 多層訓練 | 全面提升 | 複雜預測任務 |

---

## 2️⃣ Bagging 方法：Bootstrap Aggregating

### 2.1 Bagging 核心概念

**Bagging (Bootstrap Aggregating)** 是由 Leo Breiman 在 1996 年提出的集成學習方法。其核心思想是透過 Bootstrap 抽樣產生多個訓練子集，在每個子集上訓練一個基礎學習器，最後將所有學習器的預測結果進行聚合。

### 2.2 Bagging 工作流程

**步驟 1：Bootstrap 抽樣**
- 從原始訓練集 $D$ (包含 $n$ 個樣本) 中**有放回抽樣** $n$ 次，生成訓練子集 $D_1, D_2, \ldots, D_M$
- 每個子集與原始數據集大小相同，但包含約 63.2% 的不重複樣本

**步驟 2：並行訓練基礎學習器**
- 在每個 Bootstrap 樣本上訓練一個基礎學習器 $h_1, h_2, \ldots, h_M$
- 基礎學習器通常選擇高方差、低偏差的模型（如深度決策樹）

**步驟 3：預測聚合**
- **回歸任務**：對所有基礎學習器的預測結果取平均

$$
H(x) = \frac{1}{M} \sum_{i=1}^{M} h_i(x)
$$

- **分類任務**：採用投票機制，選擇得票最多的類別

$$
H(x) = \arg\max_{c} \sum_{i=1}^{M} \mathbb{1}(h_i(x) = c)
$$

### 2.3 Random Forest：Bagging 的代表算法

**Random Forest (隨機森林)** 是 Bagging 方法的改進版本，由 Leo Breiman 在 2001 年提出。它在 Bagging 的基礎上引入了**特徵隨機選擇**機制。

#### 2.3.1 Random Forest 改進之處

1. **數據隨機性**：與 Bagging 相同，使用 Bootstrap 抽樣
2. **特徵隨機性**：在每次分裂節點時，隨機選擇 $m$ 個特徵的子集（通常 $m = \sqrt{p}$ ， $p$ 為總特徵數）
3. **模型多樣性**：兩種隨機性結合，大幅提高基礎學習器之間的多樣性

#### 2.3.2 Random Forest 數學表示

對於包含 $M$ 棵決策樹的隨機森林：

**回歸預測：**

$$
\hat{y} = \frac{1}{M} \sum_{m=1}^{M} T_m(x)
$$

**分類預測（軟投票）：**

$$
\hat{p}_c(x) = \frac{1}{M} \sum_{m=1}^{M} \mathbb{1}(T_m(x) = c)
$$

其中 $T_m(x)$ 表示第 $m$ 棵決策樹的預測結果。

#### 2.3.3 Random Forest 關鍵參數

| 參數 | 說明 | 典型值 | 影響 |
|------|------|--------|------|
| `n_estimators` | 決策樹數量 | 100-500 | 數量越多，模型越穩定但計算成本增加 |
| `max_depth` | 樹的最大深度 | None 或 10-50 | 控制模型複雜度，防止過擬合 |
| `max_features` | 分裂時考慮的特徵數 | sqrt(n_features) | 控制樹之間的相關性 |
| `min_samples_split` | 分裂節點所需最小樣本數 | 2-10 | 控制樹的生長，防止過擬合 |
| `min_samples_leaf` | 葉節點最小樣本數 | 1-5 | 控制樹的深度，提高泛化能力 |

#### 2.3.4 Random Forest 優缺點

**優點：**
- ✅ 訓練速度快，可並行化
- ✅ 對異常值和噪音具有很好的魯棒性
- ✅ 可處理高維數據，無需特徵選擇
- ✅ 可提供特徵重要性評估
- ✅ 不易過擬合

**缺點：**
- ❌ 對於某些數據集，預測準確度可能不如 Boosting 方法
- ❌ 模型可解釋性較差
- ❌ 預測時間隨樹數量線性增長

### 2.4 Out-of-Bag (OOB) 評估

Bagging 的一個額外優勢是可以使用 **Out-of-Bag (OOB)** 樣本進行模型評估，無需額外的驗證集。

- 對於每個樣本，約有 37% 的基礎學習器沒有使用該樣本進行訓練
- 可以使用這些「袋外」模型對該樣本進行預測，作為交叉驗證的替代方案

**OOB 誤差估計：**

$$
\text{OOB Error} = \frac{1}{n} \sum_{i=1}^{n} L(y_i, \hat{y}_i^{\text{OOB}})
$$

其中 $\hat{y}_i^{\text{OOB}}$ 是使用未包含樣本 $i$ 的所有樹進行預測的平均值。

---
## 3️⃣ Boosting 方法：迭代式強化學習

### 3.1 Boosting 核心概念

**Boosting** 是一種序列化的集成學習方法，透過**迭代訓練弱學習器**，每次訓練都專注於前一次訓練中被錯誤分類或預測誤差較大的樣本，從而逐步提升整體模型的性能。

**核心特點：**
- **序列訓練**：基礎學習器按順序訓練，後一個學習器依賴前一個學習器的結果
- **自適應權重**：根據樣本的預測誤差動態調整樣本權重
- **加權組合**：最終預測結果是所有基礎學習器的加權和

### 3.2 Boosting 的演進

Boosting 方法經歷了多個階段的發展：

| 算法 | 提出年份 | 核心改進 | 適用場景 |
|------|---------|---------|---------|
| **AdaBoost** | 1995 | 樣本權重自適應調整 | 二元分類 |
| **Gradient Boosting** | 2001 | 使用梯度下降優化損失函數 | 回歸與分類 |
| **XGBoost** | 2016 | 正則化、並行化、處理缺失值 | 高維數據、競賽 |
| **LightGBM** | 2017 | 基於直方圖的決策樹、葉子生長策略 | 大規模數據 |
| **CatBoost** | 2018 | 原生處理類別特徵、對稱樹 | 包含類別特徵的數據 |

### 3.3 Gradient Boosting 基礎理論

Gradient Boosting 將 Boosting 問題轉化為**梯度下降優化問題**。

#### 3.3.1 基本思想

給定損失函數 $L(y, F(x))$，目標是找到一個函數 $F(x)$ 使損失最小：

$$
F^* = \arg\min_{F} \mathbb{E}_{x,y}[L(y, F(x))]
$$

Gradient Boosting 透過**加法模型**逐步逼近最優解：

$$
F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)
$$

其中：
- $F_m(x)$ 是第 $m$ 輪迭代後的模型
- $h_m(x)$ 是第 $m$ 個基礎學習器
- $\nu$ 是學習率 (learning rate)

#### 3.3.2 負梯度擬合

在每次迭代中，新的基礎學習器 $h_m(x)$ 擬合損失函數對當前模型預測的**負梯度**：

$$
r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}
$$

這個負梯度可視為**偽殘差 (pseudo-residual)**，表示當前模型需要改進的方向。

#### 3.3.3 Gradient Boosting 算法流程

**初始化：**

$$
F_0(x) = \arg\min_{\gamma} \sum_{i=1}^{n} L(y_i, \gamma)
$$

**迭代（對於 $m = 1, 2, \ldots, M$）：**

1. 計算負梯度（偽殘差）：

$$
r_{im} = -\left[\frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)}\right], \quad i = 1, 2, \ldots, n
$$

2. 訓練基礎學習器 $h_m(x)$ 擬合 $(x_i, r_{im})$

3. 計算最優步長：

$$
\gamma_m = \arg\min_{\gamma} \sum_{i=1}^{n} L(y_i, F_{m-1}(x_i) + \gamma h_m(x_i))
$$

4. 更新模型：

$$
F_m(x) = F_{m-1}(x) + \nu \gamma_m h_m(x)
$$

**最終模型：**

$$
F(x) = F_0(x) + \nu \sum_{m=1}^{M} \gamma_m h_m(x)
$$

### 3.4 XGBoost：極致梯度提升

**XGBoost (Extreme Gradient Boosting)** 是由陳天奇於 2016 年提出的高效 Boosting 算法，在 Kaggle 競賽中取得巨大成功。

#### 3.4.1 XGBoost 核心改進

**1. 正則化目標函數**

XGBoost 的目標函數包含損失項和正則化項：

$$
\text{Obj}^{(t)} = \sum_{i=1}^{n} L(y_i, \hat{y}_i^{(t)}) + \sum_{k=1}^{t} \Omega(f_k)
$$

其中正則化項為：

$$
\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2
$$

- $T$ 是葉節點數量
- $w_j$ 是第 $j$ 個葉節點的權重
- $\gamma$ 和 $\lambda$ 是正則化係數

**2. 二階泰勒展開**

XGBoost 對損失函數進行二階泰勒展開，利用一階和二階梯度信息：

$$
\text{Obj}^{(t)} \approx \sum_{i=1}^{n} [g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)] + \Omega(f_t)
$$

其中：
- $g_i = \frac{\partial L(y_i, \hat{y}^{(t-1)})}{\partial \hat{y}^{(t-1)}}$ (一階梯度)
- $h_i = \frac{\partial^2 L(y_i, \hat{y}^{(t-1)})}{\partial (\hat{y}^{(t-1)})^2}$ (二階梯度)

**3. 分裂增益計算**

對於某個節點的分裂，增益計算為：

$$
\text{Gain} = \frac{1}{2} \left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma
$$

其中 $G_L, G_R$ 和 $H_L, H_R$ 分別是左右子節點的梯度和與海森矩陣和。

#### 3.4.2 XGBoost 關鍵特性

- **處理缺失值**：自動學習缺失值的最優分裂方向
- **並行化**：在特徵層面進行並行計算，加速訓練
- **剪枝策略**：使用 max_depth 進行預剪枝，避免過擬合
- **列採樣**：類似 Random Forest，在每次迭代時隨機選擇特徵子集

#### 3.4.3 XGBoost 關鍵參數

| 參數 | 說明 | 典型值 | 影響 |
|------|------|--------|------|
| `n_estimators` | 樹的數量 | 100-1000 | 數量越多性能越好，但計算成本增加 |
| `learning_rate` (eta) | 學習率 | 0.01-0.3 | 控制每棵樹的貢獻，較小值需要更多樹 |
| `max_depth` | 樹的最大深度 | 3-10 | 控制模型複雜度，防止過擬合 |
| `min_child_weight` | 子節點最小權重和 | 1-10 | 控制分裂，較大值更保守 |
| `gamma` | 分裂所需最小損失減少 | 0-0.5 | 控制樹的複雜度，較大值更保守 |
| `subsample` | 訓練樣本採樣比例 | 0.5-1.0 | 防止過擬合，增加訓練速度 |
| `colsample_bytree` | 每棵樹的特徵採樣比例 | 0.5-1.0 | 增加模型多樣性，防止過擬合 |
| `lambda` (reg_lambda) | L2 正則化係數 | 0-10 | 控制模型複雜度 |
| `alpha` (reg_alpha) | L1 正則化係數 | 0-10 | 特徵選擇，增加稀疏性 |

### 3.5 LightGBM：輕量級梯度提升機

**LightGBM (Light Gradient Boosting Machine)** 是由微軟於 2017 年提出的高效 Boosting 框架，特別適合大規模數據。

#### 3.5.1 LightGBM 核心技術

**1. 基於直方圖的決策樹 (Histogram-based Decision Tree)**
- 將連續特徵離散化為 $k$ 個區間（直方圖），減少計算複雜度
- 時間複雜度從 $O(n \times p)$ 降低到 $O(k \times p)$
- 內存佔用大幅減少

**2. 葉子生長策略 (Leaf-wise Growth)**
- XGBoost 使用層級生長 (Level-wise)，每次分裂同一層的所有節點
- LightGBM 使用葉子生長 (Leaf-wise)，每次選擇增益最大的葉節點分裂
- 可以更快達到相同的準確度，但需要小心過擬合

**3. 單邊梯度採樣 (GOSS, Gradient-based One-Side Sampling)**
- 保留梯度較大的樣本（對模型貢獻大）
- 隨機採樣梯度較小的樣本
- 在保持準確度的同時大幅減少計算量

**4. 互斥特徵捆綁 (EFB, Exclusive Feature Bundling)**
- 將互斥的特徵（不會同時取非零值）捆綁為一個特徵
- 減少特徵數量，加速訓練

#### 3.5.2 LightGBM 關鍵參數

| 參數 | 說明 | 典型值 | 影響 |
|------|------|--------|------|
| `num_iterations` | 迭代次數 | 100-1000 | 等同於 XGBoost 的 n_estimators |
| `learning_rate` | 學習率 | 0.01-0.3 | 控制每次迭代的步長 |
| `num_leaves` | 葉節點數量 | 31-255 | LightGBM 的核心參數，控制模型複雜度 |
| `max_depth` | 最大深度 | -1（無限制）或 3-12 | 限制樹的深度，防止過擬合 |
| `min_data_in_leaf` | 葉節點最小樣本數 | 20-100 | 防止過擬合 |
| `bagging_fraction` | 數據採樣比例 | 0.5-1.0 | 類似 XGBoost 的 subsample |
| `feature_fraction` | 特徵採樣比例 | 0.5-1.0 | 類似 XGBoost 的 colsample_bytree |
| `lambda_l1` | L1 正則化 | 0-10 | 增加稀疏性 |
| `lambda_l2` | L2 正則化 | 0-10 | 控制模型複雜度 |

#### 3.5.3 LightGBM vs XGBoost

| 特性 | XGBoost | LightGBM |
|------|---------|----------|
| **樹生長策略** | Level-wise（層級） | Leaf-wise（葉子） |
| **分裂方法** | 預排序 | 基於直方圖 |
| **訓練速度** | 較慢 | 更快 |
| **內存佔用** | 較大 | 較小 |
| **小數據集** | 表現優秀 | 可能過擬合 |
| **大數據集** | 可能較慢 | 表現優秀 |
| **類別特徵** | 需要編碼 | 原生支持 |

### 3.6 CatBoost：類別特徵增強的 Boosting

**CatBoost (Categorical Boosting)** 是由 Yandex 於 2018 年提出的 Boosting 算法，專門優化了類別特徵的處理。

#### 3.6.1 CatBoost 核心技術

**1. 原生處理類別特徵 (Native Categorical Feature Handling)**
- 使用 **Ordered Target Statistics** 方法處理類別特徵
- 避免傳統 One-Hot 編碼和 Label 編碼的缺點
- 減少目標洩漏 (Target Leakage) 風險

**類別特徵編碼公式：**

$$
\hat{x}_k^i = \frac{\sum_{j=1}^{i-1} [x_j = x_i] \cdot y_j + a \cdot p}{\sum_{j=1}^{i-1} [x_j = x_i] + a}
$$

其中：
- $x_i$ 是當前樣本的類別值
- $y_j$ 是前面樣本的目標值
- $a$ 和 $p$ 是先驗參數（通常 $a=1$ ， $p$ 為目標平均值）

**2. 對稱樹 (Oblivious Trees)**
- 在同一層級的所有節點使用相同的分裂條件
- 樹結構更簡單，預測速度更快
- 減少過擬合風險

**3. 有序 Boosting (Ordered Boosting)**
- 解決傳統 Gradient Boosting 中的**預測偏移 (Prediction Shift)** 問題
- 在計算梯度時使用不同的數據子集

#### 3.6.2 CatBoost 關鍵參數

| 參數 | 說明 | 典型值 | 影響 |
|------|------|--------|------|
| `iterations` | 迭代次數 | 100-1000 | 等同於樹的數量 |
| `learning_rate` | 學習率 | 0.01-0.3 | 控制訓練步長 |
| `depth` | 樹的深度 | 4-10 | 對稱樹的深度 |
| `l2_leaf_reg` | L2 正則化係數 | 1-10 | 控制葉節點權重 |
| `cat_features` | 類別特徵索引 | None 或列表 | 指定哪些特徵是類別型 |
| `one_hot_max_size` | One-Hot 編碼閾值 | 2-255 | 類別數小於此值時使用 One-Hot |
| `bagging_temperature` | 貝葉斯 Bootstrap 溫度 | 0-1 | 控制 Bootstrap 採樣的隨機性 |
| `random_strength` | 分裂時的隨機性 | 0-10 | 增加模型多樣性 |

#### 3.6.3 CatBoost 優勢

- ✅ **無需手動編碼類別特徵**：自動處理，減少特徵工程工作
- ✅ **減少過擬合**：對稱樹結構和有序 Boosting 機制
- ✅ **高預測速度**：對稱樹結構使預測更快
- ✅ **魯棒性強**：對參數設定不敏感，默認參數通常表現良好
- ✅ **GPU 加速**：原生支持 GPU 訓練

### 3.7 三大 Boosting 算法比較

| 特性 | XGBoost | LightGBM | CatBoost |
|------|---------|----------|----------|
| **推出年份** | 2016 | 2017 | 2018 |
| **訓練速度** | 中等 | 最快 | 較慢 |
| **內存佔用** | 中等 | 最少 | 較多 |
| **小數據集** | ✅ 優秀 | ⚠️ 可能過擬合 | ✅ 優秀 |
| **大數據集** | ⚠️ 較慢 | ✅ 最佳 | ✅ 良好 |
| **類別特徵** | ❌ 需手動編碼 | ⚠️ 部分支持 | ✅ 原生支持 |
| **過擬合風險** | 中等 | 較高 | 較低 |
| **參數敏感度** | 較高 | 較高 | 較低 |
| **適用場景** | 通用競賽 | 大規模數據 | 包含類別特徵的數據 |

---
## 4️⃣ Stacking 方法：模型堆疊

### 4.1 Stacking 核心概念

**Stacking (Stacked Generalization)** 是由 David Wolpert 在 1992 年提出的集成學習方法。其核心思想是使用**多層模型結構**，將多個基礎學習器的預測結果作為新的特徵，訓練一個**元學習器 (Meta-Learner)** 來產生最終預測。

**與 Bagging/Boosting 的區別：**
- **Bagging/Boosting**：使用相同類型的基礎學習器（同質集成）
- **Stacking**：可以使用不同類型的基礎學習器（異質集成），並使用元模型學習如何最佳組合

### 4.2 Stacking 工作流程

#### 4.2.1 兩層 Stacking 架構

**第一層（基礎層）：**
1. 將訓練數據分為 K 折
2. 在每折上訓練多個不同類型的基礎學習器（如 Random Forest, XGBoost, SVM 等）
3. 使用交叉驗證生成訓練集的預測結果，作為元特徵

**第二層（元學習層）：**
1. 使用基礎學習器的預測結果作為新特徵
2. 訓練元學習器（通常選擇簡單的線性模型或邏輯回歸）
3. 元學習器學習如何組合基礎學習器的預測

#### 4.2.2 Stacking 數學表示

假設有 $M$ 個基礎學習器 $h_1, h_2, \ldots, h_M$，元學習器為 $g$。

**基礎層預測：**

$$
z_i^{(m)} = h_m(x_i), \quad m = 1, 2, \ldots, M
$$

**元特徵構造：**

$$
z_i = [z_i^{(1)}, z_i^{(2)}, \ldots, z_i^{(M)}]
$$

**最終預測：**

$$
\hat{y}_i = g(z_i) = g(h_1(x_i), h_2(x_i), \ldots, h_M(x_i))
$$

### 4.3 Stacking 優缺點

**優點：**
- ✅ 可以結合不同類型模型的優勢，性能通常優於單一模型
- ✅ 靈活性高，可以根據任務特點設計不同的堆疊結構
- ✅ 透過交叉驗證機制，降低過擬合風險
- ✅ 可以多層堆疊，構建更複雜的集成結構

**缺點：**
- ❌ 訓練和預測時間較長（需要訓練多個模型）
- ❌ 模型複雜度高，調參困難
- ❌ 可解釋性差
- ❌ 需要更多的計算資源

---

## 5️⃣ 集成學習在化工領域的應用

集成學習方法在化工領域具有廣泛的應用前景，能夠處理複雜的非線性關係和高維數據，以下是幾個典型應用場景。

### 5.1 程序控制與優化

**應用場景：**
- 反應器溫度控制
- 蒸餾塔操作優化
- 化學反應條件優化

**適用方法：**
- **Random Forest**：處理多變量控制問題，提供特徵重要性分析
- **XGBoost/LightGBM**：高精度預測最優操作條件
- **Stacking**：結合不同模型優勢，提升控制精度

**實際案例：**
- 使用 XGBoost 預測反應器出口溫度，根據進料流量、催化劑用量等操作變數進行實時調整
- 透過 Random Forest 分析影響產品純度的關鍵因素，指導蒸餾塔操作優化

### 5.2 產品質量預測

**應用場景：**
- 聚合物性質預測（分子量、黏度等）
- 產品純度預測
- 最終產品質量分類

**適用方法：**
- **CatBoost**：處理包含類別型操作條件（如催化劑類型、反應器類型）的數據
- **Stacking Regressor**：結合多種模型預測連續型質量指標
- **Stacking Classifier**：分類產品質量等級（優良、合格、不合格）

**實際案例：**
- 使用 CatBoost 預測不同配方條件下的聚合物分子量分布
- 透過 Stacking 方法預測精細化工產品的多項質量指標

### 5.3 故障診斷與預測性維護

**應用場景：**
- 設備故障檢測
- 異常操作識別
- 設備剩餘壽命預測

**適用方法：**
- **Random Forest Classifier**：多類別故障分類，識別故障類型
- **XGBoost**：不平衡數據集的故障預測（正常樣本遠多於故障樣本）
- **LightGBM**：大規模歷史運行數據的實時監控

**實際案例：**
- 使用 Random Forest 分析泵浦振動數據，識別軸承故障、對中不良等問題
- 透過 XGBoost 預測換熱器結垢程度，提前安排清洗維護

### 5.4 過程安全與風險評估

**應用場景：**
- 危險化學品洩漏風險評估
- 操作安全邊界預測
- 緊急情況響應決策支持

**適用方法：**
- **XGBoost Classifier**：預測高風險操作條件
- **Random Forest**：評估多因素對安全的影響
- **Stacking**：綜合多個安全指標的風險評估

**實際案例：**
- 使用 XGBoost 預測反應器超溫超壓風險，根據進料組成和操作條件提前預警
- 採用 Random Forest 分析歷史事故數據，識別關鍵安全影響因素

### 5.5 配方與實驗設計

**應用場景：**
- 新產品配方優化
- 實驗設計與篩選
- 原料替代方案評估

**適用方法：**
- **CatBoost**：處理類別型原料（如添加劑類型）與連續型配比的混合數據
- **Stacking**：結合多種模型預測配方性能
- **Bayesian Optimization + Ensemble Models**：高效搜索最優配方

**實際案例：**
- 使用 CatBoost 預測不同溶劑組合對塗料性能的影響
- 透過 Stacking 模型預測化妝品配方的感官評分和穩定性

---

## 6️⃣ 資料前處理與模型評估

### 6.1 資料前處理

集成學習模型對資料前處理的要求相對寬鬆，但適當的前處理仍能提升性能。

#### 6.1.1 數值特徵處理

**標準化 (Standardization)：**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**適用模型：**
- **需要標準化**：Stacking 中的線性元學習器（如 Ridge, Logistic Regression）
- **不需要標準化**：Random Forest, XGBoost, LightGBM, CatBoost（基於樹的模型對特徵尺度不敏感）

#### 6.1.2 類別特徵處理

**One-Hot 編碼：**
```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_categorical_encoded = encoder.fit_transform(X_categorical)
```

**Label 編碼：**
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X_categorical_encoded = le.fit_transform(X_categorical)
```

**模型選擇建議：**
- **Random Forest, XGBoost**：建議使用 One-Hot 或 Label 編碼
- **LightGBM**：支持原生類別特徵，設置 `categorical_feature` 參數
- **CatBoost**：原生支持類別特徵，設置 `cat_features` 參數，無需手動編碼

#### 6.1.3 缺失值處理

**簡單填充：**
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')  # 或 'median', 'most_frequent'
X_imputed = imputer.fit_transform(X)
```

**模型選擇建議：**
- **XGBoost, LightGBM, CatBoost**：可以直接處理缺失值，無需填充
- **Random Forest, Stacking**：建議先進行缺失值填充

### 6.2 模型評估指標

#### 6.2.1 回歸任務

**常用指標：**
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 均方誤差 (MSE)
mse = mean_squared_error(y_true, y_pred)

# 均方根誤差 (RMSE)
rmse = mean_squared_error(y_true, y_pred, squared=False)

# 平均絕對誤差 (MAE)
mae = mean_absolute_error(y_true, y_pred)

# R² 決定係數
r2 = r2_score(y_true, y_pred)
```

#### 6.2.2 分類任務

**常用指標：**
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 準確率
accuracy = accuracy_score(y_true, y_pred)

# 精確率
precision = precision_score(y_true, y_pred, average='weighted')

# 召回率
recall = recall_score(y_true, y_pred, average='weighted')

# F1 分數
f1 = f1_score(y_true, y_pred, average='weighted')

# ROC-AUC（二元分類）
auc = roc_auc_score(y_true, y_pred_proba)
```

### 6.3 交叉驗證 (Cross-Validation)

**K 折交叉驗證：**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f'Cross-validation RMSE: {(-scores.mean())**0.5:.4f} (+/- {scores.std():.4f})')
```

**分層 K 折交叉驗證（分類任務）：**
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
```

### 6.4 超參數調整 (Hyperparameter Tuning)

#### 6.4.1 網格搜索 (Grid Search)

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

#### 6.4.2 隨機搜索 (Random Search)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3)
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
```

---

## 7️⃣ 集成學習方法選擇指南

### 7.1 根據數據特性選擇

| 數據特性 | 推薦方法 | 理由 |
|---------|---------|------|
| **小數據集（< 10,000 樣本）** | Random Forest, XGBoost, CatBoost | 不易過擬合，穩定性好 |
| **大數據集（> 100,000 樣本）** | LightGBM | 訓練速度快，內存佔用少 |
| **包含大量類別特徵** | CatBoost | 原生處理類別特徵 |
| **高維數據（特徵數 > 100）** | Random Forest, XGBoost | 特徵選擇能力強 |
| **不平衡數據集** | XGBoost, LightGBM | 支持樣本權重和自定義損失函數 |
| **缺失值較多** | XGBoost, LightGBM, CatBoost | 可直接處理缺失值 |

### 7.2 根據任務需求選擇

| 任務需求 | 推薦方法 | 理由 |
|---------|---------|------|
| **追求最高準確度** | Stacking (XGBoost + LightGBM + CatBoost) | 結合多種模型優勢 |
| **需要快速訓練** | LightGBM | 訓練速度最快 |
| **需要快速預測** | LightGBM, CatBoost | 預測速度快 |
| **需要模型可解釋性** | Random Forest | 可提供特徵重要性 |
| **需要處理類別特徵** | CatBoost | 無需手動編碼 |
| **參數調整經驗有限** | CatBoost, Random Forest | 對參數不敏感 |

### 7.3 實戰建議

**初學者建議流程：**
1. **快速基線模型**：使用 Random Forest 建立基線
2. **嘗試 Boosting**：使用 XGBoost 或 LightGBM 提升性能
3. **處理類別特徵**：如有大量類別特徵，嘗試 CatBoost
4. **集成優化**：使用 Stacking 結合多個模型

**進階優化建議：**
1. 系統性超參數調整（Grid Search 或 Bayesian Optimization）
2. 特徵工程（特徵交互、多項式特徵等）
3. 多層 Stacking 或 Blending
4. 模型融合（加權平均多個模型的預測結果）

---

## 8️⃣ 單元總結

### 8.1 核心要點回顧

**集成學習三大方法：**
1. **Bagging**：並行訓練，降低方差
   - 代表：Random Forest
   - 特點：穩定、魯棒、易於並行化

2. **Boosting**：序列訓練，降低偏差
   - 代表：XGBoost, LightGBM, CatBoost
   - 特點：高準確度、可處理複雜關係

3. **Stacking**：多層結構，全面提升
   - 特點：靈活、可結合不同模型優勢

**關鍵概念：**
- 集成學習透過組合多個模型來提升性能
- 基礎學習器的多樣性是成功的關鍵
- 不同方法適用於不同的數據特性和任務需求

### 8.2 學習檢核表

完成本單元後，您應該能夠：
- ✅ 理解集成學習的基本原理和偏差-方差分解
- ✅ 掌握 Bagging、Boosting、Stacking 三大方法的區別
- ✅ 使用 Random Forest 進行回歸和分類任務
- ✅ 使用 XGBoost、LightGBM、CatBoost 建立高性能模型
- ✅ 理解如何選擇和調整集成學習模型的超參數
- ✅ 了解集成學習在化工領域的實際應用
- ✅ 能夠根據數據特性選擇合適的集成學習方法

### 8.3 下一步學習

**進階主題：**
- Bayesian Optimization 用於超參數優化
- Feature Engineering 特徵工程技術
- 模型解釋性工具（SHAP, LIME）
- AutoML 自動機器學習框架

**實作練習：**
- 完成 Unit13 各模型的 Jupyter Notebook 練習
- 在化工數據集上比較不同集成方法的性能
- 嘗試使用 Stacking 組合多個模型

---

## 9️⃣ 參考資源

### 9.1 學術論文

1. **Breiman, L. (1996).** "Bagging Predictors." *Machine Learning*, 24(2), 123-140.
2. **Breiman, L. (2001).** "Random Forests." *Machine Learning*, 45(1), 5-32.
3. **Friedman, J. H. (2001).** "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics*, 29(5), 1189-1232.
4. **Chen, T., & Guestrin, C. (2016).** "XGBoost: A Scalable Tree Boosting System." *Proceedings of KDD 2016*.
5. **Ke, G., et al. (2017).** "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *Proceedings of NeurIPS 2017*.
6. **Prokhorenkova, L., et al. (2018).** "CatBoost: Unbiased Boosting with Categorical Features." *Proceedings of NeurIPS 2018*.

### 9.2 線上資源

**官方文檔：**
- scikit-learn Ensemble Methods: https://scikit-learn.org/stable/modules/ensemble.html
- XGBoost Documentation: https://xgboost.readthedocs.io/
- LightGBM Documentation: https://lightgbm.readthedocs.io/
- CatBoost Documentation: https://catboost.ai/docs/

**教學資源：**
- Kaggle Learn: Intermediate Machine Learning
- Towards Data Science: Ensemble Learning 系列文章
- Machine Learning Mastery: Ensemble Learning Tutorials

### 9.3 推薦書籍

1. **《統計學習精要》（The Elements of Statistical Learning）** - Hastie, Tibshirani, Friedman
2. **《機器學習實戰》（Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow）** - Aurélien Géron
3. **《集成方法：基礎與算法》（Ensemble Methods: Foundations and Algorithms）** - Zhi-Hua Zhou

---

## 📝 課後作業

請完成以下練習來鞏固您的學習：

1. **基礎練習**：
   - 完成 Unit13_Random_Forest_Regression.ipynb
   - 完成 Unit13_XGBoost_Classification.ipynb

2. **進階挑戰**：
   - 在化工數據集上比較 XGBoost、LightGBM、CatBoost 的性能
   - 使用 Stacking 組合至少 3 個不同的模型

3. **實戰應用**：
   - 選擇一個化工領域的問題（如反應器溫度控制、產品質量預測等）
   - 使用集成學習方法建立預測模型
   - 分析不同方法的優缺點並撰寫簡短報告

---

**課程結束**

感謝您完成 Unit13 集成學習方法的學習！如有任何問題，歡迎與授課老師討論。

**作者：莊曜禎 助理教授**  
**逢甲大學 化學工程學系**  
**最後更新：2026 年 1 月**
