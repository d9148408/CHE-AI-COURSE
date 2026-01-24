# Unit07 異常檢測總覽 (Anomaly Detection Overview)

**課程名稱**: AI在化工上之應用  
**課程代碼**: CHE-AI-114  
**課程教師**: 莊曜禎 助理教授  
**單元編號**: Unit07  
**授課時間**: 114學年度第2學期

---

## 📚 單元學習目標

完成本單元後，學生將能夠：

1. 理解異常檢測 (Anomaly Detection) 的基本概念與重要性
2. 區分異常 (Anomaly)、離群值 (Outlier) 與新奇點 (Novelty) 的差異
3. 掌握異常檢測在化工領域的典型應用場景
4. 了解 sklearn 模組中主流異常檢測演算法的原理與特性
5. 能夠根據數據特性與應用需求選擇合適的異常檢測方法
6. 熟悉異常檢測模型的評估方法與實務考量

---

## 1. 異常檢測概論

### 1.1 什麼是異常檢測？

**異常檢測 (Anomaly Detection)** 是一種識別與大多數數據顯著不同的樣本的技術。這些異常樣本可能代表：

- **製程故障或設備異常**：例如反應器溫度突然升高、壓力異常波動
- **產品品質異常**：例如產品純度突然下降、雜質含量異常升高
- **感測器故障**：例如溫度感測器讀數異常、壓力計失準
- **操作錯誤**：例如原料投料錯誤、操作參數設定錯誤
- **新穎操作模式**：例如新產品試產、新製程參數探索

### 1.2 核心術語定義

#### 1.2.1 異常 vs 離群值 vs 新奇點

| 術語 | 英文 | 定義 | 化工實例 |
|------|------|------|----------|
| **異常** | Anomaly | 廣義的異常數據點，包含離群值與新奇點 | 任何偏離正常操作的數據 |
| **離群值** | Outlier | 訓練數據中的異常點，通常被視為雜訊 | 感測器雜訊、操作失誤導致的異常數據 |
| **新奇點** | Novelty | 測試數據中的異常點，訓練數據假設為正常 | 新產品試產、未見過的故障模式 |

#### 1.2.2 異常檢測的三種情境

1. **離群值檢測 (Outlier Detection)**
   - 訓練數據包含異常樣本
   - 目標：識別並移除訓練數據中的異常點
   - 應用：數據清理、品質控制

2. **新奇點檢測 (Novelty Detection)**
   - 訓練數據僅包含正常樣本
   - 目標：識別測試數據中的新奇樣本
   - 應用：製程監控、故障檢測

3. **半監督異常檢測 (Semi-Supervised Anomaly Detection)**
   - 訓練數據有部分標籤 (正常/異常)
   - 目標：學習正常與異常的邊界
   - 應用：結合歷史故障數據的異常檢測

---

## 2. 異常檢測在化工領域的應用場景

### 2.1 製程安全監控

**應用情境**：即時監控化工製程，及早發現潛在安全風險

- **反應器異常監控**
  - 溫度、壓力異常波動檢測
  - 反應熱失控早期預警
  - 異常放熱反應識別

- **管線洩漏檢測**
  - 壓力降異常檢測
  - 流量異常識別
  - 溫度分布異常

- **設備健康監控**
  - 泵浦振動異常
  - 馬達電流異常
  - 軸承溫度異常

**案例**：某石化廠使用 Isolation Forest 監控反應器溫度，成功在反應失控前 15 分鐘發出預警，避免重大事故。

### 2.2 產品品質監控

**應用情境**：即時監控產品品質，確保產品符合規格

- **線上品質檢測**
  - 產品純度異常檢測
  - 雜質含量異常識別
  - 物性參數異常監控 (黏度、密度、折射率等)

- **批次品質分析**
  - 批次間品質一致性檢查
  - 異常批次識別
  - 品質趨勢分析

**案例**：某製藥廠使用 One-Class SVM 監控藥品純度，檢測出 3 個異常批次，避免不合格產品出廠。

### 2.3 設備預測性維護

**應用情境**：提前發現設備劣化徵兆，安排預防性維護

- **設備劣化檢測**
  - 泵效率下降檢測
  - 熱交換器結垢檢測
  - 觸媒活性衰退檢測

- **故障預警**
  - 軸承磨損預警
  - 閥門洩漏預警
  - 感測器漂移檢測

**案例**：某化工廠使用 LOF 監控離心泵振動數據，提前 2 週發現軸承異常，避免停機損失。

### 2.4 能源效率監控

**應用情境**：識別能源使用異常，優化能源管理

- **能耗異常檢測**
  - 異常高耗能操作識別
  - 蒸汽系統洩漏檢測
  - 冷卻水系統效率異常

- **公用設施監控**
  - 鍋爐效率異常
  - 空壓機效率下降
  - 冷凍機異常運轉

**案例**：某煉油廠使用 Elliptic Envelope 監控蒸汽系統，發現多處洩漏點，年節省蒸汽成本 500 萬元。

### 2.5 環境排放監控

**應用情境**：監控廢氣、廢水排放，確保符合環保法規

- **廢氣排放監控**
  - VOC 排放異常檢測
  - 粉塵濃度異常識別
  - 排放管道異常

- **廢水處理監控**
  - COD、BOD 異常升高
  - pH 異常波動
  - 重金屬超標檢測

**案例**：某化工廠使用異常檢測技術監控廢水 COD，即時發現廢水處理設施異常，避免排放超標罰款。

---

## 3. sklearn 模組中的異常檢測方法

### 3.1 方法總覽

sklearn 提供了多種異常檢測演算法，各有其特點與適用場景：

| 演算法 | 類別 | 主要特點 | 適用場景 |
|--------|------|----------|----------|
| **Isolation Forest** | 基於樹的方法 | 高效、適合高維數據、無需假設分布 | 大規模數據異常檢測 |
| **One-Class SVM** | 支持向量機 | 精確邊界、非線性核函數、適合小樣本 | 小樣本異常邊界建模 |
| **Local Outlier Factor (LOF)** | 基於密度 | 局部密度比較、適合非均勻分布 | 局部密度異常檢測 |
| **Elliptic Envelope** | 基於統計 | 假設高斯分布、快速、可解釋性強 | 高斯分布數據異常檢測 |

### 3.2 孤立森林 (Isolation Forest)

#### 3.2.1 核心思想

**核心概念**：異常點容易被孤立，正常點難以被孤立

- 隨機選擇特徵與分割點建立決策樹
- 異常點在樹中的路徑較短（易被孤立）
- 正常點在樹中的路徑較長（難被孤立）

#### 3.2.2 演算法優勢

✅ **高效能**：時間複雜度低，適合大規模數據  
✅ **高維適應**：對高維數據表現良好  
✅ **無需假設**：不需假設數據分布  
✅ **參數少**：僅需設定樹的數量與汙染比例

#### 3.2.3 化工應用場景

- **大規模製程數據監控**：數千個感測器、數百萬筆歷史數據
- **多變數異常檢測**：同時監控溫度、壓力、流量、成分等數十個變數
- **即時異常檢測**：要求快速響應的線上監控系統

**實例**：某石化廠使用 Isolation Forest 監控蒸餾塔 50 個感測器，每秒處理 1000 筆數據，成功檢測出 12 起異常事件。

### 3.3 一類支持向量機 (One-Class SVM)

#### 3.3.1 核心思想

**核心概念**：在特徵空間中找到包含大部分正常數據的最小超球面或超平面

- 使用核函數將數據映射到高維空間
- 在高維空間中建立決策邊界
- 超球面外的點被視為異常

#### 3.3.2 演算法優勢

✅ **精確邊界**：能夠建立精確的決策邊界  
✅ **非線性建模**：透過核函數處理非線性關係  
✅ **小樣本友好**：適合小樣本數據  
✅ **理論基礎**：有堅實的數學理論支持

#### 3.3.3 化工應用場景

- **小批量生產監控**：歷史數據有限，需要精確邊界
- **高價值產品品質控制**：對異常檢測精確度要求高
- **新產品試產**：正常操作數據量少

**實例**：某精細化工廠使用 One-Class SVM 監控高價值產品純度，僅用 200 筆正常數據訓練，成功檢測出 5 個異常批次。

### 3.4 區域性離群因子 (Local Outlier Factor, LOF)

#### 3.4.1 核心思想

**核心概念**：比較每個點與其鄰居的局部密度，識別局部稀疏區域的點

- 計算每個點的局部可達密度
- 比較該點與其鄰居的密度比值
- LOF > 1 表示該點比鄰居稀疏（可能是異常）

#### 3.4.2 演算法優勢

✅ **局部適應**：能夠處理密度不均勻的數據  
✅ **靈活性高**：對不同密度的群集都能有效檢測  
✅ **直觀解釋**：LOF 值越大，異常程度越高

#### 3.4.3 化工應用場景

- **多模式操作監控**：製程有多種正常操作模式，密度分布不均
- **局部異常檢測**：全局正常但局部異常的情況
- **動態製程監控**：製程操作點經常變動

**實例**：某化工廠使用 LOF 監控反應器多模式操作，成功識別出在某特定操作模式下的局部異常，而該異常在全局視角下容易被忽略。

### 3.5 橢圓包絡 (Elliptic Envelope)

#### 3.5.1 核心思想

**核心概念**：假設正常數據服從多變數高斯分布，建立橢圓形決策邊界

- 估計數據的均值與協方差矩陣
- 使用馬氏距離 (Mahalanobis Distance) 衡量點到中心的距離
- 超出橢圓邊界的點被視為異常

#### 3.5.2 演算法優勢

✅ **快速運算**：計算效率高，適合即時監控  
✅ **可解釋性**：基於統計理論，容易解釋  
✅ **適合高斯數據**：當數據接近正態分布時表現優異

#### 3.5.3 化工應用場景

- **穩態操作監控**：製程在穩態操作時，數據接近正態分布
- **品質統計管制**：產品品質參數通常接近正態分布
- **快速線上檢測**：需要快速響應的即時監控

**實例**：某食品廠使用 Elliptic Envelope 監控產品水分含量與糖度，假設正常產品的品質參數服從二維高斯分布，成功檢測出 8 個異常批次。

---

## 4. 異常檢測方法比較與選擇

### 4.1 優缺點對照表

| 方法 | 優點 | 缺點 | 時間複雜度 | 適合數據規模 |
|------|------|------|------------|--------------|
| **Isolation Forest** | • 高效快速<br>• 適合高維數據<br>• 無需假設分布<br>• 參數少 | • 對局部異常不敏感<br>• 隨機性強<br>• 難以解釋 | $O(n \log n)$ | 大規模 |
| **One-Class SVM** | • 精確邊界<br>• 非線性核函數<br>• 小樣本友好<br>• 理論完善 | • 計算成本高<br>• 參數敏感<br>• 不適合高維數據 | $O(n^2)$ ~ $O(n^3)$ | 小中規模 |
| **LOF** | • 局部適應性強<br>• 處理密度不均<br>• 直觀易懂 | • 計算成本高<br>• 參數選擇困難<br>• 不適合高維 | $O(n^2)$ | 小中規模 |
| **Elliptic Envelope** | • 快速高效<br>• 可解釋性強<br>• 理論完善 | • 假設高斯分布<br>• 對離群值敏感<br>• 線性邊界 | $O(n \cdot p^2)$ | 中大規模 |

*註： $n$ = 樣本數， $p$ = 特徵數*

### 4.2 演算法選擇決策樹

```
開始
│
├─ 數據是否服從高斯分布？
│   ├─ 是 → 數據規模大嗎？
│   │   ├─ 是 → Elliptic Envelope (快速、可解釋)
│   │   └─ 否 → Elliptic Envelope 或 One-Class SVM
│   └─ 否 → 繼續
│
├─ 數據規模大嗎 (n > 10000)？
│   ├─ 是 → 特徵維度高嗎 (p > 100)？
│   │   ├─ 是 → Isolation Forest (高維高效)
│   │   └─ 否 → Isolation Forest 或 Elliptic Envelope
│   └─ 否 → 繼續
│
├─ 數據密度分布均勻嗎？
│   ├─ 否 → 需要檢測局部異常？
│   │   ├─ 是 → LOF (局部密度敏感)
│   │   └─ 否 → Isolation Forest
│   └─ 是 → 繼續
│
├─ 訓練數據量少嗎 (n < 1000)？
│   ├─ 是 → 需要精確邊界？
│   │   ├─ 是 → One-Class SVM (小樣本精確)
│   │   └─ 否 → Isolation Forest
│   └─ 否 → Isolation Forest (通用選擇)
│
└─ 綜合考量：計算資源、可解釋性、實務需求
```

### 4.3 化工應用場景對照表

| 應用場景 | 推薦方法 | 理由 | 實務考量 |
|----------|----------|------|----------|
| **大規模製程監控**<br>(>100 感測器, >百萬筆數據) | Isolation Forest | 高效、高維適應、快速響應 | 需要定期重訓練模型 |
| **小批量高價值產品**<br>(歷史數據 < 500 筆) | One-Class SVM | 小樣本友好、精確邊界 | 需仔細調整超參數 |
| **多模式操作監控**<br>(密度不均勻分布) | LOF | 局部密度適應、多模式友好 | 計算成本較高 |
| **穩態品質管制**<br>(數據接近正態分布) | Elliptic Envelope | 快速、可解釋、統計基礎 | 需驗證正態假設 |
| **反應器故障預警**<br>(即時監控、快速響應) | Isolation Forest | 快速運算、即時檢測 | 結合領域知識設定閾值 |
| **設備振動監控**<br>(非線性、局部異常) | LOF 或 One-Class SVM | 捕捉非線性關係 | 需要專家標註部分數據 |
| **感測器故障檢測**<br>(單變數或低維) | Elliptic Envelope | 快速、易於實施 | 可結合統計管制圖 |

### 4.4 選擇策略總結

#### 4.4.1 優先考量因素

1. **數據規模與維度**
   - 大規模高維 → Isolation Forest
   - 小樣本低維 → One-Class SVM 或 Elliptic Envelope

2. **數據分布特性**
   - 高斯分布 → Elliptic Envelope
   - 非均勻分布 → LOF
   - 分布未知 → Isolation Forest

3. **計算資源限制**
   - 即時監控 → Isolation Forest 或 Elliptic Envelope
   - 離線分析 → 所有方法皆可

4. **可解釋性需求**
   - 高可解釋性 → Elliptic Envelope (統計基礎)
   - 低可解釋性 → Isolation Forest (黑盒模型)

#### 4.4.2 實務建議

✅ **先從簡單方法開始**：先嘗試 Elliptic Envelope，驗證數據是否接近高斯分布  
✅ **比較多種方法**：同時訓練多個模型，比較檢測效果  
✅ **結合領域知識**：根據製程特性調整參數與閾值  
✅ **定期重訓練**：製程特性可能隨時間變化，需定期更新模型  
✅ **建立回饋機制**：收集現場工程師對異常告警的回饋，持續優化模型

---

## 5. 資料前處理 (Data Preprocessing)

### 5.1 為什麼異常檢測需要資料前處理？

異常檢測演算法對數據的尺度和分布非常敏感，適當的前處理可以：

- **消除尺度差異**：不同感測器的數值範圍可能差異極大 (如溫度 vs 壓力)
- **提升演算法效能**：標準化後的數據可以加速收斂
- **改善檢測效果**：某些演算法假設數據已標準化 (如 Elliptic Envelope)
- **處理類別變數**：將類別資訊轉換為數值特徵

### 5.2 常用前處理技術

#### 5.2.1 資料標準化 (Standardization)

**定義**：將數據轉換為均值為 0、標準差為 1 的分布

$$
z = \frac{x - \mu}{\sigma}
$$

其中 $\mu$ 為均值， $\sigma$ 為標準差

**sklearn 實作**：`StandardScaler`

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**適用情境**：
- ✅ 數據接近正態分布
- ✅ 特徵尺度差異大
- ✅ Elliptic Envelope、One-Class SVM

**化工實例**：標準化反應器數據 (溫度 300-400°C, 壓力 10-50 bar, 流量 100-500 kg/h)

#### 5.2.2 資料正規化 (Normalization / Min-Max Scaling)

**定義**：將數據縮放到指定範圍 (通常為 [0, 1])

$$
x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

**sklearn 實作**：`MinMaxScaler`

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
```

**適用情境**：
- ✅ 數據分布不是正態分布
- ✅ 數據有明確的上下界
- ✅ 需要保持數據的原始分布形狀

**化工實例**：正規化產品品質參數 (純度 95-99.9%, 水分 0.1-2%, pH 6-8)

#### 5.2.3 穩健標準化 (Robust Scaling)

**定義**：使用中位數與四分位數進行標準化，對離群值不敏感

$$
x_{\text{robust}} = \frac{x - \text{median}(x)}{\text{IQR}(x)}
$$

其中 $\text{IQR} = Q_3 - Q_1$ (四分位距)

**sklearn 實作**：`RobustScaler`

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_robust = scaler.fit_transform(X)
```

**適用情境**：
- ✅ 訓練數據包含離群值
- ✅ 離群值檢測 (Outlier Detection) 場景
- ✅ 數據分布偏斜

**化工實例**：處理含有感測器雜訊的歷史製程數據

#### 5.2.4 類別變數編碼 (Categorical Encoding)

**One-Hot Encoding**：將類別變數轉換為二元向量

**sklearn 實作**：`OneHotEncoder`

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X_categorical)
```

**適用情境**：
- ✅ 類別變數無順序關係 (如反應器類型: A, B, C)
- ✅ 類別數量不多 (< 10)

**Label Encoding**：將類別變數轉換為整數

**sklearn 實作**：`LabelEncoder`

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
X_encoded = encoder.fit_transform(X_categorical)
```

**適用情境**：
- ✅ 類別變數有順序關係 (如品質等級: Low, Medium, High)
- ✅ 樹狀模型 (Isolation Forest)

**化工實例**：編碼原料供應商 (Supplier A, B, C)、反應器類型 (CSTR, PFR, Batch)

### 5.3 前處理方法選擇指南

| 演算法 | 推薦前處理 | 理由 |
|--------|------------|------|
| **Isolation Forest** | 標準化或正規化 (可選) | 對尺度不敏感，但標準化可提升效能 |
| **One-Class SVM** | **標準化 (必要)** | 核函數對尺度敏感，必須標準化 |
| **LOF** | **標準化 (必要)** | 距離計算對尺度敏感 |
| **Elliptic Envelope** | **標準化 (必要)** | 假設標準化後的高斯分布 |

### 5.4 化工數據前處理實務流程

```
原始數據
    ↓
1. 數據探索 (EDA)
   - 檢查缺失值、離群值、數據分布
   - 視覺化特徵分布
    ↓
2. 缺失值處理
   - 刪除 (缺失率 > 30%)
   - 插補 (線性插值、前向填補、均值填補)
    ↓
3. 離群值處理 (可選)
   - 保留：離群值檢測任務
   - 移除：新奇點檢測任務
    ↓
4. 特徵工程
   - 時間特徵 (小時、星期、季節)
   - 衍生特徵 (溫差、壓差、反應熱)
    ↓
5. 尺度轉換
   - 選擇合適的標準化方法
   - fit_transform 訓練集
   - transform 測試集
    ↓
6. 類別編碼
   - One-Hot Encoding 或 Label Encoding
    ↓
準備好的數據 → 訓練異常檢測模型
```

### 5.5 前處理注意事項

⚠️ **避免數據洩漏 (Data Leakage)**
- 僅在訓練集上 `fit` 前處理器
- 測試集僅使用 `transform`

```python
# ✅ 正確做法
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform
X_test_scaled = scaler.transform(X_test)        # 僅 transform

# ❌ 錯誤做法
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)    # 錯誤！測試集不應 fit
```

⚠️ **離群值檢測 vs 新奇點檢測**
- 離群值檢測：可使用穩健標準化 (RobustScaler)
- 新奇點檢測：使用標準標準化 (StandardScaler)，假設訓練集無異常

⚠️ **保留時間資訊**
- 時間序列數據：避免打亂順序
- 考慮使用滑動窗口或時間衍生特徵

---

## 6. 模型評估 (Model Evaluation)

異常檢測的評估具有特殊挑戰：異常樣本通常極少或完全缺乏標籤。因此評估方法需根據標籤可用性分為三種情境。

### 6.1 有部分標籤時 (半監督場景)

當歷史數據中有部分異常事件已被標註時，可使用傳統分類評估指標。

#### 6.1.1 混淆矩陣 (Confusion Matrix)

|  | 預測為正常 | 預測為異常 |
|---|-----------|-----------|
| **實際為正常** | True Negative (TN) | False Positive (FP)<br>*第一類錯誤 (誤報)* |
| **實際為異常** | False Negative (FN)<br>*第二類錯誤 (漏報)* | True Positive (TP) |

**sklearn 實作**：

```python
from sklearn.metrics import confusion_matrix

# y_true: 真實標籤 (1=異常, -1=正常)
# y_pred: 模型預測 (1=異常, -1=正常)
cm = confusion_matrix(y_true, y_pred)
```

#### 6.1.2 精確率 (Precision)

**定義**：預測為異常的樣本中，實際為異常的比例

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**意義**：反映異常告警的可靠性，精確率高表示誤報少

**化工實務**：精確率低會導致現場工程師頻繁處理假告警，降低系統可信度

**sklearn 實作**：

```python
from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred, pos_label=-1)
```

#### 6.1.3 召回率 (Recall)

**定義**：實際為異常的樣本中，被正確預測的比例

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**意義**：反映異常檢測的完整性，召回率高表示漏報少

**化工實務**：召回率低會導致嚴重故障未被檢測，可能造成安全事故

**sklearn 實作**：

```python
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred, pos_label=-1)
```

#### 6.1.4 F1 分數 (F1-Score)

**定義**：精確率與召回率的調和平均數

$$
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**意義**：綜合評估檢測效果，平衡誤報與漏報

**sklearn 實作**：

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred, pos_label=-1)
```

#### 6.1.5 ROC 曲線與 AUC (ROC-AUC)

**ROC 曲線**：以不同閾值繪製真陽性率 (TPR) vs 假陽性率 (FPR)

$$
TPR = \frac{TP}{TP + FN} \quad , \quad FPR = \frac{FP}{FP + TN}
$$

**AUC (Area Under Curve)**：ROC 曲線下面積， $0.5 \leq AUC \leq 1.0$ 

- $AUC = 0.5$ ：隨機猜測
- $AUC = 1.0$ ：完美檢測
- $AUC > 0.8$ ：通常認為表現良好

**sklearn 實作**：

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 使用異常分數 (scores) 繪製 ROC 曲線
fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=-1)
auc = roc_auc_score(y_true, scores)

plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

### 6.2 完全無標籤時 (純非監督場景)

當完全沒有標籤時，需使用替代評估方法。

#### 6.2.1 視覺化檢驗

**方法**：繪製數據散點圖，標記異常點

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 降維至 2D 以便視覺化
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# 繪製散點圖，異常點用紅色標記
colors = ['red' if pred == -1 else 'blue' for pred in y_pred]
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Anomaly Detection Results')
plt.show()
```

**評估方式**：
- 異常點是否確實位於稀疏區域？
- 異常點是否在邊界或極端位置？
- 異常點數量是否合理？(通常 1-5%)

#### 6.2.2 專家領域知識驗證

**方法**：請製程工程師檢視被標記為異常的樣本

**評估方式**：
- 異常點對應的時間點是否有已知事件？(如設備故障、操作調整)
- 異常點的特徵值是否超出正常操作範圍？
- 工程師是否認同這些點為異常？

**實務流程**：
1. 列出異常分數最高的前 20 個樣本
2. 提供時間戳記與特徵值給工程師
3. 工程師回饋：真異常 / 假異常 / 不確定
4. 計算工程師認可的異常比例作為評估指標

#### 6.2.3 歷史事件回溯驗證

**方法**：檢查模型是否能檢測出已知的歷史異常事件

**評估方式**：
- 收集歷史故障報告、停機記錄
- 檢查這些事件對應的數據是否被標記為異常
- 計算歷史事件捕獲率

**範例**：
- 某化工廠過去一年有 8 起設備故障事件
- 模型成功檢測出其中 7 起
- 歷史事件捕獲率 = 7/8 = 87.5%

#### 6.2.4 統計量檢驗

**方法**：使用統計檢定驗證異常點是否顯著偏離正常分布

**Z-Score 檢驗**：

$$
z = \frac{x - \mu}{\sigma}
$$

通常 $|z| > 3$ 被視為異常 (3-sigma 規則)

**箱型圖檢驗**：

$$
\text{Outlier if } x < Q_1 - 1.5 \times IQR \text{ or } x > Q_3 + 1.5 \times IQR
$$

**sklearn 實作**：

```python
import numpy as np
from scipy import stats

# 計算 Z-Score
z_scores = np.abs(stats.zscore(X))
statistical_outliers = (z_scores > 3).any(axis=1)

# 比較模型預測與統計方法
agreement = (y_pred == -1) & statistical_outliers
agreement_rate = agreement.sum() / (y_pred == -1).sum()
print(f"Agreement with statistical method: {agreement_rate:.2%}")
```

---

### 6.3 化工領域評估的特殊考量

#### 6.3.1 成本不對稱性 (Cost Asymmetry)

**問題**：化工領域中，漏報 (FN) 與誤報 (FP) 的成本極度不對稱

- **漏報成本**：未檢測出反應器過熱 → 爆炸事故 → 數億元損失
- **誤報成本**：誤報導致停機檢查 → 生產損失 → 數十萬元損失

**對策**：應容忍較高的誤報率，以換取更低的漏報率

**sklearn 實作**：調整 `contamination` 參數，增加異常檢測的靈敏度

```python
from sklearn.ensemble import IsolationForest

# 提高 contamination，增加檢測靈敏度 (降低漏報風險)
model = IsolationForest(contamination=0.05, random_state=42)  # 5% 異常率
```

#### 6.3.2 告警抑制 (Alarm Suppression)

**問題**：過多的誤報會導致「告警疲勞」，工程師可能忽略真正的告警

**對策**：設計告警抑制機制

- **時間窗口抑制**：同一變數在 10 分鐘內只發出一次告警
- **連續確認**：連續 3 個時間點檢測為異常才發出告警
- **多變數聯合**：多個變數同時異常才發出告警

**範例**：

```python
def suppress_alarms(anomaly_flags, window_size=3):
    """連續確認：連續 window_size 個點為異常才確認"""
    confirmed = []
    for i in range(len(anomaly_flags)):
        if i < window_size - 1:
            confirmed.append(False)
        else:
            # 檢查前 window_size 個點是否全為異常
            confirmed.append(all(anomaly_flags[i-window_size+1:i+1]))
    return confirmed
```

#### 6.3.3 業務指標評估

除了統計指標，還應評估業務影響：

| 業務指標 | 定義 | 計算方式 |
|---------|------|---------|
| **事故預防率** | 成功預警的嚴重事故比例 | 預警事故數 / 總事故數 |
| **誤報率** | 假告警佔總告警的比例 | FP / (TP + FP) |
| **平均預警時間** | 事故前多久發出告警 | 平均 (事故時間 - 告警時間) |
| **可用率** | 系統正常運行時間比例 | 正常運行時間 / 總時間 |
| **響應時間** | 從告警到處置的時間 | 平均 (處置時間 - 告警時間) |

#### 6.3.4 A/B 測試

**方法**：在生產環境中進行對照實驗

- **A 組**：使用傳統方法 (如固定閾值告警)
- **B 組**：使用新的異常檢測模型

**評估指標**：
- 告警準確率提升多少？
- 漏報事故減少多少？
- 工程師滿意度如何？

**實務注意**：
- 需謹慎設計實驗，確保安全
- 通常先在非關鍵設備上試驗
- 收集至少 3-6 個月數據才能得出可靠結論

---

## 7. 單元總結 (Unit Summary)

### 7.1 核心概念回顧

本單元介紹了異常檢測的五大核心概念：

1. **異常定義的三種類型**：
   - Point Anomaly (點異常)
   - Contextual Anomaly (情境異常)
   - Collective Anomaly (集體異常)

2. **sklearn 的四大異常檢測方法**：
   - Isolation Forest：基於決策樹，適合高維數據
   - One-Class SVM：基於邊界，適合低維數據
   - Local Outlier Factor：基於密度，適合非線性分布
   - Elliptic Envelope：基於高斯分布，適合正態分布數據

3. **方法選擇原則**：
   - 數據維度、分布特性、計算資源、可解釋性需求
   - 建議從 Isolation Forest 開始嘗試

4. **資料前處理的重要性**：
   - 標準化 (StandardScaler)：最常用
   - 最小-最大縮放 (MinMaxScaler)：保留分布形狀
   - 穩健縮放 (RobustScaler)：抗異常值干擾
   - **避免資料洩漏**：訓練集與測試集分開轉換

5. **評估方法的多元性**：
   - 有標籤：Precision, Recall, F1, ROC-AUC
   - 無標籤：視覺化、專家驗證、歷史回溯、統計檢驗
   - 化工特殊考量：成本不對稱、告警抑制、業務指標

### 7.2 化工領域最佳實踐

#### 7.2.1 實施流程建議

```
步驟 1: 問題定義
  └─ 明確檢測目標 (安全、品質、能耗？)
  └─ 確定成本不對稱性 (漏報 vs 誤報成本)

步驟 2: 數據準備
  └─ 收集歷史數據 (建議至少 3 個月)
  └─ 資料清洗與前處理
  └─ 特徵工程 (時間特徵、衍生特徵)

步驟 3: 方法選擇
  └─ 根據數據特性選擇演算法
  └─ 建議至少測試 2-3 種方法

步驟 4: 模型訓練
  └─ 使用正常數據訓練模型
  └─ 調整 contamination 參數

步驟 5: 評估與驗證
  └─ 使用多種評估方法 (統計 + 專家 + 歷史)
  └─ 重點關注歷史事故的捕獲率

步驟 6: 部署與監控
  └─ 設計告警抑制機制
  └─ 持續監控模型表現
  └─ 定期使用新數據重新訓練
```

#### 7.2.2 常見陷阱與對策

| 常見陷阱 | 後果 | 對策 |
|---------|------|------|
| **使用全部數據訓練** | 異常點被學習為正常 | 僅用已確認的正常數據訓練 |
| **忽略資料前處理** | 數值範圍大的特徵主導結果 | 必須進行標準化或縮放 |
| **過度依賴默認參數** | 模型表現不佳 | 調整 contamination 等參數 |
| **忽略時間相關性** | 情境異常檢測失敗 | 加入時間衍生特徵 |
| **誤報過多導致告警疲勞** | 真正告警被忽略 | 設計告警抑制機制 |
| **模型一次部署後不更新** | 製程變化後失效 | 定期重新訓練 (建議每 1-3 個月) |

#### 7.2.3 成功案例的共同特徵

1. **領域專家深度參與**：數據科學家與製程工程師緊密合作
2. **從簡單開始**：先在單一設備或子系統上驗證，再擴展
3. **持續迭代優化**：根據現場回饋不斷調整模型
4. **重視可解釋性**：工程師能理解為什麼某點被標記為異常
5. **建立閉環反饋**：告警 → 處置 → 驗證 → 模型更新

### 7.3 學習檢核清單

完成本單元後，你應該能夠：

- [ ] 解釋異常檢測的定義與三種類型
- [ ] 說明 sklearn 四大異常檢測方法的原理
- [ ] 根據數據特性選擇合適的演算法
- [ ] 正確執行資料前處理 (標準化、縮放)
- [ ] 避免常見錯誤 (資料洩漏、參數設定不當)
- [ ] 使用多種方法評估模型表現
- [ ] 設計告警抑制機制降低誤報影響
- [ ] 說明化工領域異常檢測的特殊考量

### 7.4 下一步學習方向

- **Unit08**：將學習多種異常檢測演算法的**實作與比較**
- **Unit09**：將探討**時間序列異常檢測**的專門方法
- **後續課程**：深度學習方法 (Autoencoder, VAE) 於異常檢測的應用

---

## 8. 參考資源 (References)

### 8.1 官方文件

- **sklearn 異常檢測總覽**：
  - [Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
  
- **sklearn 各方法文件**：
  - [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
  - [One-Class SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)
  - [Local Outlier Factor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
  - [Elliptic Envelope](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html)

- **sklearn 資料前處理**：
  - [Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html)
  - [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

### 8.2 經典論文

1. **Isolation Forest**：
   - Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation forest*. ICDM.

2. **One-Class SVM**：
   - Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., & Williamson, R. C. (2001). *Estimating the support of a high-dimensional distribution*. Neural Computation.

3. **Local Outlier Factor**：
   - Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). *LOF: identifying density-based local outliers*. SIGMOD.

### 8.3 Python 套件

本單元使用的主要套件：

```python
# 異常檢測演算法
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

# 資料前處理
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 評估指標
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

# 視覺化
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
```

### 8.4 延伸閱讀

- **書籍**：
  - *Outlier Analysis* by Charu C. Aggarwal
  - *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* by Aurélien Géron (Chapter on Anomaly Detection)

- **線上資源**：
  - [PyOD (Python Outlier Detection Library)](https://pyod.readthedocs.io/)：提供更多異常檢測演算法
  - [Anomaly Detection Resources on GitHub](https://github.com/yzhao062/anomaly-detection-resources)：論文與資源整理

### 8.5 作業預告

下一份作業 (Unit08) 將要求你：

1. **實作與比較**：在化工數據集上實作本單元介紹的四種演算法
2. **參數調整**：調整 `contamination` 等參數，觀察對結果的影響
3. **評估分析**：使用視覺化與統計方法評估模型表現
4. **撰寫報告**：說明方法選擇的理由與結果分析

**準備建議**：
- 複習本單元的所有方法原理
- 熟悉 sklearn 的 API 使用方式
- 思考如何將領域知識融入模型評估

---

**文件更新日期**：2025-01-23  
**課程**：逢甲大學化學工程學系 AI課程 (CHE-AI-114)  
**授課教師**：莊曜禎助理教授  
**單元**：Part 2 - Unit07 異常檢測概論

---
