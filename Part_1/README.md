# Part 1｜Python 基礎與探索性數據分析（EDA）

**課程代號**：CHE-AI-101 Part 1  
**課程性質**：化工資料科學與機器學習基礎必修課程  
**更新日期**：2025-12-17  
**版本**：v2.0 (Restructured)

---

## 🎯 課程目標

本 Part 旨在建立化工領域**數據分析與前處理**的堅實基礎，涵蓋：

1. **Python 數據科學工具**：NumPy 張量運算、Pandas 數據處理
2. **探索性數據分析（EDA）**：視覺化、統計描述、相關性分析
3. **化工時序數據清理**：缺失值處理、時間對齊、製程特徵工程
4. **批次與連續製程數據**：數據結構化、批次品質分析
5. **機器學習前處理 SOP**：Train/Test Split、標準化、Pipeline、Data Leakage 防範

**核心特色**：
- ✅ 從通用 Python 工具延伸到**化工專業場景**
- ✅ 強調**時序數據的特殊性**（DCS/SCADA 數據處理）
- ✅ 建立**可複用的前處理 SOP**，避免後續單元重踩陷阱
- ✅ 所有案例可在 **Colab + 本地環境**執行

**為什麼化工需要 Python + ML？**

| 化工問題 | ML 方法 | 預期效益 |
|---------|--------|---------|
| 品質即時預測 | 軟測器 (Soft Sensor) | 節省 Lab 分析成本 60%，提升調控反應速度 |
| 異常提前預警 | 時序異常偵測 (Anomaly Detection) | 減少非計畫停機 40%，延長設備壽命 |
| 操作條件優化 | 貝氏優化 (Bayesian Optimization) | 提升產率 5-15%，降低能耗 |
| 設備健康管理 | 剩餘使用壽命預測 (RUL) | 從被動維修轉為預測性維護 |

---

## 📚 課程單元結構

### 📊 Unit01｜Python 工具打底 + 基本 EDA

**檔案**：
- 📄 講義：[Unit01_Python_EDA_Basics.md](Unit01_Python_EDA_Basics.md)
- 💻 實作：[Unit01_Python_EDA_Basics.ipynb](Unit01_Python_EDA_Basics.ipynb)
- 📁 輸出：`Unit01_Results/` (7 files)

**學習目標**：
- **NumPy 張量基礎**：向量化運算、reshape、廣播機制
- **張量思維**：從 1D (單變量趨勢) → 2D (多變量時序) → 3D (批次數據)
- **Pandas 數據處理**：DataFrame 操作、缺失值處理、類別編碼
- **基本 EDA 流程**：描述統計、相關性分析、視覺化

**理論深度**：
```python
# NumPy 向量化優勢：SIMD (單指令多數據流)
# Python list: 異質容器 (Pointers) → 型別檢查 + 解參考 → 慢
# NumPy ndarray: 同質連續記憶體 → CPU Cache Locality → 快 100-1000x
```

**核心概念**：
- **張量思維**：數據不只是表格，而是多維度的物理意義
  - 1D: 單一變數隨時間 $T(t)$
  - 2D: 多變量時序 $\mathbf{X} \in \mathbb{R}^{T \times d}$
  - 3D: 批次製程 $\mathcal{X} \in \mathbb{R}^{N_{batch} \times T \times d}$

**實戰案例**：
- 🚢 **Titanic 生存預測 EDA**（經典入門案例）
  - 缺失值視覺化與處理策略
  - 類別變數編碼（Gender → 0/1）
  - 相關性熱圖分析
  - 為 Part 2 建模奠定基礎

- 📊 **批次製程品質分析**（延伸案例）
  - 分子量 (Mn_kDa) vs 反應溫度/時間
  - Pairplot、箱型圖、相關性矩陣
  - 線性回歸初探

**輸出圖表**：
- `04_missing_values.png` - 缺失值視覺化
- `05_correlation_matrix.png` - 相關性熱圖
- `06_batch_pairplot.png` - 批次變數關聯圖
- `07-10_batch_*.png` - 批次 EDA 系列圖表

**技術亮點**：
- ✅ 強調**張量物理意義** > 盲目 reshape
- ✅ NumPy 向量化實測效能差異（100-1000x）
- ✅ 從通用 Titanic 延伸到化工批次數據

**預計學習時間**：4-6 小時

---

### 🕐 Unit02｜化工時間序列清理 SOP + 製程監控特徵

**檔案**：
- 📄 講義：[Unit02_TimeSeries_Cleaning.md](Unit02_TimeSeries_Cleaning.md)
- 💻 實作：[Unit02_TimeSeries_Cleaning.ipynb](Unit02_TimeSeries_Cleaning.ipynb)
- 📁 輸出：`Unit02_Results/` (6 files)

**學習目標**：
- **時間戳處理**：`to_datetime`、時區轉換、DatetimeIndex
- **缺失值與頻率差**：插值（線性/FFill/BFill）、重採樣
- **製程監控特徵**：Rolling Mean / Std / Diff
- **化工時序陷阱**：時區混亂、單位不一致、停機數據混入

**⚠️ 化工時序數據的 5 大常見陷阱**：

| 陷阱 | 錯誤行為 | 正確做法 |
|------|---------|---------|
| **時區混亂** | DCS (UTC) + Lab (Local) 混用 | 統一轉 UTC 再分析 |
| **過度插值** | 停機 12 小時也線性插值 | 設 `limit` 或標記停機段 |
| **頻率不一致** | 溫度 1min + 壓力 5min 直接合併 | `resample` 對齊或 `ffill` |
| **時間洩漏** | Rolling 窗口包含未來資料 | 只用歷史窗口 `min_periods` |
| **單位混用** | ℃ vs ℉ / bar vs psi 混在同欄 | 統一單位後再分析 |

**核心技術**：

```python
# 時間處理範例
pd.to_datetime(df['Time'], format='%Y/%m/%d %H:%M:%S', errors='coerce')
df.set_index('Time').resample('1min').mean()  # 重採樣至 1 分鐘

# 製程監控特徵
df['Temp_MA'] = df['Temp'].rolling(window=60, min_periods=1).mean()
df['Temp_Std'] = df['Temp'].rolling(window=60, min_periods=1).std()
df['Temp_Diff'] = df['Temp'].diff()  # 一階差分（變化率）
```

**化工案例**：
- 🏭 **反應器溫度監控**：Rolling Mean 平滑噪音、Diff 偵測突變
- 📊 **停機段處理**：識別長時間缺失、標記非生產時段
- 🎯 **過度插值陷阱**：實測 12 小時線性插值 vs 正確做法對比

**輸出圖表**：
- `01_interpolation.png` - 插值方法比較（線性/FFill/BFill）
- `03_rolling_mean.png` - 移動平均平滑效果
- `03_diff.png` - 一階差分（變化率偵測）
- `03_rolling_std.png` - 移動標準差（波動監控）
- `03_combined.png` - 綜合監控特徵視覺化
- `trap09_over_interpolation.png` - 過度插值警示案例

**技術亮點**：
- ✅ 完整的時區處理教學（Taipei → UTC → NY）
- ✅ **時間洩漏陷阱**：Rolling 窗口只能用歷史資料
- ✅ 實務經驗：DCS 數據常見格式與解決方案

**預計學習時間**：5-7 小時

---

### 🏭 Unit03｜化工場資料型態（Batch / Continuous）

**檔案**：
- 📄 講義：[Unit03_ChemEng_Data_Types.md](Unit03_ChemEng_Data_Types.md)
- 💻 實作：[Unit03_ChemEng_Data_Types.ipynb](Unit03_ChemEng_Data_Types.ipynb)
- 📁 輸出：`Unit03_Results/` (10 files)

**學習目標**：
- 理解化工數據的 **3 種典型型態**
- 掌握正確的數據結構表示法
- 避免批次邊界洩漏（Data Leakage）
- 進行批次品質 EDA 與非線性效應探索

**化工數據的 3 種典型型態**：

#### 1️⃣ 連續製程（Continuous Process）
```
特徵：長時間穩態運行，偶有品級切換、擾動、漂移
數據形式：(time, features) → 2D 時序數據
常見目標：軟測器、異常偵測、控制優化
案例：煉油廠蒸餾塔、化纖生產線
```

#### 2️⃣ 批次製程（Batch Process）
```
特徵：每批有明確開始/結束，分階段操作（升溫→反應→降溫）
數據形式：(batch, time, features) → 3D 張量 或 長表 (BatchID, Time, Tags)
常見目標：批次品質預測、一致性分析、批次異常偵測
案例：聚合反應、製藥發酵、精細化學品合成
```

#### 3️⃣ 混合型態（Hybrid）
```
特徵：啟停機、品級切換、多階段、Campaign
風險：不同階段資料混用 → Data Leakage
處理：明確標記階段邊界、分段建模
```

**⚠️ 批次數據的致命陷阱**：

| 陷阱 | 錯誤行為 | 後果 | 正確做法 |
|------|---------|------|---------|
| **批次邊界混合** | 把不同批次的時間軸當作連續 | 模型學到虛假關聯 | 明確 BatchID 分隔 |
| **未來資訊洩漏** | 用批次最終品質特徵預測中間狀態 | 測試失效 | 只用歷史資訊 |
| **不等長硬壓** | 強制 reshape 不等長批次 | 數據失真 | Padding + Masking |

**數據結構對比**：

```python
# 批次最終品質表（2D，易做 EDA/回歸）
BatchID | React_Temp | React_Time | Mn_kDa
--------|-----------|-----------|-------
001     | 180       | 240       | 45.2
002     | 185       | 250       | 48.1

# 批次過程曲線表（Long Format，保留動態）
BatchID | Time | Temp | Pressure | Flow
--------|------|------|----------|-----
001     | 0    | 25   | 1.0      | 0
001     | 1    | 30   | 1.1      | 5
...     | ...  | ...  | ...      | ...
```

**實戰案例**：
- 🧪 **批次品質 EDA**（2D 數據）
  - 分子量 vs 溫度/時間線性回歸
  - 相關性矩陣、Pairplot
  - 溫度分箱箱型圖

- 🔬 **進階非線性效應探索**（3D 視覺化）
  - 溫度 × 時間交互作用（3D Surface Plot）
  - 等高線圖（Contour Plot）
  - 組合效應箱型圖

**輸出圖表**：
- `01_continuous_resample.png` - 連續製程重採樣示例
- `06-10_batch_*.png` - 批次品質 EDA 系列（與 Unit01 共用）
- `11_batch_nonlinear.png` - 非線性效應視覺化
- `12_batch_interaction_3d.png` - 3D 交互作用圖
- `13_batch_contour.png` - 等高線圖
- `14_batch_combination_box.png` - 組合效應箱型圖

**技術亮點**：
- ✅ 明確區分 Batch vs Continuous 的建模策略
- ✅ **張量物理意義**重申（與 Unit01 呼應）
- ✅ 實務警示：批次邊界洩漏的真實案例

**預計學習時間**：4-5 小時

---

### 🔧 Unit04｜ML 前處理工作流程（建模前半）

**檔案**：
- 📄 講義：[Unit04_ML_Preprocessing_Workflow.md](Unit04_ML_Preprocessing_Workflow.md)
- 💻 實作：[Unit04_ML_Preprocessing_Workflow.ipynb](Unit04_ML_Preprocessing_Workflow.ipynb)
- 📁 輸出：`Unit04_Results/` (1 file)

**學習目標**：
- 理解監督式學習的**標準工作流程 SOP**
- 掌握 Train/Test Split 的原理與策略
- 理解標準化/正規化的物理意義與使用時機
- 用 Pipeline 避免前處理洩漏（Data Leakage）
- 理解交叉驗證（Cross-Validation）與時序數據的特殊處理

**監督式學習 SOP（後續所有單元通用）**：

```
1. 定義問題
   ├─ 輸入 X：特徵變數（溫度、壓力、流量...）
   ├─ 輸出 y：目標變數（品質、異常標籤...）
   └─ 評估指標：MAE/RMSE (回歸) 或 Accuracy/F1 (分類)
   
2. 切分資料
   ├─ i.i.d 資料：train_test_split(shuffle=True)
   └─ 時序資料：依時間順序切分 (shuffle=False) ⚠️ 重要！
   
3. 前處理
   ├─ 缺失值處理
   ├─ 類別編碼
   ├─ 標準化/正規化
   └─ 特徵工程
   
4. 訓練模型
   └─ 只用 Training Set
   
5. 評估模型
   └─ 只在 Test Set 評估 ⚠️ 不能看訓練資料！
   
6. 部署/保存
   └─ joblib / pickle
```

**⚠️ Train/Test Split 的關鍵原則**：

| 數據類型 | 切分方法 | 錯誤做法 | 後果 |
|---------|---------|---------|------|
| **i.i.d 資料** | `shuffle=True` 隨機切分 | 無 shuffle 可能導致特定類別偏移 | 評估偏差 |
| **時序資料** | 依時間順序 `shuffle=False` | 隨機打散 | **時間洩漏**（把未來混進訓練集） |
| **批次資料** | 依 BatchID 分組 | 同批次分散到 Train/Test | **批次洩漏** |

**理論深度**：

```
泛化誤差 (Generalization Error) 分解：
E[L(h(x), y)] = Bias² + Variance + Irreducible Error

- Bias: 模型複雜度不足（Underfitting）
- Variance: 模型對訓練資料過度敏感（Overfitting）
- Irreducible Error: 數據本身的噪音

Train/Test Split 目的：估計真實的泛化誤差
```

**標準化技術對比**：

#### 1️⃣ Z-score 標準化（Standardization）
```python
z = (x - μ) / σ

用途：
- 距離/相似度演算法（K-Means, PCA, SVM, KNN）
- 多變量異常偵測（Mahalanobis, MSPC）
- 特徵量級差異大（溫度 0-300℃ vs 壓力 0-50 bar）

化工案例：
- DCS Tag 標準化（溫度/壓力/流量同時建模）
- 主成分分析（PCA）前處理
```

#### 2️⃣ Min-Max 正規化（Normalization）
```python
x_norm = (x - x_min) / (x_max - x_min)

用途：
- 神經網路訓練（輸入範圍穩定）
- 變數有固定上下界（開度 0-100%）

化工案例：
- 閥門開度、轉速百分比等有界變數
- 深度學習前處理
```

**Pipeline 架構（避免洩漏）**：

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# 錯誤做法：先標準化全部資料再切分 → Data Leakage
scaler.fit(X)  # ❌ Test Set 資訊洩漏到 Training
X_scaled = scaler.transform(X)
X_train, X_test = train_test_split(X_scaled)

# 正確做法：Pipeline 確保只用 Training Set 統計量
pipeline = Pipeline([
    ('scaler', StandardScaler()),      # ✅ 只在 Train 計算 μ, σ
    ('model', LinearRegression())
])
pipeline.fit(X_train, y_train)        # 訓練
y_pred = pipeline.predict(X_test)     # 測試（用 Train 的 μ, σ）
```

**交叉驗證（Cross-Validation）**：

```python
# K-Fold CV：小樣本數據的可靠評估
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
print(f"平均 R² = {scores.mean():.3f} ± {scores.std():.3f}")

# 時序數據必須用 TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='neg_mean_squared_error')
```

**化工案例**：
- 📊 **時序數據切分示例**（DCS 溫度預測）
  - 正確：前 80% 訓練、後 20% 測試
  - 錯誤：隨機打散 → 未來資訊洩漏

- 🔬 **標準化效果對比**（多變量回歸）
  - 未標準化：不同量級特徵權重失衡
  - Z-score 標準化：特徵公平競爭

**輸出圖表**：
- `ts_example.png` - 時序數據切分示意圖

**技術亮點**：
- ✅ **課程核心之一**：建立後續所有單元的前處理基礎
- ✅ **時間洩漏警示**：化工 AI 最常見錯誤
- ✅ Pipeline 架構：工業部署標準做法

**預計學習時間**：3-4 小時

---

## 🎓 學習路徑建議

### 推薦學習順序

```
1. Unit01 (Python 工具 + 基本 EDA)
   ↓ 建立 NumPy/Pandas 基礎，理解張量思維
   
2. Unit02 (時序數據清理)
   ↓ 掌握化工時序數據的特殊處理技巧
   
3. Unit03 (批次 vs 連續製程)
   ↓ 理解數據型態差異與結構化方法
   
4. Unit04 (ML 前處理 SOP)
   ↓ 建立建模前的標準工作流程
   
→ 準備好進入 Part 2: 監督式學習建模
```

### 完整課程路徑

```
Part 1: Python 基礎 + EDA ← 你現在在這裡
    ↓ 學會處理化工數據、視覺化製程趨勢
    
Part 2: 監督式學習 (回歸、分類、樹模型)
    ↓ 學會建立預測模型、評估模型性能
    
Part 3: 化工特化應用 (軟測器、異常偵測)
    ↓ 解決實際化工場景問題
    
Part 4: 深度學習 (CNN、RNN、Transformer)
    ↓ 處理影像、時序、複雜模式
    
Part 5: 進階應用 (強化學習控制、RUL 預測)
    ↓ 自主優化控制、設備健康管理
```

### 先修知識檢查清單

**必備知識**：
- ✅ Python 基礎語法（變數、迴圈、函數、列表/字典）
- ✅ 基礎數學（四則運算、平均數、標準差）
- ✅ Excel 使用經驗（了解表格、欄位、篩選概念）

**建議先修（非必須）**：
- 📊 基礎統計學（相關係數、標準差、分位數）
- 🧮 線性代數基礎（向量、矩陣、維度概念）
- 🔬 化工基礎（了解 DCS、Tag、批次反應概念）

**如果完全沒有 Python 經驗**：
- 建議先完成 [Part_0/Unit00_Environment_Setup](../Part_0/Unit00_Colab_Environment_Setup.md)
- 或線上教學：[Python.org Tutorial](https://docs.python.org/3/tutorial/)

---

## 📊 課程特色與亮點

### ⭐ 核心優勢

1. **化工時序數據處理專章**（Unit02）
   - 業界罕見的完整時區/插值/重採樣教學
   - 5 大常見陷阱與解決方案
   - 實際 DCS 數據格式處理經驗

2. **批次 vs 連續製程清晰區分**（Unit03）
   - 化工領域獨有的數據型態分類
   - 批次邊界洩漏的真實案例警示
   - 3D 張量視覺化與物理意義

3. **ML 前處理 SOP 建立**（Unit04）
   - 後續所有建模單元的基礎
   - 時間洩漏陷阱的詳細說明
   - Pipeline 工業部署標準做法

4. **張量思維培養**（Unit01 + Unit03）
   - 數據不只是表格，而是物理意義
   - 從 1D → 2D → 3D 的完整思維訓練
   - NumPy 向量化實測效能差異

5. **可複現性極高**
   - 所有數據自包含（Titanic 線上下載或合成）
   - Colab + 本地環境雙支援
   - 路徑管理規範（REPO_ROOT / OUTPUT_DIR）

### 🏆 課程成果

完成本 Part 後，學生將能夠：

- ✅ 熟練使用 NumPy/Pandas 處理化工數據
- ✅ 正確處理時序數據（時區、缺失值、重採樣）
- ✅ 建立製程監控特徵（Rolling、Diff）
- ✅ 區分批次與連續製程，選擇正確建模策略
- ✅ 執行標準化 ML 前處理 SOP
- ✅ 避免 Data Leakage（時間洩漏、批次洩漏）
- ✅ 準備好進入 Part 2 監督式學習建模

---

## 📁 資料夾結構

```
Part_1/
├── README.md                                    # 本課程說明
├── IMPROVEMENT_RECOMMENDATIONS.md               # 改進建議
├── IMPROVEMENTS_COMPLETED.md                    # 已完成改進
├── NOTEBOOK_UPDATES_COMPLETED.md                # Notebook 更新紀錄
│
├── Unit01_Python_EDA_Basics.md                  # 講義：Python 工具 + EDA
├── Unit01_Python_EDA_Basics.ipynb               # 實作：Python 工具 + EDA
├── Unit01_Results/                              # 輸出檔案 (7 files)
│   ├── 04_missing_values.png
│   ├── 05_correlation_matrix.png
│   ├── 06_batch_pairplot.png
│   ├── 07_batch_mn_hist.png
│   ├── 08_batch_temp_vs_mn.png
│   ├── 09_batch_corr.png
│   └── 10_batch_box_by_temp_bin.png
│
├── Unit02_TimeSeries_Cleaning.md               # 講義：時序數據清理
├── Unit02_TimeSeries_Cleaning.ipynb            # 實作：時序數據清理
├── Unit02_Results/                              # 輸出檔案 (6 files)
│   ├── 01_interpolation.png
│   ├── 03_rolling_mean.png
│   ├── 03_diff.png
│   ├── 03_rolling_std.png
│   ├── 03_combined.png
│   └── trap09_over_interpolation.png
│
├── Unit03_ChemEng_Data_Types.md                # 講義：批次 vs 連續
├── Unit03_ChemEng_Data_Types.ipynb             # 實作：批次 vs 連續
├── Unit03_Results/                              # 輸出檔案 (10 files)
│   ├── 01_continuous_resample.png
│   ├── 06_batch_pairplot.png                    # 與 Unit01 共用批次案例
│   ├── 07_batch_mn_hist.png
│   ├── 08_batch_temp_vs_mn.png
│   ├── 09_batch_corr.png
│   ├── 10_batch_box_by_temp_bin.png
│   ├── 11_batch_nonlinear.png
│   ├── 12_batch_interaction_3d.png
│   ├── 13_batch_contour.png
│   └── 14_batch_combination_box.png
│
├── Unit04_ML_Preprocessing_Workflow.md         # 講義：ML 前處理 SOP
├── Unit04_ML_Preprocessing_Workflow.ipynb      # 實作：ML 前處理 SOP
└── Unit04_Results/                              # 輸出檔案 (1 file)
    └── ts_example.png
```

---

## 🚀 快速開始

### 環境設定

**方法一：Google Colab（推薦新手）**
```python
# 無需安裝，直接在 Colab 中開啟各單元的 .ipynb 檔案
# 所有套件（NumPy, Pandas, Matplotlib, Seaborn）預裝完成
```

**方法二：本地環境（推薦有經驗者）**
```bash
# 1. 建立虛擬環境
conda create -n chemeng_ml python=3.10
conda activate chemeng_ml

# 2. 安裝必要套件
pip install numpy pandas matplotlib seaborn scikit-learn scipy

# 3. 啟動 Jupyter Notebook
jupyter notebook
```

### 執行順序

1. 開啟 `Part_1/` 資料夾
2. 按照 Unit01 → Unit02 → Unit03 → Unit04 順序學習
3. 每個單元：
   - 📖 先閱讀 `.md` 講義（理解理論與概念）
   - 💻 再執行 `.ipynb` 實作（動手實踐）
   - 📊 檢查 `UnitXX_Results/` 輸出結果

---

## 💡 學習建議

### 對於學生

- 📝 **做筆記**：每個數學公式的物理意義（z-score、Rolling Mean 等）
- 🔬 **動手改**：修改參數觀察結果變化（窗口大小、插值方法等）
- 🤔 **多思考**：為什麼時序數據不能隨機 shuffle？
- 💬 **多討論**：與同學分享 Data Leakage 的真實案例
- ⚠️ **重點關注**：Unit02 的 5 大陷阱、Unit04 的時間洩漏

### 對於教師

- 🎯 **重點強調**：Unit02 的時序處理（最易錯）、Unit04 的 SOP（最重要）
- 📊 **課堂討論**：Data Leakage 的化工案例分享
- 🏭 **實務連結**：讓學生用實驗室數據練習
- 📚 **作業設計**：
  - Unit01: Titanic 完整 EDA 報告
  - Unit02: 實際 DCS 數據清理
  - Unit03: 批次品質影響因子分析
  - Unit04: 建立標準 Pipeline

### 進階挑戰

完成基礎學習後，可嘗試：

1. **Unit01 進階**：用 Seaborn FacetGrid 做多維度 EDA
2. **Unit02 進階**：實作 Kalman Filter 濾波（vs Rolling Mean）
3. **Unit03 進階**：Dynamic Time Warping (DTW) 對齊不等長批次
4. **Unit04 進階**：實作 Stratified Split 處理不平衡數據

---

## 🔗 與其他 Part 的連結

### Part 1 為後續課程奠定的基礎：

| Part 2 單元 | 使用 Part 1 技能 |
|-----------|----------------|
| **Unit05** (決策樹分類) | Unit01 EDA、Unit04 Train/Test Split |
| **Unit06** (交叉驗證) | Unit04 時序數據切分策略 ⚠️ |
| **Unit07** (回歸建模) | Unit01 相關性分析、Unit04 標準化 |
| **Unit08** (軟測器) | Unit02 時序特徵工程、Unit04 Pipeline |

| Part 3 單元 | 使用 Part 1 技能 |
|-----------|----------------|
| **時序異常偵測** | Unit02 Rolling Std、Diff 特徵 |
| **批次品質預測** | Unit03 批次數據結構化 |

**關鍵銜接點**：
- ✅ Unit04 建立的 SOP → Part 2 所有建模單元直接沿用
- ✅ Unit02 的時序特徵 → Part 2 Unit08 軟測器、Part 3 異常偵測
- ✅ Unit03 的批次概念 → Part 2 Unit06 GroupKFold、Part 3 批次建模

---

## 📞 支援與回饋

### 常見問題

**Q: 我沒有化工背景，可以學習本課程嗎？**
A: 可以！Part 1 是通用 Python 數據科學基礎，化工案例只是應用情境。建議先完成 Unit01-02，再決定是否深入化工領域。

**Q: NumPy 的向量化為什麼這麼快？**
A: NumPy 底層用 C 語言實作，數據連續儲存在記憶體中，CPU 可以用 SIMD（單指令多數據流）一次處理多個數據，比 Python 迴圈快 100-1000 倍。詳見 Unit01 理論說明。

**Q: 時序數據一定不能 shuffle 嗎？**
A: 對！如果 shuffle，會把「未來」資訊混進訓練集，導致模型在測試時失效。這是化工 AI 最常見的錯誤，詳見 Unit04。

**Q: Unit03 的批次數據和 Unit02 的時序數據有什麼不同？**
A: 批次數據是「多個獨立時序」，每批有自己的開始/結束。連續製程是「單一長時序」。建模策略不同：批次要避免批次邊界洩漏，連續要避免時間洩漏。

**Q: 我應該用 z-score 還是 Min-Max 標準化？**
A: 
- 距離/相似度演算法（K-Means, PCA, SVM）→ z-score
- 神經網路、有固定上下界的變數 → Min-Max
- 不確定時先用 z-score（更穩健）

### 技術支援

- 🐛 程式碼問題：檢查 Python 版本（推薦 3.9-3.11）
- 📦 套件安裝：使用 `conda` 或 `pip` 最新版本
- 💾 路徑問題：確認工作目錄在 `Part_1/` 或專案根目錄
- 🎨 圖表顯示：Colab 自動顯示，本地需 `%matplotlib inline`

---

## 📖 延伸學習資源

### 推薦教科書

- **Python 數據科學**：
  - *Python for Data Analysis* (Wes McKinney, Pandas 作者)
  - *Hands-On Machine Learning* (Aurélien Géron)

- **統計學基礎**：
  - *An Introduction to Statistical Learning* (Hastie et al.)
  - *Think Stats* (Allen B. Downey)

- **化工數據科學**：
  - *Data Science for Chemical Engineering* (專門教材較少，本課程填補此缺口)

### 線上資源

- [NumPy 官方文檔](https://numpy.org/doc/)
- [Pandas 官方文檔](https://pandas.pydata.org/docs/)
- [Scikit-learn 官方教學](https://scikit-learn.org/stable/tutorial/)
- [Seaborn 圖表範例](https://seaborn.pydata.org/examples/)

### 實務數據來源

- [Kaggle Datasets](https://www.kaggle.com/datasets)（Titanic, 時序數據）
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/)
- 化工製程數據：DCS/SCADA 匯出、實驗室批次紀錄

---

## 📋 更新紀錄

### v2.0 (2025-12-17) - 重構版本
- ✅ 建立完整 4 單元結構（Python EDA → 時序清理 → 批次/連續 → ML SOP）
- ✅ 新增 Unit02 化工時序處理 5 大陷阱
- ✅ 新增 Unit03 批次 vs 連續製程專章
- ✅ 新增 Unit04 ML 前處理 SOP（時間洩漏警示）
- ✅ 統一輸出路徑至 `UnitXX_Results/`
- ✅ 所有 notebook 測試通過

### v1.0 (2024) - 原始版本
- Python 基礎與 EDA 內容

---

**🎓 準備好開始學習了嗎？從 [Unit01](Unit01_Python_EDA_Basics.md) 開始吧！**

---

> **課程維護者**：化工資料科學教學團隊  
> **最後更新**：2025-12-17  
> **授權**：教育用途  
> **聯絡方式**：[課程問題請開 Issue 或聯繫助教]
