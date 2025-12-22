# Unit04｜Train/Test Split、標準化與 ML 流程（建模前半）

**課程名稱**：化工資料科學與機器學習實務（CHE-AI-101）  
**本單元定位**：把「資料 → 可訓練模型」之前的流程做成固定 SOP，避免後面每一個 Unit 都重踩相同的坑。  

---

## 本單元目標

- 理解為什麼要 Train/Test Split（避免 Overfitting 與 Data Leakage）  
- 會選擇合適的切分方式：i.i.d. 資料 vs 時間序列資料  
- 理解並會使用標準化/正規化（z-score、Min-Max）  
- 能把前處理與模型包進 `Pipeline`（讓訓練/測試一致）  

---

## 1. 監督式學習的最小工作流程（SOP）

典型流程（後續所有監督式單元都會沿用）：
1. 定義問題：輸入 $X$、輸出 $y$、評估指標（MAE/RMSE/Accuracy…）  
2. 切分資料：Train / Test（必要時再切 Validation）  
3. 前處理：缺值、類別編碼、標準化、特徵工程  
4. 訓練模型：用 Train 擬合  
5. 評估模型：只在 Test 評估（避免「用考題練習」）  
6. 保存模型：部署或交作業用（`joblib` / `pickle`）  

---

## 2. 為什麼要 Train/Test Split？（避免 Overfitting）

如果把所有資料都拿來訓練，模型在「訓練資料」上會看起來非常厲害，但一換到新資料就崩潰，這就是 **過擬合（Overfitting）**。

**理論背景：泛化誤差 (Generalization Error)**

我們真正關心的不是模型背誦歷史資料的能力（Training Error），而是它對未見過資料的預測能力（Test Error）。
根據 **I.I.D. 假設**（Independent and Identically Distributed），我們隨機切分出的測試集可以作為真實世界數據的無偏估計 (Unbiased Estimator)。

為了公平評估模型的泛化能力，我們通常會將資料切成兩部分：

- **訓練集（Training Set，約 70–80%）**：  
  - 用來「教」模型。
- **測試集（Test Set，約 20–30%）**：  
  - 用來「考」模型，這部分資料在訓練過程中不讓模型看到。

> 就像考試：練習題（訓練集）可以看答案反覆練，但期末考（測試集）的題目不能事先曝光。

### 2.1 i.i.d. 資料 vs 時間序列資料（很重要）

- **一般 i.i.d. 資料**（例如 Titanic）：可用隨機切分 `train_test_split(..., shuffle=True)`。  
- **時間序列資料**（製程資料最常見）：**不能隨機打散**，否則會把「未來」資訊混進訓練集（時間洩漏）。  
  - 做法：依時間順序切分（例如前 80% 訓練、後 20% 測試），或使用 `TimeSeriesSplit`。  

### 2.2 進階概念：交叉驗證 (Cross-Validation)

在化工實驗中，數據往往很珍貴且數量稀少（Small Data）。如果只切一次 Train/Test，剛好切到比較「簡單」或「極端」的測試資料，評估結果可能不準。

**理論詳解：K-Fold Cross-Validation**

為了降低評估的變異數 (Variance)，我們採用 $K$-Fold 交叉驗證。
1.  將原始資料集 $D$ 隨機分割成 $K$ 個互斥的子集 (Folds)：$D_1, D_2, \dots, D_K$。
2.  進行 $K$ 次迭代，每次取第 $k$ 個子集 $D_k$ 作為**驗證集 (Validation Set)**，其餘 $K-1$ 個子集作為**訓練集**。
3.  計算 $K$ 次的評估指標 $E_k$ (例如 Accuracy)，最終的模型效能為平均值：

$$ E_{CV} = \frac{1}{K} \sum_{k=1}^{K} E_k $$

這種方法確保了**每一筆資料都有機會被當作測試資料**，且恰好一次。這對於小樣本數據（如化工批次實驗數據）特別重要，能提供更可信的泛化誤差估計。

**結果分析：**
*   **平均準確率 (Mean Accuracy)**：這代表模型在未見過資料上的預期表現。如果這個數值與之前的單次 Test Accuracy 差異很大，代表之前的切分可能剛好運氣特別好或特別差。
*   **標準差 (Standard Deviation)**：代表模型表現的波動程度。
    *   若數值很小（例如 < 2%），代表模型很**穩定 (Stable)**，不管資料怎麼切，表現都差不多。
    *   若數值很大，代表模型對訓練資料的選取非常敏感，可能有 **Overfitting** 或資料量不足的問題。

---

## 3. 標準化 / 正規化：讓變數尺度可比較

### 3.1 z-score 標準化（Standardization）

z-score 轉換定義為：
$$
z = \frac{x - \mu}{\sigma}
$$

用途（化工情境）：
- 距離/相似度相關演算法（K-Means、PCA、SVM、KNN…）  
- 多變量異常偵測（z-score、Mahalanobis distance、MSPC）  
- 特徵量級差異很大的時候（溫度、壓力、流量混在一起）  

### 3.2 Min-Max 正規化（Normalization）
$$
x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}
$$
用途：
- 神經網路訓練（特別是輸入範圍需要穩定時）  
- 指標本來就有固定上下界（例如開度 0–100%）  

---

## 4. 用 Pipeline 避免「前處理洩漏」

重點：`Scaler/Encoder` 只能用訓練集 `fit`，測試集只能 `transform`。  
最省事、最不容易出錯的方式是用 `Pipeline` 把流程包起來：

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=200)),
])

model.fit(X_train, y_train)
print("test score:", model.score(X_test, y_test))
```

---

## 5. 本單元程式演練

請開啟並完成：`Part_1/Unit04_ML_Preprocessing_Workflow.ipynb`  
- 會示範：i.i.d. 隨機切分 vs 時間序列切分  
- 會示範：`StandardScaler` + `Pipeline` 的正確用法（避免洩漏）  

