# Unit00｜Google Colab 環境設定與基本操作

**課程**：AI 在化工上的應用（CHE-AI）  
**本單元定位**：建立「之後所有單元都共用」的 Colab 執行環境與 Notebook 操作習慣。  

---

## 0. 你要達成的 3 件事

1. **會用 Colab 上課**：開副本、掛載 Drive、切換 Runtime（CPU/GPU）、重跑/重啟、保存輸出。  
2. **會管理套件**：用 `pip` 安裝、確認版本、處理 `ModuleNotFoundError`、固定環境指紋。  
3. **建立一致的專案結構**：所有 Unit 都用同一套資料夾、路徑變數與輸出規範。  

---

## 1. Colab 上課 SOP（每次開新 Notebook 都照做）

### Step A｜開副本（避免寫在唯讀檔）
1. 用瀏覽器打開老師提供的 Colab 連結。  
2. 左上角：**檔案 (File) → 在雲端硬碟中儲存副本 (Save a copy in Drive)**。  
3. 以後都在自己的副本上操作與交作業。  

### Step B｜設定 Runtime（需要 GPU 的單元才切）
1. **執行階段 (Runtime) → 變更執行階段類型 (Change runtime type)**  
2. **硬體加速器**：選 `None`（一般單元）或 `T4 GPU`（CNN/LSTM/RUL/RL 單元常用）  
3. 用 `!nvidia-smi` 檢查 GPU 是否可用。  

### Step C｜掛載 Google Drive（保存資料、模型、圖）
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step D｜設定「課程根目錄」與統一路徑
建議每個 Notebook 都有這兩行（後續單元會一直用到）：
```python
COURSE_ROOT = "/content/drive/MyDrive/CHE-AI"
DATA_DIR = f"{COURSE_ROOT}/data"
```

### Step E｜安裝（或檢查）本課程共用套件
用 `pip` 安裝/更新：`numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `joblib`。  
（深度學習、RDKit 等會在對應單元再補充或選配）  

---

## 2. Notebook 操作習慣（會直接影響你是否「跑得起來」）

### 2.1 永遠按順序執行
- 不要跳著跑：前處理、變數、安裝套件通常在前面。  
- 如果你不確定狀態，最省時間的方法是：**Runtime → Restart runtime → Run all**。  

### 2.2 「一次只做一件事」的 Cell
把安裝、路徑、資料讀取、模型訓練、輸出畫圖分開寫，Debug 才快。  

### 2.3 你必須會看錯誤訊息
最常見的兩種：
- `ModuleNotFoundError`：沒裝套件或裝在錯的 runtime → `pip install ...` 後重啟 runtime。  
- `FileNotFoundError`：路徑沒掛載 Drive、或檔案不在你以為的位置 → 先 `ls` 確認。  

### 2.4 讓結果可重現（重要）
在需要比較模型結果時，先固定亂數種子：
```python
import os, random, numpy as np
os.environ["PYTHONHASHSEED"] = "0"
random.seed(0)
np.random.seed(0)
```

---

## 3. 推薦的課程資料夾結構（Drive）

在你的 Google Drive 建立：
```
MyDrive/
  CHE-AI/
    data/        # 原始資料、整理後資料
    outputs/     # 圖表、表格、結果
    models/      # 訓練好的模型
    notebooks/   # 自己的作業/筆記副本（可選）
```

**規則**：所有輸出（圖、模型、CSV）都寫進 `outputs/` 或 `models/`，避免散落在 `/content/`。  

---

## 4. 本單元程式演練

請開啟並完成：`Part_0/Unit00_Colab_Environment_Setup.ipynb`  
- 會自動建立資料夾、檢查版本、印出環境指紋  
- 你完成後，把 `COURSE_ROOT` 換成自己的 Drive 路徑即可沿用到後續所有 Unit  

