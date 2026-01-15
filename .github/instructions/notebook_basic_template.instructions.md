---
description: 'Notebook Basic Template Instructions'
applyTo: '**/*.ipynb'
---

# Notebook Basic Template Instructions

本文件定義 CHE-AI-COURSE 課程中所有 Notebook 的標準起始範本，包含「環境設定」與「載入相關套件」兩個核心 cells。

---

## 一、標準結構

每個新建立的 Notebook 應依序包含以下結構：

1. **Cell 1 (Markdown)**: Notebook 標題與說明
2. **Cell 2 (Markdown)**: 環境設定區段標題
3. **Cell 3 (Python)**: 環境設定程式碼
4. **Cell 4 (Markdown)**: 數據下載區段標題（選填）
5. **Cell 5 (Python)**: 數據下載程式碼（選填）
6. **Cell 6 (Python)**: 載入相關套件

---

## 二、Cell 1: Notebook 標題與說明 (Markdown)

### 格式規範
```markdown
# UnitXX_Topic | 副標題：簡短說明

本 Notebook 使用 `Part_X/data/資料夾/檔案名稱` 建立 [模型類型] 模型來 [目標說明]。

## 目標
- 目標項目 1
- 目標項目 2
- 目標項目 3

資料背景可參考：[來源連結]
```

### 變數說明
- `UnitXX_Topic`: 單元編號與主題
- `Part_X`: 對應的課程部分
- `[模型類型]`: 如 DNN、CNN、RNN 等
- `[目標說明]`: 預測目標或任務描述

---

## 三、Cell 2: 環境設定區段標題 (Markdown)

### 標準格式
```markdown
---
### 0. 環境設定
```

或

```markdown
---
## 0. 環境設定
```

---

## 四、Cell 3: 環境設定程式碼 (Python)

### 完整範本

```python
from pathlib import Path
import tensorflow as tf
import os

# ========================================
# 路徑設定 (兼容 Colab 與 Local)
# ========================================
UNIT_OUTPUT_DIR = 'P{Part}_Unit{XX}_{Topic}'
SOURCE_DATA_DIR = '{data_folder_name}'

try:
  from google.colab import drive
  IN_COLAB = True
  print("✓ 偵測到 Colab 環境，準備掛載 Google Drive...")
  drive.mount('/content/drive', force_remount=True)
except ImportError:
  IN_COLAB = False
  print("✓ 偵測到 Local 環境")
try:
  shortcut_path = '/content/CHE-AI-COURSE'
  os.remove(shortcut_path)
except FileNotFoundError:
  pass

if IN_COLAB:
  source_path = Path('/content/drive/My Drive/Colab Notebooks/CHE-AI-COURSE')
  os.symlink(source_path, shortcut_path)
  shortcut_path = Path(shortcut_path)
  if source_path.exists():
    NOTEBOOK_DIR = shortcut_path / 'Part_{Part}' / 'Unit_{XX}'
    OUTPUT_DIR = NOTEBOOK_DIR / 'outputs' / UNIT_OUTPUT_DIR
    DATA_DIR = NOTEBOOK_DIR / 'data' / SOURCE_DATA_DIR
    MODEL_DIR = OUTPUT_DIR / 'models'
    FIG_DIR = OUTPUT_DIR / 'figs'
  else:
    print(f"⚠️ 找不到路徑雲端CHE-AI-COURSE路徑，請確認自己的雲端資料夾是否正確")
  
else:
  NOTEBOOK_DIR = Path.cwd()
  OUTPUT_DIR = NOTEBOOK_DIR / 'outputs' / UNIT_OUTPUT_DIR
  DATA_DIR = NOTEBOOK_DIR / 'data' / SOURCE_DATA_DIR
  MODEL_DIR = OUTPUT_DIR / 'models'
  FIG_DIR = OUTPUT_DIR / 'figs'

NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n✓ Notebook工作目錄: {NOTEBOOK_DIR}")
print(f"✓ 數據來源目錄: {DATA_DIR}")
print(f"✓ 結果輸出目錄: {OUTPUT_DIR}")
print(f"✓ 模型輸出目錄: {MODEL_DIR}")
print(f"✓ 圖檔輸出目錄: {FIG_DIR}")


# ========================================
# 檢查 GPU 狀態
# ========================================
print(f"\nTensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ 偵測到 GPU：{gpus[0].name}")
    print("  （訓練速度將明顯快於僅用 CPU）")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("△ 未偵測到 GPU。")
    print("  訓練速度將使用 CPU（速度較慢但仍可完成）")
```

### 變數替換規則

| 變數名稱 | 說明 | 範例 |
|---------|------|------|
| `{Part}` | 課程部分編號 | `4` (Part_4) |
| `{XX}` | 單元編號 | `15` (Unit15) |
| `{Topic}` | 主題名稱 | `Example_Mining` |
| `{data_folder_name}` | 資料夾名稱 | `mining`, `redwine`, `distillation_column` |

### 輸出資料夾結構

執行後會建立以下資料夾結構：
```
Part_{Part}/Unit_{XX}/
├── data/{data_folder_name}/          # 原始資料
├── outputs/P{Part}_Unit{XX}_{Topic}/
│   ├── models/                       # 模型檔案
│   └── figs/                         # 圖檔輸出
```

---

## 五、Cell 4-5: 數據下載區段（選填）

### Cell 4 (Markdown)
```markdown
---
### 數據下載
```

或

```markdown
---
## 數據下載
```

### Cell 5 (Python) - Kaggle 資料集下載範本

```python
# 數據來源: [資料集名稱]
# [Kaggle 連結]

import requests
import os
import zipfile

# 1. 設定路徑與 URL
url = "[Kaggle API URL]"
zip_path = os.path.join(DATA_DIR, "[檔案名稱].zip")
data_file = os.path.join(DATA_DIR, "[目標檔案名稱].csv")
extract_path = DATA_DIR

def download_and_extract():
    # --- 步驟 A: 下載檔案 ---
    print(f"正在從 {url} 下載...")
    try:
        # allow_redirects=True 處理 Kaggle 的重導向
        response = requests.get(url, allow_redirects=True, stream=True)
        response.raise_for_status() # 若下載失敗會拋出異常
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"下載成功，檔案儲存於: {zip_path}")

        # --- 步驟 B: 解壓縮檔案 ---
        if zipfile.is_zipfile(zip_path):
            print(f"正在解壓縮至: {extract_path}...")
            # 確保目標資料夾存在
            if not os.path.exists(extract_path):
                os.makedirs(extract_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            print("解壓縮完成！")
            
            # (選填) 步驟 C: 刪除原始 ZIP 檔以節省空間
            # os.remove(zip_path)
            # print("已移除原始 ZIP 壓縮檔。")
        else:
            print("錯誤：下載的檔案不是有效的 ZIP 格式。")
            
    except Exception as e:
        print(f"發生錯誤: {e}")

if __name__ == "__main__" and not os.path.exists(data_file):
    download_and_extract()
else:
    print(f"檔案已存在於: {data_file}")
```

---

## 六、Cell 6: 載入相關套件 (Python)

### 完整範本
請判斷需要使用到的套件內容進行替換
```python
# 基礎套件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import joblib, pickle, json
import warnings
warnings.filterwarnings('ignore')

# sklearn套件
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

print(f"TensorFlow版本: {tf.__version__}")
print(f"Keras版本: {keras.__version__}")

# 設定隨機種子以確保結果可重現
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 設定matplotlib中文顯示
# plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial'] # Colab不支援
plt.rcParams['axes.unicode_minus'] = False
```

### 套件分類說明

1. **基礎套件**：
   - `numpy`, `pandas`: 數值與資料處理
   - `matplotlib`, `seaborn`: 視覺化
   - `os`, `datetime`: 系統與時間操作
   - `joblib`, `pickle`, `json`: 模型與資料序列化
   - `warnings`: 警告訊息控制

2. **sklearn 套件**：
   - `train_test_split`: 資料切分
   - `StandardScaler`: 標準化
   - `mean_absolute_error`, `mean_squared_error`, `r2_score`: 評估指標
   - `permutation_importance`: 特徵重要性分析
   - `Ridge`: Baseline 模型

3. **TensorFlow/Keras**：
   - `Sequential`: 序列模型
   - `Dense`, `Dropout`, `BatchNormalization`: 常用層
   - `Adam`: 優化器
   - `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, `TensorBoard`: Callbacks

4. **全域設定**：
   - 隨機種子設定 (SEED=42)
   - Matplotlib 中文顯示修正

---

## 七、使用建議

### 1. 建立新 Notebook 時
1. 複製本範本的 Cell 1-6
2. 根據實際需求替換變數：
   - `UNIT_OUTPUT_DIR`
   - `SOURCE_DATA_DIR`
   - `Part_{Part}` 和 `Unit_{XX}`
3. 根據專案需求調整套件匯入清單

### 2. 可選調整
- **不需要 GPU**：可移除 GPU 檢查區段
- **不需要下載資料**：可跳過 Cell 4-5
- **不使用 TensorFlow**：調整套件匯入，僅保留 sklearn 相關

### 3. 命名規範
- **Notebook 檔案名稱**：`UnitXX_{Topic}.ipynb`
- **輸出資料夾名稱**：`P{Part}_UnitXX_{Topic}`
- **資料夾名稱**：使用小寫底線分隔，如 `mining`, `red_wine`, `distillation_column`

### 4. 路徑相容性
- 範本同時支援 **Google Colab** 和 **本地環境**
- Colab 會自動掛載 Google Drive 並建立符號連結
- 本地環境使用 `Path.cwd()` 作為基礎路徑

---

## 八、完整範例

### 範例：Unit15_Example_RedWine.ipynb

**變數設定**：
- Part: `4`
- Unit: `15`
- Topic: `Example_RedWine`
- Data folder: `redwine`

**Cell 3 變數替換結果**：
```python
UNIT_OUTPUT_DIR = 'P4_Unit15_Example_RedWine'
SOURCE_DATA_DIR = 'redwine'
```

**路徑結構**：
```
Part_4/Unit15/
├── data/redwine/
│   └── winequality-red.csv
├── outputs/P4_Unit15_Example_RedWine/
│   ├── models/
│   │   ├── dnn_model.keras
│   │   ├── scaler_X.pkl
│   │   └── scaler_y.pkl
│   └── figs/
│       ├── quality_dist.png
│       ├── corr_bar.png
│       └── parity_plot_dnn.png
└── Unit15_Example_RedWine.ipynb
```

---

## 九、檢查清單

建立新 Notebook 時，請確認以下項目：

- [ ] Cell 1: 標題與說明完整
- [ ] Cell 2: 環境設定區段標題
- [ ] Cell 3: 路徑變數已正確替換
- [ ] Cell 3: 資料夾建立邏輯無誤
- [ ] Cell 3: GPU 檢查程式碼已包含
- [ ] Cell 4-5: 數據下載程式碼（若需要）
- [ ] Cell 6: 套件匯入完整且符合需求
- [ ] Cell 6: 隨機種子已設定
- [ ] Cell 6: Matplotlib 中文設定已包含
- [ ] 測試執行：Colab 與本地環境都能正常運作

---

## 十、注意事項
**嚴格執行逐段建立 Notebook，每次生成一部分並自檢。**
0. **逐段生成**：注意LLM模型生成文本長度限制，請逐段依序建立各Cell內容，並同步自檢確保每段程式碼能正確執行
1. **編碼規範**：所有文字輸出與檔案操作必須使用 UTF-8 編碼
2. **中文字元保留**：保持所有繁體中文字元，不要使用 ASCII 替代
3. **Matplotlib 標籤**：圖表標題、軸標籤必須使用英文（避免字型問題）
4. **路徑分隔符**：統一使用 `Path` 物件處理路徑，自動相容 Windows/Linux
5. **隨機種子**：統一設定 `SEED=42` 確保結果可重現

---

## 十一、版本記錄

| 版本 | 日期 | 說明 |
|------|------|------|
| 1.0 | 2026-01-13 | 初始版本，基於 Unit15 範例建立標準範本 |

---

**最後更新**：2026-01-13  
**維護者**：CHE-AI-COURSE Team
