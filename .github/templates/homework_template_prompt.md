# Homework Notebook Template Prompt

## 標準格式說明

此 prompt 用於生成符合課程標準格式的 Jupyter Notebook 作業檔案。

---

## Prompt Template

```
請建立 [UnitXX]_[主題名稱]_Homework.ipynb 課堂作業檔案。

### 作業基本資訊
- 單元編號: Unit XX
- 主題: [主題名稱，例如: Matplotlib, Pandas, DNN等]
- Part: [課程分類，例如: Part_1, Part_4等]
- 輸出目錄命名: P[X]_Unit[XX]_HW

### 格式要求

#### 1. 標題資訊 (第1個cell - markdown)
```markdown
## Unit XX. [主題名稱] 課堂作業
- 課程編號: CHEXXXX
- 學年度: 114下
- 上課時間: 每週四 09:00-12:00
-
- 指導教授: ＯＯＯ 教授
- 學生姓名: ＯＯＯ
- 學號: m12345678
- email帳號: fcu.m12345678@gmail.com
```

**注意:** 使用佔位符 CHEXXXX 和 ＯＯＯ，不要使用真實資訊

---

#### 2. 環境設定 (第2-4個cells)

**Cell 2 - markdown:**
```markdown
---
### 環境設定
```

**Cell 3 - python (路徑設定，兼容Colab與Local):**
```python
from pathlib import Path
import os

# ========================================
# 路徑設定 (兼容 Colab 與 Local)
# ========================================
UNIT_OUTPUT_DIR = 'P[X]_Unit[XX]_HW'

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
    NOTEBOOK_DIR = shortcut_path / 'Part_[X]' / 'Unit[XX]'
    OUTPUT_DIR = NOTEBOOK_DIR / 'outputs' / UNIT_OUTPUT_DIR
    [DATA_DIR = NOTEBOOK_DIR / 'data' / SOURCE_DATA_DIR]  # 如需要數據
    [MODEL_DIR = OUTPUT_DIR / 'models']  # 如需要模型
    FIG_DIR = OUTPUT_DIR / 'figs'  # 如需要圖檔
  else:
    print(f"⚠️ 找不到路徑雲端CHE-AI-COURSE路徑，請確認自己的雲端資料夾是否正確")
  
else:
  NOTEBOOK_DIR = Path.cwd()
  OUTPUT_DIR = NOTEBOOK_DIR / 'outputs' / UNIT_OUTPUT_DIR
  [DATA_DIR = NOTEBOOK_DIR / 'data' / SOURCE_DATA_DIR]  # 如需要
  [MODEL_DIR = OUTPUT_DIR / 'models']  # 如需要
  FIG_DIR = OUTPUT_DIR / 'figs'  # 如需要

NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
[DATA_DIR.mkdir(parents=True, exist_ok=True)]  # 如需要
[MODEL_DIR.mkdir(parents=True, exist_ok=True)]  # 如需要
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n✓ Notebook工作目錄: {NOTEBOOK_DIR}")
[print(f"✓ 數據來源目錄: {DATA_DIR}")]  # 如需要
print(f"✓ 結果輸出目錄: {OUTPUT_DIR}")
[print(f"✓ 模型輸出目錄: {MODEL_DIR}")]  # 如需要
print(f"✓ 圖檔輸出目錄: {FIG_DIR}")
```

**注意:**
- 方括號 `[]` 內的內容視需求決定是否包含
- 如果是深度學習相關作業，需要 MODEL_DIR
- 如果需要載入數據，需要 DATA_DIR 和 SOURCE_DATA_DIR
- 如果是視覺化作業，需要 FIG_DIR

**Cell 4 - python (套件載入):**
根據主題調整，例如:
```python
# 基礎套件
import numpy as np
import pandas as pd
[import matplotlib.pyplot as plt]  # 如需視覺化
[import tensorflow as tf]  # 如需深度學習
import warnings
warnings.filterwarnings('ignore')

# 設定隨機種子以確保結果可重現
SEED = 42
np.random.seed(SEED)
[tf.random.set_seed(SEED)]  # 如使用 TensorFlow

print("✓ 套件載入完成")
```

**如果是深度學習作業，Cell 3 需要加入 GPU 檢查:**
```python
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

---

#### 3. 主要練習題 - I. 練習題 (markdown)

**結構:**
```markdown
---
### **I. 練習題: [主題描述]**
 - [簡短說明作業目標]
 - [其他重要說明]
```

**接著是多個子任務，每個子任務包含:**
1. **說明 cell (markdown):** 
   - 子任務標題: `### **1. [子任務名稱]**`
   - 詳細要求列表
   - 若需儲存檔案，明確指定檔名

2. **數據/程式碼 cell (python):**
   ```python
   # 提供的數據 (如有需要)
   [數據定義]
   
   # ========== 請在下方撰寫程式碼 ==========
   ```
   
   或者只有空白 cell 讓學生自行撰寫

**子任務數量:** 通常 3-7 個子任務，視主題複雜度調整

---

#### 4. 額外加分作業 - II. 額外加分作業 (markdown)

```markdown
---
## II. 額外加分作業 (自由繳交)
- 學生可自由選擇是否繳交加分作業
- 分數會加在最後總成績上, 每份作業加0.1 ~ [X]分
- 加分作業不一定要全部都做完, 想繳交至少完成其中一項實驗即可
- 請務必自行完成! 你可以問AI, 問同學, 問學長姊, 甚至參考以前別人的作業, 但一定要自行[吸收][執行][整理]結果!
- 不要貼AI的答案寄給老師看, 你自己看就好
- 如果系統自動比對發現作業內容(與他人雷同, 原創性低, 抄襲比例過高), 作業的分數會0分
- 如果你直接100%複製別人的作業繳交, 你會直接被當掉(提供作業給別人複製的也一樣)
- 老師看你作業要花時間, 真的有心想寫加分作業, 請好好自己做
```

**接著是 3-5 個實驗題目，每個包含:**
1. **實驗說明 cell (markdown):** 
   ```markdown
   ### **實驗[N]: [實驗主題]**
   
   [實驗目標描述]
   
   **要求**:
   - [具體要求1]
   - [具體要求2]
   - ...
   ```

2. **空白程式碼 cell (python)**

---

#### 5. 思考題 (markdown + 空白markdown)

```markdown
### 💭 思考題

1. [問題1]
2. [問題2]
3. [問題3]
...
[N]. [問題N]
```

**接著一個空白 markdown cell 供學生作答**

**思考題設計原則:**
- 通常 5-10 題
- 涵蓋原理理解、應用場景、實務考量
- 鼓勵批判性思考
- 與主題緊密相關

---

#### 6. 結尾 - 想對老師說的話

**兩個 markdown cells:**
1. 標題 cell:
```markdown
---
# 想對老師說的話
```

2. 空白 cell (供學生填寫)

---

## 完整結構概覽

```
Cell 1: [M] 標題與學生資訊
Cell 2: [M] --- 環境設定
Cell 3: [P] 路徑設定程式碼 (+ GPU檢查，如適用)
Cell 4: [P] 套件載入

Cell 5: [M] --- I. 練習題: [主題]
Cell 6: [M] 1. 子任務1說明
Cell 7: [P] 子任務1程式碼/數據
Cell 8: [M] 2. 子任務2說明
Cell 9: [P] 子任務2程式碼/數據
... (重複子任務結構)

Cell N: [M] --- II. 額外加分作業
Cell N+1: [M] 實驗1說明
Cell N+2: [P] 實驗1空白程式碼
Cell N+3: [M] 實驗2說明
Cell N+4: [P] 實驗2空白程式碼
... (重複實驗結構)

Cell X: [M] 💭 思考題
Cell X+1: [M] 空白 (供作答)

Cell Y: [M] --- 想對老師說的話
Cell Y+1: [M] 空白 (供填寫)
```

**圖例:**
- [M] = Markdown cell
- [P] = Python code cell

---

## 重要注意事項

### ✅ 必須遵守:
1. 使用佔位符 (CHEXXXX, ＯＯＯ) 而非真實資訊
2. 路徑設定必須兼容 Colab 與 Local 環境
3. 主練習題採用「一個大題目，多個子任務」結構，而非多個獨立題目
4. 必須包含「額外加分作業」章節
5. 必須包含「思考題」章節  
6. 必須包含「想對老師說的話」章節
7. 所有圖表標籤必須使用英文
8. 每個程式碼 cell 要有明確的分隔註解 `# ========== 請在下方撰寫程式碼 ==========`

### ❌ 禁止項目:
1. 不要有「作業提交說明」章節
2. 不要將主題拆成多個獨立的練習題 (I, II, III, IV, V)
3. 不要在標題使用真實教授名稱或課程編號
4. 不要在主練習題中混用「練習題一、二、三」和「1, 2, 3」的編號

### 📋 彈性調整:
- 子任務數量: 依主題複雜度，通常 3-7 個
- 實驗數量: 依主題深度，通常 3-5 個
- 思考題數量: 依主題廣度，通常 5-10 題
- 是否需要 DATA_DIR, MODEL_DIR: 依主題需求決定
- 是否需要 GPU 檢查: 僅深度學習相關主題需要

---

## 範例參考

### 已完成的標準範例:
1. **Unit04_Matplotlib_Homework.ipynb** (Part_1)
   - 視覺化主題
   - 5個子任務 (折線圖、散佈圖、長條圖、直方圖、多子圖)
   - 4個實驗 (進階樣式、雙Y軸、圓餅圖、箱型圖)

2. **Unit15_DNN_MLP_Homework.ipynb** (Part_4)
   - 深度學習主題
   - 7個子任務 (數據準備、模型建立、編譯、訓練、視覺化、評估、保存)
   - 4個實驗 (網路深度、Dropout、Batch Size、學習率)

---

## 使用此 Template Prompt

**步驟:**
1. 確定單元編號、主題、Part分類
2. 根據主題設計 3-7 個循序漸進的子任務
3. 設計 3-5 個進階實驗題目
4. 準備 5-10 個思考題
5. 根據需求調整路徑設定 (DATA_DIR, MODEL_DIR, FIG_DIR)
6. 確保所有 cell 類型與順序正確
7. 檢查是否符合所有必須遵守事項

**生成指令範例:**
```
請依照 homework_template_prompt.md 的格式規範，建立 Unit05_Pandas_Homework.ipynb 作業檔案，包含以下子任務: [列出子任務]，實驗題目: [列出實驗]。
```

---

**文件版本:** v1.0  
**最後更新:** 2026-01-27  
**維護者:** CHE-AI-COURSE 課程團隊
```
