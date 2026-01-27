# Unit00 Windows本地環境設定教學

## 課程目標

本單元將帶領學生在Windows本地電腦上建立完整的Python開發環境，透過Miniconda進行環境管理，並使用Jupyter Notebook或JupyterLab進行程式碼開發與實驗。

### 學習目標
- 理解Python環境管理的重要性與基本概念
- 學會在Windows系統上安裝Miniconda
- 掌握使用conda指令建立與管理虛擬環境
- 學會使用YAML檔案批次安裝套件
- 了解Jupyter Notebook與JupyterLab的基本操作
- 實作基本的Python程式碼執行與資料分析流程

---

## 1. 本地開發環境簡介

### 1.1 為什麼需要本地開發環境？

雖然Google Colab提供了便利的雲端開發環境，但在許多情況下，建立本地開發環境仍然是必要的：

**本地環境的優勢：**
- **無網路限制**：離線也可以進行開發與測試
- **無時間限制**：沒有執行時間或閒置中斷的限制
- **完整控制**：可以自由安裝任何套件與版本
- **資料安全**：敏感數據不需上傳至雲端
- **效能穩定**：不受雲端資源分配影響
- **專業開發**：符合實際工作環境的需求

### 1.2 什麼是虛擬環境？

虛擬環境（Virtual Environment）是Python開發中的重要概念，它允許我們為不同的專案建立獨立的Python環境，每個環境可以有不同的套件版本。

**使用虛擬環境的好處：**
1. **隔離性**：不同專案的套件不會互相干擾
2. **可重現性**：可以精確記錄專案所需的套件版本
3. **避免衝突**：解決不同專案需要不同版本套件的問題
4. **乾淨管理**：方便移除整個環境而不影響系統

### 1.3 Miniconda vs Anaconda

在Python環境管理工具中，Anaconda和Miniconda是最受歡迎的選擇：

| 特性 | Miniconda | Anaconda |
|------|-----------|----------|
| 安裝大小 | ~50 MB | ~3 GB |
| 預裝套件 | 最小化 | 1500+ 科學計算套件 |
| 下載時間 | 快速 | 較慢 |
| 適合對象 | 進階使用者、客製化需求 | 初學者、快速開始 |
| 彈性 | 高 | 中 |

**本課程選擇Miniconda的原因：**
- 安裝檔案小，下載快速
- 只安裝需要的套件，避免浪費空間
- 培養精確管理環境的好習慣
- 符合實際工作環境的需求

---

## 2. Miniconda安裝教學

### 2.1 下載Miniconda

1. 開啟Miniconda官方網站：[https://www.anaconda.com/download/success](https://www.anaconda.com/download/success)

2. 選擇適合您系統的安裝檔：
   - **作業系統**：Windows
   - **Python版本**：建議選擇Python 3.10或以上版本
   - **架構**：
     - 大多數現代電腦：64-bit (x86_64)
     - ARM架構電腦：ARM64 (較少見)

3. 下載安裝檔（約50 MB）

### 2.2 安裝步驟

1. **執行安裝檔**
   - 雙擊下載的 `.exe` 檔案
   - 如果出現「使用者帳戶控制」視窗，點選「是」

2. **安裝設定**
   - 點選「Next」開始安裝
   - 閱讀授權條款，點選「I Agree」
   - 選擇安裝類型：
     - **Just Me (recommended)**：只為當前使用者安裝
     - All Users：為所有使用者安裝（需要管理員權限）

3. **選擇安裝路徑**
   - 預設路徑：`C:\Users\<您的使用者名稱>\miniconda3`
   - 可以保持預設，或選擇其他位置
   - **注意**：路徑中不要包含中文或特殊符號

4. **進階選項**（重要）
   - ☐ Add Miniconda3 to my PATH environment variable
     - **建議不勾選**：避免與其他Python安裝衝突
     - 我們將使用Anaconda Prompt進行操作
   - ☑ Register Miniconda3 as my default Python 3.10
     - **建議勾選**：讓系統識別此Python為預設版本

5. **完成安裝**
   - 點選「Install」開始安裝
   - 等待安裝完成（約1-2分鐘）
   - 點選「Next」然後「Finish」

### 2.3 驗證安裝

安裝完成後，我們需要驗證Miniconda是否正確安裝：

1. **開啟Anaconda Prompt**
   - 在Windows搜尋列輸入「Anaconda Prompt」
   - 以系統管理員身分執行

2. **檢查conda版本**
   ```powershell
   conda --version
   ```
   應該會顯示類似：`conda 23.x.x`

3. **檢查Python版本**
   ```powershell
   python --version
   ```
   應該會顯示類似：`Python 3.10.x`

4. **更新conda（建議）**
   ```powershell
   conda update conda
   ```
   輸入 `y` 確認更新

---

## 3. 建立虛擬環境

### 3.1 理解YAML環境檔案

本課程提供了 `PY310_environment.yml` 檔案，這是一個環境定義檔案，包含了課程所需的所有套件與版本資訊。

**YAML檔案結構說明：**

```yaml
name: PY310                    # 環境名稱
channels:                      # 套件來源
  - conda-forge               # conda-forge頻道（推薦）
dependencies:                  # 套件列表
  - python=3.10               # Python版本
  - numpy=1.23.5              # 指定版本
  - pandas                    # 最新版本
  - scikit-learn              # 機器學習套件
  - matplotlib                # 視覺化套件
  - jupyterlab                # 開發環境
  - pip:                      # 透過pip安裝的套件
      - tensorflow==2.10.1    # 深度學習框架
```

### 3.2 使用YAML檔案建立環境

1. **下載環境檔案**
   - 從課程資料夾取得 `PY310_environment.yml` 檔案
   - 將檔案放在容易存取的位置（例如：`D:\MyPython\`）

2. **開啟Anaconda Prompt**
   - 以系統管理員身分執行

3. **切換到檔案所在目錄**
   ```powershell
   cd D:\MyPython
   ```
   或是使用檔案的完整路徑

4. **建立環境**
   ```powershell
   conda env create -f PY310_environment.yml
   ```
   
   這個指令會：
   - 讀取YAML檔案內容
   - 建立名為 `PY310` 的虛擬環境
   - 安裝所有指定的套件
   - **注意**：第一次建立需要較長時間（約10-30分鐘），視網路速度而定

5. **等待安裝完成**
   - 終端機會顯示下載與安裝進度
   - 看到「done」或類似訊息表示完成

### 3.3 手動建立環境（替代方案）

如果您想要手動建立環境，可以使用以下步驟：

```powershell
# 1. 建立基礎環境
conda create -n PY310 python=3.10

# 2. 啟動環境
conda activate PY310

# 3. 逐一安裝套件
conda install numpy pandas scipy scikit-learn
conda install matplotlib seaborn
conda install jupyterlab notebook

# 4. 使用pip安裝特定套件
pip install tensorflow==2.10.1
```

---

## 4. 環境管理常用指令

### 4.1 基本環境操作

**啟動環境**
```powershell
conda activate PY310
```
啟動後，命令提示字元前面會顯示 `(PY310)`，表示目前在此環境中。

**退出環境**
```powershell
conda deactivate
```

**列出所有環境**
```powershell
conda env list
```
或
```powershell
conda info --envs
```
目前啟動的環境前面會有 `*` 標記。

**刪除環境**
```powershell
conda env remove -n PY310
```
**注意**：刪除前請確認不再需要此環境！

### 4.2 套件管理

**列出環境中的所有套件**
```powershell
conda activate PY310
conda list
```

**安裝新套件**
```powershell
# 使用conda安裝
conda install 套件名稱

# 使用pip安裝
pip install 套件名稱

# 安裝特定版本
conda install numpy=1.23.5
pip install tensorflow==2.10.1
```

**更新套件**
```powershell
# 更新特定套件
conda update 套件名稱

# 更新所有套件
conda update --all
```

**移除套件**
```powershell
conda remove 套件名稱
```

### 4.3 匯出與分享環境

**匯出環境到YAML檔案**
```powershell
conda activate PY310
conda env export > my_environment.yml
```

**匯出簡化版本（只包含手動安裝的套件）**
```powershell
conda env export --from-history > my_environment_simple.yml
```

這個功能非常實用，可以：
- 備份您的開發環境
- 在其他電腦上重建相同環境
- 與團隊成員分享環境設定
- 確保專案的可重現性

---

## 5. Jupyter Notebook / JupyterLab 使用教學

### 5.1 Jupyter Notebook vs JupyterLab

兩者都是互動式Python開發環境，但有一些差異：

| 特性 | Jupyter Notebook | JupyterLab |
|------|------------------|------------|
| 介面 | 單一文件介面 | 多分頁IDE介面 |
| 檔案瀏覽 | 基本 | 進階（側邊欄） |
| 多檔案操作 | 需要多個瀏覽器分頁 | 同一視窗多分頁 |
| 擴充功能 | 較少 | 豐富 |
| 啟動速度 | 快 | 稍慢 |
| 適合對象 | 初學者、簡單任務 | 進階使用者、複雜專案 |

**本課程建議：**
- 初期學習：使用Jupyter Notebook（介面簡單）
- 進階開發：使用JupyterLab（功能強大）

### 5.2 啟動Jupyter Notebook

1. **開啟Anaconda Prompt**

2. **啟動環境**
   ```powershell
   conda activate PY310
   ```

3. **切換到工作目錄**
   ```powershell
   cd D:\MyProject
   ```

4. **啟動Jupyter Notebook**
   ```powershell
   jupyter notebook
   ```

5. **瀏覽器自動開啟**
   - 預設網址：`http://localhost:8888`
   - 顯示目前目錄的檔案列表

### 5.3 啟動JupyterLab

步驟與Jupyter Notebook相同，只是最後的啟動指令改為：

```powershell
jupyter lab
```

### 5.4 建立新的Notebook

1. 在Jupyter介面中，點選右上角的「New」按鈕
2. 選擇「Python 3 (ipykernel)」
3. 新的Notebook會在新分頁中開啟

### 5.5 Notebook基本操作

**Cell類型**
- **Code Cell**：撰寫Python程式碼
- **Markdown Cell**：撰寫文字說明（支援Markdown語法）

**快捷鍵（命令模式）**
- `Enter`：進入編輯模式
- `Esc`：進入命令模式
- `A`：在上方插入新Cell
- `B`：在下方插入新Cell
- `D D`：刪除Cell（按兩次D）
- `M`：轉換為Markdown Cell
- `Y`：轉換為Code Cell
- `Shift + Enter`：執行Cell並移到下一個
- `Ctrl + Enter`：執行Cell但停留在原位

**選單操作**
- `File → Save and Checkpoint`：儲存檔案
- `Kernel → Restart`：重新啟動Python核心
- `Kernel → Restart & Clear Output`：重啟並清除所有輸出
- `Cell → Run All`：執行所有Cell

### 5.6 關閉Jupyter

1. **在瀏覽器中**
   - 關閉所有Notebook分頁
   - 在首頁勾選要關閉的Notebook
   - 點選「Shutdown」

2. **在Anaconda Prompt中**
   - 按 `Ctrl + C` 兩次
   - 或輸入 `y` 確認關閉

---

## 6. 環境設定驗證與測試

### 6.1 檢查關鍵套件

建立一個新的Notebook，執行以下程式碼來驗證環境：

```python
# 檢查Python版本
import sys
print(f"Python version: {sys.version}")

# 檢查關鍵套件
import numpy as np
import pandas as pd
import matplotlib
import sklearn
import tensorflow as tf

print(f"\nNumpy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"TensorFlow version: {tf.__version__}")

# 檢查GPU可用性（如果有NVIDIA GPU）
print(f"\nGPU available: {tf.config.list_physical_devices('GPU')}")
```

**預期輸出範例：**
```
Python version: 3.10.x ...

Numpy version: 1.23.5
Pandas version: 1.x.x
Matplotlib version: 3.x.x
Scikit-learn version: 1.x.x
TensorFlow version: 2.10.1

GPU available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### 6.2 測試基本功能

**測試數值運算**
```python
import numpy as np

# 建立陣列
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Mean: {arr.mean()}")
print(f"Sum: {arr.sum()}")
```

**測試資料處理**
```python
import pandas as pd

# 建立DataFrame
data = {
    'Temperature': [25, 30, 35, 40],
    'Pressure': [1.0, 1.2, 1.4, 1.6],
    'Yield': [85, 88, 92, 90]
}
df = pd.DataFrame(data)
print(df)
print(f"\nAverage Yield: {df['Yield'].mean()}")
```

**測試視覺化**
```python
import matplotlib.pyplot as plt
import numpy as np

# 建立簡單圖表
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sine Wave')
plt.grid(True)
plt.show()
```

---

## 7. 常見問題與解決方案

### 7.1 安裝相關問題

**問題1：找不到Anaconda Prompt**

**解決方案：**
- 方法1：使用Windows搜尋列搜尋「Anaconda Prompt」
- 方法2：到開始選單 → Anaconda3 資料夾 → Anaconda Prompt
- 方法3：使用一般的PowerShell或CMD，然後手動啟動conda：
  ```powershell
  C:\Users\<您的使用者名稱>\miniconda3\Scripts\activate.bat
  ```

**問題2：conda指令無法執行**

**解決方案：**
- 確認使用Anaconda Prompt（不是一般的CMD）
- 如果必須使用一般CMD，需要手動初始化：
  ```powershell
  conda init cmd.exe
  ```
  然後重新開啟CMD

**問題3：安裝環境時出現SSL錯誤**

**解決方案：**
```powershell
# 方法1：暫時停用SSL驗證（不建議長期使用）
conda config --set ssl_verify false

# 方法2：更新conda
conda update conda

# 方法3：使用鏡像站點
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
```

### 7.2 套件安裝問題

**問題4：某些套件無法安裝**

**解決方案：**
```powershell
# 方法1：指定channel
conda install -c conda-forge 套件名稱

# 方法2：改用pip安裝
pip install 套件名稱

# 方法3：建立新環境重新安裝
conda create -n test_env python=3.10
conda activate test_env
conda install 套件名稱
```

**問題5：TensorFlow安裝失敗**

**解決方案：**
```powershell
# 使用pip安裝（建議）
pip install tensorflow==2.10.1

# 確認CUDA版本相容性（如果使用GPU）
# TensorFlow 2.10需要CUDA 11.2和cuDNN 8.1
```

### 7.3 Jupyter相關問題

**問題6：Jupyter無法啟動**

**解決方案：**
```powershell
# 確認已安裝Jupyter
conda list jupyter

# 重新安裝
conda install jupyterlab notebook

# 清除配置
jupyter notebook --generate-config
```

**問題7：Kernel無法連接**

**解決方案：**
```powershell
# 重新安裝ipykernel
conda install ipykernel

# 手動註冊kernel
python -m ipykernel install --user --name PY310 --display-name "Python 3.10 (PY310)"
```

**問題8：Notebook無法匯入已安裝的套件**

**解決方案：**
- 確認環境已正確啟動
- 在Notebook中檢查Python路徑：
  ```python
  import sys
  print(sys.executable)
  ```
- 確認路徑指向正確的環境

### 7.4 效能相關問題

**問題9：Jupyter執行很慢**

**解決方案：**
- 關閉不需要的Notebook
- 重啟Kernel
- 檢查電腦資源使用狀況
- 考慮增加記憶體或使用更快的硬碟

**問題10：GPU無法使用**

**解決方案：**
```python
# 檢查TensorFlow是否偵測到GPU
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# 如果沒有GPU，檢查：
# 1. NVIDIA驅動程式是否正確安裝
# 2. CUDA和cuDNN版本是否相容
# 3. TensorFlow版本是否支援GPU
```

---

## 8. 最佳實務建議

### 8.1 環境管理

1. **為不同專案建立不同環境**
   - 避免套件版本衝突
   - 保持環境乾淨簡潔

2. **定期備份環境**
   ```powershell
   conda env export > backup_$(Get-Date -Format "yyyyMMdd").yml
   ```

3. **使用環境名稱規範**
   - 有意義的名稱：`ML_Project1`, `DL_TensorFlow2`
   - 包含Python版本：`PY310_ML`

4. **記錄套件變更**
   - 在專案中維護 `requirements.txt` 或 `environment.yml`
   - 使用版本控制（Git）追蹤變更

### 8.2 Notebook使用

1. **檔案組織**
   ```
   專案資料夾/
   ├── data/              # 資料檔案
   ├── notebooks/         # Jupyter notebooks
   ├── scripts/           # Python腳本
   ├── results/           # 輸出結果
   └── environment.yml    # 環境定義
   ```

2. **Cell組織原則**
   - 第一個Cell：匯入所有套件
   - 第二個Cell：設定常數與路徑
   - 適時加入Markdown說明
   - 保持每個Cell功能單一

3. **定期儲存與檢查點**
   - 使用 `Ctrl + S` 經常儲存
   - 重要階段使用 `File → Save and Checkpoint`

4. **輸出管理**
   - 大量輸出考慮儲存到檔案
   - 定期清除不需要的輸出：`Cell → All Output → Clear`

### 8.3 開發習慣

1. **版本控制**
   - 使用Git管理程式碼
   - `.gitignore` 排除大型檔案和環境資料夾

2. **程式碼風格**
   - 遵循PEP 8規範
   - 使用有意義的變數名稱
   - 加入適當的註解

3. **錯誤處理**
   - 使用try-except捕捉例外
   - 記錄錯誤訊息

4. **效能優化**
   - 使用向量化運算（Numpy）
   - 避免不必要的迴圈
   - 適時使用進度條（tqdm）

---

## 9. 下一步學習

### 9.1 環境設定完成檢查清單

- ✅ Miniconda已成功安裝
- ✅ PY310環境已建立並包含所有必要套件
- ✅ Jupyter Notebook/Lab可以正常啟動
- ✅ 所有關鍵套件版本正確
- ✅ 可以執行基本的Python程式碼
- ✅ 可以進行資料視覺化

### 9.2 推薦學習資源

**官方文件**
- [Conda官方文件](https://docs.conda.io/)
- [Jupyter官方文件](https://jupyter.org/documentation)
- [Python官方教學](https://docs.python.org/3/tutorial/)

**線上教學**
- [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)
- [Jupyter Notebook教學](https://jupyter-notebook.readthedocs.io/en/stable/)

**社群資源**
- Stack Overflow：解決常見問題
- GitHub：參考其他人的專案結構

### 9.3 課程預告

完成本單元後，接下來的課程將涵蓋：

- **Unit01**：AI與機器學習概論
- **Unit02**：Python程式語言基礎
- **Unit03**：Numpy與Pandas模組
- **Unit04**：Matplotlib與Seaborn視覺化

請確保您的環境已正確設定，以便順利進行後續的學習！

---

## 總結

本單元介紹了如何在Windows本地電腦上建立Python開發環境：

1. **環境管理**：使用Miniconda管理Python環境，實現套件隔離與版本控制
2. **批次安裝**：使用YAML檔案快速建立包含所有必要套件的環境
3. **開發工具**：掌握Jupyter Notebook/Lab的基本操作
4. **故障排除**：了解常見問題的解決方案
5. **最佳實務**：學習專業的開發習慣

擁有穩定的本地開發環境是進行AI與機器學習專案的重要基礎。請務必完成環境設定，並透過配套的Notebook進行實際操作練習！

---

**課程資訊**
- 課程：AI在化工上之應用 (CHE-AI-114)
- 教師：莊曜禎 助理教授
- 學校：逢甲大學 化學工程學系

