# Unit00 Python學習環境設定教學 (Environment Setup)

## 📚 單元簡介

在開始AI與機器學習的學習之旅前，建立一個穩定且功能完整的Python開發環境是至關重要的基礎工作。本單元將帶領您完成兩種主流的Python開發環境設定：**Google Colab雲端環境**與**Windows本地開發環境**。

Google Colab提供了免費的雲端運算資源與GPU加速，適合快速開始學習與原型開發；而本地環境則提供了更高的自由度與穩定性，適合長期專案開發與實際工作需求。掌握這兩種環境的設定與使用，將為您後續的課程學習奠定堅實基礎。

本單元涵蓋從環境安裝、套件管理、到開發工具使用的完整流程，並提供實際的驗證測試程式，確保您的環境已準備就緒，可以順利進行後續的AI課程學習。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **理解Python環境管理的核心概念**：虛擬環境、套件管理、相依性控制
2. **掌握雙重開發環境**：Google Colab雲端環境與Windows本地環境
3. **選擇合適的開發環境**：根據專案需求、運算資源、協作需求選擇最適合的環境
4. **實作環境建立與驗證**：使用Miniconda、Jupyter Notebook/Lab進行實際操作
5. **應用於化工領域AI學習**：確保所有必要套件正確安裝，可進行數據分析、機器學習與深度學習開發

---

## 📖 單元內容架構

### 1️⃣ Google Colab 雲端環境設定

**檔案**：
- 講義：[Unit00_Colab_Environment_Setup.md](Unit00_Colab_Environment_Setup.md)
- 程式範例：[Unit00_Colab_Environment_Setup.ipynb](Unit00_Colab_Environment_Setup.ipynb)

**內容重點**：
- **Google Colab 簡介**：
  - 什麼是Google Colab及其主要特色
  - 為什麼選擇Colab作為學習平台
  - Colab的限制與注意事項
  
- **基本操作教學**：
  - 建立與管理Colab筆記本
  - Notebook介面與功能說明
  - Cell的類型與執行方式
  - 快捷鍵與效率技巧
  
- **檔案與資料管理**：
  - 上傳與下載檔案的多種方法
  - Google Drive掛載與使用
  - 資料持久化策略
  - 檔案路徑管理技巧
  
- **套件與環境管理**：
  - 預裝套件清單
  - 安裝額外套件（pip、apt-get）
  - 套件版本管理
  - 環境變數設定
  
- **GPU/TPU加速運算**：
  - 如何啟用GPU/TPU
  - 運算資源監控
  - GPU記憶體管理
  - 效能優化建議
  
- **實作練習**：
  - 基本Python程式執行
  - NumPy、Pandas數據處理
  - Matplotlib資料視覺化
  - 簡單的機器學習模型訓練

**適合場景**：
- 初學者快速開始學習
- 需要GPU加速的訓練任務
- 團隊協作與程式碼分享
- 不想在本地安裝環境
- 需要從不同裝置存取專案

---

### 2️⃣ Windows 本地環境設定 ⭐

**檔案**：
- 講義：[Unit00_Local_Environment_Setup.md](Unit00_Local_Environment_Setup.md)
- 程式範例：[Unit00_Local_Environment_Setup.ipynb](Unit00_Local_Environment_Setup.ipynb)

**內容重點**：
- **本地開發環境簡介**：
  - 為什麼需要本地開發環境
  - 虛擬環境（Virtual Environment）概念
  - Miniconda vs Anaconda比較
  
- **Miniconda安裝教學**：
  - 下載與安裝步驟（含圖文說明）
  - 安裝選項說明
  - 驗證安裝是否成功
  - 初始設定與更新
  
- **虛擬環境建立與管理**：
  - 使用YAML檔案批次建立環境
  - `PY310_environment.yml` 檔案結構說明
  - 環境啟動、切換、刪除指令
  - 套件安裝、更新、移除
  - 環境匯出與分享
  
- **Jupyter Notebook/Lab 使用教學**：
  - Jupyter Notebook vs JupyterLab 比較
  - 啟動與關閉Jupyter
  - 建立與管理Notebook
  - Cell操作與快捷鍵
  - Kernel管理
  
- **環境驗證與測試**：
  - Python版本檢查
  - 核心套件版本確認（NumPy、Pandas、Matplotlib等）
  - 機器學習套件測試（Scikit-learn、TensorFlow、XGBoost等）
  - GPU可用性檢查
  - 基本功能測試範例
  
- **常見問題與解決方案**：
  - 10個常見安裝與設定問題
  - 詳細的故障排除步驟
  - 套件衝突解決方法
  
- **最佳實務建議**：
  - 環境管理策略
  - Notebook使用技巧
  - 專案檔案組織
  - 開發習慣養成

**適合場景**：
- 長期專案開發
- 需要離線工作
- 敏感資料處理
- 完整的環境控制權
- 符合實際工作環境需求

---

### 3️⃣ 環境配置檔案

**檔案**：[PY310_environment.yml](PY310_environment.yml)

**內容說明**：
- Python 3.10 基礎環境
- **數值運算**：numpy, scipy
- **資料處理**：pandas
- **視覺化**：matplotlib, seaborn, kaleido
- **機器學習**：scikit-learn, imbalanced-learn, mlxtend
- **進階演算法**：xgboost, lightgbm, catboost
- **降維與分群**：umap-learn
- **深度學習**：tensorflow, tensorboard
- **開發工具**：jupyterlab, notebook, ipykernel
- **GPU支援**：cudatoolkit, cudnn（選配）

**使用方式**：
```powershell
conda env create -f PY310_environment.yml
conda activate PY310
```

---

## 🎓 環境比較與選擇指南

| 特性 | Google Colab | Windows 本地環境 |
|------|-------------|----------------|
| **安裝難度** | 無需安裝 | 需要安裝設定 |
| **成本** | 免費 | 免費（需有電腦） |
| **網路需求** | 必需穩定網路 | 離線可用 |
| **運算資源** | GPU/TPU免費配額 | 依電腦硬體而定 |
| **執行時間** | 最長12小時 | 無限制 |
| **閒置限制** | 90分鐘自動中斷 | 無限制 |
| **儲存空間** | Google Drive配額 | 本地硬碟空間 |
| **套件自由度** | 中等（需要sudo權限的受限） | 完全自由 |
| **協作功能** | ✅ 優秀（即時協作） | ❌ 需額外工具 |
| **資料安全** | 雲端儲存 | 本地儲存 |
| **環境持久性** | Session結束即清除 | 永久保存 |
| **適合對象** | 初學者、快速原型 | 進階使用者、正式開發 |

**選擇建議**：
1. **初學階段** → Google Colab（快速開始，無需設定）
2. **課程作業** → Google Colab 或 本地環境皆可
3. **大型模型訓練** → Google Colab（免費GPU）
4. **長時間運算** → 本地環境（無時間限制）
5. **實際工作專案** → 本地環境（完整控制）
6. **團隊協作** → Google Colab（即時協作功能）
7. **敏感資料處理** → 本地環境（資料安全）

**建議學習策略**：
- **同時掌握兩種環境**，根據需求彈性切換
- 初期使用Colab快速學習，後期轉向本地環境深入開發
- Colab用於實驗與快速驗證，本地環境用於正式專案

---

## 💻 實作環境需求

### 本地環境硬體建議

**最低需求**：
- CPU: Intel i3 或同等級
- RAM: 8 GB
- 硬碟空間: 10 GB 可用空間
- 作業系統: Windows 10/11 (64-bit)

**建議配置**：
- CPU: Intel i5/i7 或 AMD Ryzen 5/7
- RAM: 16 GB 以上
- 硬碟: SSD 20 GB 以上可用空間
- GPU: NVIDIA GPU（選配，用於深度學習加速）
- 作業系統: Windows 10/11 (64-bit)

**GPU支援**（選配）：
- NVIDIA GPU with CUDA Compute Capability 3.5+
- CUDA Toolkit 11.2
- cuDNN 8.1

### 必要套件清單

```yaml
# 核心套件
python=3.10
numpy=1.23.5
pandas
scipy
matplotlib
seaborn

# 機器學習
scikit-learn
imbalanced-learn
mlxtend

# 進階演算法
xgboost
lightgbm
catboost
umap-learn

# 深度學習
tensorflow==2.10.1
tensorboard==2.10.1

# 開發工具
jupyterlab
notebook
ipykernel
ipywidgets
```

---

## 📈 學習路徑建議

### 🔰 第一階段：快速開始（建議時間：1-2小時）

**選項A：使用 Google Colab（推薦初學者）**
1. 閱讀 [Unit00_Colab_Environment_Setup.md](Unit00_Colab_Environment_Setup.md)（30分鐘）
2. 執行 [Unit00_Colab_Environment_Setup.ipynb](Unit00_Colab_Environment_Setup.ipynb)（30分鐘）
3. 熟悉Colab基本操作與功能
4. 完成環境驗證測試

**選項B：建立本地環境（建議有基礎者）**
1. 閱讀 [Unit00_Local_Environment_Setup.md](Unit00_Local_Environment_Setup.md)（40分鐘）
2. 依照步驟安裝Miniconda與建立環境（30-60分鐘）
3. 執行 [Unit00_Local_Environment_Setup.ipynb](Unit00_Local_Environment_Setup.ipynb)（30分鐘）
4. 完成所有驗證測試

### 🔧 第二階段：環境熟悉與驗證（建議時間：2-3小時）

1. **理解環境管理概念**
   - 虛擬環境的重要性
   - 套件相依性管理
   - 環境隔離與重現性

2. **熟練基本操作**
   - 啟動/關閉環境
   - 安裝/更新/移除套件
   - 建立/執行Notebook
   - Cell的編輯與執行

3. **驗證核心功能**
   - Python基本語法測試
   - NumPy數值運算
   - Pandas資料處理
   - Matplotlib視覺化
   - Scikit-learn機器學習

4. **故障排除練習**
   - 熟悉常見錯誤訊息
   - 學習查找解決方案
   - 了解套件版本衝突處理

### 🎯 第三階段：進階設定（選修，建議時間：1-2小時）

1. **建立第二種環境**
   - 如果已完成Colab，嘗試本地環境
   - 如果已完成本地環境，了解Colab用法

2. **環境客製化**
   - 安裝額外需要的套件
   - 設定Jupyter擴充功能
   - 客製化程式碼自動完成

3. **專案結構建立**
   - 建立標準化的專案資料夾結構
   - 學習版本控制（Git）基礎
   - 了解.gitignore的設定

---

## 🔍 核心操作指令速查

### Google Colab 指令

**檔案操作**：
```python
# 掛載 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 上傳檔案
from google.colab import files
uploaded = files.upload()

# 下載檔案
files.download('filename.csv')

# 檢查GPU
!nvidia-smi
```

**套件管理**：
```python
# 安裝套件
!pip install package_name

# 安裝特定版本
!pip install package_name==1.2.3

# 升級套件
!pip install --upgrade package_name

# 列出已安裝套件
!pip list
```

### Windows 本地環境指令

**環境管理**：
```powershell
# 啟動環境
conda activate PY310

# 退出環境
conda deactivate

# 列出所有環境
conda env list

# 建立新環境
conda create -n env_name python=3.10

# 從YAML建立環境
conda env create -f environment.yml

# 刪除環境
conda env remove -n env_name

# 匯出環境
conda env export > environment.yml
```

**套件管理**：
```powershell
# 安裝套件
conda install package_name
pip install package_name

# 安裝特定版本
conda install package_name=1.2.3
pip install package_name==1.2.3

# 更新套件
conda update package_name
pip install --upgrade package_name

# 移除套件
conda remove package_name
pip uninstall package_name

# 列出已安裝套件
conda list
pip list
```

**Jupyter操作**：
```powershell
# 啟動 Jupyter Notebook
jupyter notebook

# 啟動 JupyterLab
jupyter lab

# 指定埠口
jupyter notebook --port 8889

# 列出執行中的服務
jupyter notebook list

# 停止服務（在Anaconda Prompt中按 Ctrl + C）
```

---

## 📝 環境驗證檢查清單

### ✅ Google Colab 環境檢查

執行以下程式碼確認環境正常：

```python
import sys
print(f"Python version: {sys.version}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Matplotlib: {plt.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")

# 檢查GPU
import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
```

**預期結果**：
- Python 3.10.x
- 所有套件成功匯入
- 版本號正常顯示
- GPU狀態顯示（如有）

### ✅ Windows 本地環境檢查

在 [Unit00_Local_Environment_Setup.ipynb](Unit00_Local_Environment_Setup.ipynb) 中執行完整的驗證測試：

**檢查項目**：
- ✓ Python 3.10+ 已安裝
- ✓ NumPy 可用
- ✓ Pandas 可用
- ✓ Matplotlib 可用
- ✓ Scikit-learn 可用
- ✓ TensorFlow 可用
- ✓ Jupyter Notebook 正常運作
- ✓ 基本數值運算測試通過
- ✓ 資料處理功能正常
- ✓ 視覺化功能正常
- ✓ 機器學習模型可訓練

**如果所有項目顯示 PASS，表示環境設定完成！**

---

## 🚀 故障排除指南

### 常見問題 Top 10

#### 1. **找不到 Anaconda Prompt**
**症狀**：安裝完Miniconda後找不到Anaconda Prompt  
**解決方案**：
- 使用Windows搜尋列搜尋「Anaconda Prompt」
- 到開始選單 → Anaconda3 資料夾內查找
- 使用一般PowerShell，手動啟動conda

#### 2. **conda 指令無法執行**
**症狀**：輸入conda指令出現「找不到命令」  
**解決方案**：
```powershell
# 初始化conda
conda init cmd.exe
# 或
conda init powershell
# 重新開啟終端機
```

#### 3. **環境建立時出現 SSL 錯誤**
**症狀**：下載套件時出現SSL憑證錯誤  
**解決方案**：
```powershell
# 方法1：暫時停用SSL驗證（不建議長期使用）
conda config --set ssl_verify false

# 方法2：使用鏡像站點
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
```

#### 4. **TensorFlow 無法安裝或啟動**
**症狀**：安裝失敗或匯入時出錯  
**解決方案**：
```powershell
# 使用pip安裝（推薦）
pip install tensorflow==2.10.1

# 確認Python版本相容（TensorFlow 2.10需要Python 3.7-3.10）
python --version
```

#### 5. **Jupyter 無法啟動**
**症狀**：執行jupyter notebook沒有反應  
**解決方案**：
```powershell
# 重新安裝Jupyter
conda install jupyterlab notebook -y

# 清除配置
jupyter notebook --generate-config
```

#### 6. **Kernel 無法連接**
**症狀**：Notebook中Kernel連接失敗  
**解決方案**：
```powershell
# 重新安裝ipykernel
conda install ipykernel -y

# 手動註冊kernel
python -m ipykernel install --user --name PY310 --display-name "Python 3.10 (PY310)"
```

#### 7. **無法匯入已安裝的套件**
**症狀**：套件已安裝但import失敗  
**解決方案**：
- 確認環境已啟動（命令提示字元前有環境名稱）
- 檢查Python路徑：
```python
import sys
print(sys.executable)
```
- 確認路徑指向正確的環境

#### 8. **GPU 無法使用**
**症狀**：TensorFlow無法偵測到GPU  
**解決方案**：
```python
# 檢查GPU狀態
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```
- 確認NVIDIA驅動程式已安裝
- 確認CUDA和cuDNN版本相容
- TensorFlow 2.10需要CUDA 11.2和cuDNN 8.1

#### 9. **Colab連線中斷**
**症狀**：Colab執行時突然中斷  
**解決方案**：
- 確認網路連線穩定
- 避免長時間閒置（90分鐘限制）
- 使用程式碼保持連線活躍：
```python
# 定期執行小任務
import time
while True:
    print("Keep alive")
    time.sleep(60)  # 每分鐘執行一次
```

#### 10. **記憶體不足錯誤**
**症狀**：執行大型數據或模型時記憶體耗盡  
**解決方案**：
- Colab：使用Colab Pro或減少batch size
- 本地：增加實體記憶體或使用較小的數據集
- 優化程式碼，釋放不需要的變數：
```python
import gc
del large_variable
gc.collect()
```

---

## 📚 參考資源

### 官方文件
1. [Conda Documentation](https://docs.conda.io/) - Conda完整使用指南
2. [Jupyter Documentation](https://jupyter.org/documentation) - Jupyter官方文件
3. [Google Colab FAQ](https://research.google.com/colaboratory/faq.html) - Colab常見問題
4. [Python Documentation](https://docs.python.org/3/) - Python官方文件

### 線上教學資源
- [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html) - Conda指令速查表
- [Jupyter Notebook Tutorial](https://jupyter-notebook.readthedocs.io/en/stable/) - Jupyter入門教學
- [Google Colab Tutorials](https://colab.research.google.com/notebooks/intro.ipynb) - Colab官方教學

### 套件文件
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)

### 社群支援
- [Stack Overflow](https://stackoverflow.com/) - 程式設計問題解答
- [GitHub](https://github.com/) - 開源專案與程式碼範例
- [Reddit r/learnpython](https://www.reddit.com/r/learnpython/) - Python學習社群

---

## 💡 學習建議與最佳實務

### 環境管理最佳實務

1. **為不同專案建立獨立環境**
   - 避免套件版本衝突
   - 保持環境乾淨整潔
   - 便於環境重現與分享

2. **定期備份環境設定**
   ```powershell
   conda env export > environment_backup_$(Get-Date -Format "yyyyMMdd").yml
   ```

3. **使用有意義的環境名稱**
   - 好的命名：`ML_Project1`, `DL_TensorFlow2`, `PY310_ChemEng`
   - 避免：`test`, `env1`, `temp`

4. **記錄環境變更**
   - 在專案中維護 `requirements.txt` 或 `environment.yml`
   - 使用版本控制（Git）追蹤環境變更
   - 在README中記錄環境設定步驟

### Notebook 使用最佳實務

1. **合理組織Cell**
   - 第一個Cell：匯入所有套件
   - 第二個Cell：設定常數、路徑、種子
   - 適時加入Markdown說明
   - 每個Cell保持功能單一

2. **定期儲存工作**
   - 養成 `Ctrl + S` 的習慣
   - 重要階段使用 `File → Save and Checkpoint`
   - 考慮自動備份機制

3. **管理輸出內容**
   - 大量輸出考慮儲存到檔案
   - 定期清除不需要的輸出
   - 使用進度條（tqdm）追蹤長時間運算

4. **程式碼品質**
   - 遵循PEP 8命名規範
   - 使用有意義的變數名稱
   - 加入適當的註解
   - 避免在Notebook中寫過長的函式

### 專案檔案組織建議

```
專案資料夾/
├── data/                    # 原始數據
│   ├── raw/                # 未處理數據
│   └── processed/          # 處理後數據
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/                    # Python模組
│   ├── __init__.py
│   ├── data_processing.py
│   └── models.py
├── results/                # 輸出結果
│   ├── figures/           # 圖表
│   └── models/            # 訓練好的模型
├── environment.yml         # 環境定義檔
├── requirements.txt        # pip套件清單
├── README.md              # 專案說明
└── .gitignore            # Git忽略清單
```

---

## ✍️ 學習檢核與下一步

### 完成本單元後，您應該能夠：

- [ ] 說明虛擬環境的概念與重要性
- [ ] 在Google Colab中建立並執行Notebook
- [ ] 在本地環境成功安裝Miniconda
- [ ] 使用YAML檔案建立Python虛擬環境
- [ ] 熟練使用conda基本指令
- [ ] 啟動並使用Jupyter Notebook/Lab
- [ ] 執行環境驗證測試並確認所有套件正常
- [ ] 了解常見問題的排除方法
- [ ] 根據需求選擇合適的開發環境

### 準備進入下一階段

完成環境設定後，您已準備好開始：

- **Part 1 - Unit01**：[AI與機器學習概論](../../Part_1/Unit01/README.md)
- **Part 1 - Unit02**：[Python程式語言基礎](../../Part_1/Unit02/README.md)
- **Part 1 - Unit03**：[NumPy與Pandas模組](../../Part_1/Unit03/README.md)
- **Part 1 - Unit04**：[Matplotlib與Seaborn視覺化](../../Part_1/Unit04/README.md)

**重要提醒**：
- 每次開始工作前，記得啟動環境：`conda activate PY310`
- 確保Jupyter Notebook能正常開啟並執行程式碼
- 如遇到問題，先參考本單元的故障排除章節

---

## 📞 技術支援

如有疑問或需要協助，歡迎透過以下方式聯繫：
- **課程討論區**：提出問題與同學交流
- **指導教授 Office Hour**：面對面深入討論
- **課程助教**：協助解決技術問題

### 提問時請提供：
1. 作業系統版本（Windows 10/11）
2. Python版本（執行 `python --version`）
3. 完整的錯誤訊息（截圖或文字）
4. 已嘗試的解決方法
5. 相關的程式碼（如適用）

---

## 🎉 開始您的 AI 學習之旅！

環境設定是第一步，也是最重要的基礎。完成本單元後，您將擁有一個穩定、功能完整的開發環境，可以順利進行後續的AI與機器學習課程。

記住：遇到問題是正常的，關鍵是學會如何找到解決方案。祝您學習順利！🚀

---

**課程資訊**
- 課程名稱：AI在化工上之應用
- 課程單元：Part 0 學習環境設定
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-01-28

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---