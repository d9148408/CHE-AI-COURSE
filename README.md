# AI在化工上之應用課程 (AI Applications in Chemical Engineering)

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-green.svg)
![Course](https://img.shields.io/badge/Course-CHE--AI--114-orange.svg)

---

## 📚 課程簡介

本課程專為**化學工程學系**學生設計，旨在培養學生將人工智慧 (AI) 與機器學習 (Machine Learning) 技術應用於化工領域的實作能力。課程涵蓋從基礎Python程式設計、非監督式學習、監督式學習、到深度學習的完整學習路徑，並透過豐富的化工實際案例進行教學。

- **課程名稱**：AI在化工上之應用
- **課程製作**：逢甲大學 化工系 智慧程序系統工程實驗室
- **授課教師**：莊曜禎 助理教授  
- **適合對象**：化工系大三、大四及研究所學生
- **預備知識**：基礎Python程式設計概念 (如變數、迴圈、函式)
- **課程授權**：[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## 🎯 課程目標

完成本課程後，您將能夠：

1. **紮實的Python基礎**：未學習過Python程式設計的人員，可快速掌握Python程式設計基礎、科學計算 (NumPy/Pandas)、資料視覺化 (Matplotlib/Seaborn)
2. **AI理論與實務**：理解AI與機器學習的核心概念，並能選擇合適的演算法解決問題
3. **豐富的演算法工具箱**：從K-Means、PCA到XGBoost、LSTM，掌握20+種主流演算法
4. **化工應用實戰能力**：能將AI技術應用於製程優化、品質控制、異常檢測、預測建模等實際問題
5. **完整的ML工作流程**：從資料前處理、模型訓練、評估、到部署的端到端能力

---

## 📖 課程結構

本課程分為五大部分，涵蓋 17 個單元的完整學習內容：

### [Part 0: Python學習環境設定](Part_0/)
**主題**：建立穩定且功能完整的Python開發環境

- **[Unit00: 環境設定教學](Part_0/README.md)**
  - Google Colab 雲端環境設定
  - Windows 本地環境設定 (Miniconda + Jupyter)
  - 套件安裝與環境驗證
  - GPU 加速配置

---

### [Part 1: AI與機器學習概論](Part_1/)
**主題**：AI與機器學習的理論基礎、Python程式設計、科學計算與資料視覺化

- **[Unit01: AI與機器學習概論](Part_1/Unit01/)** - AI/ML基礎理論、化工應用案例

***已修讀過大一Python程式設計課程的學生可入複習或快跳過以下單元***
- **[Unit02: Python程式語言基礎 (選讀)](Part_1/Unit02/)** - 語法、流程控制、函式、例外處理
- **[Unit03: NumPy與Pandas (選讀)](Part_1/Unit03/)** - 數值運算、表格資料處理、時間序列
- **[Unit04: Matplotlib與Seaborn (選讀)](Part_1/Unit04/)** - 資料視覺化、統計繪圖

**應用場景**：化工資料分析、實驗數據處理、結果視覺化

---

### [Part 2: 非監督式學習](Part_2/)
**主題**：從未標記數據中自動發現模式與結構

- **[Unit05: 分群分析 (Clustering)](Part_2/Unit05/)** 
  - K-Means、階層式分群、DBSCAN、高斯混合模型
  - **應用**：製程操作模式識別、產品品質分類
  
- **[Unit06: 降維 (Dimensionality Reduction)](Part_2/Unit06/)**
  - PCA、t-SNE、UMAP、LDA
  - **應用**：高維數據視覺化、特徵提取
  
- **[Unit07: 異常檢測 (Anomaly Detection)](Part_2/Unit07/)**
  - Isolation Forest、One-Class SVM、LOF、Autoencoders
  - **應用**：製程異常偵測、設備故障診斷
  
- **[Unit08: 關聯規則學習 (Association Rule Learning)](Part_2/Unit08/)**
  - Apriori、FP-Growth
  - **應用**：製程參數關聯分析、故障模式探索
  
- **[Unit09: 綜合案例研究](Part_2/Unit09/)**
  - 整合多種方法進行端到端分析

---

### [Part 3: 監督式學習](Part_3/)
**主題**：從已標記數據中學習預測模型

#### 回歸 (Regression) - 預測連續數值

- **[Unit10: 線性模型回歸](Part_3/Unit10/)**
  - Linear Regression、Ridge、Lasso、ElasticNet、SGD
  - **應用**：產量預測、能耗估算
  
- **[Unit11: 非線性模型回歸](Part_3/Unit11/)**
  - Polynomial、Decision Tree、Random Forest、SVM、Gaussian Process、XGBoost
  - **應用**：複雜製程建模、品質預測

#### 分類 (Classification) - 預測離散類別

- **[Unit12: 分類模型](Part_3/Unit12/)**
  - Logistic Regression、Decision Tree、Random Forest、SVM、Naive Bayes、XGBoost
  - **應用**：良品不良品分類、安全風險評估

#### 集成學習與模型優化

- **[Unit13: 集成學習 (Ensemble Learning)](Part_3/Unit13/)**
  - Random Forest、XGBoost、LightGBM、CatBoost、Stacking
  - **應用**：高精度預測、競賽級建模
  
- **[Unit14: 模型評估與選擇](Part_3/Unit14/)**
  - 評估指標、交叉驗證、學習曲線、模型比較
  - **應用**：系統性模型優化與選擇

---

### [Part 4: 深度學習](Part_4/)
**主題**：使用深度神經網路處理複雜非線性問題

- **[Unit15: 深度神經網路 (DNN/MLP)](Part_4/Unit15/)**
  - 前向傳播、反向傳播、激活函數、正則化
  - **應用**：軟感測器設計、多變量製程預測
  - **實際案例**：蒸餾塔控制、燃料氣排放預測、採礦優化、紅酒品質預測
  
- **[Unit16: 卷積神經網路 (CNN)](Part_4/Unit16/)**
  - 卷積層、池化層、遷移學習
  - **應用**：產品缺陷檢測、顯微影像分析
  - **實際案例**：腦瘤分類、胸部X光診斷、鑄造缺陷檢測
  
- **[Unit17: 循環神經網路 (RNN/LSTM/GRU)](Part_4/Unit17/)**
  - 時間序列建模、序列預測
  - **應用**：製程動態建模、故障預測、剩餘壽命預測
  - **實際案例**：空氣品質預測、家庭用電預測、鋰電池RUL預測

---

### [Part 5: 進階課程 (研究所選修)](Part_5/)
**主題**：前沿AI技術與應用

- **強化式學習 (Reinforcement Learning)**：製程自動控制、操作策略優化
- **生成式AI (Generative AI)**：分子設計、製程合成、數據增強
- **大型語言模型 (Large Language Models)**：知識萃取、文獻分析、智慧助手

*註：進階課程適合對AI技術有深入興趣的研究所學生。*

---

## 🛠️ 技術棧

### 核心工具
- **Python 3.10** - 主要程式語言
- **Jupyter Notebook/Lab** - 互動式開發環境
- **Google Colab** - 雲端運算平台

### 科學計算與資料處理
- **NumPy** - 數值運算
- **Pandas** - 資料處理與分析
- **SciPy** - 科學計算與統計

### 資料視覺化
- **Matplotlib** - 基礎繪圖
- **Seaborn** - 統計視覺化
- **Plotly** - 互動式視覺化

### 機器學習
- **scikit-learn** - 傳統機器學習演算法
- **XGBoost / LightGBM / CatBoost** - 梯度提升演算法

### 深度學習
- **TensorFlow 2.x** - 深度學習框架
- **Keras** - 高階神經網路API

### 其他工具
- **imbalanced-learn** - 不平衡數據處理
- **UMAP** - 降維與視覺化

---

## 📂 資料夾結構

```
CHE-AI-COURSE/
│
├── Part_0/                    # Python學習環境設定
│   ├── README.md
│   ├── Unit00_Colab_Environment_Setup.md
│   ├── Unit00_Colab_Environment_Setup.ipynb
│   ├── Unit00_Local_Environment_Setup.md
│   ├── Unit00_Local_Environment_Setup.ipynb
│   └── PY310_environment.yml
│
├── Part_1/                    # Python基礎與資料處理
│   ├── README.md
│   ├── Unit01/               # AI與機器學習概論
│   ├── Unit02/               # Python程式語言基礎
│   ├── Unit03/               # NumPy與Pandas
│   └── Unit04/               # Matplotlib與Seaborn
│
├── Part_2/                    # 非監督式學習
│   ├── README.md
│   ├── Unit05/               # 分群分析
│   ├── Unit06/               # 降維
│   ├── Unit07/               # 異常檢測
│   ├── Unit08/               # 關聯規則學習
│   └── Unit09/               # 綜合案例研究
│
├── Part_3/                    # 監督式學習
│   ├── README.md
│   ├── Unit10/               # 線性模型回歸
│   ├── Unit11/               # 非線性模型回歸
│   ├── Unit12/               # 分類模型
│   ├── Unit13/               # 集成學習
│   └── Unit14/               # 模型評估與選擇
│
├── Part_4/                    # 深度學習
│   ├── README.md
│   ├── Unit15/               # 深度神經網路 (DNN/MLP)
│   ├── Unit16/               # 卷積神經網路 (CNN)
│   └── Unit17/               # 循環神經網路 (RNN/LSTM/GRU)
│
├── Part_5/                    # 進階課程 (研究所選修)
│
└── README.md                  # 本文件
```

---

## 🚀 快速開始

### 方法一：使用 Google Colab (推薦新手)

1. 瀏覽至 [Google Colab](https://colab.research.google.com/)
2. 開啟本專案的任一 `.ipynb` 檔案
3. 點選「在 Colab 中開啟」
4. 開始學習！

**優點**：
- 免費GPU/TPU運算資源
- 無需安裝，瀏覽器即可使用
- 適合快速學習與原型開發

---

### 方法二：本地環境設定 (推薦長期學習)

#### 步驟1：安裝 Miniconda

下載並安裝 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

#### 步驟2：建立虛擬環境

```powershell
# 複製專案
git clone https://github.com/fcuycchuang/CHE-AI-COURSE.git
cd CHE-AI-COURSE

# 建立 Python 3.10 虛擬環境
conda env create -f Part_0/PY310_environment.yml

# 啟動環境
conda activate PY310
```

#### 步驟3：啟動 Jupyter

```powershell
# 啟動 Jupyter Notebook
jupyter notebook

# 或啟動 Jupyter Lab (推薦)
jupyter lab
```

#### 步驟4：開啟 Notebook 開始學習

瀏覽至對應的 Part 資料夾，開啟 `.ipynb` 檔案即可開始學習。

---

## 📝 學習路徑建議

### 🎓 初學者路徑 (建議學習順序)

1. **Part 0** → 環境設定 (必修)
2. **Part 1** → Python基礎與資料處理 (必修)
3. **Part 2** → 非監督式學習 (建議完整學習)
4. **Part 3** → 監督式學習 (建議完整學習)
5. **Part 4** → 深度學習 (建議完整學習)

**預估時間**：約 54 小時 (每單元 3 小時)

---

### 🏆 進階學習者路徑

如果您已具備Python與機器學習基礎：

1. 快速複習 **Part 1** (僅複習不熟悉的部分)
2. 深入學習 **Part 2-4** 的進階演算法
3. 專注於 **實際案例** 與 **作業練習**
4. 挑戰 **Part 5** 的前沿主題

---

### 🎯 專題導向路徑

根據您的興趣選擇特定主題深入學習：

- **製程優化** → Unit05 (分群)、Unit10-11 (回歸)、Unit13 (集成學習)
- **品質控制** → Unit07 (異常檢測)、Unit12 (分類)、Unit15 (DNN)
- **預測建模** → Unit11 (非線性回歸)、Unit13 (集成學習)、Unit17 (RNN/LSTM)
- **影像分析** → Unit16 (CNN)
- **故障診斷** → Unit07 (異常檢測)、Unit12 (分類)、Unit17 (RNN)

---

## 📚 每個單元包含什麼？

每個單元通常包含以下內容：

1. **📄 教學講義 (.md)**
   - 完整的理論說明與數學推導
   - 演算法原理與流程圖
   - 化工應用案例介紹
   - 參數選擇建議與注意事項

2. **📓 範例程式碼 (.ipynb)**
   - 完整的程式實作範例
   - 逐步說明與註解
   - 視覺化結果
   - 化工實際數據應用

3. **📝 作業練習 (Homework.ipynb)**
   - 實作練習題
   - 引導式問題設計
   - 提供部分程式碼框架

4. **📊 結果資料夾 (Results/)**
   - 執行結果圖表
   - 模型評估報告
   - 參數調整記錄

---

## 💡 學習建議

### 理論與實作並重
- 先閱讀講義理解原理，再執行程式碼實作
- 修改程式碼參數，觀察結果變化
- 嘗試將方法應用到自己的數據

### 善用視覺化
- 每個演算法都提供視覺化範例
- 圖表比數字更能幫助理解

### 完成作業練習
- 作業是鞏固學習的最佳方式
- 提供部分程式碼框架，降低難度

### 化工應用導向
- 每個單元都包含化工實際案例
- 思考如何應用到您的研究或工作中

### 建立學習筆記
- 記錄重要概念與心得
- 整理各演算法的適用場景與參數

---

## 🤝 貢獻

歡迎提出問題、建議或貢獻：

1. **Issue**：發現錯誤或有建議，請開 Issue
2. **Pull Request**：歡迎提交程式碼改進或新範例
3. **討論**：歡迎在討論區分享學習心得與應用案例

---

## 📧 聯絡方式

**授課教師**：莊曜禎 助理教授  
**單位**：逢甲大學 化學工程學系  
**Email**：yaocchuang@o365.fcu.edu.tw

---

## 📄 授權

本課程內容採用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 授權。
允許他人以非商業方式使用、修改、再創作作品，但必須標明原作者，並以相同的CC BY-NC-SA授權條款發布衍生作品。 

---

## 🙏 致謝

感謝以下開源社群與工具：
- Python Software Foundation
- scikit-learn、TensorFlow、Keras 開發團隊
- Jupyter Project
- 所有提供開源數據集的研究機構
- ChatGPT、Gemini、Claude 提供的技術支援、內容校對以及內容優化

---

## 📈 課程更新記錄

- **2026-01** - Unit01-17 完成初版

---

**祝您學習順利！💪**

如有任何問題，歡迎隨時聯繫或在討論區提問。讓我們一起探索AI在化工領域的無限可能！
