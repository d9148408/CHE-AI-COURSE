# Unit17 循環神經網路 (Recurrent Neural Networks, RNN)

## 📚 單元簡介

在 Unit15 和 Unit16 中，我們學習了 DNN（處理結構化數據）和 CNN（處理影像數據）。這些架構都有一個共同特點：**輸入是固定大小的向量或矩陣，各個輸入之間相互獨立**。但在化工製程中，大量數據是**時間序列 (Time Series)**：溫度、壓力、流量、濃度隨時間變化，前後時刻存在強烈的相依性。

**循環神經網路 (Recurrent Neural Networks, RNN)** 及其進階變體 **LSTM** 和 **GRU** 是專為序列數據設計的深度學習架構，核心創新是：

- **記憶機制 (Memory Mechanism)**：透過隱藏狀態 (Hidden State) 記住過去的資訊
- **參數共享 (Parameter Sharing)**：同一組權重在時間軸上重複使用
- **可變長度輸入 (Variable-Length Input)**：處理不同長度的序列
- **時序依賴建模 (Temporal Dependency)**：捕捉長短期依賴關係

RNN 解決了傳統 DNN 無法處理的問題：
- **時序預測**：基於過去 N 個時刻的數據，預測未來 M 個時刻
- **異常檢測**：識別時間序列中的異常模式
- **序列分類**：將整個序列分類（正常 vs. 故障）
- **序列到序列 (Seq2Seq)**：翻譯、摘要、對話生成

在化工領域，RNN 特別適合：
- **製程動態建模**：反應器、蒸餾塔、熱交換器的動態響應預測
- **故障預測與診斷**：設備健康監測、剩餘壽命預測 (RUL)
- **品質預測**：基於歷史操作軌跡預測產品品質
- **預測性控制 (MPC)**：多步預測，優化控制策略
- **批次製程監控**：追蹤批次進程，早期異常檢測

本單元涵蓋：
- **RNN 核心原理**：前向傳播、BPTT、梯度消失/爆炸問題
- **進階架構**：LSTM、GRU、雙向 RNN (BiRNN)、Seq2Seq、Attention
- **TensorFlow/Keras 實作**：SimpleRNN、LSTM、GRU、return_sequences、Stateful RNN
- **三個實際案例**：鍋爐 NOx 排放、脫丁烷塔控制、NASA 渦輪風扇 RUL 預測

本單元是 Part_4 深度學習系列的時間序列專題，建議在完成 Unit15 (DNN) 基礎後學習。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **理解 RNN 的核心概念**：隱藏狀態、時間展開、BPTT、梯度消失/爆炸
2. **掌握 LSTM/GRU 原理**：門控機制、細胞狀態、遺忘門、輸入門、輸出門
3. **熟練使用 TensorFlow/Keras**：SimpleRNN、LSTM、GRU、TimeDistributed、Bidirectional
4. **解決化工時序問題**：製程預測、故障診斷、RUL 預測、動態建模
5. **應用進階技術**：Seq2Seq、Attention、雙向 RNN、多變量時間序列

---

## 📖 單元內容架構

### 1️⃣ 總覽篇：RNN、LSTM、GRU 基礎 ⭐

**檔案**：
- 講義：[Unit17_RNN_Overview.md](Unit17_RNN_Overview.md)
- 範例：[Unit17_RNN_Overview.ipynb](Unit17_RNN_Overview.ipynb)

**內容重點**：
- **為什麼需要 RNN？**：
  - DNN 的限制：無法處理序列、固定輸入長度、無時序記憶
  - 化工時序數據的特點：溫度、壓力、流量的時間相依性
  - RNN 的核心優勢：記憶機制、參數共享、可變長度
  
- **RNN 基礎架構**：
  - 隱藏狀態 (Hidden State)：$h_t = f(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$
  - 輸出 (Output)：$y_t = W_{hy}h_t + b_y$
  - 時間展開 (Unrolling in Time)：RNN 如何處理序列
  - 參數共享：$W_{hh}, W_{xh}, W_{hy}$ 在所有時刻重複使用
  
- **RNN 的挑戰：梯度消失與爆炸**：
  - **梯度消失 (Vanishing Gradient)**：長序列訓練時梯度趨近於 0，無法學習長期依賴
  - **梯度爆炸 (Exploding Gradient)**：梯度指數增長，導致訓練不穩定
  - 解決方案：LSTM、GRU、梯度裁剪 (Gradient Clipping)、BPTT 截斷
  
- **LSTM (Long Short-Term Memory)**：
  - **細胞狀態 (Cell State)**：$C_t$，長期記憶通道
  - **遺忘門 (Forget Gate)**：$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$，決定遺忘多少過去資訊
  - **輸入門 (Input Gate)**：$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$，決定記住多少新資訊
  - **輸出門 (Output Gate)**：$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$，決定輸出多少資訊
  - **細胞更新**：$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
  - **隱藏狀態更新**：$h_t = o_t \odot \tanh(C_t)$
  
- **GRU (Gated Recurrent Unit)**：
  - 簡化版 LSTM，只有兩個門（更新門、重置門）
  - **更新門 (Update Gate)**：$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$
  - **重置門 (Reset Gate)**：$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$
  - 參數更少、訓練更快，性能接近 LSTM
  
- **RNN 變體架構**：
  - **雙向 RNN (Bidirectional RNN)**：同時從前向後、從後向前處理序列
  - **多層 RNN (Stacked RNN)**：堆疊多層 LSTM/GRU，提升表達能力
  - **Seq2Seq (Sequence-to-Sequence)**：編碼器-解碼器架構，用於序列映射
  - **Attention 機制**：讓模型關注序列中的重要部分
  
- **時間序列預測任務類型**：
  - **Many-to-One**：序列 → 單一輸出（整個序列分類、最終值預測）
  - **Many-to-Many (同步)**：序列 → 等長序列（時間序列預測、異常檢測）
  - **Many-to-Many (異步)**：序列 → 不同長度序列（Seq2Seq、翻譯）
  - **One-to-Many**：單一輸入 → 序列（影像描述生成）
  
- **TensorFlow/Keras 實作**：
  - `SimpleRNN`：基礎 RNN 層
  - `LSTM`：LSTM 層（`units`, `return_sequences`, `return_state`）
  - `GRU`：GRU 層
  - `Bidirectional`：雙向包裝器
  - `TimeDistributed`：對每個時刻應用相同層
  - `Stateful RNN`：跨 batch 保留狀態
  
- **數據準備關鍵**：
  - 時間窗口切分 (Sliding Window)：過去 N 步預測未來 M 步
  - 數據標準化：MinMaxScaler、StandardScaler（逐特徵標準化）
  - 3D 輸入格式：`(samples, timesteps, features)`
  - 訓練/驗證/測試集分割：**順序分割**（不能隨機打亂）
  
- **化工領域應用場景**：
  - **製程動態建模**：ARX、NARX 模型的神經網路版
  - **故障預測**：設備健康監測、異常檢測、剩餘壽命預測
  - **品質預測**：基於操作歷史預測產品規格
  - **預測性控制**：多步預測，MPC 優化

**適合讀者**：所有學員，**建議最先閱讀**以建立完整的 RNN 理論基礎

---

### 2️⃣ 進階專題：雙向 RNN、Seq2Seq、Attention ⭐

**檔案**：
- 講義：[Unit17_Advanced_BiRNN_Seq2Seq_Attention.md](Unit17_Advanced_BiRNN_Seq2Seq_Attention.md)
- 程式範例：[Unit17_Advanced_BiRNN_Seq2Seq_Attention.ipynb](Unit17_Advanced_BiRNN_Seq2Seq_Attention.ipynb)

**內容重點**：
- **雙向 RNN (Bidirectional RNN)**：
  - 動機：有些任務需要未來資訊（如序列標註、異常檢測）
  - 架構：前向 LSTM + 反向 LSTM，拼接輸出
  - Keras 實作：`Bidirectional(LSTM(units))`
  - 適用場景：序列分類、異常檢測、自然語言處理
  - 限制：無法用於即時預測（需要完整序列）
  
- **Seq2Seq (Sequence-to-Sequence)**：
  - 架構：**編碼器 (Encoder)** + **解碼器 (Decoder)**
  - 編碼器：將輸入序列壓縮成固定長度向量（Context Vector）
  - 解碼器：基於 Context Vector 生成輸出序列
  - 應用：機器翻譯、對話系統、摘要生成
  - 化工應用：操作軌跡生成、配方優化、多步預測
  
- **Attention 機制**：
  - 問題：Seq2Seq 的 Context Vector 是瓶頸（長序列資訊丟失）
  - 解決：Attention 讓解碼器在每一步都關注編碼器的不同部分
  - 權重計算：$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}$
  - Keras 實作：自定義 Attention 層
  - 可解釋性：視覺化 Attention 權重，理解模型決策
  
- **Transformer 簡介**：
  - 完全基於 Attention，無 RNN 結構
  - Self-Attention、Multi-Head Attention
  - 平行化訓練，速度快
  - 應用：NLP（BERT、GPT）、時間序列（Temporal Fusion Transformer）
  
- **實作案例**：
  - 雙向 LSTM 用於異常檢測
  - Seq2Seq 用於多步預測
  - Attention 用於時間序列預測可解釋性

**適合讀者**：完成基礎 RNN 後，對進階架構有興趣的學員

---

### 3️⃣ 實際案例 1：鍋爐 NOx 排放預測 (Boiler Emission) ⭐

**檔案**：
- 講義：[Unit17_Example_Boiler.md](Unit17_Example_Boiler.md)
- 程式範例：[Unit17_Example_Boiler.ipynb](Unit17_Example_Boiler.ipynb)

**內容重點**：
- **問題背景**：
  - 鍋爐燃燒過程產生 NOx 排放，受多變數影響（燃料流量、空氣量、溫度）
  - 傳統 DNN 無法捕捉動態響應（滯後、慣性）
  - 目標：基於過去 10 個時刻的操作數據，預測當前 NOx 排放
  
- **數據特性**：
  - 多變量時間序列（燃料流量、空氣流量、煙氣溫度等）
  - 時間相依性強（前後時刻高度相關）
  - 非線性、動態系統
  
- **模型設計**：
  - LSTM 架構：2 層 LSTM (64 → 32 units)
  - 時間窗口：過去 10 步（timesteps=10）
  - Many-to-One 預測（輸出單一值）
  
- **關鍵技術**：
  - 滑動窗口數據準備：`create_sequences(data, timesteps=10)`
  - 多變量輸入處理：`input_shape=(10, n_features)`
  - 順序分割：訓練/驗證/測試集不能打亂
  - LSTM vs. GRU 性能對比
  
- **工程意義**：
  - 實時排放預測，滿足環保法規
  - 優化操作參數降低 NOx
  - 預警異常排放

**適合場景**：燃燒過程建模、排放預測、多變量時序回歸

---

### 4️⃣ 實際案例 2：脫丁烷塔軟感測器 (Debutanizer Column)

**檔案**：
- 講義：[Unit17_Example_Debutanizer.md](Unit17_Example_Debutanizer.md)
- 程式範例：[Unit17_Example_Debutanizer.ipynb](Unit17_Example_Debutanizer.ipynb)

**內容重點**：
- **問題背景**：
  - 脫丁烷塔（Debutanizer）是煉油廠關鍵單元，分離 C4（丁烷）
  - 線上測量丁烷濃度成本高、時間延遲大
  - 目標：基於溫度、壓力、流量等易測變數預測丁烷濃度
  
- **數據特性**：
  - 真實煉油廠操作數據
  - 7 個輸入變數（溫度 4 個、流量 2 個、壓力 1 個）
  - 1 個輸出變數（丁烷濃度 mol%）
  - 動態響應顯著（滯後約 5-10 分鐘）
  
- **模型設計**：
  - LSTM 軟感測器：3 層 LSTM (128 → 64 → 32 units)
  - 時間窗口：過去 20 步（考慮滯後效應）
  - Dropout (0.2) 防止過擬合
  
- **關鍵技術**：
  - 與 DNN 對比：RNN 捕捉動態優於靜態模型
  - 滯後分析：找出最佳時間窗口長度
  - 殘差分析：檢查預測誤差分佈
  - Early Stopping、Learning Rate Reduction
  
- **工程意義**：
  - 軟感測器實現即時監控
  - 降低分析成本
  - 提升控制性能（更快響應）

**適合場景**：軟感測器設計、蒸餾塔建模、製程動態預測

---

### 5️⃣ 實際案例 3：NASA 渦輪風扇 RUL 預測 (Remaining Useful Life) ⭐

**檔案**：
- 講義：[Unit17_Example_NASA_Turbofan.md](Unit17_Example_NASA_Turbofan.md)
- 程式範例：[Unit17_Example_NASA_Turbofan.ipynb](Unit17_Example_NASA_Turbofan.ipynb)

**內容重點**：
- **問題背景**：
  - 渦輪風扇引擎 (Turbofan Engine) 的健康監測與維護
  - 目標：預測設備剩餘壽命 (Remaining Useful Life, RUL)，實現預測性維護
  - RUL 定義：設備從當前狀態到故障的時間（週期數）
  
- **數據特性**：
  - NASA C-MAPSS 數據集（標準 PHM 數據集）
  - 多變量時間序列（21 個感測器，溫度、壓力、轉速等）
  - 多個引擎（Run-to-Failure 數據）
  - 標籤：每個時刻的 RUL（從大到小遞減）
  
- **模型設計**：
  - 深層 LSTM：3-4 層 LSTM (128 → 64 → 32 units)
  - 時間窗口：過去 50 步（捕捉退化趨勢）
  - RUL 目標裁剪（常見設定：max_RUL=125）
  - Many-to-One 回歸（預測單一 RUL 值）
  
- **關鍵技術**：
  - **特徵工程**：滾動統計（均值、標準差、趨勢）
  - **RUL 標籤生成**：從 run-to-failure 數據計算 RUL
  - **分段訓練**：將每個引擎視為獨立序列
  - **評估指標**：RMSE、Score Function（對晚期預測誤差懲罰更重）
  - **健康指標 (HI) 構建**：PCA 降維後的綜合健康分數
  
- **工程意義**：
  - 預測性維護（Predictive Maintenance）
  - 避免非計劃停機
  - 優化維護排程
  - 降低維護成本
  
- **化工類比應用**：
  - 泵浦 RUL 預測
  - 熱交換器性能衰退預測
  - 催化劑失活預測
  - 閥門健康監測

**適合場景**：設備健康監測、RUL 預測、預測性維護、PHM

---

### 6️⃣ 實作練習

**檔案**：[Unit17_RNN_Homework.ipynb](Unit17_RNN_Homework.ipynb)

**練習內容**：
- 建立完整的 LSTM/GRU 時間序列預測模型
- 比較 SimpleRNN、LSTM、GRU 的性能
- 實驗不同時間窗口長度的影響
- 多步預測 (Multi-Step Forecasting)
- 雙向 LSTM 應用
- Attention 機制實作
- 模型可解釋性分析
- 應用於自己的化工時序數據

---

## 📊 數據集說明

### 1. 鍋爐 NOx 排放數據 (`data/boiler/`)
- 鍋爐燃燒過程操作數據
- 多變量時間序列（燃料流量、空氣流量、溫度等）
- 目標變數：NOx 排放濃度 (ppm)
- 用於多變量時序回歸演示

### 2. 脫丁烷塔數據 (`data/debutanizer/`)
- 真實煉油廠操作數據
- 7 個輸入變數（溫度、流量、壓力）
- 1 個輸出變數（丁烷濃度 mol%）
- 用於軟感測器建模與動態預測

### 3. NASA 渦輪風扇數據 (`data/nasa_turbofan/`)
- NASA C-MAPSS 數據集（PHM08 Challenge）
- 多個引擎的 Run-to-Failure 數據
- 21 個感測器時間序列
- 標籤：RUL（剩餘壽命）
- 用於設備健康監測與 RUL 預測

---

## 🎓 RNN 建模決策指南

### RNN 變體選擇

| 模型 | 參數量 | 訓練速度 | 長期記憶 | 適用場景 |
|------|-------|---------|---------|---------|
| **SimpleRNN** | 低 | 快 | 差 | 短序列（< 10 步），教學演示 |
| **LSTM** | 高 | 慢 | 優 | **長序列（> 20 步），複雜依賴** |
| **GRU** | 中 | 中 | 優 | **通用首選**（LSTM 簡化版） |
| **Bidirectional** | 2× | 慢 | 優 | 序列分類、異常檢測（需完整序列） |

### 時間窗口設計

| 序列特性 | 建議窗口長度 | 備註 |
|---------|-------------|------|
| **快速響應** | 5-10 步 | 反應速度快的系統 |
| **中等滯後** | 10-30 步 | 大多數化工過程 |
| **慢速動態** | 30-100 步 | 熱交換器、大容積反應器 |
| **長期依賴** | 100+ 步 | 批次製程、設備退化 |

### 網路深度設計

| 層數 | 適用場景 | 參數量 | 過擬合風險 |
|-----|---------|-------|-----------|
| **1 層** | 簡單時序、小數據集 | 低 | 低 |
| **2 層** | **通用首選** | 中 | 中 |
| **3-4 層** | 複雜依賴、大數據集 | 高 | 高（需 Dropout） |
| **5+ 層** | 超複雜任務（少見） | 極高 | 極高 |

### 超參數建議

| 參數 | 建議值 | 說明 |
|------|-------|------|
| **units** | 32-128 | LSTM/GRU 單元數（64 常用） |
| **timesteps** | 10-50 | 時間窗口長度（根據動態響應） |
| **batch_size** | 32-128 | 時序數據建議較小（32-64） |
| **learning_rate** | 0.001 | Adam 預設值，時序模型較敏感 |
| **dropout** | 0.2-0.4 | 防止過擬合（0.2 常用） |
| **epochs** | 50-200 | 搭配 Early Stopping |

### 輸入格式

```python
# 3D 輸入格式：(samples, timesteps, features)
# samples: 樣本數（滑動窗口數量）
# timesteps: 時間窗口長度（過去幾步）
# features: 特徵數（多變量）

# 範例：100 個樣本，每個樣本過去 10 步，5 個特徵
X.shape = (100, 10, 5)
```

---

## 💻 實作環境需求

### 必要套件
```python
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scikit-learn >= 1.0.0
tensorflow >= 2.10.0             # 深度學習框架
keras >= 2.10.0                  # 高階 API
```

### 選用套件
```python
statsmodels >= 0.13.0            # 時間序列分析
pmdarima >= 2.0.0                # Auto ARIMA
tslearn >= 0.5.0                 # 時間序列機器學習
sktime >= 0.13.0                 # 時間序列工具包
tensorboard >= 2.10.0            # 訓練視覺化
```

---

## 📈 學習路徑建議

### 第一階段：理論基礎建立（必讀）
1. 閱讀 [Unit17_RNN_Overview.md](Unit17_RNN_Overview.md)
2. 執行 [Unit17_RNN_Overview.ipynb](Unit17_RNN_Overview.ipynb)
3. 重點掌握：
   - RNN 隱藏狀態更新機制
   - LSTM 的三個門（遺忘門、輸入門、輸出門）
   - 梯度消失/爆炸問題與解決方案
   - 時間序列數據準備（滑動窗口）

### 第二階段：從簡單到複雜的案例學習

**建議學習順序**：

1. **鍋爐 NOx 排放**（建議最先學習，多變量時序回歸）
   - 閱讀 [Unit17_Example_Boiler.md](Unit17_Example_Boiler.md)
   - 執行 [Unit17_Example_Boiler.ipynb](Unit17_Example_Boiler.ipynb)
   - 重點：多變量輸入、時間窗口設計、LSTM 基礎應用
   
2. **脫丁烷塔軟感測器**（軟感測器經典應用）
   - 閱讀 [Unit17_Example_Debutanizer.md](Unit17_Example_Debutanizer.md)
   - 執行 [Unit17_Example_Debutanizer.ipynb](Unit17_Example_Debutanizer.ipynb)
   - 重點：動態系統建模、滯後效應、RNN vs. DNN 對比
   
3. **NASA 渦輪風扇 RUL**（進階應用，RUL 預測）
   - 閱讀 [Unit17_Example_NASA_Turbofan.md](Unit17_Example_NASA_Turbofan.md)
   - 執行 [Unit17_Example_NASA_Turbofan.ipynb](Unit17_Example_NASA_Turbofan.ipynb)
   - 重點：RUL 標籤生成、健康指標、預測性維護

### 第三階段：進階架構（選修）
1. 閱讀 [Unit17_Advanced_BiRNN_Seq2Seq_Attention.md](Unit17_Advanced_BiRNN_Seq2Seq_Attention.md)
2. 執行 [Unit17_Advanced_BiRNN_Seq2Seq_Attention.ipynb](Unit17_Advanced_BiRNN_Seq2Seq_Attention.ipynb)
3. 重點：雙向 RNN、Seq2Seq、Attention、Transformer

### 第四階段：綜合練習
1. 完成 [Unit17_RNN_Homework.ipynb](Unit17_RNN_Homework.ipynb)
2. 比較 LSTM vs. GRU vs. SimpleRNN
3. 實驗不同時間窗口長度
4. 嘗試多步預測（預測未來 5 步）
5. 應用於自己的化工時序數據

---

## 🔍 化工領域核心應用

### 1. 製程動態建模與軟感測器 ⭐
- **目標**：基於歷史操作數據預測當前/未來狀態
- **RNN 優勢**：
  - 捕捉動態響應（滯後、慣性）
  - 處理多變量耦合
  - 自適應模型（持續學習）
- **典型應用**：
  - 蒸餾塔組成推論（脫丁烷塔案例）
  - 反應器溫度預測
  - 產品品質即時估計
  - 能耗預測
- **關鍵技術**：
  - 時間窗口選擇（根據時間常數）
  - 多變量輸入設計
  - 滯後分析（找出最佳延遲）
  - 與靜態 DNN 性能對比

### 2. 故障預測與診斷
- **目標**：早期檢測異常、預測故障發生
- **RNN 優勢**：
  - 學習正常操作模式
  - 識別異常趨勢
  - 時序異常檢測
- **典型應用**：
  - 設備健康監測
  - 過程異常檢測
  - 洩漏早期預警
  - 觸媒失活檢測
- **關鍵技術**：
  - 正常數據訓練（異常為測試集）
  - 重建誤差閾值設定
  - 異常分數計算
  - 報警邏輯設計

### 3. 剩餘壽命預測 (RUL) ⭐
- **目標**：預測設備從當前狀態到故障的時間
- **RNN 優勢**：
  - 學習退化趨勢
  - 多感測器融合
  - 個體化預測
- **典型應用**：
  - 泵浦軸承壽命
  - 熱交換器性能衰退
  - 觸媒失活時間
  - 閥門健康監測
- **關鍵技術**：
  - RUL 標籤生成
  - 健康指標構建（HI）
  - 滾動統計特徵
  - Score Function 評估
- **化工設備應用**：
  - 泵浦、壓縮機、風機
  - 熱交換器、冷凝器
  - 觸媒、吸附劑
  - 閥門、密封件

### 4. 批次製程監控 (Batch Process Monitoring)
- **目標**：追蹤批次進程，早期異常檢測
- **RNN 優勢**：
  - 處理批次軌跡（可變長度）
  - 階段識別
  - 終點預測
- **典型應用**：
  - 批次反應器
  - 發酵過程
  - 聚合反應
  - 製藥批次
- **關鍵技術**：
  - 批次對齊（DTW）
  - 階段分割
  - 終點預測
  - 批次到批次變異分析

### 5. 預測性控制 (Model Predictive Control, MPC)
- **目標**：多步預測，優化未來控制動作
- **RNN 優勢**：
  - 準確的多步預測
  - 非線性動態建模
  - 實時計算
- **典型應用**：
  - 蒸餾塔控制
  - 反應器溫度控制
  - 能源管理系統
- **關鍵技術**：
  - 多步預測（Many-to-Many）
  - 滾動時域優化
  - 約束處理
  - 閉環穩定性

### 6. 多步時序預測 (Multi-Step Forecasting)
- **策略選擇**：
  - **Direct Strategy**：訓練多個模型，每個預測一步
  - **Recursive Strategy**：單一模型，預測值作為下一步輸入
  - **Direct-Recursive Hybrid**：結合兩者優勢
  - **Seq2Seq**：編碼器-解碼器，一次生成多步
- **適用場景**：
  - 需求預測（原料需求、能源需求）
  - 生產排程
  - 庫存管理

---

## 📝 評估指標總結

### 回歸任務（時序預測）
- **MSE/RMSE**：均方誤差，最常用
- **MAE**：平均絕對誤差，對異常值不敏感
- **MAPE**：平均絕對百分比誤差，相對誤差
- **R² Score**：解釋變異比例
- **時序特定指標**：
  - **sMAPE**：對稱 MAPE，解決零值問題
  - **MASE**：平均絕對比例誤差（與基準方法比較）

### RUL 預測專用指標
- **RMSE**：標準均方根誤差
- **Score Function**（NASA PHM）：
  - 早期預測偏高懲罰輕：$s_i = e^{-\frac{d_i}{13}} - 1$ (if $d_i < 0$)
  - 晚期預測偏低懲罰重：$s_i = e^{+\frac{d_i}{10}} - 1$ (if $d_i \geq 0$)
  - 總分：$Score = \sum_{i=1}^{n} s_i$

### 異常檢測
- **重建誤差 (Reconstruction Error)**：$\text{Error} = |y_{\text{true}} - y_{\text{pred}}|$
- **異常分數 (Anomaly Score)**：標準化重建誤差
- **閾值設定**：$\text{threshold} = \mu + k\sigma$ (常用 $k=3$)
- **分類指標**：Precision、Recall、F1-Score、ROC-AUC

### 訓練監控
- **Training Loss vs. Validation Loss**：過擬合診斷
- **Gradient Norm**：梯度消失/爆炸檢測
- **Learning Rate Schedule**：學習率調整記錄

---

## 🚀 進階學習方向

完成本單元後，可以進一步探索：

1. **Transformer for Time Series**：Temporal Fusion Transformer (TFT)、Informer
2. **神經 ODE (Neural ODEs)**：連續時間動態系統建模
3. **WaveNet**：因果卷積用於時間序列
4. **TCN (Temporal Convolutional Networks)**：CNN 用於時序（平行化訓練）
5. **Prophet / NeuralProphet**：時間序列預測工具
6. **ARIMA vs. LSTM 混合模型**：結合統計與深度學習
7. **聯邦學習 (Federated Learning)**：分佈式設備數據隱私保護
8. **強化學習 + RNN**：動態優化控制

---

## 📚 參考資源

### 教科書
1. *Deep Learning* by Goodfellow, Bengio, Courville（第 10 章 RNN）
2. *Time Series Analysis and Its Applications* by Shumway & Stoffer
3. *Introduction to Time Series Forecasting with Python* by Jason Brownlee

### 經典論文
- **LSTM** (1997): "Long Short-Term Memory" - Hochreiter & Schmidhuber
- **GRU** (2014): "Learning Phrase Representations using RNN Encoder-Decoder" - Cho et al.
- **Seq2Seq** (2014): "Sequence to Sequence Learning with Neural Networks" - Sutskever et al.
- **Attention** (2015): "Neural Machine Translation by Jointly Learning to Align and Translate" - Bahdanau et al.
- **Transformer** (2017): "Attention Is All You Need" - Vaswani et al.

### 線上資源
- [Colah's Blog: Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [TensorFlow Time Series Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Keras RNN Guide](https://keras.io/guides/working_with_rnns/)

### 化工領域應用論文
- "LSTM networks for predictive maintenance in manufacturing" (Journal of Manufacturing Systems, 2020)
- "Soft sensor development using LSTM for chemical processes" (Chemical Engineering Science, 2019)
- "RUL prediction for industrial equipment using deep learning" (Reliability Engineering & System Safety, 2021)

---

## ✍️ 課後習題提示

1. **架構對比**：在同一數據集上比較 SimpleRNN、LSTM、GRU 的性能（RMSE、訓練時間）
2. **時間窗口實驗**：測試 timesteps = [5, 10, 20, 50] 對預測精度的影響
3. **多步預測**：實作預測未來 5 步（Recursive Strategy vs. Seq2Seq）
4. **雙向 LSTM**：比較單向 vs. 雙向 LSTM 在異常檢測上的效果
5. **Attention 視覺化**：實作 Attention 機制，視覺化模型關注哪些時刻
6. **RUL 預測**：應用於化工設備（泵浦、熱交換器），計算 Score Function
7. **化工應用**：將 LSTM 應用於自己的製程時序數據，撰寫完整建模報告

---

## 📞 技術支援

如有疑問或需要協助，歡迎透過以下方式聯繫：
- 課程討論區
- 指導教授 Office Hour
- 課程助教

---

**🎉 RNN/LSTM 是時間序列深度學習的核心！掌握 RNN 讓化工製程的動態預測與健康監測更加精準！**

---

**課程資訊**
- 課程名稱：AI在化工上之應用
- 課程單元：Unit17 - 循環神經網路 (RNN) 與時間序列預測
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-01-28

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---