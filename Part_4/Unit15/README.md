# Unit15 深度神經網路與多層感知機 (Deep Neural Networks & Multi-Layer Perceptron)

## 📚 單元簡介

歡迎來到深度學習的世界！在前面的 Part_3 中，我們學習了傳統機器學習方法（線性模型、樹模型、集成學習）。這些方法在結構化數據上表現優異，但面對複雜的非線性關係、高維數據、或需要自動特徵學習的場景時，深度學習展現出革命性的優勢。

**深度神經網路 (Deep Neural Networks, DNN)** 和 **多層感知機 (Multi-Layer Perceptron, MLP)** 是深度學習的基礎架構，也是理解後續 CNN、RNN 等進階架構的關鍵。DNN 透過多層神經元的堆疊，能夠自動學習數據的階層式特徵表示，從簡單到複雜，層層抽象。

在化工領域，DNN 特別適合：
- **軟感測器 (Soft Sensor)**：從易測量變數預測難測量變數（如從溫壓流量預測產品品質）
- **複雜非線性建模**：反應動力學、相平衡、傳遞現象的高精度預測
- **高維數據分析**：光譜數據、過程數據的降維與特徵提取
- **實時製程監控**：基於歷史數據的快速預測與決策

本單元涵蓋：
- **DNN/MLP 理論基礎**：神經元、前向傳播、反向傳播、激活函數、損失函數
- **TensorFlow/Keras 實作**：模型建立、訓練、評估、調參、部署
- **四個實際案例**：蒸餾塔控制、燃料氣排放預測、採礦過程優化、紅酒品質預測

本單元是 Part_4 深度學習系列的基石，為後續的 CNN (Unit16) 和 RNN (Unit17) 打下堅實基礎。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **理解深度學習的核心概念**：神經元數學模型、前向傳播、反向傳播、梯度下降、激活函數
2. **掌握 DNN 建模流程**：數據準備、模型設計、訓練策略、過擬合防止、超參數調整
3. **熟練使用 TensorFlow/Keras**：Sequential API、Functional API、模型編譯、訓練、評估、儲存載入
4. **解決實際化工問題**：軟感測器設計、製程預測、品質控制、排放監測
5. **優化模型性能**：Early Stopping、Dropout、Batch Normalization、學習率調整

---

## 📖 單元內容架構

### 1️⃣ 總覽篇：DNN 與 MLP 基礎 ⭐

**檔案**：
- 講義：[Unit15_DNN_MLP_Overview.md](Unit15_DNN_MLP_Overview.md)
- 範例：[Unit15_DNN_MLP_Overview.ipynb](Unit15_DNN_MLP_Overview.ipynb)

**內容重點**：
- **DNN 與 MLP 基礎理論**：
  - 什麼是神經網路？歷史發展與核心概念
  - 神經元數學模型：加權和、偏差、激活函數
  - 多層網路架構：輸入層、隱藏層、輸出層
  - 前向傳播 (Forward Propagation)：如何進行預測
  
- **深度學習的核心機制**：
  - 損失函數 (Loss Function)：MSE、MAE、Cross-Entropy、Huber Loss
  - 反向傳播 (Backpropagation)：如何從錯誤中學習
  - 梯度下降 (Gradient Descent)：參數更新策略（SGD、Adam、RMSprop）
  - 激活函數選擇：ReLU、Leaky ReLU、ELU、Sigmoid、Tanh、Softmax
  
- **TensorFlow/Keras 基礎**：
  - Keras 建模 API：Sequential、Functional、Model Subclassing
  - 模型編譯：optimizer、loss、metrics
  - 模型訓練：fit()、callbacks、validation_split
  - 模型評估與預測：evaluate()、predict()
  
- **模型優化技巧**：
  - 防止過擬合：Dropout、L1/L2 Regularization、Early Stopping
  - 加速訓練：Batch Normalization、學習率調度
  - 超參數調整：網路深度、寬度、學習率、batch size
  
- **化工領域應用場景**：
  - 軟感測器設計（推論難測變數）
  - 製程優化（尋找最佳操作點）
  - 品質預測（產品規格預測）
  - 故障診斷（設備健康監測）

**適合讀者**：所有學員，**建議最先閱讀**以建立完整的深度學習理論基礎

---

### 2️⃣ 實際案例 1：蒸餾塔乙醇濃度預測 (Distillation Column) ⭐

**檔案**：
- 講義：[Unit15_Example_Distillation.md](Unit15_Example_Distillation.md)
- 程式範例：[Unit15_Example_Distillation.ipynb](Unit15_Example_Distillation.ipynb)

**內容重點**：
- **問題背景**：
  - 蒸餾塔是化工分離單元操作的核心設備
  - 線上測量乙醇濃度成本高、時間延遲大
  - 目標：基於溫度、流量、壓力等易測變數預測乙醇濃度
  
- **數據特性**：
  - 真實蒸餾塔操作數據
  - 多個連續特徵（溫度、流量、壓力）
  - 回歸任務（預測連續數值）
  
- **模型設計**：
  - DNN 架構設計：輸入層、3-4 層隱藏層、輸出層
  - 激活函數選擇：ReLU for hidden layers
  - 損失函數：MSE（均方誤差）
  
- **關鍵技術**：
  - 數據標準化 (StandardScaler)
  - Early Stopping 避免過擬合
  - 學習曲線分析
  - 預測誤差分析與殘差圖
  
- **工程意義**：
  - 軟感測器實現即時監控
  - 降低分析成本
  - 提升控制響應速度

**適合場景**：軟感測器設計、製程變數推論、實時監控系統

---

### 3️⃣ 實際案例 2：燃料氣 NOx 排放預測 (Fuel Gas Emission)

**檔案**：
- 講義：[Unit15_Example_FuelGasEmission.md](Unit15_Example_FuelGasEmission.md)
- 程式範例：[Unit15_Example_FuelGasEmission.ipynb](Unit15_Example_FuelGasEmission.ipynb)

**內容重點**：
- **問題背景**：
  - 燃氣輪機 (Gas Turbine) 發電過程中會產生 NOx 排放
  - 環保法規嚴格限制 NOx 排放量
  - 目標：根據操作參數預測 NOx 排放，實現主動控制
  
- **數據特性**：
  - UCI Gas Turbine 數據集
  - 特徵包含：環境溫度、濕度、壓力、燃料流量等
  - 回歸任務（預測 NOx 濃度 ppm）
  
- **模型設計**：
  - 多層 DNN 捕捉複雜非線性關係
  - Dropout 防止過擬合
  - Adam 優化器加速收斂
  
- **關鍵技術**：
  - 訓練集/測試集分割策略
  - Batch Normalization 穩定訓練
  - 超參數網格搜索
  - 特徵重要性分析
  
- **工程意義**：
  - 滿足環保法規要求
  - 優化操作參數降低排放
  - 預測性維護與異常檢測

**適合場景**：環保排放監測、燃燒過程優化、法規遵循預測

---

### 4️⃣ 實際案例 3：採礦過程浮選優化 (Mining Process)

**檔案**：
- 講義：[Unit15_Example_Mining.md](Unit15_Example_Mining.md)
- 程式範例：[Unit15_Example_Mining.ipynb](Unit15_Example_Mining.ipynb)

**內容重點**：
- **問題背景**：
  - 浮選 (Flotation) 是礦物分離的關鍵單元操作
  - 目標：預測鐵精礦 (Iron Concentrate) 的二氧化矽雜質含量
  - 優化操作參數提升產品品質
  
- **數據特性**：
  - 採礦浮選過程的真實數據
  - 多個操作變數（流量、pH、藥劑用量等）
  - 回歸任務（預測 SiO2 含量 %）
  
- **模型設計**：
  - DNN vs. Random Forest 性能比較
  - 深層網路捕捉複雜交互作用
  - 正則化技術防止過擬合
  
- **關鍵技術**：
  - 與傳統機器學習方法對比
  - 交叉驗證評估泛化能力
  - 敏感度分析（哪些變數最重要）
  - 操作窗口優化建議
  
- **工程意義**：
  - 提升產品純度
  - 降低雜質含量
  - 優化藥劑使用降低成本

**適合場景**：礦物加工、化學分離過程、產品品質優化

---

### 5️⃣ 實際案例 4：紅酒品質預測 (Red Wine Quality)

**檔案**：
- 講義：[Unit15_Example_RedWine.md](Unit15_Example_RedWine.md)
- 程式範例：[Unit15_Example_RedWine.ipynb](Unit15_Example_RedWine.ipynb)

**內容重點**：
- **問題背景**：
  - 基於化學分析結果預測紅酒品質等級
  - 目標：自動化品質評估，替代昂貴的人工品鑑
  
- **數據特性**：
  - UCI Wine Quality 數據集
  - 特徵：酸度、糖分、酒精度、pH、硫酸鹽等
  - 分類任務（品質評分 3-8 分，可視為多類別分類或回歸）
  
- **模型設計**：
  - DNN 分類器設計
  - Softmax 輸出層（多類別分類）
  - 類別不平衡處理（Class Weights）
  
- **關鍵技術**：
  - 回歸 vs. 分類問題的選擇
  - 混淆矩陣分析
  - 分類報告（Precision、Recall、F1-Score）
  - 錯誤樣本分析
  
- **工程意義**：
  - 食品品質自動化檢測
  - 化學成分與感官品質的關聯分析
  - 生產過程控制指導

**適合場景**：食品品質分類、化學成分分析、感官品質預測

---

### 6️⃣ 實作練習

**檔案**：[Unit15_DNN_MLP_Homework.ipynb](Unit15_DNN_MLP_Homework.ipynb)

**練習內容**：
- 建立完整的 DNN 回歸/分類模型
- 比較不同網路架構的性能（深度、寬度）
- 實驗不同激活函數的效果
- 超參數調整（學習率、batch size、epochs）
- 過擬合診斷與防止技巧應用
- 模型儲存、載入與部署
- 結果解讀與工程決策建議

---

## 📊 數據集說明

### 1. 蒸餾塔數據 (`data/distillation_column/`)
- 蒸餾塔操作數據，包含溫度、流量、壓力等變數
- 目標變數：乙醇濃度 (%)
- 用於軟感測器建模演示

### 2. 燃料氣排放數據 (`data/fuel_gas/`)
- 燃氣輪機操作數據（train.csv、test.csv）
- 特徵：環境參數、操作參數
- 目標變數：NOx 排放濃度 (ppm)

### 3. 採礦浮選數據 (`data/mining/`)
- 採礦浮選過程數據（Mining.csv）
- 特徵：流量、pH、藥劑用量等
- 目標變數：SiO2 含量 (%)

### 4. 紅酒品質數據 (`data/redwine/`)
- 紅酒化學成分分析數據（winequality-red.csv）
- 特徵：酸度、糖分、酒精度等
- 目標變數：品質評分 (3-8 分)

---

## 🎓 DNN 建模決策指南

### 網路架構設計原則

| 問題類型 | 輸出層設計 | 激活函數 | 損失函數 | 評估指標 |
|---------|-----------|---------|---------|---------|
| **回歸** | 1 個神經元 | Linear (無) | MSE, MAE, Huber | RMSE, MAE, R² |
| **二元分類** | 1 個神經元 | Sigmoid | Binary Crossentropy | Accuracy, AUC |
| **多類別分類** | N 個神經元 | Softmax | Categorical Crossentropy | Accuracy, F1 |

### 隱藏層設計建議

| 資料複雜度 | 隱藏層數 | 每層神經元數 | 備註 |
|-----------|---------|------------|------|
| **簡單線性** | 1-2 層 | 32-64 | 接近傳統機器學習 |
| **中等非線性** | 2-3 層 | 64-128 | 大多數化工問題 |
| **複雜非線性** | 3-5 層 | 128-256 | 高維、複雜交互 |
| **超高維** | 5+ 層 | 256-512 | 需大量數據避免過擬合 |

### 優化器選擇建議

| 優化器 | 特點 | 學習率建議 | 適用場景 |
|--------|------|----------|----------|
| **SGD** | 簡單穩定 | 0.01-0.1 | 需手動調整 |
| **Adam** | 自適應、快速 | 0.001 (預設) | **通用首選** |
| **RMSprop** | 適合 RNN | 0.001 | 時間序列 |
| **AdaGrad** | 適合稀疏數據 | 0.01 | 文本、稀疏特徵 |

### 防止過擬合策略

1. **Dropout** (推薦 0.2-0.5)：隨機丟棄神經元，強制學習穩健特徵
2. **L1/L2 Regularization**：懲罰大權重，促進簡潔模型
3. **Early Stopping**：監控驗證集，最佳時停止訓練
4. **Batch Normalization**：標準化層間輸出，穩定訓練
5. **數據增強**：增加訓練樣本多樣性（主要用於影像）

---

## 💻 實作環境需求

### 必要套件
```python
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scikit-learn >= 1.0.0
tensorflow >= 2.10.0          # 深度學習框架
keras >= 2.10.0               # 高階 API（TensorFlow 內建）
```

### 選用套件
```python
tensorboard >= 2.10.0         # 訓練視覺化工具
keras-tuner >= 1.1.0          # 超參數自動調整
scikeras >= 0.9.0             # Keras 與 sklearn 整合
joblib >= 1.0.0               # 模型儲存
```

---

## 📈 學習路徑建議

### 第一階段：理論基礎建立（必讀）
1. 閱讀 [Unit15_DNN_MLP_Overview.md](Unit15_DNN_MLP_Overview.md)
2. 執行 [Unit15_DNN_MLP_Overview.ipynb](Unit15_DNN_MLP_Overview.ipynb)
3. 重點掌握：
   - 神經元運算原理
   - 前向傳播與反向傳播
   - 損失函數與激活函數選擇
   - Keras 建模 API

### 第二階段：實際案例學習
1. **蒸餾塔案例**（建議最先學習，最典型的軟感測器應用）
   - 閱讀 [Unit15_Example_Distillation.md](Unit15_Example_Distillation.md)
   - 執行 [Unit15_Example_Distillation.ipynb](Unit15_Example_Distillation.ipynb)
   - 重點：數據預處理、模型設計、Early Stopping
   
2. **燃料氣排放案例**（環保監測應用）
   - 閱讀 [Unit15_Example_FuelGasEmission.md](Unit15_Example_FuelGasEmission.md)
   - 執行 [Unit15_Example_FuelGasEmission.ipynb](Unit15_Example_FuelGasEmission.ipynb)
   - 重點：Dropout、Batch Normalization、超參數調整
   
3. **採礦浮選案例**（與傳統 ML 對比）
   - 閱讀 [Unit15_Example_Mining.md](Unit15_Example_Mining.md)
   - 執行 [Unit15_Example_Mining.ipynb](Unit15_Example_Mining.ipynb)
   - 重點：DNN vs. Random Forest、交叉驗證、特徵重要性
   
4. **紅酒品質案例**（分類問題）
   - 閱讀 [Unit15_Example_RedWine.md](Unit15_Example_RedWine.md)
   - 執行 [Unit15_Example_RedWine.ipynb](Unit15_Example_RedWine.ipynb)
   - 重點：多類別分類、混淆矩陣、類別不平衡處理

### 第三階段：綜合練習
1. 完成 [Unit15_DNN_MLP_Homework.ipynb](Unit15_DNN_MLP_Homework.ipynb)
2. 比較不同架構（深度、寬度）的性能
3. 實驗各種正則化技術的效果
4. 嘗試將 DNN 應用於自己的化工數據

### 第四階段：進階優化
1. 學習 TensorBoard 視覺化訓練過程
2. 使用 Keras Tuner 自動調整超參數
3. 模型集成：結合多個 DNN 模型
4. 模型部署：TensorFlow Serving、ONNX

---

## 🔍 化工領域核心應用

### 1. 軟感測器設計 (Soft Sensor) ⭐
- **目標**：基於易測變數（溫度、壓力、流量）推論難測變數（濃度、品質）
- **DNN 優勢**：
  - 捕捉複雜非線性關係
  - 自動學習特徵交互
  - 即時預測無延遲
- **關鍵技術**：
  - 數據標準化（輸入特徵尺度統一）
  - 時間窗口設計（考慮滯後效應）
  - 模型定期更新（適應製程漂移）
- **典型應用**：
  - 蒸餾塔組成推論
  - 反應器轉化率預測
  - 產品品質線上監測

### 2. 製程優化與最佳化
- **目標**：尋找最佳操作條件（最大產率、最小能耗、最佳品質）
- **DNN 角色**：建立精準的製程模型，作為優化目標函數
- **關鍵技術**：
  - 代理模型 (Surrogate Model) 設計
  - 結合優化算法（GA、PSO、Bayesian Optimization）
  - 多目標優化（Pareto 前沿）
- **典型應用**：
  - 反應條件優化（溫度、壓力、停留時間）
  - 配方優化（原料配比、催化劑用量）
  - 能源管理（最小化能耗）

### 3. 品質預測與控制
- **目標**：預測產品品質，實現預測性控制
- **DNN 優勢**：
  - 處理多變數影響
  - 非線性關係建模
  - 快速響應（毫秒級）
- **關鍵技術**：
  - 分類 vs. 回歸問題選擇
  - 閾值設定（品質合格標準）
  - 誤判成本考量
- **典型應用**：
  - 聚合物品質分級
  - 藥品品質預測
  - 食品品質評估

### 4. 故障診斷與預測性維護
- **目標**：早期檢測設備異常，預測故障發生時間
- **DNN 角色**：從正常操作數據學習模式，識別異常
- **關鍵技術**：
  - 異常檢測（正常 vs. 異常）
  - 故障分類（故障類型識別）
  - 剩餘壽命預測 (RUL)
- **典型應用**：
  - 泵浦健康監測
  - 熱交換器效率衰退預警
  - 催化劑失活預測

### 5. 光譜數據分析
- **目標**：從光譜數據推論化學成分或物性
- **DNN 優勢**：
  - 處理高維光譜數據（數百到數千個波長點）
  - 自動特徵提取
  - 優於傳統偏最小二乘法 (PLS)
- **關鍵技術**：
  - 光譜預處理（平滑、導數、標準化）
  - 降維技術結合
  - 校正集與驗證集設計
- **典型應用**：
  - 近紅外光譜 (NIR) 成分分析
  - 拉曼光譜物相鑑定
  - 色譜數據定量分析

---

## 📝 評估指標總結

### 回歸任務
- **MSE (Mean Squared Error)**： $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ ，懲罰大誤差
- **RMSE (Root Mean Squared Error)**： $\sqrt{\text{MSE}}$ ，與目標變數同單位
- **MAE (Mean Absolute Error)**： $\frac{1}{n}\sum|y_i - \hat{y}_i|$ ，對異常值較不敏感
- **R² (Coefficient of Determination)**： $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ ，解釋變異比例
- **MAPE (Mean Absolute Percentage Error)**： $\frac{100\%}{n}\sum\frac{|y_i - \hat{y}_i|}{|y_i|}$ ，相對誤差

### 分類任務
- **Accuracy**：整體預測正確率
- **Precision**：預測為正類中真正為正類的比例
- **Recall**：真實正類中被正確預測的比例
- **F1-Score**：Precision 與 Recall 的調和平均
- **ROC-AUC**：分類能力的綜合評估

### 訓練監控指標
- **Training Loss vs. Validation Loss**：過擬合診斷
- **Epoch**：訓練輪數
- **Learning Rate**：學習率變化
- **Gradient Norm**：梯度大小（梯度爆炸/消失檢測）

---

## 🚀 進階學習方向

完成本單元後，可以進一步探索：

1. **卷積神經網路 (CNN)**：Unit16，處理影像數據（缺陷檢測、顯微影像分析）
2. **循環神經網路 (RNN/LSTM)**：Unit17，處理時間序列數據（製程動態建模、預測控制）
3. **自動編碼器 (Autoencoder)**：降維、異常檢測、數據去噪
4. **生成對抗網路 (GAN)**：數據增強、製程模擬
5. **遷移學習 (Transfer Learning)**：利用預訓練模型加速訓練
6. **神經架構搜索 (NAS)**：自動化網路架構設計
7. **可解釋 AI**：SHAP、LIME 應用於 DNN 模型解釋

---

## 📚 參考資源

### 教科書
1. *Deep Learning* by Ian Goodfellow, Yoshua Bengio, Aaron Courville（深度學習聖經）
2. *Neural Networks and Deep Learning* by Michael Nielsen（線上免費）
3. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* by Aurélien Géron（實作導向）

### 線上資源
- [TensorFlow 官方教學](https://www.tensorflow.org/tutorials)
- [Keras 官方文件](https://keras.io/)
- [Deep Learning Specialization by Andrew Ng (Coursera)](https://www.coursera.org/specializations/deep-learning)

### 化工領域應用論文
- "Deep learning for smart manufacturing: Methods and applications" (Journal of Manufacturing Systems, 2018)
- "Soft sensor development using deep learning" (Chemical Engineering Science, 2019)
- "Process monitoring using deep neural networks" (AIChE Journal, 2020)

---

## ✍️ 課後習題提示

1. **架構實驗**：比較 2 層、3 層、5 層網路在相同數據上的性能
2. **激活函數對比**：嘗試 ReLU、Leaky ReLU、ELU、Tanh 的效果差異
3. **正則化效果**：測試 Dropout (0.2, 0.5)、L2 regularization 對過擬合的影響
4. **優化器比較**：比較 SGD、Adam、RMSprop 的收斂速度與最終性能
5. **學習率調度**：實驗 ReduceLROnPlateau、ExponentialDecay、CosineDecay
6. **Batch Size 影響**：測試 16、32、64、128 對訓練穩定性與速度的影響
7. **化工應用**：將 DNN 應用於自己的化工數據，撰寫完整建模報告

---

## 📞 技術支援

如有疑問或需要協助，歡迎透過以下方式聯繫：
- 課程討論區
- 指導教授 Office Hour
- 課程助教

---

**🎉 恭喜您踏入深度學習的世界！掌握 DNN 是理解 CNN、RNN 等進階架構的關鍵基礎！**

---

**課程資訊**
- 課程名稱：AI在化工上之應用
- 課程單元：Unit15 深度神經網路與多層感知機 (Deep Neural Networks & Multi-Layer Perceptron)
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-01-28

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---