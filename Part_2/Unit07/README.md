# Unit07 異常檢測 (Anomaly Detection)

## 📚 單元簡介

本單元系統性地介紹異常檢測 (Anomaly Detection) 在化學工程領域的理論與應用。異常檢測是識別與大多數數據顯著不同樣本的技術，在製程安全監控、產品品質控制、設備健康管理等領域扮演關鍵角色。本單元涵蓋四種主流的異常檢測演算法，從基於決策樹的方法到基於密度和支持向量機的方法，全面掌握異常檢測技術。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **理解異常檢測的核心概念**：掌握異常、離群值、新奇點的差異
2. **掌握多種異常檢測演算法**：理解不同演算法的原理、優缺點與適用場景
3. **選擇合適的異常檢測方法**：根據數據特性與應用需求選擇最佳演算法
4. **實作異常檢測模型**：使用 scikit-learn 實作各種異常檢測演算法
5. **應用於化工領域**：解決製程安全監控、產品品質控制、設備故障診斷等實際問題

---

## 📖 單元內容架構

### 1️⃣ 總覽篇：異常檢測基礎

**檔案**：[Unit07_Anomaly_Detection_Overview.md](Unit07_Anomaly_Detection_Overview.md)

**內容重點**：
- 異常檢測的定義與重要性
- 核心術語：異常 vs 離群值 vs 新奇點
- 異常檢測的三種情境：
  - 離群值檢測 (Outlier Detection)
  - 新奇點檢測 (Novelty Detection)
  - 半監督異常檢測
- 化工領域應用場景：
  - 製程安全監控（反應器異常、管線洩漏、設備健康）
  - 產品品質監控（線上檢測、批次分析）
  - 感測器故障診斷
  - 操作異常識別
- 異常檢測演算法分類：統計方法、基於距離、基於密度、基於模型

**適合讀者**：所有學員，建議先閱讀此篇以建立整體概念

---

### 2️⃣ 孤立森林 (Isolation Forest) ⭐ 最廣泛應用

**檔案**：
- 講義：[Unit07_Isolation_Forest.md](Unit07_Isolation_Forest.md)
- 程式範例：[Unit07_Isolation_Forest.ipynb](Unit07_Isolation_Forest.ipynb)

**內容重點**：
- **演算法原理**：
  - 核心理念：異常點更容易被「孤立」
  - 孤立樹 (iTree) 的建構：隨機選擇特徵與切分點
  - 異常分數計算：路徑長度與正規化
  - 孤立森林：多棵 iTree 的集成
  
- **實作技術**：
  - scikit-learn `IsolationForest` 類別使用
  - 關鍵超參數：contamination、n_estimators、max_samples
  - 異常分數解讀與閾值設定
  - 決策函數與預測
  
- **化工應用案例**：
  - 反應器異常監控（溫度驟升、壓力波動）
  - 產品品質異常檢測（純度異常、雜質超標）
  - 設備健康監控（振動異常、電流異常）
  - 製程安全監控（多變數綜合判斷）
  - 感測器故障診斷（漂移、卡死、雜訊）
  
- **演算法特性**：
  - ✅ 優點：速度快、適合高維數據、無需標籤、可擴展性好
  - ❌ 缺點：對參數敏感、難以解釋、不適合局部異常

**適合場景**：大規模數據、高維數據、即時監控、通用異常檢測

---

### 3️⃣ 區域性離群因子 (Local Outlier Factor, LOF)

**檔案**：
- 講義：[Unit07_LOF.md](Unit07_LOF.md)
- 程式範例：[Unit07_LOF.ipynb](Unit07_LOF.ipynb)

**內容重點**：
- **演算法原理**：
  - 核心理念：異常點的局部密度明顯低於鄰近點
  - k-距離與 k-鄰域
  - 可達距離 (Reachability Distance)
  - 局部可達密度 (Local Reachability Density)
  - LOF 分數計算與解讀
  
- **實作技術**：
  - scikit-learn `LocalOutlierFactor` 類別使用
  - 關鍵超參數：n_neighbors、contamination、metric
  - 離群值檢測 vs 新奇點檢測模式
  - LOF 分數視覺化與分析
  
- **化工應用案例**：
  - 多模式操作製程監控（不同負載、不同階段）
  - 非均勻密度的感測器數據
  - 局部異常檢測（單一設備、特定時段）
  - 設備群組健康監控（不同設備型號）
  - 品質控制中的局部缺陷檢測
  
- **演算法特性**：
  - ✅ 優點：檢測局部異常、處理不均勻密度、直觀易懂
  - ❌ 缺點：計算成本高 (O(n²))、對 k 值敏感、不適合高維數據

**適合場景**：多模式製程、不均勻密度分布、局部異常檢測

---

### 4️⃣ 一類支持向量機 (One-Class SVM)

**檔案**：
- 講義：[Unit07_OneClass_SVM.md](Unit07_OneClass_SVM.md)
- 程式範例：[Unit07_OneClass_SVM.ipynb](Unit07_OneClass_SVM.ipynb)

**內容重點**：
- **演算法原理**：
  - 核心理念：在特徵空間中找最小超球面包含正常數據
  - 支持向量與決策邊界
  - 核函數：RBF、多項式、Sigmoid
  - 優化目標：最大化邊界半徑與最小化違反點
  
- **實作技術**：
  - scikit-learn `OneClassSVM` 類別使用
  - 關鍵超參數：nu、kernel、gamma
  - 決策邊界視覺化
  - 支持向量分析
  
- **化工應用案例**：
  - 高價值產品品質監控（精細化工、小批量）
  - 新產品試產監控（訓練數據有限）
  - 關鍵設備健康監控（壓縮機、離心機）
  - 特殊製程安全監控（高風險製程）
  - 小樣本感測器校正
  
- **演算法特性**：
  - ✅ 優點：精確決策邊界、理論扎實、適合小樣本、處理非線性
  - ❌ 缺點：計算成本高、超參數調整複雜、對核函數選擇敏感

**適合場景**：小樣本數據、高精度需求、非線性決策邊界

---

### 5️⃣ 橢圓包絡 (Elliptic Envelope)

**檔案**：
- 講義：[Unit07_Elliptic_Envelope.md](Unit07_Elliptic_Envelope.md)
- 程式範例：[Unit07_Elliptic_Envelope.ipynb](Unit07_Elliptic_Envelope.ipynb)

**內容重點**：
- **演算法原理**：
  - 核心理念：假設正常數據服從多變量高斯分布
  - 馬氏距離 (Mahalanobis Distance) 計算
  - 協方差矩陣估計
  - 穩健協方差估計 (Robust Covariance Estimation)
  - 橢圓決策邊界
  
- **實作技術**：
  - scikit-learn `EllipticEnvelope` 類別使用
  - 關鍵超參數：contamination、support_fraction
  - 馬氏距離視覺化
  - 協方差矩陣分析
  
- **化工應用案例**：
  - 產品品質控制（多指標聯合監控）
  - 穩態製程監控（連續製程正常區域）
  - 感測器故障診斷（多感測器聯合驗證）
  - 批次一致性檢驗（批次特性評估）
  - 實驗設計與數據驗證
  
- **演算法特性**：
  - ✅ 優點：理論清晰、可解釋性強、適合常態分布、穩健性好
  - ❌ 缺點：假設數據服從高斯分布、不適合非線性、對維度敏感

**適合場景**：數據近似常態分布、需要可解釋性、多變數聯合監控

---

### 6️⃣ 實作練習

**檔案**：[Unit07_Anomaly_Detection_Homework.ipynb](Unit07_Anomaly_Detection_Homework.ipynb)

**練習內容**：
- 應用多種異常檢測演算法於實際化工數據
- 比較不同演算法的檢測效果
- 參數調整與模型優化
- 結果解讀與工程意義分析

---

## 📊 數據集說明

### 1. 反應器操作數據 (`data/reactor_operation/`)
- 包含正常與異常操作條件的時間序列數據
- 用於製程監控與異常檢測

### 2. 批次反應器數據 (`data/batch_reactor/`)
- 批次製程的操作軌跡數據
- 用於批次品質監控與異常批次識別

---

## 🎓 演算法比較與選擇指南

| 演算法 | 檢測類型 | 計算複雜度 | 高維適用性 | 局部異常 | 可解釋性 | 適用場景 |
|--------|---------|-----------|-----------|---------|---------|----------|
| **Isolation Forest** | 全局 | 低 (O(n log n)) | ✅ 優秀 | ❌ 弱 | ⚠️ 中等 | 大規模、高維、通用 |
| **LOF** | 局部 | 高 (O(n²)) | ❌ 差 | ✅ 優秀 | ✅ 良好 | 多模式、不均勻密度 |
| **One-Class SVM** | 全局 | 高 (O(n²~n³)) | ⚠️ 中等 | ⚠️ 中等 | ⚠️ 中等 | 小樣本、高精度 |
| **Elliptic Envelope** | 全局 | 中等 (O(n²)) | ⚠️ 中等 | ❌ 弱 | ✅ 優秀 | 常態分布、可解釋性 |

**選擇建議**：
1. **大規模數據、通用場景** → Isolation Forest
2. **多模式製程、局部異常** → LOF
3. **小樣本、高精度需求** → One-Class SVM
4. **常態分布、需要可解釋性** → Elliptic Envelope
5. **組合使用**：多種方法投票決策，提高可靠性

---

## 💻 實作環境需求

### 必要套件
```python
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scikit-learn >= 1.0.0
scipy >= 1.7.0
```

### 選用套件
```python
plotly >= 5.0.0  # 互動式視覺化
pyod >= 1.0.0  # Python Outlier Detection 工具箱
```

---

## 📈 學習路徑建議

### 第一階段：基礎概念建立
1. 閱讀 [Unit07_Anomaly_Detection_Overview.md](Unit07_Anomaly_Detection_Overview.md)
2. 理解異常檢測的核心概念與應用場景
3. 區分離群值檢測與新奇點檢測

### 第二階段：演算法學習與實作
1. **Isolation Forest**（建議最先學習，最實用）
   - 閱讀講義 [Unit07_Isolation_Forest.md](Unit07_Isolation_Forest.md)
   - 執行 [Unit07_Isolation_Forest.ipynb](Unit07_Isolation_Forest.ipynb)
   
2. **LOF**
   - 閱讀講義 [Unit07_LOF.md](Unit07_LOF.md)
   - 執行 [Unit07_LOF.ipynb](Unit07_LOF.ipynb)
   
3. **One-Class SVM**
   - 閱讀講義 [Unit07_OneClass_SVM.md](Unit07_OneClass_SVM.md)
   - 執行 [Unit07_OneClass_SVM.ipynb](Unit07_OneClass_SVM.ipynb)
   
4. **Elliptic Envelope**
   - 閱讀講義 [Unit07_Elliptic_Envelope.md](Unit07_Elliptic_Envelope.md)
   - 執行 [Unit07_Elliptic_Envelope.ipynb](Unit07_Elliptic_Envelope.ipynb)

### 第三階段：綜合應用與練習
1. 完成 [Unit07_Anomaly_Detection_Homework.ipynb](Unit07_Anomaly_Detection_Homework.ipynb)
2. 比較不同演算法在相同數據集上的表現
3. 嘗試將異常檢測技術應用於自己的化工數據

---

## 🔍 化工領域核心應用

### 1. 製程安全監控 ⭐ 最重要應用
- **目標**：即時監控製程，及早發現安全風險
- **演算法建議**：Isolation Forest（快速、適合多變數）
- **關鍵技術**：
  - 多變數綜合監控
  - 異常分數閾值設定
  - 分級告警系統
  - 根因分析（貢獻度分析）

### 2. 產品品質監控
- **目標**：確保產品符合規格，識別異常批次
- **演算法建議**：Elliptic Envelope、Isolation Forest
- **關鍵技術**：線上監控、批次一致性檢驗、趨勢分析

### 3. 設備健康管理
- **目標**：預測設備故障，實現預防性維護
- **演算法建議**：LOF（處理不同運行模式）
- **關鍵技術**：振動分析、溫度監控、劣化趨勢識別

### 4. 感測器故障診斷
- **目標**：識別感測器漂移、故障、雜訊
- **演算法建議**：Elliptic Envelope（多感測器聯合驗證）
- **關鍵技術**：一致性檢查、冗餘感測器比對

### 5. 操作異常識別
- **目標**：識別偏離正常操作的條件
- **演算法建議**：LOF（多模式操作）、Isolation Forest（單模式）
- **關鍵技術**：操作窗口定義、異常模式分類

---

## 📝 評估指標總結

### 有標籤數據（監督式評估）
- **準確率 (Accuracy)**：正確分類的比例
- **精確率 (Precision)**：預測為異常中真正異常的比例
- **召回率 (Recall)**：真實異常被正確識別的比例
- **F1 分數**：精確率與召回率的調和平均
- **ROC-AUC**：接收者操作特徵曲線下面積
- **混淆矩陣**：True/False Positive/Negative 分析

### 無標籤數據（非監督式評估）
- **異常分數分布**：正常與異常的分數分離度
- **域專家驗證**：工程師判斷檢測結果的合理性
- **穩定性測試**：模型在不同時段的一致性
- **誤報率 (FAR)**：False Alarm Rate
- **漏報率 (MDR)**：Missed Detection Rate

### 工業應用評估
- **檢測延遲**：從異常發生到檢測出的時間
- **告警頻率**：單位時間內的告警次數
- **根因追溯能力**：能否指出異常的主要變數
- **經濟效益**：避免的損失 vs 系統成本

---

## 🚀 進階學習方向

完成本單元後，可以進一步探索：

1. **深度學習異常檢測**：Autoencoder、VAE、GAN
2. **時間序列異常檢測**：LSTM-Autoencoder、Matrix Profile
3. **半監督異常檢測**：結合少量標籤數據
4. **在線異常檢測**：增量學習、概念漂移處理
5. **多模態異常檢測**：結合感測器、圖像、文本數據
6. **因果異常檢測**：從相關性到因果關係

---

## 📚 參考資源

### 教科書
1. *Outlier Analysis* by Charu C. Aggarwal
2. *Anomaly Detection: A Survey* by Varun Chandola et al.
3. *Introduction to Data Mining* by Tan et al.（第 10 章）

### 線上資源
- [scikit-learn Anomaly Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [PyOD Documentation](https://pyod.readthedocs.io/)
- [Awesome Anomaly Detection](https://github.com/hoya012/awesome-anomaly-detection)

### 化工領域應用論文
- Fault Detection and Diagnosis in Chemical Processes
- Process Monitoring Using Statistical Methods
- Machine Learning for Predictive Maintenance

---

## ✍️ 課後習題提示

1. **比較分析**：使用同一數據集比較四種演算法的檢測效果
2. **參數優化**：探討 contamination 參數對結果的影響
3. **實際應用**：將異常檢測技術應用於您的化工數據集
4. **組合策略**：設計多演算法投票決策的異常檢測系統
5. **根因分析**：當檢測到異常時，如何找出主要原因變數

---

## 📞 技術支援

如有疑問或需要協助，歡迎透過以下方式聯繫：
- 課程討論區
- 指導教授 Office Hour
- 課程助教

---

**版本資訊**：Unit07 v1.0 | 最後更新：2026-01-27
