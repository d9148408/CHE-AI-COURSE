# Unit08 關聯規則學習 (Association Rule Learning)

## 📚 單元簡介

本單元系統性地介紹關聯規則學習 (Association Rule Learning) 在化學工程領域的理論與應用。關聯規則學習是一種非監督式學習方法，旨在發現數據集中項目之間的有趣關聯或相關模式。最早應用於市場購物籃分析，在化工領域可用於發現製程變數之間的關聯性、配方成分之間的協同效應、操作條件與產品品質之間的關係等，為製程優化和配方設計提供數據驅動的洞察。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **理解關聯規則學習的核心概念**：掌握項目集、支持度、置信度、提升度等基本概念
2. **掌握主流關聯規則演算法**：理解 Apriori 和 FP-Growth 的原理與差異
3. **評估關聯規則的品質**：使用多種指標評估規則的強度和可信度
4. **實作關聯規則挖掘**：使用 mlxtend 和 pyfpgrowth 套件實作演算法
5. **應用於化工領域**：解決配方優化、製程知識挖掘、品質預測等實際問題

---

## 📖 單元內容架構

### 1️⃣ 總覽篇：關聯規則學習基礎

**檔案**：[Unit08_Association_Rule_Learning_Overview.md](Unit08_Association_Rule_Learning_Overview.md)

**內容重點**：
- 關聯規則學習的定義與核心概念
- 基本術語：項目、項目集、交易
- 關聯規則的形式： $X \Rightarrow Y$ 
- 三大評估指標：
  - **支持度 (Support)**：規則在數據中的普遍性
  - **置信度 (Confidence)**：規則的可靠性
  - **提升度 (Lift)**：規則的關聯強度
- 化工領域應用場景：
  - 配方優化與成分協同效應
  - 製程知識挖掘
  - 品質預測與根因分析
  - 故障診斷與預防
  - 操作條件關聯分析
- 關聯規則學習的目標與挑戰

**適合讀者**：所有學員，建議先閱讀此篇以建立整體概念

---

### 2️⃣ Apriori 演算法

**檔案**：
- 講義：[Unit08_Apriori_Algorithm.md](Unit08_Apriori_Algorithm.md)
- 程式範例：[Unit08_Apriori_Algorithm.ipynb](Unit08_Apriori_Algorithm.ipynb)

**內容重點**：
- **演算法原理**：
  - Apriori 性質（向下封閉性）：頻繁項目集的子集也是頻繁的
  - 候選項目集生成與剪枝策略
  - 迭代過程：從 1-項目集到 k-項目集
  - 關聯規則生成：從頻繁項目集提取規則
  
- **實作技術**：
  - mlxtend 套件使用：`apriori` 和 `association_rules` 函數
  - 支持度與置信度閾值設定
  - 規則過濾與排序
  - 規則視覺化：網絡圖、熱圖
  
- **化工應用案例**：
  - **聚合物配方優化**：發現單體、引發劑、鏈轉移劑的最佳組合
  - **催化劑配方設計**：識別主催化劑、助催化劑、載體的協同效應
  - **製程條件關聯分析**：發現導致高產率的操作條件組合
  - **品質問題根因分析**：識別導致品質不良的因素組合
  
- **演算法特性**：
  - ✅ 優點：簡單易懂、易於實作、可解釋性強
  - ❌ 缺點：多次掃描數據、候選集爆炸、不適合大規模數據

**適合場景**：中小規模數據、需要可解釋性、配方優化

---

### 3️⃣ FP-Growth 演算法

**檔案**：
- 講義：[Unit08_FP_Growth_Algorithm.md](Unit08_FP_Growth_Algorithm.md)
- 程式範例：[Unit08_FP_Growth_Algorithm.ipynb](Unit08_FP_Growth_Algorithm.ipynb)

**內容重點**：
- **演算法原理**：
  - 核心資料結構：FP-Tree (Frequent Pattern Tree)
  - 分而治之策略：遞歸挖掘頻繁模式
  - 條件模式基 (Conditional Pattern Base)
  - 條件 FP-Tree 構建
  - 頻繁項目集挖掘過程
  
- **實作技術**：
  - pyfpgrowth 或 mlxtend 套件使用
  - FP-Tree 視覺化與分析
  - 與 Apriori 的性能比較
  - 記憶體管理與優化
  
- **化工應用案例**：
  - **大規模歷史配方分析**：從數萬筆配方記錄中快速挖掘成功模式
  - **高維製程數據挖掘**：處理包含數百個操作參數的製程數據
  - **實時配方推薦系統**：快速響應配方查詢需求
  - **跨工廠數據整合分析**：整合多個工廠的歷史數據
  
- **演算法特性**：
  - ✅ 優點：速度快、適合大規模數據、只需掃描數據兩次、無候選集
  - ❌ 缺點：實作複雜、記憶體需求較高、FP-Tree 構建有學習曲線

**適合場景**：大規模數據、高維數據、實時應用、性能要求高

---

### 4️⃣ 實作練習

**檔案**：[Unit08_Association_Rule_Learning_Homework.ipynb](Unit08_Association_Rule_Learning_Homework.ipynb)

**練習內容**：
- 應用關聯規則學習於化工配方數據
- 比較 Apriori 和 FP-Growth 的性能
- 規則品質評估與過濾
- 結果解讀與工程意義分析

---

## 📊 數據集說明

### 化工配方數據 (`data/`)
- 包含配方成分、操作條件、產品品質等信息
- 以交易格式組織（每筆配方為一個交易）
- 用於挖掘配方規則、成分協同效應

---

## 🎓 演算法比較與選擇指南

| 特性 | Apriori | FP-Growth |
|------|---------|-----------|
| **數據掃描次數** | k+1 次（k 為最大項目集大小） | 2 次 |
| **候選集生成** | 需要 | 不需要 |
| **資料結構** | 簡單（列表、集合） | 複雜（FP-Tree） |
| **記憶體需求** | 中等 | 較高（需存儲 FP-Tree） |
| **計算速度** | 慢（大數據） | 快 |
| **實作難度** | 簡單 | 中等 |
| **可擴展性** | 差 | 優秀 |
| **適用數據規模** | 小到中等 | 中到大規模 |

**選擇建議**：
1. **小規模數據（< 10,000 筆）** → Apriori（簡單易用）
2. **大規模數據（> 10,000 筆）** → FP-Growth（性能優勢）
3. **需要快速原型開發** → Apriori（實作簡單）
4. **生產環境、實時應用** → FP-Growth（效率高）
5. **學習目的** → 兩者都學（Apriori 易懂，FP-Growth 實用）

**性能對比參考**：
- 1,000 筆配方：Apriori 2.3 秒，FP-Growth 0.8 秒（2.9× 加速）
- 10,000 筆配方：Apriori 45.6 秒，FP-Growth 5.2 秒（8.8× 加速）
- 100,000 筆配方：Apriori > 30 分鐘，FP-Growth 78 秒（> 23× 加速）

---

## 💻 實作環境需求

### 必要套件
```python
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
mlxtend >= 0.19.0  # Apriori 和 FP-Growth
```

### 選用套件
```python
pyfpgrowth >= 1.0  # 另一種 FP-Growth 實作
networkx >= 2.6.0  # 規則網絡視覺化
plotly >= 5.0.0  # 互動式視覺化
```

---

## 📈 學習路徑建議

### 第一階段：基礎概念建立
1. 閱讀 [Unit08_Association_Rule_Learning_Overview.md](Unit08_Association_Rule_Learning_Overview.md)
2. 理解支持度、置信度、提升度等核心概念
3. 了解關聯規則在化工領域的應用價值

### 第二階段：演算法學習與實作
1. **Apriori 演算法**（建議先學習）
   - 閱讀講義 [Unit08_Apriori_Algorithm.md](Unit08_Apriori_Algorithm.md)
   - 執行 [Unit08_Apriori_Algorithm.ipynb](Unit08_Apriori_Algorithm.ipynb)
   - 理解 Apriori 性質與剪枝策略
   
2. **FP-Growth 演算法**
   - 閱讀講義 [Unit08_FP_Growth_Algorithm.md](Unit08_FP_Growth_Algorithm.md)
   - 執行 [Unit08_FP_Growth_Algorithm.ipynb](Unit08_FP_Growth_Algorithm.ipynb)
   - 理解 FP-Tree 結構與分治策略

### 第三階段：綜合應用與練習
1. 完成 [Unit08_Association_Rule_Learning_Homework.ipynb](Unit08_Association_Rule_Learning_Homework.ipynb)
2. 比較 Apriori 和 FP-Growth 的性能差異
3. 嘗試將關聯規則學習應用於自己的化工配方數據

---

## 🔍 化工領域核心應用

### 1. 配方優化與成分協同效應
- **目標**：發現配方成分之間的最佳組合
- **方法**：挖掘高支持度、高置信度的配方規則
- **應用**：
  - 聚合物配方設計（單體、引發劑、助劑組合）
  - 催化劑配方開發（主催化劑、助催化劑、載體協同）
  - 塗料配方優化（樹脂、溶劑、添加劑搭配）
- **關鍵指標**：提升度 > 1.5 表示強協同效應

### 2. 製程知識挖掘
- **目標**：從歷史數據中提取隱藏的製程規則
- **方法**：分析操作條件與產品品質的關聯
- **應用**：
  - 反應條件優化（溫度、壓力、時間組合）
  - 分離條件設定（塔板數、回流比、進料位置）
  - 多變數操作窗口定義
- **關鍵技術**：連續變數離散化、領域知識結合

### 3. 品質預測與根因分析
- **目標**：識別影響產品品質的關鍵因素組合
- **方法**：挖掘原料特性、製程參數與品質的關聯規則
- **應用**：
  - 高品質產品的操作特徵識別
  - 品質不良的多因素根因分析
  - 關鍵品質指標的影響因子挖掘
- **關鍵技術**：品質標籤化、多層次規則挖掘

### 4. 故障診斷與預防
- **目標**：發現導致設備故障或製程異常的條件模式
- **方法**：分析故障前的製程狀態組合
- **應用**：
  - 設備故障前兆識別
  - 製程異常的多變數模式
  - 預防性維護決策支援
- **關鍵技術**：時間窗口設定、序列規則挖掘

### 5. 溶劑與原料篩選
- **目標**：從候選物質中篩選出最佳組合
- **方法**：分析物質性質與性能指標的關聯
- **應用**：
  - 綠色溶劑篩選（環境友善性、溶解能力、成本）
  - 原料供應商評估（品質、價格、交期）
  - 添加劑選擇（功能性、相容性、穩定性）
- **關鍵技術**：多目標規則挖掘、帕累托最優化

---

## 📝 評估指標詳解

### 基本指標
1. **支持度 (Support)**
   - 計算： $\text{Support}(X \Rightarrow Y) = \frac{|X \cup Y|}{N}$ 
   - 意義：規則的普遍性，出現頻率
   - 閾值建議：化工配方 0.05-0.20，製程數據 0.10-0.30

2. **置信度 (Confidence)**
   - 計算： $\text{Confidence}(X \Rightarrow Y) = \frac{|X \cup Y|}{|X|}$ 
   - 意義：規則的可靠性，條件機率
   - 閾值建議：通常 > 0.70，高精度應用 > 0.85

3. **提升度 (Lift)**
   - 計算： $\text{Lift}(X \Rightarrow Y) = \frac{\text{Confidence}(X \Rightarrow Y)}{\text{Support}(Y)}$ 
   - 意義：X 和 Y 的關聯強度
   - 判斷：Lift > 1 正相關，Lift = 1 獨立，Lift < 1 負相關
   - 閾值建議：Lift > 1.2 表示有意義的關聯

### 進階指標
4. **確信度 (Conviction)**
   - 計算： $\text{Conviction}(X \Rightarrow Y) = \frac{1 - \text{Support}(Y)}{1 - \text{Confidence}(X \Rightarrow Y)}$ 
   - 意義：規則的方向性強度
   - 特點：考慮反向關係

5. **槓桿率 (Leverage)**
   - 計算： $\text{Leverage}(X \Rightarrow Y) = \text{Support}(X \cup Y) - \text{Support}(X) \times \text{Support}(Y)$ 
   - 意義：相對於獨立性的支持度增量

6. **Zhang's Metric**
   - 計算： $\text{Zhang} = \frac{\text{Confidence}(X \Rightarrow Y) - \text{Confidence}(\bar{X} \Rightarrow Y)}{\max\{\text{Confidence}(X \Rightarrow Y), \text{Confidence}(\bar{X} \Rightarrow Y)\}}$ 
   - 意義：對稱性好，考慮正反兩向關聯

---

## 🚀 進階學習方向

完成本單元後，可以進一步探索：

1. **序列規則挖掘 (Sequential Pattern Mining)**：挖掘時間序列中的模式
2. **多層次關聯規則 (Multi-Level Association Rules)**：考慮概念層次結構
3. **量化關聯規則 (Quantitative Association Rules)**：處理連續變數
4. **罕見規則挖掘 (Rare Rule Mining)**：發現低支持度但重要的規則
5. **約束關聯規則 (Constrained Association Rules)**：結合領域知識約束
6. **關聯規則分類 (Associative Classification)**：用於分類任務

---

## 📚 參考資源

### 教科書
1. *Data Mining: Concepts and Techniques* by Jiawei Han et al.（第 6-7 章）
2. *Introduction to Data Mining* by Tan et al.（第 6 章）
3. *Machine Learning* by Tom Mitchell（Association Rule Learning）

### 線上資源
- [mlxtend Documentation - Association Rules](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/)
- [FP-Growth Algorithm Tutorial](https://www.cs.sfu.ca/~jpei/publications/sigmod00.pdf)

### 化工領域應用論文
- Association Rule Mining for Chemical Process Optimization
- Knowledge Discovery in Chemical Engineering Databases
- Data-Driven Recipe Optimization using Association Rules

---

## ✍️ 課後習題提示

1. **配方規則挖掘**：從化工配方數據中挖掘成分協同效應規則
2. **演算法比較**：比較 Apriori 和 FP-Growth 在不同數據規模下的性能
3. **參數調整**：探討支持度、置信度閾值對規則數量和品質的影響
4. **規則評估**：使用多種指標評估規則，篩選出最有價值的規則
5. **實際應用**：將關聯規則學習應用於您的化工數據集

---

## 📞 技術支援

如有疑問或需要協助，歡迎透過以下方式聯繫：
- 課程討論區
- 指導教授 Office Hour
- 課程助教

---

**課程資訊**
- 課程名稱：AI在化工上之應用
- 課程單元：Unit08 關聯規則學習 (Association Rule Learning)
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-01-28

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---