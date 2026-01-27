# Unit04 Matplotlib與Seaborn (Matplotlib and Seaborn)

## 📚 單元簡介

「一圖勝千言」，資料視覺化是資料科學與機器學習工作流程中不可或缺的環節。無論是探索性資料分析（EDA）、模型結果呈現，還是向非技術背景人員溝通發現，優秀的視覺化都能讓資訊更清晰、更具說服力。

**Matplotlib** 是Python最經典、最強大的資料視覺化套件，提供完整的繪圖功能與精細的自訂控制能力。從簡單的折線圖到複雜的多圖佈局，Matplotlib幾乎可以繪製所有類型的靜態圖表。

**Seaborn** 建立在Matplotlib之上，提供更高階、更美觀的統計視覺化介面。其簡潔的語法、內建的統計計算、以及精美的預設樣式，讓資料探索與統計分析變得更加高效且專業。

本單元將系統性地學習這兩個套件的核心功能，從基本的折線圖、長條圖、散佈圖，到進階的分佈圖、關係圖、熱力圖等。透過化工領域的實例（製程趨勢圖、品質控制圖、實驗結果比較、相關性分析等），您將學會如何選擇合適的圖表類型，以及如何製作清晰、美觀、專業的資料視覺化。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **掌握Matplotlib繪圖架構**：理解Figure、Axes、Artist的層次結構，熟悉pyplot與物件導向兩種繪圖介面
2. **製作基本圖表**：繪製折線圖、長條圖、散佈圖、直方圖、圓餅圖等常用圖表
3. **自訂圖表元素**：設定標題、軸標籤、圖例、網格、顏色、線型、標記等視覺元素
4. **建立多圖佈局**：使用subplot建立多圖排列，掌握複雜圖表的佈局設計
5. **使用Seaborn統計視覺化**：繪製分佈圖、類別圖、關係圖、熱力圖等統計圖表
6. **美化圖表外觀**：應用Seaborn主題樣式與調色盤，提升圖表專業度
7. **應用於化工資料分析**：將視覺化技術應用於製程監控、實驗數據分析、品質管制等實際場景

---

## 📖 單元內容架構

### 1️⃣ Matplotlib 資料視覺化 ⭐

**檔案**：
- 講義：[Unit04_Matplotlib.md](Unit04_Matplotlib.md)
- 程式範例：[Unit04_Matplotlib.ipynb](Unit04_Matplotlib.ipynb)

**內容重點**：
- **Matplotlib基礎**：
  - Matplotlib架構（Backend、Artist、Scripting三層）
  - pyplot介面 vs 物件導向介面
  - Figure與Axes的概念
  - 在Jupyter Notebook中的顯示設定
  
- **基本圖表類型**：
  - **折線圖（Line Plot）**：時間序列、趨勢圖
  - **散佈圖（Scatter Plot）**：變數關係、分布探索
  - **長條圖（Bar Chart）**：類別比較、數量展示
  - **直方圖（Histogram）**：資料分布、頻率統計
  - **圓餅圖（Pie Chart）**：比例展示、組成分析
  - **箱型圖（Box Plot）**：統計分布、離群值偵測
  
- **圖表自訂**：
  - 標題與軸標籤（title、xlabel、ylabel）
  - 圖例（legend）位置與樣式
  - 網格（grid）設定
  - 顏色、線型、標記樣式
  - 軸範圍與刻度（xlim、ylim、xticks、yticks）
  - 中文字型設定
  
- **多圖佈局**：
  - subplot()建立子圖網格
  - subplots()建立Figure與Axes陣列
  - GridSpec進階佈局
  - 子圖間距調整
  
- **進階功能**：
  - 雙Y軸圖表（twinx）
  - 註解與箭頭（annotate）
  - 填充區域（fill_between）
  - 圖片儲存（savefig）與格式選擇
  
- **化工應用範例**：
  - 製程趨勢圖（溫度、壓力、流量隨時間變化）
  - 反應動力學曲線（濃度-時間關係）
  - 品質控制圖（控制上下限、異常點標記）
  - 實驗結果比較（多組條件的長條圖比較）
  - 設備效率分析（圓餅圖、箱型圖）

**適合場景**：需要精細控制圖表細節、製作出版品質圖表、複雜客製化佈局

---

### 2️⃣ Seaborn 統計資料視覺化 ⭐

**檔案**：
- 講義：[Unit04_Seaborn.md](Unit04_Seaborn.md)
- 程式範例：[Unit04_Seaborn.ipynb](Unit04_Seaborn.ipynb)

**內容重點**：
- **Seaborn基礎**：
  - Seaborn設計理念與優勢
  - 與Matplotlib的關係
  - 樣式系統（5種style、4種context）
  - 調色盤系統
  
- **分佈視覺化**：
  - **histplot()**：直方圖與KDE（核密度估計）
  - **kdeplot()**：平滑的密度曲線
  - **rugplot()**：資料點地毯圖
  - **distplot()**：綜合分布圖（已淘汰，但仍常見）
  - **jointplot()**：雙變數聯合分布
  
- **類別資料視覺化**：
  - **barplot()**：長條圖（自動計算平均值與信賴區間）
  - **countplot()**：計數長條圖
  - **boxplot()**：箱型圖（顯示四分位數）
  - **violinplot()**：小提琴圖（結合箱型圖與KDE）
  - **stripplot()**：散佈圖（顯示所有資料點）
  - **swarmplot()**：蜂群圖（避免資料點重疊）
  
- **關係視覺化**：
  - **scatterplot()**：散佈圖（支援hue、size、style編碼）
  - **lineplot()**：折線圖（自動計算信賴區間）
  - **relplot()**：關係圖（支援facet分面）
  - **regplot()**：迴歸圖（自動擬合線性迴歸）
  - **lmplot()**：線性模型圖（支援facet）
  
- **矩陣視覺化**：
  - **heatmap()**：熱力圖（相關性矩陣、混淆矩陣）
  - **clustermap()**：階層式聚類熱力圖
  
- **多變數視覺化**：
  - **pairplot()**：成對關係圖（矩陣散佈圖）
  - **FacetGrid()**：分面網格（按類別分割子圖）
  - **PairGrid()**：自訂成對圖
  
- **樣式與美化**：
  - 主題樣式（darkgrid、whitegrid、dark、white、ticks）
  - 繪圖環境（paper、notebook、talk、poster）
  - 調色盤（deep、muted、pastel、bright、dark、colorblind）
  - 自訂調色盤
  
- **化工應用範例**：
  - 製程參數分布分析（多變數histplot）
  - 品質數據的統計比較（boxplot、violinplot）
  - 製程變數相關性分析（heatmap）
  - 實驗設計結果探索（pairplot）
  - 多條件製程數據比較（FacetGrid）

**適合場景**：快速探索性資料分析、統計視覺化、需要美觀預設樣式的場合

---

### 3️⃣ Matplotlib實作練習

**檔案**：[Unit04_Matplotlib_Homework.ipynb](Unit04_Matplotlib_Homework.ipynb)

**練習內容**：
- 基本圖表繪製（折線圖、長條圖、散佈圖）
- 圖表自訂與美化
- 多圖佈局設計
- 化工製程數據視覺化
- 圖表儲存與格式優化

---

### 4️⃣ Seaborn實作練習

**檔案**：[Unit04_Seaborn_Homework.ipynb](Unit04_Seaborn_Homework.ipynb)

**練習內容**：
- 分佈視覺化實作
- 類別資料比較
- 相關性分析與熱力圖
- 多變數關係探索
- 統計圖表在化工數據分析中的應用

---

## 📊 輸出資料夾

### Matplotlib圖表輸出
- **資料夾**：`outputs/P1_Unit04_Matplotlib/figs/`
- **內容**：Matplotlib教學範例產生的所有圖表檔案

### Seaborn圖表輸出
- **資料夾**：`outputs/P1_Unit04_Seaborn/figs/`
- **內容**：Seaborn教學範例產生的所有圖表檔案

---

## 💻 實作環境需求

### 必要套件
```python
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

### 選用套件
```python
scipy >= 1.7.0       # Seaborn統計功能需要
pillow >= 8.0.0      # 圖片處理
```

### 安裝指令
```bash
# 使用pip安裝
pip install matplotlib seaborn

# 使用conda安裝
conda install matplotlib seaborn
```

---

## 📈 學習路徑建議

### 第一階段：Matplotlib基礎
1. 閱讀 [Unit04_Matplotlib.md](Unit04_Matplotlib.md) 第1-3節
2. 執行 [Unit04_Matplotlib.ipynb](Unit04_Matplotlib.ipynb) 對應章節
3. 重點掌握：
   - Matplotlib的基本繪圖流程
   - 常用圖表類型（折線圖、散佈圖、長條圖、直方圖）
   - 圖表標題、標籤、圖例的設定
   - 顏色、線型、標記的自訂

### 第二階段：Matplotlib進階
1. 閱讀講義第4-6節
2. 執行對應的程式範例
3. 重點掌握：
   - subplot多圖佈局
   - 物件導向繪圖介面
   - 圖表的精細自訂
   - 圖片儲存與格式選擇

### 第三階段：Seaborn基礎
1. 閱讀 [Unit04_Seaborn.md](Unit04_Seaborn.md) 第1-3節
2. 執行 [Unit04_Seaborn.ipynb](Unit04_Seaborn.ipynb) 對應章節
3. 重點掌握：
   - Seaborn的樣式系統
   - 分佈視覺化（histplot、kdeplot）
   - 類別資料視覺化（barplot、boxplot、violinplot）
   - 關係視覺化（scatterplot、regplot）

### 第四階段：Seaborn進階
1. 閱讀講義第4-6節
2. 執行對應的程式範例
3. 重點掌握：
   - 熱力圖（heatmap）應用
   - 成對關係圖（pairplot）
   - 分面網格（FacetGrid）
   - 調色盤的選擇與自訂

### 第五階段：綜合應用
1. 完成 [Unit04_Matplotlib_Homework.ipynb](Unit04_Matplotlib_Homework.ipynb)
2. 完成 [Unit04_Seaborn_Homework.ipynb](Unit04_Seaborn_Homework.ipynb)
3. 將視覺化技術應用於化工資料探索
4. 建立個人的視覺化圖表庫

---

## 🎓 Matplotlib vs Seaborn 比較與選擇指南

| 比較維度 | Matplotlib | Seaborn |
|---------|-----------|---------|
| **抽象層次** | 低階，精細控制 | 高階，快速繪圖 |
| **語法風格** | 需要更多程式碼 | 簡潔直觀 |
| **預設樣式** | 較基本 | 現代化、美觀 |
| **統計功能** | 需手動計算 | 內建統計計算（信賴區間、迴歸線） |
| **資料格式** | 主要使用陣列/列表 | 原生支援DataFrame |
| **圖表類型** | 通用圖表 | 統計導向圖表 |
| **學習曲線** | 陡峭 | 平緩 |
| **客製化程度** | 極高 | 中等（但可結合Matplotlib） |

**選擇建議**：
1. **快速探索性分析** → Seaborn
2. **統計視覺化（分布、相關性）** → Seaborn
3. **需要精細控制每個細節** → Matplotlib
4. **複雜佈局與客製化** → Matplotlib
5. **時間序列趨勢圖** → Matplotlib或Seaborn均可
6. **出版品質圖表** → Matplotlib（精細調整）或Seaborn（美觀預設）

**實務上**：兩者常搭配使用，Seaborn快速繪圖，Matplotlib進行細部調整。

---

## 🔍 化工領域核心應用

### 1. 製程監控趨勢圖 ⭐
- **目標**：視覺化製程參數隨時間的變化，監控製程穩定性
- **圖表建議**：折線圖（Matplotlib）
- **關鍵技術**：
  - 多條曲線在同一圖表（溫度、壓力、流量）
  - 雙Y軸（不同單位的參數）
  - 標記異常時間點
  - 填充控制上下限區間

### 2. 品質數據統計分析
- **目標**：比較不同批次、不同產品的品質分布
- **圖表建議**：箱型圖、小提琴圖（Seaborn）
- **關鍵技術**：
  - 多組數據並排比較
  - 顯示四分位數與離群值
  - 結合分布密度視覺化
  - 統計檢定結果標記

### 3. 實驗結果比較
- **目標**：比較不同實驗條件的效果
- **圖表建議**：長條圖（Seaborn barplot）
- **關鍵技術**：
  - 自動計算平均值與誤差棒
  - 分組比較（hue參數）
  - 統計顯著性標記
  - 專業配色方案

### 4. 製程變數相關性分析
- **目標**：探索多個製程參數之間的相關性
- **圖表建議**：熱力圖（Seaborn heatmap）、成對關係圖（pairplot）
- **關鍵技術**：
  - 相關係數矩陣視覺化
  - 顏色編碼相關強度
  - 散佈圖矩陣（所有變數兩兩關係）
  - 對角線顯示分布

### 5. 反應動力學曲線
- **目標**：展示反應物濃度隨時間的變化
- **圖表建議**：折線圖 + 迴歸曲線（Matplotlib + Seaborn）
- **關鍵技術**：
  - 實驗數據點（散佈圖）
  - 擬合曲線（折線圖）
  - 信賴區間填充
  - 半對數或雙對數座標

---

## 📝 圖表類型選擇指南

### 基於資料類型

| 資料特性 | 建議圖表類型 | Matplotlib | Seaborn |
|---------|------------|-----------|---------|
| **時間序列** | 折線圖 | `plot()` | `lineplot()` |
| **類別比較** | 長條圖、箱型圖 | `bar()`, `boxplot()` | `barplot()`, `boxplot()` |
| **分布探索** | 直方圖、KDE | `hist()` | `histplot()`, `kdeplot()` |
| **變數關係** | 散佈圖 | `scatter()` | `scatterplot()`, `regplot()` |
| **比例組成** | 圓餅圖 | `pie()` | ❌ 不支援 |
| **相關性** | 熱力圖 | `imshow()` | `heatmap()` |
| **多變數探索** | 成對圖 | ❌ 需手動建立 | `pairplot()` |

### 基於分析目的

1. **比較** → 長條圖、箱型圖
2. **分布** → 直方圖、KDE、小提琴圖
3. **關係** → 散佈圖、迴歸圖
4. **趨勢** → 折線圖
5. **組成** → 圓餅圖、堆疊長條圖
6. **相關性** → 熱力圖、成對圖

---

## 🎨 圖表設計最佳實踐

### 1. 清晰性原則
- ✅ 使用清楚的標題與軸標籤（單位必須標示）
- ✅ 圖例位置適當，不遮擋數據
- ✅ 字體大小適中（title: 14-16pt, labels: 12pt）
- ❌ 避免3D圖表（除非必要）
- ❌ 避免過多顏色或線型（≤7種）

### 2. 專業性原則
- ✅ 使用專業配色（ColorBrewer、Seaborn調色盤）
- ✅ 保持圖表風格一致
- ✅ 去除不必要的網格線與邊框
- ✅ 使用適當的圖表比例（寬高比）
- ❌ 避免預設的鮮豔配色

### 3. 誠實性原則
- ✅ Y軸從0開始（長條圖、面積圖）
- ✅ 誤差棒顯示不確定性
- ✅ 不裁切數據範圍（除非有充分理由）
- ❌ 不扭曲比例誤導讀者

### 4. 化工領域慣例
- ✅ 溫度使用°C或K（標註清楚）
- ✅ 壓力使用bar、atm或Pa（統一單位）
- ✅ 時間序列橫軸為時間
- ✅ 標註操作條件與實驗參數

---

## 🚀 進階學習方向

完成本單元後，可以進一步探索：

1. **互動式視覺化**：
   - Plotly（網頁互動圖表）
   - Bokeh（大數據視覺化）
   - Altair（宣告式視覺化）

2. **專業領域視覺化**：
   - Matplotlib動畫（animation）
   - 3D繪圖（mplot3d）
   - 地理資訊視覺化（Cartopy、Folium）

3. **視覺化設計理論**：
   - 色彩理論（色盲友善配色）
   - 資訊圖表設計
   - Edward Tufte的視覺化原則

4. **整合到報告與簡報**：
   - LaTeX整合
   - PowerPoint自動化
   - 網頁儀表板（Dash、Streamlit）

---

## 📚 參考資源

### 教科書
1. *Matplotlib for Python Developers* by Sandro Tosi（深入Matplotlib）
2. *Python Data Science Handbook* by Jake VanderPlas（第四章視覺化）
3. *Storytelling with Data* by Cole Nussbaumer Knaflic（視覺化設計理念）

### 線上資源
- [Matplotlib官方範例庫](https://matplotlib.org/stable/gallery/index.html)
- [Seaborn官方範例庫](https://seaborn.pydata.org/examples/index.html)
- [Python Graph Gallery](https://python-graph-gallery.com/) - 大量範例程式碼
- [ColorBrewer](https://colorbrewer2.org/) - 專業配色工具

### 視訊教學
- [Matplotlib Tutorial by Corey Schafer](https://www.youtube.com/watch?v=UO98lJQ3QGI&list=PL-osiE80TeTvipOqomVEeZ1HRrcEvtZB_)
- [Seaborn Tutorial by Keith Galli](https://www.youtube.com/watch?v=6GUZXDef2U0)

---

## ✍️ 課後習題提示

1. **基本繪圖**：繪製化工製程的溫度-時間曲線
2. **多圖佈局**：在一個Figure中展示4個不同的製程參數
3. **資料分布**：使用Seaborn分析品質數據的分布特性
4. **相關性分析**：製作製程變數的相關性熱力圖
5. **統計比較**：比較不同操作條件下的產率（含誤差棒）
6. **圖表美化**：將預設圖表改造成出版品質的專業圖表

---

## 📞 技術支援

如有疑問或需要協助，歡迎透過以下方式聯繫：
- 課程討論區
- 指導教授 Office Hour
- 課程助教

---

**版本資訊**：Unit04 v1.0 | 最後更新：2026-01-27

**上一單元**：[Unit03 Numpy與Pandas](../Unit03/)  
**下一階段**：完成Part 1後，前往 [Part 2 非監督式學習](../../Part_2/) 開始機器學習實作

---

## 🎉 恭喜完成Part 1！

您已經完成「AI與機器學習概論」Part的所有單元，建立了紮實的Python基礎、資料處理能力與視覺化技術。這些是機器學習的核心基礎，接下來您將進入Part 2，開始學習非監督式學習的演算法與應用！
