# Part_1 課程改進建議（專家評估）

**評估者視角**：資深機器學習課程教學專家 + 化學工程實務專家  
**評估日期**：2025-12-17  
**課程版本**：Part_1 新重構版本

---

## 一、整體評估摘要

### ✅ 優點
1. **技術深度適當**：NumPy 張量思維、Pandas EDA 流程清晰
2. **理論基礎扎實**：SIMD、記憶體局部性、皮爾森係數有數學推導
3. **結構邏輯清楚**：Unit01→02→03→04 進階合理
4. **程式碼品質高**：註解清楚、錯誤處理完善

### ⚠️ 待改進點
1. **缺少學習動機建立**（Critical）
2. **化工案例接觸太晚**（Major）
3. **實務陷阱提醒不足**（Major）
4. **批次案例過於簡化**（Minor）

---

## 二、具體改進建議

### 建議 1：Unit01 增加「課程開場白」（Critical Priority）

**位置**：Unit01_Python_EDA_Basics.md 開頭，在「本堂課目標」之前

**新增章節**：「0. 為什麼化工需要 Python + ML？」

**內容要點**：
```markdown
## 0. 為什麼化工需要 Python + ML？

### 0.1 化工場景的數據挑戰

**傳統化工面臨的問題**：
- 📊 **海量時序數據**：DCS 每秒記錄數千個 Tag，人工無法即時監控
- 🔬 **品質預測困難**：批次最終品質需等反應結束，無法即時調整
- ⚠️ **異常偵測延遲**：設備故障徵兆藏在數千變數中，難以提前發現
- 💰 **操作優化靠經驗**：最佳操作條件依賴資深工程師，難以量化

**機器學習的解決方案**：
| 化工問題 | ML 方法 | 預期效益 |
|---------|--------|---------|
| 品質即時預測 | 軟測器 (Soft Sensor) | 節省 Lab 分析成本 60% |
| 異常提前預警 | 時序異常偵測 | 減少非計畫停機 40% |
| 操作條件優化 | 貝氏優化 | 提升產率 5-15% |
| 設備健康管理 | RUL 預測 | 延長設備壽命 20% |

### 0.2 本課程的學習路徑

```
Part_1: Python 基礎 + EDA
    ↓ (你現在在這裡)
Part_2: ML 建模基礎 (回歸、分類、樹模型)
    ↓
Part_3: 化工特化應用 (軟測器、異常偵測)
    ↓
Part_4: 深度學習 (CNN、RNN、Transformer)
    ↓
Part_5: 進階應用 (強化學習控制、RUL 預測)
```

**你將學到的核心技能**：
1. ✅ 用 NumPy/Pandas 處理化工時序數據
2. ✅ 用 Matplotlib/Seaborn 視覺化製程趨勢
3. ✅ 用 Scikit-learn 建立預測模型
4. ✅ 用 TensorFlow/PyTorch 做深度學習
5. ✅ 部署模型到實際化工場景

### 0.3 第一個化工 ML 案例預覽（先看看終點）

**案例**：反應器溫度軟測器
- **問題**：Lab 每 2 小時才能測一次產物濃度，無法即時調控
- **解決**：用溫度/壓力/流量等即時變數預測濃度
- **效果**：R² = 0.92，誤差 < 5%

（這裡可以放一張圖：真實值 vs 預測值的時序圖）

> **重點**：不用擔心現在看不懂！Part_1 會先打好基礎，Part_2/3 就能做出這個案例。
```

---

### 建議 2：Unit01 每個技術點增加「化工類比」（Major Priority）

**修改位置**：在每個 NumPy/Pandas 技術說明後立即添加化工應用

#### 範例 1：向量化運算的化工意義
```markdown
**化工應用場景**：
```python
# 化工案例：計算 1000 個反應器的即時轉化率
reactor_temps = np.array([...])  # 1000 個反應器溫度
activation_energy = 50000  # J/mol
R = 8.314

# ❌ Python loop：需要 0.5 秒
conversion = []
for T in reactor_temps:
    k = np.exp(-activation_energy / (R * T))
    conversion.append(1 - np.exp(-k * residence_time))

# ✅ NumPy 向量化：只需 0.01 秒
k = np.exp(-activation_energy / (R * reactor_temps))
conversion = 1 - np.exp(-k * residence_time)
```
**實務意義**：DCS 每秒可能需要計算數千次，向量化是必需而非可選。
```

#### 範例 2：Reshape 的化工意義
```markdown
**化工應用場景**：
```python
# 案例：將 24 小時的分鐘級數據重組成小時級矩陣
minute_data = np.arange(24 * 60)  # 1440 分鐘數據
hourly_matrix = minute_data.reshape(24, 60)  # (小時, 該小時內的分鐘)

# 計算每小時統計量
hourly_mean = hourly_matrix.mean(axis=1)
hourly_std = hourly_matrix.std(axis=1)
```
**物理意義**：
- axis=0（沿行）：跨小時的同一分鐘位置平均（通常無意義）
- axis=1（沿列）：單一小時內的分鐘級平均（有意義）
- **錯誤示範**：把 (1440,) reshape 成 (60, 24) 會混淆時間順序！
```

---

### 建議 3：Unit02 增加「化工時序數據陷阱清單」（Major Priority）

**新增章節**：在 Unit02_TimeSeries_Cleaning.md 開頭

```markdown
## 化工時序數據的十大陷阱（實務經驗總結）

### ⚠️ 數據品質陷阱

1. **單位不一致**
   - 問題：溫度混用 °C 和 K，壓力混用 bar 和 psi
   - 檢查：`df.describe()` 看極值是否合理
   - 解法：建立單位轉換字典，統一標準化

2. **時區混亂**
   - 問題：DCS 用 UTC，Lab 用本地時間，夏令時切換時出現重複/跳躍
   - 檢查：`df.index.is_monotonic_increasing`
   - 解法：全部轉 UTC，用 `pd.to_datetime(utc=True)`

3. **感測器漂移 (Sensor Drift)**
   - 問題：溫度計長期使用後系統性偏移 +2°C
   - 檢查：與校驗記錄比對，看趨勢是否線性漂移
   - 解法：用校驗點做線性修正，或標註「不可信區間」

4. **停機期間的異常值**
   - 問題：設備停機時感測器讀數變成 0 或 -9999
   - 檢查：`df[df['Flow'] < 0]`
   - 解法：用製程狀態標籤 (running/idle/maintenance) 過濾

5. **採樣頻率不一致**
   - 問題：正常 1 分鐘/筆，故障時變成 10 秒/筆
   - 檢查：`df.index.to_series().diff().describe()`
   - 解法：統一重採樣到固定頻率

### ⚠️ 物理意義陷阱

6. **因果倒置 (Causality Violation)**
   - 問題：用「未來的產物濃度」預測「過去的反應溫度」
   - 檢查：確保 X 的時間戳 < y 的時間戳
   - 解法：建立嚴格的時間窗口規則

7. **批次邊界洩漏 (Batch Leakage)**
   - 問題：把批次 A 的結尾和批次 B 的開頭當作連續時序
   - 檢查：視覺化確認批次切分點
   - 解法：用 `groupby('BatchID')` 分開處理

8. **穩態假設錯誤**
   - 問題：用穩態模型處理開車/切換品級的動態過程
   - 檢查：計算滑動標準差，識別非穩態區間
   - 解法：分段建模或用動態模型 (RNN/LSTM)

### ⚠️ 統計陷阱

9. **假相關 (Spurious Correlation)**
   - 問題：「夏天冰淇淋銷量」與「溺水人數」高度相關
   - 化工案例：「戶外溫度」與「反應器溫度」相關（實際無因果）
   - 解法：用領域知識過濾，做因果推論檢驗

10. **倖存者偏差 (Survivorship Bias)**
    - 問題：只分析「成功批次」，忽略「失敗批次」
    - 檢查：確認數據集包含所有批次（包括報廢的）
    - 解法：建立完整的批次追蹤系統

---

**實務檢查清單（每次拿到新數據都要過一遍）**：

```python
# === 化工數據健康檢查 SOP ===
def chemeng_data_health_check(df):
    print("=== 1. 基本資訊 ===")
    print(f"形狀: {df.shape}")
    print(f"時間範圍: {df.index.min()} ~ {df.index.max()}")
    print(f"缺失值: {df.isnull().sum().sum()} / {df.size}")
    
    print("\n=== 2. 時間檢查 ===")
    print(f"時間單調遞增: {df.index.is_monotonic_increasing}")
    time_diff = df.index.to_series().diff()
    print(f"採樣間隔 (median): {time_diff.median()}")
    print(f"採樣間隔 (std): {time_diff.std()}")
    
    print("\n=== 3. 數值範圍檢查 ===")
    for col in df.select_dtypes(include=[np.number]).columns:
        q1, q99 = df[col].quantile([0.01, 0.99])
        print(f"{col}: [{q1:.2f}, {q99:.2f}] (1%-99%)")
        
    print("\n=== 4. 異常值檢查 ===")
    for col in df.select_dtypes(include=[np.number]).columns:
        n_neg = (df[col] < 0).sum()
        n_zero = (df[col] == 0).sum()
        if n_neg > 0:
            print(f"⚠️ {col} 有 {n_neg} 個負值")
        if n_zero > len(df) * 0.1:
            print(f"⚠️ {col} 有 {n_zero} 個零值 ({n_zero/len(df)*100:.1f}%)")
```
```

---

### 建議 4：Unit03 批次 EDA 增加進階案例（Minor Priority）

**新增內容**：在現有簡單線性案例後，添加「進階案例：溫度-時間交互作用」

```markdown
### 4.6 進階案例：非線性與交互作用

**化工現實**：實際批次製程常有：
1. **二次效應**：溫度過高或過低都不好（最佳點在中間）
2. **交互作用**：高溫 + 長時間 = 過度反應
3. **階段行為**：升溫階段 vs 恆溫階段的影響不同

```python
# 更真實的批次數據生成
np.random.seed(42)
n = 50
df_real = pd.DataFrame({
    'Temp': np.random.uniform(75, 85, n),
    'Time': np.random.uniform(100, 140, n),
})

# 非線性 + 交互作用
df_real['Mn'] = (
    50  # 基線
    + 15 * (df_real['Temp'] - 80)  # 線性溫度效應
    - 0.5 * (df_real['Temp'] - 80)**2  # 二次項（過高過低都不好）
    + 0.3 * (df_real['Time'] - 120)  # 線性時間效應
    - 0.02 * (df_real['Temp'] - 80) * (df_real['Time'] - 120)  # 交互作用
    + np.random.normal(0, 2, n)  # 噪音
)

# 視覺化交互作用
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 4))

# 1. 3D 散佈圖
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(df_real['Temp'], df_real['Time'], df_real['Mn'], c=df_real['Mn'], cmap='viridis')
ax1.set_xlabel('Temperature')
ax1.set_ylabel('Time')
ax1.set_zlabel('Mn')

# 2. 溫度效應（分時間組）
ax2 = fig.add_subplot(132)
for time_group in ['Short (<115)', 'Medium (115-125)', 'Long (>125)']:
    if time_group == 'Short (<115)':
        mask = df_real['Time'] < 115
    elif time_group == 'Medium (115-125)':
        mask = (df_real['Time'] >= 115) & (df_real['Time'] <= 125)
    else:
        mask = df_real['Time'] > 125
    ax2.scatter(df_real.loc[mask, 'Temp'], df_real.loc[mask, 'Mn'], label=time_group, alpha=0.6)
ax2.set_xlabel('Temperature')
ax2.set_ylabel('Mn')
ax2.legend()
ax2.set_title('溫度效應隨時間改變（交互作用）')

# 3. 熱圖（溫度 × 時間）
ax3 = fig.add_subplot(133)
from scipy.interpolate import griddata
temp_grid = np.linspace(75, 85, 20)
time_grid = np.linspace(100, 140, 20)
T_mesh, Time_mesh = np.meshgrid(temp_grid, time_grid)
Mn_mesh = griddata(
    (df_real['Temp'], df_real['Time']), 
    df_real['Mn'], 
    (T_mesh, Time_mesh), 
    method='cubic'
)
im = ax3.contourf(T_mesh, Time_mesh, Mn_mesh, levels=15, cmap='RdYlGn')
ax3.set_xlabel('Temperature')
ax3.set_ylabel('Time')
ax3.set_title('Mn 響應曲面')
plt.colorbar(im, ax=ax3)

plt.tight_layout()
plt.savefig('Unit03_Results/11_advanced_batch_interaction.png')
plt.show()
```

**關鍵洞察**：
1. **最佳操作點**：≈ 80°C × 120 min（響應曲面的峰值）
2. **溫度容忍度**：長時間時對溫度更敏感（交互作用）
3. **風險區域**：高溫 + 長時間 = Mn 下降（過度反應）

**實務啟示**：
- 簡單線性模型可能漏掉最佳點
- 需要用多項式回歸或樹模型捕捉非線性
- DoE（實驗設計）應包含交互作用項
```

---

## 三、其他細節建議

### 3.1 程式碼品質

**建議**：在每個 notebook 開頭增加版本檢查
```python
import sys
print(f"Python: {sys.version}")
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")

# 確保版本相容
assert pd.__version__ >= '1.3.0', "請升級 Pandas 到 1.3.0 以上"
```

### 3.2 學習評估

**建議**：每個 Unit 結尾增加「自我檢測題」

**Unit01 自測範例**：
```markdown
## 自我檢測（完成後請勾選）

### NumPy 張量
- [ ] 我能解釋為什麼向量化比 for loop 快
- [ ] 我能將 (60, 3) reshape 成 (3, 20, 3) 並解釋物理意義
- [ ] 我能識別 reshape 操作中可能的批次邊界混淆

### Pandas EDA
- [ ] 我能判斷缺失值應該填補還是刪除
- [ ] 我能解釋 One-Hot Encoding 為什麼要 drop_first
- [ ] 我能讀懂相關係數熱圖並提出假設

### 化工應用
- [ ] 我能列舉 3 個化工場景中的 ML 應用
- [ ] 我能說明批次數據與連續數據的差異
- [ ] 我能設計一個簡單的批次 EDA 流程
```

### 3.3 實務案例庫

**建議**：建立「化工 ML 案例速查表」（獨立 markdown）

```markdown
# 化工 ML 案例速查表

| 製程類型 | 問題 | ML 方法 | 數據需求 | 預期效益 |
|---------|-----|---------|---------|---------|
| 聚合反應 | Mn 預測 | RF/XGBoost | 溫度/時間/催化劑 | R²>0.9 |
| 精餾塔 | 塔頂純度軟測器 | MLP/LSTM | 溫度分佈/迴流比 | 減少Lab分析 |
| 熱交換器 | 結垢預測 | 時序異常偵測 | 壓降/傳熱係數 | 提前 2 週預警 |
| 壓縮機 | RUL 預測 | CNN-LSTM | 振動/溫度 | 減少停機 30% |
```

---

## 四、實施優先級

### 🔴 Critical（必須做）
1. **建議 1**：Unit01 增加課程開場白（估計 2 小時）
   - 影響：直接關係學生學習動機

### 🟡 Major（強烈建議）
2. **建議 2**：每個技術點增加化工類比（估計 4 小時）
   - 影響：提升學習遷移效果
3. **建議 3**：Unit02 增加陷阱清單（估計 3 小時）
   - 影響：減少實務應用時的常見錯誤

### 🟢 Minor（時間允許再做）
4. **建議 4**：Unit03 增加進階批次案例（估計 2 小時）
   - 影響：提升案例真實性

---

## 五、總結

當前 Part_1 課程在**技術深度**和**邏輯結構**上已經非常優秀，主要需要補強的是：

1. **學習動機**：讓學生一開始就看到「終點在哪裡」
2. **領域連結**：每個技術點都要有化工案例支撐
3. **實務經驗**：傳授數據陷阱的識別與處理

這些改進將使課程從「技術培訓」提升為「化工 ML 專業訓練」。

---

**評估者簽名**：GitHub Copilot (Claude Sonnet 4.5)  
**複審建議**：請化工系教授和資深數據科學家各審閱一次
