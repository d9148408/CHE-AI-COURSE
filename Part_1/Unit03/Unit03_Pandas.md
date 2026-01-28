# Unit03 Pandas 資料處理與分析

## 課程簡介

Pandas 是 Python 中最重要的資料處理與分析套件，提供高效能、易用的資料結構與資料分析工具。在資料科學、機器學習與化學工程的資料分析中，Pandas 是不可或缺的核心工具。

### 為什麼需要學習 Pandas？

1. **強大的資料結構**：提供 DataFrame 與 Series 兩種核心資料結構，直觀地處理表格式資料
2. **高效的資料操作**：基於 NumPy 建立，運算效能優異，適合處理大規模資料
3. **豐富的功能**：支援資料讀取、清理、轉換、合併、分組、時間序列分析等完整功能
4. **資料科學標準工具**：是 Python 資料科學生態系統的基礎，與 NumPy、Matplotlib、Scikit-learn 等套件無縫整合
5. **化工資料分析利器**：適用於製程數據分析、實驗數據整理、品質管制、時間序列分析等應用

### 學習目標

完成本單元後，您將能夠：

- ✓ 理解 Pandas 的核心資料結構 (Series 與 DataFrame)
- ✓ 從各種來源讀取與寫入資料 (CSV、Excel、JSON 等)
- ✓ 進行資料選取、篩選、排序與索引操作
- ✓ 執行資料清理與處理缺失值
- ✓ 處理時間序列資料與日期時間格式
- ✓ 進行資料合併、分組與聚合運算
- ✓ 應用 Pandas 於化工領域的實際資料分析

---

## 1. Pandas 基本概念

### 1.1 什麼是 Pandas？

Pandas (Panel Data) 是建立在 NumPy 之上的高階資料處理套件，主要特點：

| 特性 | 說明 |
|------|------|
| **DataFrame** | 二維表格結構，類似 Excel 工作表或 SQL 資料表 |
| **Series** | 一維陣列結構，具有標籤索引 |
| **索引功能** | 強大的標籤索引系統，支援多層索引 |
| **資料對齊** | 自動依據索引對齊資料 |
| **缺失值處理** | 內建缺失值檢測與處理功能 |
| **時間序列** | 完整的日期時間處理與時間序列分析功能 |

### 1.2 安裝與匯入 Pandas

```python
# 安裝 Pandas（若尚未安裝）
# !pip install pandas

# 匯入 Pandas（慣例使用 pd 別名）
import pandas as pd
import numpy as np

# 檢查版本
print(f"Pandas 版本: {pd.__version__}")
```

### 1.3 Pandas 與 NumPy 的關係

```python
# NumPy 提供底層陣列運算
import numpy as np
numpy_array = np.array([1, 2, 3, 4, 5])

# Pandas 在 NumPy 基礎上提供標籤索引與高階功能
pandas_series = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
print(pandas_series)
print(f"\n透過標籤存取: pandas_series['c'] = {pandas_series['c']}")
```

---

## 2. Pandas 核心資料結構

### 2.1 Series：一維標籤陣列

Series 是帶有標籤索引的一維陣列，可視為增強版的 NumPy 陣列或是單欄的 DataFrame。

#### 2.1.1 建立 Series

```python
# 從 list 建立
s1 = pd.Series([10, 20, 30, 40, 50])
print("從 list 建立：")
print(s1)
print(f"索引: {s1.index}")
print(f"數值: {s1.values}")
print()

# 指定自訂索引
s2 = pd.Series([10, 20, 30, 40, 50], 
               index=['A', 'B', 'C', 'D', 'E'])
print("自訂索引：")
print(s2)
print()

# 從字典建立
temp_data = {'Mon': 25.5, 'Tue': 26.3, 'Wed': 24.8, 'Thu': 27.1, 'Fri': 25.9}
s3 = pd.Series(temp_data)
print("從字典建立：")
print(s3)
print(f"資料型態: {s3.dtype}")
```

#### 2.1.2 Series 索引與選取

```python
# 化工應用範例：反應器溫度記錄
temperatures = pd.Series(
    [350, 355, 360, 358, 362, 365, 368],
    index=['0h', '1h', '2h', '3h', '4h', '5h', '6h'],
    name='Reactor Temperature (°C)'
)

print("反應器溫度記錄：")
print(temperatures)
print()

# 單一索引存取
print(f"第 2 小時溫度: {temperatures['2h']}°C")
print(f"位置索引 [2]: {temperatures.iloc[2]}°C")
print()

# 切片操作
print("前 3 小時溫度：")
print(temperatures['0h':'2h'])
print()

# 條件篩選
print("溫度 > 360°C 的時段：")
print(temperatures[temperatures > 360])
```

#### 2.1.3 Series 運算

```python
# 向量化運算
pressures_bar = pd.Series([1.0, 1.5, 2.0, 2.5, 3.0])
pressures_psi = pressures_bar * 14.5038  # bar 轉 psi
print("壓力轉換 (bar → psi)：")
print(pressures_psi)
print()

# Series 間運算（自動對齊索引）
s1 = pd.Series([10, 20, 30], index=['A', 'B', 'C'])
s2 = pd.Series([5, 10, 15, 20], index=['A', 'B', 'C', 'D'])
result = s1 + s2
print("Series 自動對齊相加：")
print(result)  # D 的結果為 NaN（缺失值）
```

### 2.2 DataFrame：二維標籤表格

DataFrame 是 Pandas 最核心的資料結構，是二維的標籤資料表，可以想像成：
- Excel 試算表
- SQL 資料表
- 多個 Series 的集合（每欄都是一個 Series）

#### 2.2.1 建立 DataFrame

```python
# 方法 1：從字典建立
data = {
    'Temperature': [350, 355, 360, 358, 362],
    'Pressure': [2.0, 2.1, 2.2, 2.15, 2.25],
    'Conversion': [0.75, 0.78, 0.82, 0.80, 0.85]
}
df1 = pd.DataFrame(data)
print("從字典建立 DataFrame：")
print(df1)
print()

# 方法 2：從嵌套列表建立
data_list = [
    [350, 2.0, 0.75],
    [355, 2.1, 0.78],
    [360, 2.2, 0.82]
]
df2 = pd.DataFrame(data_list, 
                   columns=['Temperature', 'Pressure', 'Conversion'],
                   index=['Run1', 'Run2', 'Run3'])
print("從列表建立 DataFrame：")
print(df2)
print()

# 方法 3：從 NumPy 陣列建立
np.random.seed(42)
data_array = np.random.randn(5, 3)
df3 = pd.DataFrame(data_array,
                   columns=['A', 'B', 'C'],
                   index=['r1', 'r2', 'r3', 'r4', 'r5'])
print("從 NumPy 陣列建立 DataFrame：")
print(df3)
```

#### 2.2.2 DataFrame 基本屬性與方法

```python
# 建立化工製程數據範例
process_data = pd.DataFrame({
    'Time': pd.date_range('2024-01-01', periods=6, freq='h'),
    'Temp_C': [350, 355, 360, 358, 362, 365],
    'Press_bar': [2.0, 2.1, 2.2, 2.15, 2.25, 2.3],
    'Flow_L_min': [100, 105, 110, 108, 112, 115],
    'Conv_pct': [75.0, 78.0, 82.0, 80.0, 85.0, 87.0]
})

print("化工製程數據：")
print(process_data)
print()

# 基本屬性
print(f"資料形狀 (列, 欄): {process_data.shape}")
print(f"總元素數量: {process_data.size}")
print(f"欄位名稱: {list(process_data.columns)}")
print(f"索引: {list(process_data.index)}")
print(f"資料型態:\n{process_data.dtypes}")
print()

# 常用方法
print("前 3 列資料 (head)：")
print(process_data.head(3))
print()

print("後 2 列資料 (tail)：")
print(process_data.tail(2))
print()

print("資料摘要資訊 (info)：")
process_data.info()
print()

print("數值欄位統計摘要 (describe)：")
print(process_data.describe())
```

---

## 3. 資料讀取與寫入

Pandas 支援多種資料格式的讀取與寫入，是資料分析的第一步。

### 3.1 讀取 CSV 檔案

CSV (Comma-Separated Values) 是最常用的資料交換格式。

```python
# 讀取 CSV 檔案
df = pd.read_csv('process_data.csv')

# 常用參數
df = pd.read_csv(
    'process_data.csv',
    sep=',',              # 分隔符號（預設為逗號）
    header=0,             # 表頭列位置（預設為 0）
    index_col=0,          # 將第 0 欄設為索引
    encoding='utf-8',     # 編碼方式
    parse_dates=['Time'], # 將指定欄位解析為日期時間
    na_values=['NA', 'N/A', '']  # 自訂缺失值標記
)

print(df.head())
```

### 3.2 讀取 Excel 檔案

```python
# 讀取 Excel 檔案（需安裝 openpyxl 或 xlrd）
df_excel = pd.read_excel('experimental_data.xlsx', sheet_name='Sheet1')

# 讀取多個工作表
excel_file = pd.ExcelFile('experimental_data.xlsx')
sheet_names = excel_file.sheet_names
print(f"工作表名稱: {sheet_names}")

# 讀取特定工作表
df_sheet1 = pd.read_excel(excel_file, sheet_name='Experiment1')
df_sheet2 = pd.read_excel(excel_file, sheet_name='Experiment2')
```

### 3.3 寫入檔案

```python
# 寫入 CSV 檔案
df.to_csv('output_data.csv', index=False, encoding='utf-8-sig')

# 寫入 Excel 檔案
df.to_excel('output_data.xlsx', sheet_name='Results', index=False)

# 寫入多個工作表
with pd.ExcelWriter('multi_sheet_output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Experiment1', index=False)
    df2.to_excel(writer, sheet_name='Experiment2', index=False)
```

### 3.4 其他常用格式

```python
# JSON 格式
df.to_json('data.json', orient='records', indent=2)
df_json = pd.read_json('data.json', orient='records')

# SQL 資料庫（需安裝 sqlalchemy）
import sqlite3
conn = sqlite3.connect('database.db')
df.to_sql('table_name', conn, if_exists='replace', index=False)
df_sql = pd.read_sql('SELECT * FROM table_name', conn)

# HTML 表格
df.to_html('table.html', index=False)
```

---

## 4. 資料選取與索引

### 4.1 選取欄位

```python
# 建立範例 DataFrame
df = pd.DataFrame({
    'Time': pd.date_range('2024-01-01', periods=5, freq='h'),
    'Temp': [350, 355, 360, 358, 362],
    'Press': [2.0, 2.1, 2.2, 2.15, 2.25],
    'Flow': [100, 105, 110, 108, 112]
})

# 選取單一欄位（回傳 Series）
temp_series = df['Temp']
print(temp_series)
print()

# 選取多個欄位（回傳 DataFrame）
subset = df[['Temp', 'Press']]
print(subset)
print()

# 新增計算欄位
df['Temp_K'] = df['Temp'] + 273.15
df['Press_psi'] = df['Press'] * 14.5038
print(df)
```

### 4.2 選取列 (Row)

```python
# 使用 loc（標籤索引）
print("使用 loc 選取：")
print(df.loc[0])      # 選取索引為 0 的列
print(df.loc[0:2])    # 選取索引 0 到 2 的列（包含端點）
print()

# 使用 iloc（位置索引）
print("使用 iloc 選取：")
print(df.iloc[0])     # 選取第 1 列
print(df.iloc[0:3])   # 選取前 3 列（不包含端點）
print()

# 布林索引（條件篩選）
print("溫度 > 355°C 的資料：")
high_temp = df[df['Temp'] > 355]
print(high_temp)
print()

# 多條件篩選
print("溫度 > 355°C 且壓力 > 2.1 bar：")
filtered = df[(df['Temp'] > 355) & (df['Press'] > 2.1)]
print(filtered)
```

### 4.3 loc 與 iloc 的綜合應用

```python
# loc：標籤索引，[列, 欄]
print("loc 選取特定列與欄：")
print(df.loc[1:3, ['Temp', 'Press']])
print()

# iloc：位置索引，[列, 欄]
print("iloc 選取特定位置：")
print(df.iloc[1:4, 1:3])
print()

# 混合選取
print("選取溫度 > 355 的資料，只顯示 Temp 和 Flow：")
print(df.loc[df['Temp'] > 355, ['Temp', 'Flow']])
```

### 4.4 設定索引

```python
# 將時間欄位設為索引
df_indexed = df.set_index('Time')
print("以 Time 為索引：")
print(df_indexed)
print()

# 重設索引
df_reset = df_indexed.reset_index()
print("重設索引：")
print(df_reset)
print()

# 多層索引
df_multi = df.set_index(['Time', 'Temp'])
print("多層索引：")
print(df_multi)
```

---

## 5. 資料清理與處理

### 5.1 處理缺失值 (Missing Values)

```python
# 建立包含缺失值的 DataFrame
data_with_nan = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [10, np.nan, 30, 40, 50],
    'C': [100, 200, 300, np.nan, 500]
})

print("原始資料：")
print(data_with_nan)
print()

# 檢測缺失值
print("缺失值標記：")
print(data_with_nan.isnull())
print()

print("每欄缺失值數量：")
print(data_with_nan.isnull().sum())
print()

# 刪除包含缺失值的列
print("刪除任何欄位有缺失值的列：")
print(data_with_nan.dropna())
print()

print("刪除所有欄位都是缺失值的列：")
print(data_with_nan.dropna(how='all'))
print()

# 刪除包含缺失值的欄
print("刪除包含缺失值的欄：")
print(data_with_nan.dropna(axis=1))
print()

# 填補缺失值
print("用 0 填補缺失值：")
print(data_with_nan.fillna(0))
print()

print("用前一個值填補（forward fill）：")
print(data_with_nan.fillna(method='ffill'))
print()

print("用後一個值填補（backward fill）：")
print(data_with_nan.fillna(method='bfill'))
print()

print("用平均值填補：")
print(data_with_nan.fillna(data_with_nan.mean()))
```

### 5.2 處理重複值

```python
# 建立包含重複值的 DataFrame
data_dup = pd.DataFrame({
    'ID': [1, 2, 2, 3, 4, 4, 5],
    'Value': [10, 20, 20, 30, 40, 40, 50]
})

print("原始資料：")
print(data_dup)
print()

# 檢測重複列
print("重複列標記：")
print(data_dup.duplicated())
print()

# 刪除重複列（保留第一次出現）
print("刪除重複列（保留首次）：")
print(data_dup.drop_duplicates())
print()

# 刪除重複列（保留最後一次出現）
print("刪除重複列（保留最後）：")
print(data_dup.drop_duplicates(keep='last'))
print()

# 根據特定欄位判斷重複
print("根據 Value 欄位刪除重複：")
print(data_dup.drop_duplicates(subset=['Value']))
```

### 5.3 資料型態轉換

```python
# 建立範例資料
df_types = pd.DataFrame({
    'A': ['1', '2', '3', '4'],
    'B': ['10.5', '20.3', '30.8', '40.1'],
    'C': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']
})

print("原始資料型態：")
print(df_types.dtypes)
print()

# 轉換為數值型態
df_types['A'] = df_types['A'].astype(int)
df_types['B'] = df_types['B'].astype(float)

# 轉換為日期時間型態
df_types['C'] = pd.to_datetime(df_types['C'])

print("轉換後的資料型態：")
print(df_types.dtypes)
print()
print(df_types)
```

### 5.4 字串處理

```python
# 建立包含字串的 DataFrame
df_str = pd.DataFrame({
    'Name': ['  Reactor A  ', 'REACTOR B', 'reactor c'],
    'Status': ['Running', 'Stopped', 'Maintenance']
})

print("原始資料：")
print(df_str)
print()

# 字串方法（需使用 .str 存取器）
df_str['Name_clean'] = df_str['Name'].str.strip()        # 移除前後空白
df_str['Name_lower'] = df_str['Name'].str.lower()        # 轉小寫
df_str['Name_upper'] = df_str['Name'].str.upper()        # 轉大寫
df_str['Name_title'] = df_str['Name'].str.strip().str.title()  # 標題格式

print("字串處理結果：")
print(df_str)
print()

# 字串搜尋與取代
print("包含 'reactor' 的列（不區分大小寫）：")
print(df_str[df_str['Name'].str.contains('reactor', case=False)])
print()

# 字串取代
df_str['Name_replaced'] = df_str['Name'].str.replace('reactor', 'Tank', case=False)
print("取代後：")
print(df_str[['Name', 'Name_replaced']])
```

---

## 6. 時間序列資料處理

時間序列資料在化工製程監控中極為常見，Pandas 提供完整的日期時間處理功能。

### 6.1 建立日期時間物件

```python
import pandas as pd
from datetime import datetime, timedelta

# 建立單一日期時間
dt1 = pd.Timestamp('2024-01-01 10:30:00')
print(f"單一時間點: {dt1}")
print()

# 建立日期時間範圍
date_range = pd.date_range(
    start='2024-01-01',
    end='2024-01-10',
    freq='D'  # 'D'=日, 'h'=小時, 'min'=分鐘, 'S'=秒
)
print("日期範圍（每日）：")
print(date_range)
print()

# 建立時間序列 DataFrame
ts_data = pd.DataFrame({
    'Time': pd.date_range('2024-01-01', periods=24, freq='h'),
    'Temperature': np.random.uniform(350, 370, 24),
    'Pressure': np.random.uniform(2.0, 2.5, 24)
})
ts_data = ts_data.set_index('Time')
print("時間序列資料：")
print(ts_data.head())
```

### 6.2 日期時間索引與選取

```python
# 使用日期範圍選取
print("2024-01-01 的所有資料：")
print(ts_data['2024-01-01'])
print()

# 使用時間範圍選取
print("2024-01-01 00:00 到 05:00 的資料：")
print(ts_data['2024-01-01 00:00':'2024-01-01 05:00'])
print()

# 提取日期時間屬性
ts_data['Hour'] = ts_data.index.hour
ts_data['Day'] = ts_data.index.day
ts_data['Weekday'] = ts_data.index.day_name()
print("加入時間屬性：")
print(ts_data.head())
```

### 6.3 時間序列運算

```python
# 時間偏移
print("時間往後移動 2 小時：")
print(ts_data.shift(2).head())
print()

# 時間差分（計算變化率）
ts_data['Temp_diff'] = ts_data['Temperature'].diff()
print("溫度變化率：")
print(ts_data[['Temperature', 'Temp_diff']].head())
print()

# 滾動視窗計算（移動平均）
ts_data['Temp_MA3'] = ts_data['Temperature'].rolling(window=3).mean()
print("3 小時移動平均：")
print(ts_data[['Temperature', 'Temp_MA3']].head(10))
```

### 6.4 時間重採樣 (Resampling)

```python
# 建立分鐘級資料
minute_data = pd.DataFrame({
    'Time': pd.date_range('2024-01-01', periods=60, freq='min'),
    'Value': np.random.randn(60).cumsum()
}).set_index('Time')

# 下採樣：從分鐘降到小時（聚合）
hourly_mean = minute_data.resample('h').mean()
print("小時平均值：")
print(hourly_mean.head())
print()

hourly_sum = minute_data.resample('h').sum()
print("小時總和：")
print(hourly_sum.head())
print()

# 上採樣：從小時升到分鐘（插值）
hourly_data = pd.DataFrame({
    'Time': pd.date_range('2024-01-01', periods=5, freq='h'),
    'Value': [10, 20, 30, 40, 50]
}).set_index('Time')

minute_interp = hourly_data.resample('30min').interpolate(method='linear')
print("線性插值至 30 分鐘：")
print(minute_interp)
```

### 6.5 時間序列可視化

```python
import matplotlib.pyplot as plt

# 建立範例時間序列資料
ts = pd.DataFrame({
    'Time': pd.date_range('2024-01-01', periods=168, freq='h'),
    'Temperature': 350 + 10 * np.sin(np.linspace(0, 4*np.pi, 168)) + np.random.randn(168) * 2
}).set_index('Time')

# 計算移動平均
ts['MA_6h'] = ts['Temperature'].rolling(window=6).mean()
ts['MA_24h'] = ts['Temperature'].rolling(window=24).mean()

# 繪圖
plt.figure(figsize=(12, 6))
plt.plot(ts.index, ts['Temperature'], label='Raw Data', alpha=0.5, linewidth=1)
plt.plot(ts.index, ts['MA_6h'], label='6-hour MA', linewidth=2)
plt.plot(ts.index, ts['MA_24h'], label='24-hour MA', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Temperature (Celsius)')
plt.title('Temperature Time Series with Moving Averages')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 7. 資料合併與分組

### 7.1 資料合併 (Merge)

```python
# 建立兩個 DataFrame
df_temp = pd.DataFrame({
    'Reactor': ['R1', 'R2', 'R3'],
    'Temperature': [350, 360, 355]
})

df_press = pd.DataFrame({
    'Reactor': ['R1', 'R2', 'R4'],
    'Pressure': [2.0, 2.2, 2.1]
})

print("溫度資料：")
print(df_temp)
print()
print("壓力資料：")
print(df_press)
print()

# 內連接 (Inner Join)：只保留兩邊都有的
inner_merge = pd.merge(df_temp, df_press, on='Reactor', how='inner')
print("內連接 (Inner Join)：")
print(inner_merge)
print()

# 左連接 (Left Join)：保留左邊所有資料
left_merge = pd.merge(df_temp, df_press, on='Reactor', how='left')
print("左連接 (Left Join)：")
print(left_merge)
print()

# 外連接 (Outer Join)：保留所有資料
outer_merge = pd.merge(df_temp, df_press, on='Reactor', how='outer')
print("外連接 (Outer Join)：")
print(outer_merge)
```

### 7.2 資料串接 (Concatenate)

```python
# 縱向串接（堆疊列）
df1 = pd.DataFrame({
    'A': [1, 2],
    'B': [3, 4]
})

df2 = pd.DataFrame({
    'A': [5, 6],
    'B': [7, 8]
})

vertical_concat = pd.concat([df1, df2], axis=0, ignore_index=True)
print("縱向串接：")
print(vertical_concat)
print()

# 橫向串接（擴展欄位）
df3 = pd.DataFrame({
    'C': [10, 20],
    'D': [30, 40]
})

horizontal_concat = pd.concat([df1, df3], axis=1)
print("橫向串接：")
print(horizontal_concat)
```

### 7.3 分組運算 (GroupBy)

```python
# 建立化工實驗資料
experiment_data = pd.DataFrame({
    'Catalyst': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'Temperature': [350, 360, 370, 350, 360, 370, 350, 360, 370],
    'Conversion': [75, 80, 85, 70, 78, 82, 72, 79, 84]
})

print("原始實驗資料：")
print(experiment_data)
print()

# 依催化劑分組，計算平均值
catalyst_avg = experiment_data.groupby('Catalyst')['Conversion'].mean()
print("各催化劑的平均轉化率：")
print(catalyst_avg)
print()

# 多重聚合函數
catalyst_stats = experiment_data.groupby('Catalyst')['Conversion'].agg(['mean', 'std', 'min', 'max'])
print("各催化劑的統計摘要：")
print(catalyst_stats)
print()

# 多欄位分組
temp_catalyst_avg = experiment_data.groupby(['Temperature', 'Catalyst'])['Conversion'].mean()
print("溫度與催化劑組合的平均轉化率：")
print(temp_catalyst_avg)
print()

# 使用 transform（保持原始資料形狀）
experiment_data['Conv_Mean'] = experiment_data.groupby('Catalyst')['Conversion'].transform('mean')
experiment_data['Conv_Deviation'] = experiment_data['Conversion'] - experiment_data['Conv_Mean']
print("加入群組平均值與偏差：")
print(experiment_data)
```

### 7.4 透視表 (Pivot Table)

```python
# 建立銷售資料範例
sales_data = pd.DataFrame({
    'Product': ['A', 'A', 'B', 'B', 'C', 'C'] * 2,
    'Quarter': ['Q1', 'Q2', 'Q1', 'Q2', 'Q1', 'Q2'] * 2,
    'Region': ['North'] * 6 + ['South'] * 6,
    'Sales': [100, 120, 80, 90, 110, 130, 95, 115, 75, 85, 105, 125]
})

print("原始銷售資料：")
print(sales_data)
print()

# 建立透視表
pivot = pd.pivot_table(
    sales_data,
    values='Sales',
    index='Product',
    columns='Quarter',
    aggfunc='sum'
)
print("銷售透視表：")
print(pivot)
print()

# 多維透視表
pivot_multi = pd.pivot_table(
    sales_data,
    values='Sales',
    index=['Product', 'Region'],
    columns='Quarter',
    aggfunc='mean'
)
print("多維透視表：")
print(pivot_multi)
```

---

## 8. 化工領域應用案例

### 8.1 案例一：反應器批次數據分析

```python
# 建立批次反應器數據
batch_data = pd.DataFrame({
    'Batch_ID': ['B001', 'B001', 'B001', 'B002', 'B002', 'B002', 'B003', 'B003', 'B003'],
    'Time_h': [0, 2, 4, 0, 2, 4, 0, 2, 4],
    'Temp_C': [25, 80, 85, 25, 82, 87, 25, 78, 83],
    'Conversion': [0, 65, 88, 0, 68, 90, 0, 62, 85]
})

print("批次反應器數據：")
print(batch_data)
print()

# 計算每批次的平均轉化率
batch_avg = batch_data.groupby('Batch_ID')['Conversion'].mean()
print("各批次平均轉化率：")
print(batch_avg)
print()

# 計算每批次的最終轉化率
final_conversion = batch_data.groupby('Batch_ID')['Conversion'].last()
print("各批次最終轉化率：")
print(final_conversion)
print()

# 計算轉化率增長速率
batch_data['Conv_rate'] = batch_data.groupby('Batch_ID')['Conversion'].diff() / batch_data.groupby('Batch_ID')['Time_h'].diff()
print("轉化率變化速率：")
print(batch_data)
```

### 8.2 案例二：製程監控數據清理

```python
# 建立包含異常值的製程數據
process_raw = pd.DataFrame({
    'Time': pd.date_range('2024-01-01', periods=20, freq='h'),
    'Temperature': [350, 355, 360, 999, 358, 362, 365, -99, 368, 370,
                   372, 375, 378, 380, 999, 382, 385, 388, 390, 392],
    'Pressure': [2.0, 2.1, np.nan, 2.2, 2.15, 2.25, np.nan, 2.3, 2.35, 2.4,
                2.45, np.nan, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85]
})

print("原始數據（包含異常值與缺失值）：")
print(process_raw)
print()

# 步驟 1：識別並移除異常值
temp_mask = (process_raw['Temperature'] > 300) & (process_raw['Temperature'] < 500)
process_clean = process_raw[temp_mask].copy()
print("移除溫度異常值後：")
print(process_clean)
print()

# 步驟 2：處理缺失值（使用線性插值）
process_clean['Pressure'] = process_clean['Pressure'].interpolate(method='linear')
print("插值處理缺失值後：")
print(process_clean)
print()

# 步驟 3：計算移動平均（平滑數據）
process_clean['Temp_MA'] = process_clean['Temperature'].rolling(window=3, center=True).mean()
process_clean['Press_MA'] = process_clean['Pressure'].rolling(window=3, center=True).mean()
print("計算移動平均後：")
print(process_clean[['Time', 'Temperature', 'Temp_MA', 'Pressure', 'Press_MA']])
```

### 8.3 案例三：多反應器性能比較

```python
# 建立多反應器數據
reactor_data = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=10),
    'R1_Temp': np.random.uniform(350, 360, 10),
    'R1_Conv': np.random.uniform(75, 85, 10),
    'R2_Temp': np.random.uniform(355, 365, 10),
    'R2_Conv': np.random.uniform(78, 88, 10),
    'R3_Temp': np.random.uniform(345, 355, 10),
    'R3_Conv': np.random.uniform(70, 80, 10)
})

print("多反應器數據：")
print(reactor_data.head())
print()

# 重塑數據為長格式（Tidy Data）
reactor_long = pd.melt(
    reactor_data,
    id_vars=['Date'],
    value_vars=['R1_Conv', 'R2_Conv', 'R3_Conv'],
    var_name='Reactor',
    value_name='Conversion'
)
reactor_long['Reactor'] = reactor_long['Reactor'].str.replace('_Conv', '')

print("長格式數據：")
print(reactor_long.head())
print()

# 計算各反應器統計摘要
reactor_stats = reactor_long.groupby('Reactor')['Conversion'].agg([
    ('Average', 'mean'),
    ('StdDev', 'std'),
    ('Min', 'min'),
    ('Max', 'max')
]).round(2)

print("各反應器性能統計：")
print(reactor_stats)
```

---

## 9. 總結與最佳實踐

### 9.1 Pandas 核心概念回顧

1. **Series 與 DataFrame**：Pandas 的兩種核心資料結構
2. **索引系統**：強大的標籤索引，支援 loc 和 iloc
3. **資料讀寫**：支援 CSV、Excel、JSON、SQL 等多種格式
4. **資料清理**：處理缺失值、重複值、異常值
5. **時間序列**：完整的日期時間處理與重採樣功能
6. **資料轉換**：合併、分組、透視表等進階操作

### 9.2 Pandas 使用最佳實踐

```python
# 1. 總是檢查資料概況
df.info()
df.describe()
df.head()

# 2. 使用向量化運算取代迴圈
# 避免：
# for i in range(len(df)):
#     df.loc[i, 'new_col'] = df.loc[i, 'col1'] * df.loc[i, 'col2']

# 推薦：
df['new_col'] = df['col1'] * df['col2']

# 3. 使用 copy() 避免意外修改原始資料
df_subset = df[df['value'] > 0].copy()

# 4. 鏈式方法調用提高可讀性
result = (df
    .dropna()
    .query('Temperature > 350')
    .groupby('Catalyst')['Conversion']
    .mean()
    .sort_values(ascending=False)
)

# 5. 使用 inplace=False（預設）保持數據不可變性
df_clean = df.dropna()  # 回傳新 DataFrame
# 避免：df.dropna(inplace=True)  # 修改原始 DataFrame
```

### 9.3 常見陷阱與解決方案

| 陷阱 | 問題 | 解決方案 |
|------|------|----------|
| **SettingWithCopyWarning** | 在 DataFrame 切片上賦值 | 使用 `.copy()` 或 `.loc[]` |
| **索引未對齊** | 運算時索引不匹配 | 使用 `.reset_index()` 或明確對齊 |
| **記憶體溢出** | 處理大型資料集 | 使用 `chunksize` 分批讀取或 `dtype` 優化 |
| **混用 loc 與 iloc** | 標籤與位置索引混淆 | 明確使用 `loc` 或 `iloc` |
| **忽略缺失值** | NaN 影響運算結果 | 明確處理缺失值（dropna、fillna） |

### 9.4 延伸學習資源

- **官方文檔**：[pandas.pydata.org](https://pandas.pydata.org/)
- **教學書籍**：《Python for Data Analysis》by Wes McKinney
- **實戰練習**：Kaggle 資料集、化工製程數據
- **進階主題**：多層索引、自訂函數應用、效能優化

---

## 10. 練習題

### 練習 1：基本操作

建立一個 DataFrame 記錄 5 天的反應器操作數據，包含日期、溫度、壓力、轉化率，並完成以下任務：
1. 計算平均溫度和轉化率
2. 找出轉化率最高的那天
3. 新增一欄「效率」= 轉化率 / 溫度

### 練習 2：資料清理

給定包含缺失值和異常值的製程數據，完成以下任務：
1. 識別缺失值的數量和位置
2. 使用適當方法填補缺失值
3. 移除異常值（定義：超出平均值 ± 3 個標準差）

### 練習 3：時間序列分析

建立 48 小時的時間序列溫度數據，完成以下任務：
1. 計算 6 小時移動平均
2. 找出溫度變化最劇烈的時段（最大差分值）
3. 將資料重採樣為 4 小時間隔

### 練習 4：分組分析

給定多批次實驗數據（包含批次編號、催化劑類型、溫度、轉化率），完成以下任務：
1. 計算每種催化劑的平均轉化率
2. 找出每種催化劑的最佳操作溫度
3. 建立催化劑 × 溫度的透視表

---

## 11. 參考資料

- McKinney, W. (2022). *Python for Data Analysis, 3rd Edition*. O'Reilly Media.
- Pandas Official Documentation: https://pandas.pydata.org/docs/
- Pandas API Reference: https://pandas.pydata.org/docs/reference/index.html
- Real Python Pandas Tutorials: https://realpython.com/learning-paths/pandas-data-science/
- Kaggle Learn - Pandas: https://www.kaggle.com/learn/pandas

---

**下一單元預告**：在 Unit04 中，我們將學習使用 Matplotlib 與 Seaborn 進行資料視覺化，將 Pandas 處理後的資料繪製成各種專業圖表。

---

**課程資訊**
- 課程名稱：AI在化工上之應用
- 課程單元：Unit03 Pandas資料處理與分析
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-01-28

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---