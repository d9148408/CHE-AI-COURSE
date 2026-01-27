# Unit04 Matplotlib 資料視覺化

## 學習目標

完成本單元後，學生將能夠：
- 理解 Matplotlib 的基本架構與繪圖邏輯
- 掌握常用的基本圖表類型（折線圖、長條圖、散佈圖等）
- 熟悉圖表自訂功能（標題、標籤、圖例、網格等）
- 學會多圖表佈局與子圖設計
- 應用 Matplotlib 於化工領域的數據視覺化

---

## 1. Matplotlib 簡介

### 1.1 什麼是 Matplotlib？

Matplotlib 是 Python 中最廣泛使用的資料視覺化套件，由 John Hunter 於 2003 年開發。它提供了豐富的繪圖功能，可以繪製各種靜態、動態、互動式圖表。

**主要特點：**
- **功能強大**：支援各種圖表類型（折線圖、長條圖、散佈圖、直方圖、圓餅圖等）
- **高度自訂**：可精細控制圖表的每個元素（顏色、線型、標記、字體等）
- **跨平台**：支援多種輸出格式（PNG、PDF、SVG、EPS 等）
- **整合性佳**：與 Numpy、Pandas 等套件無縫整合

### 1.2 Matplotlib 的架構

Matplotlib 採用階層式架構，主要包含三個層次：

1. **Backend Layer（後端層）**：負責實際的繪圖與輸出
2. **Artist Layer（藝術家層）**：處理所有圖表元素（線條、文字、圖例等）
3. **Scripting Layer（腳本層）**：提供簡單的繪圖介面（pyplot）

**兩種繪圖介面：**
- **pyplot 介面**（狀態機風格）：適合快速繪圖與互動式分析
- **物件導向介面**：適合進階自訂與複雜佈局

### 1.3 在化工領域的應用

Matplotlib 在化工領域有廣泛應用：
- 製程監控數據視覺化（溫度、壓力、流量趨勢圖）
- 實驗數據分析（反應動力學曲線、相平衡圖）
- 模型結果呈現（擬合曲線、預測結果比較）
- 品質管制圖表（控制圖、柏拉圖）
- 設備性能監測（效率分析、能耗分析）

---

## 2. 安裝與匯入

### 2.1 安裝 Matplotlib

```bash
# 使用 pip 安裝
pip install matplotlib

# 使用 conda 安裝
conda install matplotlib
```

### 2.2 基本匯入方式

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 在 Jupyter Notebook 中顯示圖表
%matplotlib inline
```

**常見匯入慣例：**
- `matplotlib.pyplot` 通常簡寫為 `plt`
- 配合 Numpy 與 Pandas 使用

---

## 3. 基本繪圖流程

### 3.1 最簡單的折線圖

```python
import matplotlib.pyplot as plt

# 準備數據
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# 繪製圖表
plt.plot(x, y)
plt.show()
```

### 3.2 完整的繪圖流程

```python
import matplotlib.pyplot as plt
import numpy as np

# 1. 準備數據
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 2. 建立圖表
plt.figure(figsize=(8, 5))

# 3. 繪製數據
plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)

# 4. 添加標題與標籤（必須使用英文）
plt.title('Sine Wave Function', fontsize=14, fontweight='bold')
plt.xlabel('X-axis', fontsize=12)
plt.ylabel('Y-axis', fontsize=12)

# 5. 添加圖例與網格
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 6. 顯示圖表
plt.tight_layout()
plt.show()
```

**重要提醒：** 根據專案規範，所有圖表的標題、軸標籤必須使用英文，以確保相容性與清晰度。

---

## 4. 常用圖表類型

### 4.1 折線圖 (Line Plot)

折線圖適用於顯示數據隨時間或連續變數的變化趨勢。

**基本用法：**

```python
import matplotlib.pyplot as plt
import numpy as np

# 準備數據
time = np.linspace(0, 24, 100)
temperature = 25 + 5 * np.sin(2 * np.pi * time / 24)

# 繪製折線圖
plt.figure(figsize=(10, 5))
plt.plot(time, temperature, color='red', linewidth=2, linestyle='-', marker='o', markersize=3)

plt.title('Temperature Variation Over 24 Hours', fontsize=14)
plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**線條樣式參數：**
- `color`：顏色（'red', 'blue', '#FF5733' 等）
- `linewidth` 或 `lw`：線寬
- `linestyle` 或 `ls`：線型（'-', '--', '-.', ':' 等）
- `marker`：標記樣式（'o', 's', '^', 'D' 等）
- `markersize` 或 `ms`：標記大小

**化工應用範例：反應器溫度監控**

```python
import matplotlib.pyplot as plt
import numpy as np

# 模擬反應器溫度數據
time = np.linspace(0, 8, 100)  # 8 hours
reactor_temp = 80 + 20 * np.exp(-0.5 * time) + np.random.normal(0, 1, 100)

plt.figure(figsize=(10, 6))
plt.plot(time, reactor_temp, color='orangered', linewidth=1.5, label='Reactor Temperature')
plt.axhline(y=80, color='green', linestyle='--', linewidth=2, label='Target Temperature')
plt.fill_between(time, 75, 85, color='green', alpha=0.1, label='Safe Range')

plt.title('Reactor Temperature Monitoring', fontsize=14, fontweight='bold')
plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.legend(loc='best')
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()
```

### 4.2 散佈圖 (Scatter Plot)

散佈圖用於顯示兩個變數之間的關係，適合探索相關性。

**基本用法：**

```python
import matplotlib.pyplot as plt
import numpy as np

# 準備數據
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5

# 繪製散佈圖
plt.figure(figsize=(8, 6))
plt.scatter(x, y, c='blue', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

plt.title('Scatter Plot Example', fontsize=14)
plt.xlabel('Variable X', fontsize=12)
plt.ylabel('Variable Y', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**參數說明：**
- `c`：顏色（可以是單一顏色或數組）
- `s`：點的大小
- `alpha`：透明度（0-1）
- `edgecolors`：邊框顏色
- `linewidth`：邊框寬度

**化工應用範例：產品品質與製程參數關係**

```python
import matplotlib.pyplot as plt
import numpy as np

# 模擬數據：反應溫度 vs 產率
np.random.seed(42)
temperature = np.random.uniform(60, 100, 50)
yield_rate = 30 + 0.8 * temperature + np.random.normal(0, 5, 50)

# 根據產率分組著色
colors = ['red' if y < 70 else 'yellow' if y < 85 else 'green' for y in yield_rate]

plt.figure(figsize=(10, 6))
plt.scatter(temperature, yield_rate, c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=1)

# 添加趨勢線
z = np.polyfit(temperature, yield_rate, 1)
p = np.poly1d(z)
plt.plot(temperature, p(temperature), "b--", linewidth=2, label='Trend Line')

plt.title('Reaction Temperature vs Yield Rate', fontsize=14, fontweight='bold')
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Yield Rate (%)', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()
```

### 4.3 長條圖 (Bar Chart)

長條圖用於比較不同類別的數值。

**基本用法：**

```python
import matplotlib.pyplot as plt

# 準備數據
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

# 繪製長條圖
plt.figure(figsize=(8, 6))
plt.bar(categories, values, color='steelblue', alpha=0.8, edgecolor='black')

plt.title('Bar Chart Example', fontsize=14)
plt.xlabel('Categories', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```

**水平長條圖：**

```python
plt.barh(categories, values, color='coral', alpha=0.8, edgecolor='black')
```

**化工應用範例：不同催化劑效能比較**

```python
import matplotlib.pyplot as plt
import numpy as np

# 催化劑效能數據
catalysts = ['Cat-A', 'Cat-B', 'Cat-C', 'Cat-D', 'Cat-E']
conversion_rate = [78, 85, 92, 88, 81]
selectivity = [82, 88, 90, 85, 83]

x = np.arange(len(catalysts))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, conversion_rate, width, label='Conversion Rate', color='skyblue', edgecolor='black')
bars2 = ax.bar(x + width/2, selectivity, width, label='Selectivity', color='lightcoral', edgecolor='black')

# 在長條上添加數值標籤
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=9)

ax.set_title('Catalyst Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xlabel('Catalyst Type', fontsize=12)
ax.set_ylabel('Performance (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(catalysts)
ax.legend()
ax.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()
```

### 4.4 直方圖 (Histogram)

直方圖用於顯示數據的分布情況。

**基本用法：**

```python
import matplotlib.pyplot as plt
import numpy as np

# 生成隨機數據
data = np.random.randn(1000)

# 繪製直方圖
plt.figure(figsize=(8, 6))
plt.hist(data, bins=30, color='teal', alpha=0.7, edgecolor='black')

plt.title('Histogram Example', fontsize=14)
plt.xlabel('Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```

**化工應用範例：產品粒徑分布**

```python
import matplotlib.pyplot as plt
import numpy as np

# 模擬產品粒徑數據（微米）
np.random.seed(42)
particle_size = np.random.lognormal(3, 0.5, 1000)

plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(particle_size, bins=50, color='gold', alpha=0.7, edgecolor='black')

# 添加統計線
mean_size = np.mean(particle_size)
median_size = np.median(particle_size)
plt.axvline(mean_size, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_size:.1f} μm')
plt.axvline(median_size, color='blue', linestyle='--', linewidth=2, label=f'Median = {median_size:.1f} μm')

plt.title('Product Particle Size Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Particle Size (μm)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()
```

### 4.5 圓餅圖 (Pie Chart)

圓餅圖用於顯示各部分佔整體的比例。

**基本用法：**

```python
import matplotlib.pyplot as plt

# 準備數據
labels = ['Category A', 'Category B', 'Category C', 'Category D']
sizes = [30, 25, 20, 25]
colors = ['gold', 'lightblue', 'lightcoral', 'lightgreen']
explode = (0.1, 0, 0, 0)  # 突出第一塊

# 繪製圓餅圖
plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)

plt.title('Pie Chart Example', fontsize=14)
plt.axis('equal')  # 確保圓形
plt.tight_layout()
plt.show()
```

**化工應用範例：原料成本結構**

```python
import matplotlib.pyplot as plt

# 原料成本數據
materials = ['Raw Material A', 'Raw Material B', 'Catalyst', 'Solvent', 'Others']
costs = [35, 28, 15, 12, 10]
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']

plt.figure(figsize=(10, 8))
wedges, texts, autotexts = plt.pie(costs, labels=materials, colors=colors, autopct='%1.1f%%',
                                     startangle=140, pctdistance=0.85)

# 美化文字
for text in texts:
    text.set_fontsize(11)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(10)
    autotext.set_fontweight('bold')

plt.title('Production Cost Structure', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.show()
```

---

## 5. 圖表自訂與美化

### 5.1 標題、標籤與圖例

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)', linewidth=2, color='blue')
plt.plot(x, y2, label='cos(x)', linewidth=2, color='red')

# 標題設定
plt.title('Trigonometric Functions', fontsize=16, fontweight='bold', pad=20)

# 軸標籤設定
plt.xlabel('X-axis (radians)', fontsize=13, fontweight='bold')
plt.ylabel('Y-axis (amplitude)', fontsize=13, fontweight='bold')

# 圖例設定
plt.legend(loc='upper right', fontsize=11, frameon=True, shadow=True, fancybox=True)

# 網格設定
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
```

**圖例位置參數：**
- `'upper right'`, `'upper left'`, `'lower right'`, `'lower left'`
- `'center'`, `'center left'`, `'center right'`
- `'best'`（自動選擇最佳位置）

### 5.2 顏色與樣式

**常用顏色表示法：**

```python
# 1. 顏色名稱
plt.plot(x, y, color='red')

# 2. 簡寫代碼
plt.plot(x, y, color='r')  # r, g, b, c, m, y, k, w

# 3. 十六進位碼
plt.plot(x, y, color='#FF5733')

# 4. RGB 元組
plt.plot(x, y, color=(0.1, 0.2, 0.5))

# 5. 內建色彩映射
plt.plot(x, y, color='tab:blue')  # tab:blue, tab:orange, tab:green 等
```

**線條樣式：**

```python
plt.plot(x, y, linestyle='-')   # 實線
plt.plot(x, y, linestyle='--')  # 虛線
plt.plot(x, y, linestyle='-.')  # 點劃線
plt.plot(x, y, linestyle=':')   # 點線
```

### 5.3 軸範圍與刻度

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2)

# 設定軸範圍
plt.xlim(0, 10)
plt.ylim(-1.5, 1.5)

# 設定刻度
plt.xticks(np.arange(0, 11, 1))
plt.yticks(np.arange(-1.5, 2, 0.5))

# 自訂刻度標籤
plt.xticks([0, np.pi, 2*np.pi, 3*np.pi], ['0', 'π', '2π', '3π'])

plt.title('Customized Axis Range and Ticks', fontsize=14)
plt.xlabel('X-axis', fontsize=12)
plt.ylabel('Y-axis', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 5.4 圖表尺寸與解析度

```python
# 設定圖表尺寸（單位：英吋）
plt.figure(figsize=(12, 6))

# 儲存高解析度圖片
plt.savefig('output.png', dpi=300, bbox_inches='tight')
```

**儲存格式：**
- PNG：`plt.savefig('figure.png')`
- PDF：`plt.savefig('figure.pdf')`
- SVG：`plt.savefig('figure.svg')`
- JPG：`plt.savefig('figure.jpg')`

---

## 6. 多圖表佈局

### 6.1 使用 subplot

**基本用法：**

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 第一個子圖
axes[0, 0].plot(x, np.sin(x), 'r-')
axes[0, 0].set_title('sin(x)')
axes[0, 0].grid(True, alpha=0.3)

# 第二個子圖
axes[0, 1].plot(x, np.cos(x), 'b-')
axes[0, 1].set_title('cos(x)')
axes[0, 1].grid(True, alpha=0.3)

# 第三個子圖
axes[1, 0].plot(x, np.tan(x), 'g-')
axes[1, 0].set_title('tan(x)')
axes[1, 0].set_ylim(-5, 5)
axes[1, 0].grid(True, alpha=0.3)

# 第四個子圖
axes[1, 1].plot(x, np.exp(-x/5), 'm-')
axes[1, 1].set_title('exp(-x/5)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 6.2 化工應用範例：反應器多參數監控

```python
import matplotlib.pyplot as plt
import numpy as np

# 生成模擬數據
time = np.linspace(0, 24, 200)
temperature = 80 + 5 * np.sin(2 * np.pi * time / 12) + np.random.normal(0, 0.5, 200)
pressure = 2.5 + 0.3 * np.sin(2 * np.pi * time / 12 + 1) + np.random.normal(0, 0.05, 200)
flow_rate = 100 + 10 * np.sin(2 * np.pi * time / 8) + np.random.normal(0, 1, 200)
concentration = 0.8 + 0.1 * np.exp(-time/10) + np.random.normal(0, 0.01, 200)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 溫度
axes[0, 0].plot(time, temperature, color='orangered', linewidth=1.5)
axes[0, 0].axhline(y=80, color='green', linestyle='--', linewidth=2, alpha=0.7)
axes[0, 0].fill_between(time, 75, 85, color='green', alpha=0.1)
axes[0, 0].set_title('Temperature Monitoring', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Time (hours)', fontsize=11)
axes[0, 0].set_ylabel('Temperature (°C)', fontsize=11)
axes[0, 0].grid(True, linestyle=':', alpha=0.5)

# 壓力
axes[0, 1].plot(time, pressure, color='steelblue', linewidth=1.5)
axes[0, 1].axhline(y=2.5, color='green', linestyle='--', linewidth=2, alpha=0.7)
axes[0, 1].fill_between(time, 2.2, 2.8, color='green', alpha=0.1)
axes[0, 1].set_title('Pressure Monitoring', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Time (hours)', fontsize=11)
axes[0, 1].set_ylabel('Pressure (bar)', fontsize=11)
axes[0, 1].grid(True, linestyle=':', alpha=0.5)

# 流量
axes[1, 0].plot(time, flow_rate, color='forestgreen', linewidth=1.5)
axes[1, 0].axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.7)
axes[1, 0].fill_between(time, 90, 110, color='yellow', alpha=0.1)
axes[1, 0].set_title('Flow Rate Monitoring', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Time (hours)', fontsize=11)
axes[1, 0].set_ylabel('Flow Rate (L/min)', fontsize=11)
axes[1, 0].grid(True, linestyle=':', alpha=0.5)

# 濃度
axes[1, 1].plot(time, concentration, color='purple', linewidth=1.5)
axes[1, 1].axhline(y=0.8, color='orange', linestyle='--', linewidth=2, alpha=0.7)
axes[1, 1].set_title('Concentration Monitoring', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Time (hours)', fontsize=11)
axes[1, 1].set_ylabel('Concentration (mol/L)', fontsize=11)
axes[1, 1].grid(True, linestyle=':', alpha=0.5)

fig.suptitle('Reactor Multi-Parameter Monitoring Dashboard', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()
```

### 6.3 不規則佈局

```python
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(12, 8))

# 使用 subplot2grid 建立不規則佈局
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
ax4 = plt.subplot2grid((3, 3), (2, 0))
ax5 = plt.subplot2grid((3, 3), (2, 1))

x = np.linspace(0, 10, 100)

ax1.plot(x, np.sin(x), 'b-')
ax1.set_title('Main Plot', fontsize=12)
ax1.grid(True, alpha=0.3)

ax2.plot(x, np.cos(x), 'r-')
ax2.set_title('Secondary Plot', fontsize=12)
ax2.grid(True, alpha=0.3)

ax3.plot(x, np.exp(-x/5), 'g-')
ax3.set_title('Side Plot', fontsize=12)
ax3.grid(True, alpha=0.3)

ax4.hist(np.random.randn(1000), bins=30, color='orange', alpha=0.7)
ax4.set_title('Histogram', fontsize=12)

ax5.scatter(np.random.randn(100), np.random.randn(100), alpha=0.6)
ax5.set_title('Scatter', fontsize=12)
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 7. 進階應用：雙 Y 軸圖表

在化工製程中，常需要在同一時間軸上顯示不同單位或數量級的參數，這時雙 Y 軸圖表非常有用。

```python
import matplotlib.pyplot as plt
import numpy as np

# 生成模擬數據
time = np.linspace(0, 24, 100)
temperature = 80 + 10 * np.sin(2 * np.pi * time / 12)
conversion = 50 + 30 * (1 - np.exp(-time/5))

fig, ax1 = plt.subplots(figsize=(12, 6))

# 第一個 Y 軸（溫度）
color = 'tab:red'
ax1.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Temperature (°C)', color=color, fontsize=12, fontweight='bold')
ax1.plot(time, temperature, color=color, linewidth=2, label='Temperature')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle=':', alpha=0.5)

# 第二個 Y 軸（轉化率）
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Conversion Rate (%)', color=color, fontsize=12, fontweight='bold')
ax2.plot(time, conversion, color=color, linewidth=2, linestyle='--', label='Conversion')
ax2.tick_params(axis='y', labelcolor=color)

# 標題與圖例
fig.suptitle('Reactor Temperature and Conversion Rate', fontsize=14, fontweight='bold')

# 合併圖例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.show()
```

---

## 8. 圖表儲存與輸出

### 8.1 基本儲存

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2)
plt.title('Sample Plot', fontsize=14)
plt.xlabel('X-axis', fontsize=12)
plt.ylabel('Y-axis', fontsize=12)
plt.grid(True, alpha=0.3)

# 儲存圖表
plt.savefig('sample_plot.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 8.2 儲存參數說明

```python
plt.savefig('output.png',
            dpi=300,                    # 解析度（每英吋點數）
            bbox_inches='tight',        # 自動裁切空白邊距
            pad_inches=0.1,             # 邊距大小
            transparent=False,          # 透明背景
            facecolor='white',          # 背景顏色
            edgecolor='none')           # 邊框顏色
```

**建議設定：**
- **螢幕顯示**：dpi=100
- **簡報投影**：dpi=150
- **論文發表**：dpi=300
- **印刷品質**：dpi=600

---

## 9. 常見錯誤與除錯

### 9.1 中文顯示問題

**問題：** Matplotlib 預設不支援中文字體，會顯示方框。

**解決方案（僅供理解，本課程使用英文標籤）：**

```python
import matplotlib.pyplot as plt
import matplotlib as mpl

# 設定中文字體（僅作範例說明）
mpl.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 微軟正黑體
mpl.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
```

**本課程規範：** 所有圖表標題、軸標籤必須使用英文，無需處理中文顯示問題。

### 9.2 圖表重疊問題

**問題：** 多個子圖或標籤重疊。

**解決方案：**

```python
# 使用 tight_layout 自動調整
plt.tight_layout()

# 或手動調整
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)
```

### 9.3 圖表不顯示

**問題：** 程式執行後沒有顯示圖表。

**解決方案：**

```python
# 確保加上 show()
plt.show()

# 在 Jupyter Notebook 中使用
%matplotlib inline
```

### 9.4 記憶體問題

**問題：** 繪製大量圖表後記憶體不足。

**解決方案：**

```python
# 關閉圖表釋放記憶體
plt.close()

# 關閉所有圖表
plt.close('all')
```

---

## 10. 最佳實踐建議

### 10.1 圖表設計原則

1. **簡潔明瞭**：避免過多裝飾，突出重點數據
2. **適當配色**：使用對比明顯但不刺眼的顏色
3. **清晰標註**：確保標題、標籤、圖例完整且易讀
4. **合理比例**：選擇適當的圖表尺寸與軸範圍
5. **資訊完整**：包含必要的單位、標籤與說明

### 10.2 化工領域圖表規範

1. **單位標示**：必須標註所有物理量的單位
2. **安全範圍**：標示操作安全範圍或目標值
3. **時間軸**：製程數據通常使用時間為 X 軸
4. **數據品質**：標示數據來源與測量精度
5. **趨勢線**：適時添加趨勢線或參考線

### 10.3 程式碼組織建議

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_reactor_temperature(time, temperature, save_path=None):
    """
    繪製反應器溫度監控圖
    
    Parameters:
    -----------
    time : array-like
        時間數據（小時）
    temperature : array-like
        溫度數據（攝氏度）
    save_path : str, optional
        圖表儲存路徑
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time, temperature, color='orangered', linewidth=2, label='Reactor Temp')
    plt.axhline(y=80, color='green', linestyle='--', linewidth=2, label='Target Temp')
    plt.fill_between(time, 75, 85, color='green', alpha=0.1, label='Safe Range')
    
    plt.title('Reactor Temperature Monitoring', fontsize=14, fontweight='bold')
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# 使用函數
time = np.linspace(0, 8, 100)
temp = 80 + 10 * np.exp(-0.5 * time) + np.random.normal(0, 1, 100)
plot_reactor_temperature(time, temp, save_path='reactor_temp.png')
```

---

## 11. 小結

本單元介紹了 Matplotlib 的基本概念與常用功能：

1. **基本架構**：理解 Matplotlib 的層次結構與繪圖邏輯
2. **常用圖表**：掌握折線圖、散佈圖、長條圖、直方圖、圓餅圖等基本圖表
3. **圖表自訂**：學習標題、標籤、圖例、顏色、樣式等自訂方法
4. **多圖佈局**：能夠使用 subplot 建立多圖表佈局
5. **進階應用**：掌握雙 Y 軸、不規則佈局等進階技巧
6. **化工應用**：了解如何將 Matplotlib 應用於化工領域的數據視覺化

### 學習要點

- **所有標籤必須使用英文**：遵循專案規範，確保相容性
- **適當選擇圖表類型**：根據數據特性選擇最合適的圖表
- **重視圖表美化**：專業的圖表能更有效傳達資訊
- **實際應用練習**：多練習化工領域的實際案例

### 下一步

- 完成 Unit04_Matplotlib.ipynb 程式演練
- 練習 Unit04_Matplotlib_Homework.ipynb 作業
- 學習 Unit04_Seaborn 統計視覺化進階內容

---

## 12. 參考資源

### 官方文件
- [Matplotlib 官方文件](https://matplotlib.org/stable/contents.html)
- [Matplotlib 圖表範例庫](https://matplotlib.org/stable/gallery/index.html)
- [Matplotlib 教學](https://matplotlib.org/stable/tutorials/index.html)

### 推薦閱讀
- *Python Data Science Handbook* by Jake VanderPlas
- *Matplotlib for Python Developers* by Sandro Tosi

### 線上資源
- [Real Python - Matplotlib Guide](https://realpython.com/python-matplotlib-guide/)
- [DataCamp - Matplotlib Tutorial](https://www.datacamp.com/tutorial/matplotlib-tutorial-python)

---

**課程製作：** 逢甲大學化學工程學系  
**授課教師：** 莊曜禎 助理教授  
**最後更新：** 2026年1月
