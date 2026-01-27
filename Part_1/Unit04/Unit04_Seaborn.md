# Unit04 Seaborn 統計資料視覺化

## 學習目標

完成本單元後，學生將能夠：
- 理解 Seaborn 的設計理念與優勢
- 掌握 Seaborn 與 Matplotlib 的關係與整合
- 熟悉常用的統計視覺化圖表（分佈圖、類別圖、關係圖、熱力圖等）
- 學會使用主題樣式與調色盤美化圖表
- 應用 Seaborn 於化工領域的統計分析與視覺化

---

## 1. Seaborn 簡介

### 1.1 什麼是 Seaborn？

Seaborn 是基於 Matplotlib 建立的高階統計視覺化套件，由 Michael Waskom 開發。它提供了更簡潔的語法和更美觀的預設樣式，特別適合探索性數據分析和統計視覺化。

**主要特點：**
- **高階介面**：用更少的程式碼建立複雜的統計圖表
- **美觀設計**：內建多種精美主題與調色盤
- **統計整合**：自動進行統計計算與視覺化（如迴歸線、信賴區間等）
- **Pandas 整合**：直接支援 DataFrame，語法更直觀
- **分面繪圖**：輕鬆建立多維度數據的子圖矩陣

### 1.2 Seaborn vs Matplotlib

| 特性 | Matplotlib | Seaborn |
|------|-----------|---------|
| **抽象層次** | 低階，需要更多程式碼 | 高階，語法簡潔 |
| **預設樣式** | 較基本 | 現代化、美觀 |
| **統計功能** | 需手動計算 | 內建統計視覺化 |
| **數據格式** | 主要使用陣列 | 原生支援 DataFrame |
| **適用場景** | 精細控制、客製化 | 快速探索、統計分析 |

**關係：** Seaborn 是建立在 Matplotlib 之上的，所有 Matplotlib 的功能都可以在 Seaborn 圖表中使用。

### 1.3 在化工領域的應用

Seaborn 在化工領域的統計分析中非常實用：
- **實驗數據分布分析**：快速檢視數據分布、離群值
- **製程參數關係探索**：視覺化多變數之間的相關性
- **品質控制統計**：批次數據的統計比較
- **實驗設計結果分析**：多因子實驗的視覺化
- **製程優化探索**：操作條件與產出的統計關係

---

## 2. 安裝與基本設定

### 2.1 安裝 Seaborn

```bash
# 使用 pip 安裝
pip install seaborn

# 使用 conda 安裝
conda install seaborn
```

### 2.2 基本匯入與設定

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 設定 Seaborn 樣式
sns.set_theme()  # 使用預設主題

# 或指定特定樣式
sns.set_style("whitegrid")  # 白色網格背景

# 設定圖表大小
sns.set_context("notebook")  # 適合 Jupyter Notebook

# 在 Jupyter Notebook 中顯示圖表
%matplotlib inline
```

### 2.3 Seaborn 樣式系統

**五種內建樣式：**
1. **darkgrid**：深色網格（預設）
2. **whitegrid**：白色網格
3. **dark**：深色背景無網格
4. **white**：白色背景無網格
5. **ticks**：有刻度標記的白色背景

**四種繪圖環境：**
1. **paper**：適合論文發表
2. **notebook**：適合 Jupyter Notebook（預設）
3. **talk**：適合演講投影片
4. **poster**：適合海報展示

```python
# 組合使用
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.2)
```

---

## 3. Seaborn 圖表分類

Seaborn 的圖表可以分為以下幾大類：

### 3.1 關係圖 (Relational Plots)
- `scatterplot()`：散佈圖
- `lineplot()`：折線圖
- `relplot()`：關係圖的通用介面

### 3.2 分佈圖 (Distribution Plots)
- `histplot()`：直方圖
- `kdeplot()`：核密度估計圖
- `ecdfplot()`：經驗累積分佈圖
- `rugplot()`：地毯圖
- `displot()`：分佈圖的通用介面

### 3.3 類別圖 (Categorical Plots)
- `stripplot()`：散點分類圖
- `swarmplot()`：蜂群圖
- `boxplot()`：箱型圖
- `violinplot()`：小提琴圖
- `barplot()`：長條圖
- `pointplot()`：點圖
- `catplot()`：類別圖的通用介面

### 3.4 回歸圖 (Regression Plots)
- `regplot()`：回歸圖
- `lmplot()`：線性模型圖

### 3.5 矩陣圖 (Matrix Plots)
- `heatmap()`：熱力圖
- `clustermap()`：階層式聚類熱力圖

### 3.6 多圖網格 (Multi-plot Grids)
- `FacetGrid`：分面網格
- `PairGrid`：配對網格
- `JointGrid`：聯合分佈網格

---

## 4. 分佈圖 (Distribution Plots)

分佈圖用於探索單變數或多變數的數據分佈特性。

### 4.1 直方圖 (Histogram)

直方圖顯示數據在不同區間的頻率分佈。

**基本用法：**

```python
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 產生模擬數據
np.random.seed(42)
data = np.random.normal(100, 15, 1000)

# 繪製直方圖
plt.figure(figsize=(10, 6))
sns.histplot(data, bins=30, kde=True, color='steelblue')

plt.title('Distribution of Data', fontsize=14, fontweight='bold')
plt.xlabel('Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.show()
```

**參數說明：**
- `bins`：區間數量
- `kde`：是否顯示核密度估計曲線
- `color`：顏色
- `stat`：統計類型（'count', 'frequency', 'density', 'probability'）

**化工應用範例：產品純度分佈分析**

```python
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 模擬產品純度數據（百分比）
np.random.seed(42)
purity_batch1 = np.random.normal(98.5, 0.8, 150)
purity_batch2 = np.random.normal(97.8, 1.2, 150)

plt.figure(figsize=(12, 6))

# 繪製雙直方圖比較
sns.histplot(purity_batch1, bins=25, kde=True, color='skyblue', label='Batch 1', alpha=0.6)
sns.histplot(purity_batch2, bins=25, kde=True, color='salmon', label='Batch 2', alpha=0.6)

plt.axvline(x=98.0, color='green', linestyle='--', linewidth=2, label='Specification Limit')
plt.title('Product Purity Distribution Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Purity (%)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()
```

### 4.2 核密度估計圖 (KDE Plot)

KDE 圖顯示數據的平滑機率密度估計。

**基本用法：**

```python
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 產生模擬數據
np.random.seed(42)
data = np.random.gamma(2, 2, 1000)

# 繪製 KDE 圖
plt.figure(figsize=(10, 6))
sns.kdeplot(data, fill=True, color='purple', alpha=0.6)

plt.title('Kernel Density Estimation', fontsize=14, fontweight='bold')
plt.xlabel('Value', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.tight_layout()
plt.show()
```

**多維度 KDE：**

```python
# 二維 KDE
x = np.random.normal(0, 1, 1000)
y = np.random.normal(0, 1, 1000)

plt.figure(figsize=(10, 8))
sns.kdeplot(x=x, y=y, cmap='Blues', fill=True, levels=10)

plt.title('2D Kernel Density Estimation', fontsize=14, fontweight='bold')
plt.xlabel('X Variable', fontsize=12)
plt.ylabel('Y Variable', fontsize=12)
plt.tight_layout()
plt.show()
```

### 4.3 箱型圖 (Box Plot)

箱型圖顯示數據的五數概括（最小值、第一四分位數、中位數、第三四分位數、最大值）。

**基本用法：**

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 產生模擬數據
np.random.seed(42)
data = pd.DataFrame({
    'Category': ['A']*50 + ['B']*50 + ['C']*50,
    'Value': np.concatenate([
        np.random.normal(100, 10, 50),
        np.random.normal(110, 15, 50),
        np.random.normal(95, 8, 50)
    ])
})

# 繪製箱型圖
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Category', y='Value', palette='Set2')

plt.title('Box Plot by Category', fontsize=14, fontweight='bold')
plt.xlabel('Category', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.tight_layout()
plt.show()
```

**化工應用範例：不同反應器批次產率比較**

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 模擬四個反應器的產率數據
np.random.seed(42)
n_samples = 30

data = pd.DataFrame({
    'Reactor': ['R1']*n_samples + ['R2']*n_samples + ['R3']*n_samples + ['R4']*n_samples,
    'Yield (%)': np.concatenate([
        np.random.normal(85, 3, n_samples),
        np.random.normal(88, 4, n_samples),
        np.random.normal(82, 5, n_samples),
        np.random.normal(90, 2.5, n_samples)
    ])
})

plt.figure(figsize=(12, 7))
sns.boxplot(data=data, x='Reactor', y='Yield (%)', palette='pastel', linewidth=2)
sns.swarmplot(data=data, x='Reactor', y='Yield (%)', color='black', alpha=0.5, size=3)

plt.axhline(y=85, color='red', linestyle='--', linewidth=2, label='Target Yield')
plt.title('Yield Comparison Across Reactors', fontsize=14, fontweight='bold')
plt.xlabel('Reactor ID', fontsize=12)
plt.ylabel('Yield (%)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()
```

### 4.4 小提琴圖 (Violin Plot)

小提琴圖結合了箱型圖和核密度估計圖的特點。

**基本用法：**

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 使用上面的數據
plt.figure(figsize=(10, 6))
sns.violinplot(data=data, x='Category', y='Value', palette='muted')

plt.title('Violin Plot by Category', fontsize=14, fontweight='bold')
plt.xlabel('Category', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.tight_layout()
plt.show()
```

**化工應用範例：不同操作條件下的產品品質分布**

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 模擬不同溫度條件下的產品品質數據
np.random.seed(42)
n = 100

data = pd.DataFrame({
    'Temperature': ['Low']*n + ['Medium']*n + ['High']*n,
    'Quality Score': np.concatenate([
        np.random.normal(75, 8, n),
        np.random.normal(85, 5, n),
        np.random.normal(80, 10, n)
    ])
})

plt.figure(figsize=(12, 7))
sns.violinplot(data=data, x='Temperature', y='Quality Score', 
               palette='Set3', inner='quartile', linewidth=2)

plt.title('Product Quality Distribution by Temperature', fontsize=14, fontweight='bold')
plt.xlabel('Operating Temperature', fontsize=12)
plt.ylabel('Quality Score', fontsize=12)
plt.tight_layout()
plt.show()
```

---

## 5. 類別圖 (Categorical Plots)

類別圖用於比較不同類別的數據分佈或統計量。

### 5.1 長條圖 (Bar Plot)

Seaborn 的長條圖會自動計算並顯示平均值和信賴區間。

**基本用法：**

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 產生模擬數據
np.random.seed(42)
data = pd.DataFrame({
    'Method': ['A']*30 + ['B']*30 + ['C']*30,
    'Efficiency': np.concatenate([
        np.random.normal(85, 5, 30),
        np.random.normal(90, 4, 30),
        np.random.normal(80, 6, 30)
    ])
})

# 繪製長條圖（自動計算平均值）
plt.figure(figsize=(10, 6))
sns.barplot(data=data, x='Method', y='Efficiency', palette='viridis', 
            errorbar='sd', capsize=0.1)

plt.title('Average Efficiency by Method', fontsize=14, fontweight='bold')
plt.xlabel('Method', fontsize=12)
plt.ylabel('Efficiency (%)', fontsize=12)
plt.tight_layout()
plt.show()
```

**參數說明：**
- `errorbar`：誤差線類型（'sd', 'se', 'ci', None）
- `capsize`：誤差線端點寬度
- `estimator`：統計函數（預設為 mean）

### 5.2 點圖 (Point Plot)

點圖用點和線顯示不同類別的統計估計值。

**化工應用範例：不同催化劑在不同溫度下的效能**

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 模擬催化劑效能數據
np.random.seed(42)
n = 20

data = pd.DataFrame({
    'Catalyst': ['Cat-A']*60 + ['Cat-B']*60 + ['Cat-C']*60,
    'Temperature': ['Low', 'Medium', 'High']*60,
    'Conversion (%)': np.concatenate([
        # Cat-A
        np.random.normal(70, 3, n), np.random.normal(85, 3, n), np.random.normal(78, 4, n),
        # Cat-B
        np.random.normal(65, 4, n), np.random.normal(88, 2, n), np.random.normal(92, 3, n),
        # Cat-C
        np.random.normal(75, 2, n), np.random.normal(82, 3, n), np.random.normal(85, 4, n)
    ])
})

plt.figure(figsize=(12, 7))
sns.pointplot(data=data, x='Temperature', y='Conversion (%)', hue='Catalyst',
              palette='Set2', markers=['o', 's', '^'], linestyles=['-', '--', '-.'],
              errorbar='ci', capsize=0.1)

plt.title('Catalyst Performance vs Temperature', fontsize=14, fontweight='bold')
plt.xlabel('Operating Temperature', fontsize=12)
plt.ylabel('Conversion Rate (%)', fontsize=12)
plt.legend(title='Catalyst Type')
plt.tight_layout()
plt.show()
```

### 5.3 蜂群圖 (Swarm Plot)

蜂群圖顯示所有數據點，避免重疊，適合中小規模數據。

**基本用法：**

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 產生模擬數據
np.random.seed(42)
data = pd.DataFrame({
    'Process': ['A']*25 + ['B']*25 + ['C']*25,
    'Yield': np.concatenate([
        np.random.normal(80, 5, 25),
        np.random.normal(85, 4, 25),
        np.random.normal(78, 6, 25)
    ])
})

# 繪製蜂群圖
plt.figure(figsize=(10, 6))
sns.swarmplot(data=data, x='Process', y='Yield', palette='Set1', size=6)

plt.title('Yield Distribution by Process', fontsize=14, fontweight='bold')
plt.xlabel('Process Type', fontsize=12)
plt.ylabel('Yield (%)', fontsize=12)
plt.tight_layout()
plt.show()
```

### 5.4 組合圖表

將不同類型的圖表組合，提供更豐富的資訊。

**範例：箱型圖 + 蜂群圖**

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 化工應用：反應時間對產率的影響
np.random.seed(42)
n = 30

data = pd.DataFrame({
    'Reaction Time (h)': ['2h']*n + ['4h']*n + ['6h']*n + ['8h']*n,
    'Yield (%)': np.concatenate([
        np.random.normal(65, 5, n),
        np.random.normal(80, 4, n),
        np.random.normal(88, 3, n),
        np.random.normal(87, 4, n)
    ])
})

plt.figure(figsize=(12, 7))

# 繪製箱型圖作為底層
sns.boxplot(data=data, x='Reaction Time (h)', y='Yield (%)', 
            palette='pastel', linewidth=2)

# 疊加蜂群圖顯示所有數據點
sns.swarmplot(data=data, x='Reaction Time (h)', y='Yield (%)', 
              color='black', alpha=0.5, size=4)

plt.title('Yield vs Reaction Time', fontsize=14, fontweight='bold')
plt.xlabel('Reaction Time', fontsize=12)
plt.ylabel('Yield (%)', fontsize=12)
plt.tight_layout()
plt.show()
```

---

## 6. 關係圖 (Relational Plots)

關係圖用於探索兩個或多個連續變數之間的關係。

### 6.1 散佈圖 (Scatter Plot)

Seaborn 的散佈圖可以輕鬆加入第三、第四維度的資訊。

**基本用法：**

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 產生模擬數據
np.random.seed(42)
n = 100

data = pd.DataFrame({
    'Temperature': np.random.uniform(60, 100, n),
    'Pressure': np.random.uniform(1, 5, n),
    'Yield': np.random.uniform(70, 95, n),
    'Catalyst': np.random.choice(['A', 'B', 'C'], n)
})

# 繪製散佈圖
plt.figure(figsize=(12, 7))
sns.scatterplot(data=data, x='Temperature', y='Yield', 
                hue='Catalyst', size='Pressure',
                palette='deep', sizes=(50, 300), alpha=0.7)

plt.title('Process Parameters Relationship', fontsize=14, fontweight='bold')
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Yield (%)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

**參數說明：**
- `hue`：用顏色區分類別
- `size`：用大小表示數值
- `style`：用標記形狀區分類別
- `alpha`：透明度

### 6.2 折線圖 (Line Plot)

Seaborn 的折線圖會自動計算並顯示信賴區間。

**化工應用範例：批次反應動力學監控**

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 模擬批次反應數據（3 次重複實驗）
np.random.seed(42)
time = np.linspace(0, 10, 50)
batches = []

for batch in range(3):
    for t in time:
        conversion = 95 * (1 - np.exp(-0.5 * t)) + np.random.normal(0, 2)
        batches.append({
            'Time (h)': t,
            'Conversion (%)': max(0, min(100, conversion)),
            'Batch': f'Batch {batch+1}'
        })

data = pd.DataFrame(batches)

plt.figure(figsize=(12, 7))
sns.lineplot(data=data, x='Time (h)', y='Conversion (%)', 
             hue='Batch', style='Batch',
             markers=True, dashes=False, palette='tab10')

plt.title('Batch Reaction Kinetics', fontsize=14, fontweight='bold')
plt.xlabel('Time (h)', fontsize=12)
plt.ylabel('Conversion (%)', fontsize=12)
plt.legend(title='Batch ID')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 6.3 回歸圖 (Regression Plot)

回歸圖自動擬合線性回歸線並顯示信賴區間。

**基本用法：**

```python
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 產生模擬數據
np.random.seed(42)
x = np.random.uniform(0, 100, 100)
y = 2 * x + 10 + np.random.normal(0, 15, 100)

# 繪製回歸圖
plt.figure(figsize=(10, 6))
sns.regplot(x=x, y=y, scatter_kws={'alpha':0.5}, 
            line_kws={'color':'red', 'linewidth':2})

plt.title('Linear Regression Plot', fontsize=14, fontweight='bold')
plt.xlabel('X Variable', fontsize=12)
plt.ylabel('Y Variable', fontsize=12)
plt.tight_layout()
plt.show()
```

**化工應用範例：濃度與反應速率關係**

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 模擬濃度與反應速率數據
np.random.seed(42)
concentration = np.linspace(0.1, 2.0, 50)
rate = 5 * concentration + np.random.normal(0, 0.5, 50)

data = pd.DataFrame({
    'Concentration (mol/L)': concentration,
    'Reaction Rate (mol/L/min)': rate
})

plt.figure(figsize=(10, 6))
sns.regplot(data=data, x='Concentration (mol/L)', y='Reaction Rate (mol/L/min)',
            scatter_kws={'alpha':0.6, 's':50}, 
            line_kws={'color':'darkred', 'linewidth':2.5})

plt.title('Reaction Rate vs Concentration', fontsize=14, fontweight='bold')
plt.xlabel('Concentration (mol/L)', fontsize=12)
plt.ylabel('Reaction Rate (mol/L/min)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 7. 熱力圖與相關性矩陣

熱力圖是視覺化矩陣數據的強大工具，特別適合顯示相關性矩陣。

### 7.1 基本熱力圖

**基本用法：**

```python
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 產生隨機矩陣數據
np.random.seed(42)
data = np.random.rand(10, 12)

# 繪製熱力圖
plt.figure(figsize=(12, 8))
sns.heatmap(data, annot=True, fmt='.2f', cmap='YlOrRd', 
            linewidths=0.5, linecolor='gray')

plt.title('Basic Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

**參數說明：**
- `annot`：是否在格子中顯示數值
- `fmt`：數值格式
- `cmap`：色彩映射
- `linewidths`：格子之間的線寬
- `vmin`, `vmax`：色彩範圍
- `cbar`：是否顯示色彩條

### 7.2 相關性矩陣熱力圖

相關性矩陣熱力圖是探索多變數關係的重要工具。

**化工應用範例：製程參數相關性分析**

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 模擬製程數據
np.random.seed(42)
n = 100

data = pd.DataFrame({
    'Temperature (°C)': np.random.uniform(60, 100, n),
    'Pressure (bar)': np.random.uniform(1, 5, n),
    'Flow Rate (L/min)': np.random.uniform(10, 50, n),
    'Catalyst Conc (%)': np.random.uniform(0.5, 2, n),
    'Residence Time (min)': np.random.uniform(5, 30, n),
    'Yield (%)': np.random.uniform(70, 95, n)
})

# 加入一些相關性
data['Yield (%)'] = (0.3 * data['Temperature (°C)'] + 
                      0.2 * data['Pressure (bar)'] + 
                      0.15 * data['Catalyst Conc (%)'] + 
                      np.random.normal(0, 5, n))

# 計算相關性矩陣
corr_matrix = data.corr()

# 繪製相關性熱力圖
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, 
            cbar_kws={'shrink': 0.8})

plt.title('Process Parameters Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### 7.3 遮罩三角熱力圖

由於相關性矩陣是對稱的，可以只顯示一半。

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 使用上面的數據和相關性矩陣
# 建立遮罩（上三角）
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# 繪製遮罩熱力圖
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
            cmap='RdBu_r', center=0, square=True, 
            linewidths=1, cbar_kws={'shrink': 0.8})

plt.title('Correlation Matrix (Lower Triangle)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### 7.4 階層式聚類熱力圖

階層式聚類熱力圖自動對數據進行聚類並重新排序。

**化工應用範例：批次數據相似性分析**

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 模擬不同批次的製程數據
np.random.seed(42)
batches = [f'Batch_{i:02d}' for i in range(1, 21)]
variables = ['Temp', 'Press', 'Flow', 'pH', 'Conc', 'Yield']

data = pd.DataFrame(
    np.random.randn(20, 6) * 10 + [80, 3, 25, 7, 1.5, 85],
    index=batches,
    columns=variables
)

# 繪製階層式聚類熱力圖
plt.figure(figsize=(12, 10))
sns.clustermap(data, cmap='viridis', standard_scale=1, 
               figsize=(12, 10), linewidths=0.5,
               cbar_kws={'label': 'Standardized Value'})

plt.suptitle('Hierarchical Clustering of Batch Data', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()
```

**參數說明：**
- `standard_scale`：標準化方式（0=行, 1=列）
- `method`：聚類方法（'average', 'single', 'complete' 等）
- `metric`：距離度量（'euclidean', 'correlation' 等）

---

## 8. 配對圖與多變數探索

### 8.1 配對圖 (Pair Plot)

配對圖顯示多個變數兩兩之間的關係，對角線顯示單變數分佈。

**基本用法：**

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 產生模擬數據
np.random.seed(42)
n = 100

data = pd.DataFrame({
    'Var1': np.random.normal(100, 15, n),
    'Var2': np.random.normal(50, 10, n),
    'Var3': np.random.normal(75, 12, n),
    'Category': np.random.choice(['A', 'B', 'C'], n)
})

# 繪製配對圖
sns.pairplot(data, hue='Category', palette='Set2', 
             diag_kind='kde', plot_kws={'alpha':0.6})

plt.suptitle('Pair Plot with Categories', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
```

**化工應用範例：反應條件多參數探索**

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 模擬反應條件數據
np.random.seed(42)
n = 150

data = pd.DataFrame({
    'Temperature': np.random.uniform(60, 100, n),
    'Pressure': np.random.uniform(1, 5, n),
    'Catalyst': np.random.uniform(0.5, 2, n),
    'Yield': np.random.uniform(70, 95, n),
    'Quality': np.random.choice(['Low', 'Medium', 'High'], n)
})

# 加入變數間的關聯
data['Yield'] = (0.4 * data['Temperature'] + 
                 5 * data['Pressure'] + 
                 10 * data['Catalyst'] + 
                 np.random.normal(0, 5, n))

# 繪製配對圖
sns.pairplot(data, hue='Quality', palette='viridis',
             diag_kind='kde', plot_kws={'alpha':0.5, 's':30},
             corner=True)

plt.suptitle('Process Parameters Pair Plot', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
```

**參數說明：**
- `hue`：用顏色區分類別
- `diag_kind`：對角線圖表類型（'hist', 'kde'）
- `corner`：只顯示下三角（避免重複）
- `vars`：選擇特定變數

### 8.2 聯合分佈圖 (Joint Plot)

聯合分佈圖同時顯示兩個變數的散佈圖和各自的邊際分佈。

**基本用法：**

```python
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 產生相關數據
np.random.seed(42)
x = np.random.normal(100, 15, 200)
y = x + np.random.normal(0, 20, 200)

# 繪製聯合分佈圖
sns.jointplot(x=x, y=y, kind='scatter', color='steelblue', alpha=0.6)

plt.suptitle('Joint Distribution Plot', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
```

**不同類型的聯合圖：**

```python
# kind='hex': 六邊形密度圖
sns.jointplot(x=x, y=y, kind='hex', color='purple')

# kind='kde': 核密度估計圖
sns.jointplot(x=x, y=y, kind='kde', cmap='Blues')

# kind='reg': 回歸圖
sns.jointplot(x=x, y=y, kind='reg', color='darkred')
```

**化工應用範例：溫度與產率的聯合分析**

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 模擬數據
np.random.seed(42)
temperature = np.random.uniform(60, 100, 200)
yield_rate = 30 + 0.6 * temperature + np.random.normal(0, 8, 200)

data = pd.DataFrame({
    'Temperature (°C)': temperature,
    'Yield (%)': yield_rate
})

# 繪製聯合分佈圖
g = sns.jointplot(data=data, x='Temperature (°C)', y='Yield (%)',
                  kind='scatter', color='orangered', alpha=0.6,
                  marginal_kws={'bins':20, 'fill':True})

# 添加回歸線
g.plot_joint(sns.regplot, scatter=False, color='blue', line_kws={'linewidth':2})

plt.suptitle('Temperature vs Yield Joint Distribution', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
```

---

## 9. 樣式與調色盤

### 9.1 調色盤類型

Seaborn 提供多種調色盤：

**1. 類別調色盤（Qualitative）：**
```python
import seaborn as sns
import matplotlib.pyplot as plt

# 內建類別調色盤
palettes = ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind']

fig, axes = plt.subplots(3, 2, figsize=(12, 8))
axes = axes.flatten()

for i, pal in enumerate(palettes):
    colors = sns.color_palette(pal)
    axes[i].barh(range(len(colors)), [1]*len(colors), color=colors)
    axes[i].set_title(f'{pal}', fontsize=12)
    axes[i].axis('off')

plt.suptitle('Categorical Color Palettes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

**2. 連續調色盤（Sequential）：**
```python
# 適合表示連續數值
sequential_palettes = ['Blues', 'Greens', 'Reds', 'viridis', 'plasma', 'rocket']

fig, axes = plt.subplots(3, 2, figsize=(12, 8))
axes = axes.flatten()

for i, pal in enumerate(sequential_palettes):
    colors = sns.color_palette(pal, 10)
    axes[i].barh(range(len(colors)), [1]*len(colors), color=colors)
    axes[i].set_title(f'{pal}', fontsize=12)
    axes[i].axis('off')

plt.suptitle('Sequential Color Palettes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

**3. 發散調色盤（Diverging）：**
```python
# 適合表示有中心點的數據（如相關性）
diverging_palettes = ['coolwarm', 'RdBu', 'RdYlGn', 'Spectral', 'vlag', 'icefire']

fig, axes = plt.subplots(3, 2, figsize=(12, 8))
axes = axes.flatten()

for i, pal in enumerate(diverging_palettes):
    colors = sns.color_palette(pal, 11)
    axes[i].barh(range(len(colors)), [1]*len(colors), color=colors)
    axes[i].set_title(f'{pal}', fontsize=12)
    axes[i].axis('off')

plt.suptitle('Diverging Color Palettes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### 9.2 自訂調色盤

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 從特定顏色建立調色盤
custom_palette = sns.color_palette(['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])

# 使用調色盤
sns.set_palette(custom_palette)

# 或在繪圖時指定
sns.boxplot(data=data, x='Category', y='Value', palette=custom_palette)
```

### 9.3 化工領域常用色彩方案

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 製程安全相關：紅-黃-綠
safety_palette = sns.color_palette(['#d62728', '#ff7f0e', '#2ca02c'])

# 溫度梯度：藍-白-紅
temperature_palette = sns.diverging_palette(240, 10, as_cmap=True)

# 產品品質等級：淺-深
quality_palette = sns.light_palette("seagreen", as_cmap=True)
```

---

## 10. 進階技巧與最佳實踐

### 10.1 圖表尺寸與解析度

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 設定預設圖表大小
sns.set(rc={'figure.figsize':(12, 8)})

# 或在繪圖時指定
plt.figure(figsize=(14, 10), dpi=150)

# 儲存高解析度圖片
plt.savefig('high_res_plot.png', dpi=300, bbox_inches='tight')
```

### 10.2 字體與文字設定

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 設定字體
sns.set(font='Arial', font_scale=1.2)

# 或使用 Matplotlib 設定
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# 設定軸標籤字體
plt.xlabel('X Label', fontsize=14, fontweight='bold')
plt.ylabel('Y Label', fontsize=14, fontweight='bold')
```

### 10.3 子圖佈局

**使用 FacetGrid：**

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 模擬數據
np.random.seed(42)
data = pd.DataFrame({
    'Time': np.tile(np.arange(24), 12),
    'Temperature': np.random.normal(25, 5, 288),
    'Month': np.repeat(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 24)
})

# 建立 FacetGrid
g = sns.FacetGrid(data, col='Month', col_wrap=4, height=3, aspect=1.2)
g.map(sns.lineplot, 'Time', 'Temperature')
g.set_axis_labels('Hour of Day', 'Temperature (°C)')
g.set_titles(col_template='{col_name}')

plt.suptitle('Temperature Variation by Month', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
```

### 10.4 儲存圖表

```python
import matplotlib.pyplot as plt

# 儲存為不同格式
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.savefig('plot.pdf', bbox_inches='tight')
plt.savefig('plot.svg', bbox_inches='tight')

# 透明背景
plt.savefig('plot.png', transparent=True, dpi=300, bbox_inches='tight')
```

---

## 11. 綜合化工應用案例

### 案例：製程優化數據視覺化分析

整合多種 Seaborn 圖表，進行完整的製程數據分析。

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 模擬完整的製程數據
np.random.seed(42)
n = 200

data = pd.DataFrame({
    'Batch_ID': [f'B{i:03d}' for i in range(1, n+1)],
    'Temperature (°C)': np.random.uniform(70, 90, n),
    'Pressure (bar)': np.random.uniform(2, 4, n),
    'Catalyst (%)': np.random.uniform(0.8, 1.5, n),
    'Residence_Time (min)': np.random.uniform(20, 40, n),
    'Yield (%)': 0,
    'Quality': '',
    'Reactor': np.random.choice(['R1', 'R2', 'R3'], n)
})

# 建立變數關係
data['Yield (%)'] = (0.5 * data['Temperature (°C)'] + 
                     8 * data['Pressure (bar)'] + 
                     15 * data['Catalyst (%)'] + 
                     0.3 * data['Residence_Time (min)'] + 
                     np.random.normal(0, 5, n))

# 品質分類
data['Quality'] = pd.cut(data['Yield (%)'], 
                         bins=[0, 75, 85, 100], 
                         labels=['Low', 'Medium', 'High'])

# 建立綜合分析圖表
fig = plt.figure(figsize=(18, 12))

# 1. 產率分佈直方圖
ax1 = plt.subplot(2, 3, 1)
sns.histplot(data=data, x='Yield (%)', bins=30, kde=True, 
             color='skyblue', ax=ax1)
ax1.set_title('Yield Distribution', fontsize=12, fontweight='bold')

# 2. 不同反應器的產率比較
ax2 = plt.subplot(2, 3, 2)
sns.boxplot(data=data, x='Reactor', y='Yield (%)', 
            palette='Set2', ax=ax2)
sns.swarmplot(data=data, x='Reactor', y='Yield (%)', 
              color='black', alpha=0.3, size=3, ax=ax2)
ax2.set_title('Yield by Reactor', fontsize=12, fontweight='bold')

# 3. 溫度與產率關係
ax3 = plt.subplot(2, 3, 3)
sns.scatterplot(data=data, x='Temperature (°C)', y='Yield (%)', 
                hue='Quality', palette='RdYlGn', s=60, alpha=0.7, ax=ax3)
ax3.set_title('Temperature vs Yield', fontsize=12, fontweight='bold')

# 4. 相關性矩陣
ax4 = plt.subplot(2, 3, 4)
corr_cols = ['Temperature (°C)', 'Pressure (bar)', 
             'Catalyst (%)', 'Residence_Time (min)', 'Yield (%)']
corr_matrix = data[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, ax=ax4, cbar_kws={'shrink': 0.8})
ax4.set_title('Correlation Matrix', fontsize=12, fontweight='bold')

# 5. 多參數點圖
ax5 = plt.subplot(2, 3, 5)
quality_order = ['Low', 'Medium', 'High']
sns.pointplot(data=data, x='Quality', y='Yield (%)', hue='Reactor',
              palette='Set1', markers=['o', 's', '^'], ax=ax5,
              order=quality_order)
ax5.set_title('Yield by Quality and Reactor', fontsize=12, fontweight='bold')

# 6. 小提琴圖
ax6 = plt.subplot(2, 3, 6)
sns.violinplot(data=data, x='Quality', y='Yield (%)', 
               palette='muted', inner='quartile', ax=ax6,
               order=quality_order)
ax6.set_title('Yield Distribution by Quality', fontsize=12, fontweight='bold')

plt.suptitle('Process Optimization: Comprehensive Data Analysis', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

## 12. 總結

### 12.1 Seaborn 的主要優勢

1. **簡潔的語法**：用更少的程式碼建立複雜的統計圖表
2. **美觀的預設樣式**：專業的視覺效果
3. **統計功能整合**：自動計算統計量與信賴區間
4. **Pandas 整合**：直接處理 DataFrame，語法更直觀
5. **豐富的圖表類型**：涵蓋大部分統計視覺化需求

### 12.2 何時使用 Seaborn vs Matplotlib

**使用 Seaborn 的情境：**
- 探索性數據分析（EDA）
- 統計視覺化（分佈、相關性、比較等）
- 快速原型與報告
- 需要美觀的預設樣式

**使用 Matplotlib 的情境：**
- 需要精細控制每個圖表元素
- 客製化的複雜佈局
- 動態或互動式圖表
- 特殊的科學圖表（如等高線圖、3D 圖等）

**最佳實踐：** 結合兩者使用，用 Seaborn 快速建立圖表，用 Matplotlib 進行精細調整。

### 12.3 化工領域應用建議

1. **實驗數據探索**：使用配對圖、分佈圖快速檢視數據特性
2. **製程比較分析**：使用箱型圖、小提琴圖比較不同條件
3. **相關性分析**：使用熱力圖視覺化多變數相關性
4. **優化結果呈現**：使用回歸圖、散佈圖展示參數-產出關係
5. **品質監控**：使用時序圖、控制圖追蹤製程穩定性

### 12.4 學習資源

- **官方文件**：https://seaborn.pydata.org/
- **官方教學**：https://seaborn.pydata.org/tutorial.html
- **圖庫範例**：https://seaborn.pydata.org/examples/index.html

---

## 練習題

1. **基礎練習**：使用 Seaborn 繪製直方圖和 KDE 圖，比較兩批產品的品質分佈。

2. **進階練習**：建立一個配對圖，探索溫度、壓力、催化劑濃度與產率之間的關係。

3. **綜合應用**：使用熱力圖視覺化製程參數的相關性矩陣，並找出與產率最相關的三個參數。

4. **實戰演練**：整合多種 Seaborn 圖表，建立一個完整的製程數據分析儀表板。

---

**延伸閱讀**：
- Matplotlib 官方文件：https://matplotlib.org/
- Seaborn 官方文件：https://seaborn.pydata.org/
- Python Data Visualization Cookbook (書籍)
- Effective Data Visualization (書籍)
