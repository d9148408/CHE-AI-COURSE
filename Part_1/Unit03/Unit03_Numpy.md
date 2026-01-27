# Unit03 Numpy 數值計算基礎

## 課程簡介

NumPy (Numerical Python) 是 Python 中最重要的科學計算套件之一，提供高效能的多維陣列物件以及處理這些陣列的工具。在資料科學、機器學習與化學工程的數值計算中，NumPy 扮演著核心的角色。

### 為什麼需要學習 NumPy？

1. **高效能運算**：NumPy 的核心運算以 C 語言實作，比 Python 原生 list 快 10-100 倍
2. **向量化運算**：支援陣列層級的運算，無需明確撰寫迴圈，程式碼更簡潔高效
3. **廣泛的數學函式**：提供線性代數、統計分析、隨機數生成等豐富功能
4. **機器學習基礎**：是 Pandas、Scikit-learn、TensorFlow 等套件的基礎
5. **化工應用**：適用於製程數據處理、數值模擬、反應動力學計算等

### 學習目標

完成本單元後，您將能夠：

- ✓ 理解 NumPy 陣列 (ndarray) 的基本概念與特性
- ✓ 建立與操作多維陣列
- ✓ 運用向量化運算提升程式效能
- ✓ 使用 NumPy 的數學函式進行科學計算
- ✓ 應用 NumPy 於化工數據處理與分析
- ✓ 掌握陣列索引、切片與變形技巧

---

## 1. NumPy 基本概念

### 1.1 什麼是 NumPy 陣列？

NumPy 陣列 (ndarray) 是一個多維度的同質資料容器，與 Python 原生的 list 相比具有以下特點：

| 特性 | NumPy 陣列 | Python List |
|------|-----------|-------------|
| **資料型態** | 所有元素必須相同型態 | 可包含不同型態 |
| **效能** | 高效能（C 語言實作） | 較慢（純 Python） |
| **記憶體** | 連續記憶體配置 | 分散儲存 |
| **運算** | 支援向量化運算 | 需要明確迴圈 |
| **功能** | 豐富的數學函式 | 基本操作 |

### 1.2 安裝與匯入 NumPy

```python
# 安裝 NumPy（若尚未安裝）
# !pip install numpy

# 匯入 NumPy（慣例使用 np 別名）
import numpy as np

# 檢查版本
print(f"NumPy 版本: {np.__version__}")
```

### 1.3 為什麼使用向量化運算？

**範例：計算溫度轉換**

使用 Python list 需要明確迴圈：

```python
# Python list 方式（慢）
celsius_list = [0, 10, 20, 30, 40, 50]
fahrenheit_list = []
for temp in celsius_list:
    fahrenheit_list.append(temp * 9/5 + 32)
print(fahrenheit_list)
```

使用 NumPy 陣列可直接運算：

```python
# NumPy 向量化運算（快）
celsius_array = np.array([0, 10, 20, 30, 40, 50])
fahrenheit_array = celsius_array * 9/5 + 32
print(fahrenheit_array)
```

---

## 2. 建立 NumPy 陣列

### 2.1 從 Python List 建立

```python
# 一維陣列
arr_1d = np.array([1, 2, 3, 4, 5])
print(arr_1d)
print(f"維度: {arr_1d.ndim}")
print(f"形狀: {arr_1d.shape}")
print(f"大小: {arr_1d.size}")
print(f"資料型態: {arr_1d.dtype}")
```

```python
# 二維陣列
arr_2d = np.array([[1, 2, 3], 
                   [4, 5, 6]])
print(arr_2d)
print(f"維度: {arr_2d.ndim}")
print(f"形狀: {arr_2d.shape}")
```

```python
# 三維陣列
arr_3d = np.array([[[1, 2], [3, 4]], 
                   [[5, 6], [7, 8]]])
print(arr_3d)
print(f"形狀: {arr_3d.shape}")  # (2, 2, 2)
```

### 2.2 使用內建函式建立陣列

**全零陣列與全一陣列**

```python
# 建立全零陣列
zeros = np.zeros((3, 4))
print(zeros)

# 建立全一陣列
ones = np.ones((2, 3, 4))
print(ones.shape)

# 建立指定值的陣列
full_array = np.full((3, 3), 7.5)
print(full_array)
```

**等差數列與等比數列**

```python
# np.arange：類似 Python 的 range
# arange(start, stop, step)
arr_range = np.arange(0, 10, 2)
print(arr_range)  # [0 2 4 6 8]

# np.linspace：在指定範圍內產生均勻分佈的數值
# linspace(start, stop, num)
arr_linspace = np.linspace(0, 1, 5)
print(arr_linspace)  # [0.   0.25 0.5  0.75 1.  ]

# 化工應用：建立溫度範圍
temperature = np.linspace(25, 100, 11)  # 25°C 到 100°C，11個點
print(temperature)
```

**單位矩陣與對角矩陣**

```python
# 單位矩陣（對角線為 1，其餘為 0）
identity = np.eye(4)
print(identity)

# 對角矩陣
diagonal = np.diag([1, 2, 3, 4])
print(diagonal)
```

### 2.3 隨機數生成

NumPy 提供強大的隨機數生成功能，適用於模擬與取樣。

```python
# 設定隨機種子（確保結果可重現）
np.random.seed(42)

# 均勻分佈 [0, 1)
uniform = np.random.rand(3, 3)
print(uniform)

# 標準常態分佈（平均 0，標準差 1）
normal = np.random.randn(1000)
print(f"平均值: {normal.mean():.4f}")
print(f"標準差: {normal.std():.4f}")

# 指定範圍的隨機整數
integers = np.random.randint(0, 100, size=(3, 4))
print(integers)

# 化工應用：模擬測量誤差
true_value = 50.0  # 真實值
measurement_noise = np.random.normal(0, 0.5, 100)  # 平均 0，標準差 0.5
measurements = true_value + measurement_noise
print(f"測量平均值: {measurements.mean():.4f}")
```

---

## 3. 陣列索引與切片

### 3.1 一維陣列索引

```python
arr = np.array([10, 20, 30, 40, 50])

# 正向索引
print(arr[0])    # 10（第一個元素）
print(arr[2])    # 30

# 反向索引
print(arr[-1])   # 50（最後一個元素）
print(arr[-2])   # 40

# 切片 [start:stop:step]
print(arr[1:4])      # [20 30 40]
print(arr[:3])       # [10 20 30]（前三個）
print(arr[2:])       # [30 40 50]（從第三個到最後）
print(arr[::2])      # [10 30 50]（每隔一個）
print(arr[::-1])     # [50 40 30 20 10]（反轉）
```

### 3.2 二維陣列索引

```python
# 建立二維陣列
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

# 索引單一元素 [row, column]
print(arr_2d[0, 0])    # 1
print(arr_2d[1, 2])    # 7
print(arr_2d[-1, -1])  # 12

# 索引整列或整欄
print(arr_2d[1, :])    # [5 6 7 8]（第二列）
print(arr_2d[:, 2])    # [3 7 11]（第三欄）

# 切片範圍
print(arr_2d[0:2, 1:3])  # [[2 3]
                          #  [6 7]]

# 化工應用：提取製程數據的特定時段與變數
# 假設 arr_2d 每列代表一個時間點，每欄代表一個變數
time_slice = arr_2d[5:10, :]    # 第 5-9 個時間點的所有變數
variable_2_and_3 = arr_2d[:, 1:3]  # 所有時間點的變數 2 和 3
```

### 3.3 布林索引

布林索引允許根據條件篩選資料，在數據處理中非常實用。

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 建立布林遮罩
mask = arr > 5
print(mask)  # [False False False False False  True  True  True  True  True]

# 使用遮罩篩選
print(arr[mask])  # [6 7 8 9 10]

# 簡寫
print(arr[arr > 5])  # [6 7 8 9 10]

# 多條件組合（& 表示 AND，| 表示 OR）
print(arr[(arr > 3) & (arr < 8)])  # [4 5 6 7]
print(arr[(arr < 3) | (arr > 8)])  # [1 2 9 10]

# 化工應用：篩選異常數據
temperature = np.array([25.1, 25.3, 98.5, 25.0, 24.9, 25.2])
# 篩選溫度異常（> 30°C）
abnormal_temp = temperature[temperature > 30]
print(f"異常溫度: {abnormal_temp}")
```

### 3.4 花式索引

使用整數陣列進行索引，可以任意順序提取元素。

```python
arr = np.array([10, 20, 30, 40, 50])

# 使用整數陣列索引
indices = np.array([0, 2, 4])
print(arr[indices])  # [10 30 50]

# 二維陣列的花式索引
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

rows = np.array([0, 2])
cols = np.array([1, 2])
print(arr_2d[rows, cols])  # [2 9]
```

---

## 4. 陣列運算

### 4.1 基本算術運算

NumPy 陣列支援元素層級 (element-wise) 的運算。

```python
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])

# 基本運算
print(arr1 + arr2)  # [ 6  8 10 12]
print(arr1 - arr2)  # [-4 -4 -4 -4]
print(arr1 * arr2)  # [ 5 12 21 32]
print(arr1 / arr2)  # [0.2   0.333 0.428 0.5]
print(arr1 ** 2)    # [ 1  4  9 16]

# 與純量運算
print(arr1 + 10)    # [11 12 13 14]
print(arr1 * 2)     # [2 4 6 8]

# 化工應用：濃度轉換
concentration_ppm = np.array([100, 200, 300, 400])  # ppm
concentration_percent = concentration_ppm / 10000   # 轉換為 %
print(f"濃度 (%): {concentration_percent}")
```

### 4.2 比較運算

```python
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([5, 4, 3, 2, 1])

print(arr1 > 3)       # [False False False  True  True]
print(arr1 == arr2)   # [False False  True False False]
print(arr1 < arr2)    # [ True  True False False False]

# 化工應用：檢查製程參數是否在規格範圍內
pressure = np.array([100, 105, 98, 110, 102])
spec_lower = 95
spec_upper = 108

within_spec = (pressure >= spec_lower) & (pressure <= spec_upper)
print(f"符合規格: {within_spec}")
print(f"符合規格數量: {within_spec.sum()}")
```

### 4.3 通用函式 (Universal Functions, ufunc)

NumPy 提供大量快速的元素層級數學函式。

```python
arr = np.array([1, 4, 9, 16, 25])

# 平方根
print(np.sqrt(arr))  # [1. 2. 3. 4. 5.]

# 指數與對數
print(np.exp(arr))   # e^x
print(np.log(arr))   # 自然對數
print(np.log10(arr)) # 常用對數

# 三角函式
angles = np.array([0, np.pi/4, np.pi/2, np.pi])
print(np.sin(angles))
print(np.cos(angles))

# 化工應用：Arrhenius 方程式計算反應速率常數
# k = A * exp(-Ea / (R * T))
A = 1e10              # 頻率因子
Ea = 50000            # 活化能 (J/mol)
R = 8.314             # 氣體常數 (J/mol·K)
T = np.array([300, 350, 400, 450, 500])  # 溫度 (K)

k = A * np.exp(-Ea / (R * T))
print(f"反應速率常數: {k}")
```

---

## 5. 陣列變形與組合

### 5.1 改變陣列形狀

```python
# 原始陣列
arr = np.arange(12)
print(arr)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]

# reshape：改變形狀（不改變數據）
arr_2d = arr.reshape(3, 4)
print(arr_2d)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

arr_3d = arr.reshape(2, 3, 2)
print(arr_3d.shape)  # (2, 3, 2)

# 自動計算維度（使用 -1）
arr_auto = arr.reshape(4, -1)  # 自動計算為 (4, 3)
print(arr_auto.shape)

# flatten：攤平為一維陣列
arr_flat = arr_2d.flatten()
print(arr_flat)

# ravel：類似 flatten，但可能回傳 view（不複製）
arr_ravel = arr_2d.ravel()
print(arr_ravel)
```

### 5.2 轉置與軸交換

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# 轉置
arr_T = arr.T
print(arr_T)
# [[1 4]
#  [2 5]
#  [3 6]]

# transpose：指定軸的順序
arr_3d = np.arange(24).reshape(2, 3, 4)
print(arr_3d.shape)  # (2, 3, 4)

arr_transposed = np.transpose(arr_3d, (2, 0, 1))
print(arr_transposed.shape)  # (4, 2, 3)
```

### 5.3 陣列堆疊

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# 垂直堆疊（沿著列方向）
v_stack = np.vstack([arr1, arr2])
print(v_stack)
# [[1 2 3]
#  [4 5 6]]

# 水平堆疊（沿著欄方向）
h_stack = np.hstack([arr1, arr2])
print(h_stack)  # [1 2 3 4 5 6]

# 通用堆疊函式
# concatenate：指定軸進行堆疊
arr2d_1 = np.array([[1, 2], [3, 4]])
arr2d_2 = np.array([[5, 6], [7, 8]])

concat_0 = np.concatenate([arr2d_1, arr2d_2], axis=0)  # 垂直
print(concat_0)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

concat_1 = np.concatenate([arr2d_1, arr2d_2], axis=1)  # 水平
print(concat_1)
# [[1 2 5 6]
#  [3 4 7 8]]
```

### 5.4 陣列分割

```python
arr = np.arange(16).reshape(4, 4)

# 水平分割
h_split = np.hsplit(arr, 2)  # 分成 2 個部分
print(h_split[0])
print(h_split[1])

# 垂直分割
v_split = np.vsplit(arr, 2)
print(v_split[0])

# 通用分割
split = np.split(arr, 2, axis=0)  # 沿著列分割
```

---

## 6. 統計與聚合函式

### 6.1 基本統計量

```python
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 總和與乘積
print(f"總和: {data.sum()}")          # 55
print(f"累積和: {data.cumsum()}")     # [ 1  3  6 10 15 21 28 36 45 55]
print(f"乘積: {data.prod()}")         # 3628800

# 平均值
print(f"平均值: {data.mean()}")       # 5.5
print(f"中位數: {np.median(data)}")   # 5.5

# 變異數與標準差
print(f"變異數: {data.var()}")        # 8.25
print(f"標準差: {data.std()}")        # 2.872

# 最大值、最小值與範圍
print(f"最大值: {data.max()}")        # 10
print(f"最小值: {data.min()}")        # 1
print(f"範圍: {data.ptp()}")          # peak to peak = 9

# 最大值與最小值的索引位置
print(f"最大值索引: {data.argmax()}") # 9
print(f"最小值索引: {data.argmin()}") # 0
```

### 6.2 多維陣列的統計

```python
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

# 整個陣列的統計
print(f"總和: {arr_2d.sum()}")        # 78

# 沿著特定軸的統計
# axis=0：沿著列方向（對每一欄做統計）
print(f"每欄總和: {arr_2d.sum(axis=0)}")    # [15 18 21 24]

# axis=1：沿著欄方向（對每一列做統計）
print(f"每列總和: {arr_2d.sum(axis=1)}")    # [10 26 42]

# 平均值
print(f"每欄平均: {arr_2d.mean(axis=0)}")   # [5. 6. 7. 8.]
print(f"每列平均: {arr_2d.mean(axis=1)}")   # [2.5 6.5 10.5]

# 化工應用：計算多變數製程數據的統計量
# 假設每列代表一個時間點，每欄代表一個製程變數
process_data = np.random.randn(100, 4)  # 100 時間點，4 個變數

variable_means = process_data.mean(axis=0)
variable_stds = process_data.std(axis=0)

print(f"各變數平均值: {variable_means}")
print(f"各變數標準差: {variable_stds}")
```

### 6.3 百分位數與分位數

```python
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 百分位數
print(f"25% 百分位數: {np.percentile(data, 25)}")  # 3.25
print(f"50% 百分位數: {np.percentile(data, 50)}")  # 5.5
print(f"75% 百分位數: {np.percentile(data, 75)}")  # 7.75

# 分位數（0-1 之間）
print(f"第一四分位數: {np.quantile(data, 0.25)}")   # 3.25
print(f"第三四分位數: {np.quantile(data, 0.75)}")   # 7.75

# 化工應用：計算製程數據的控制界限
# 使用 3-sigma 規則
process_values = np.random.normal(100, 5, 1000)
mean = process_values.mean()
std = process_values.std()

ucl = mean + 3 * std  # Upper Control Limit
lcl = mean - 3 * std  # Lower Control Limit

print(f"中心線: {mean:.2f}")
print(f"上控制界限 (UCL): {ucl:.2f}")
print(f"下控制界限 (LCL): {lcl:.2f}")
```

---

## 7. 線性代數運算

### 7.1 矩陣乘法

```python
# 元素層級乘法 (element-wise)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

element_wise = A * B
print(element_wise)
# [[ 5 12]
#  [21 32]]

# 矩陣乘法（內積）
matrix_product = np.dot(A, B)  # 或 A @ B
print(matrix_product)
# [[19 22]
#  [43 50]]

# 使用 @ 運算子（Python 3.5+）
matrix_product_2 = A @ B
print(matrix_product_2)
```

### 7.2 矩陣運算

```python
A = np.array([[1, 2], 
              [3, 4]])

# 轉置
print(A.T)

# 行列式 (determinant)
det = np.linalg.det(A)
print(f"行列式: {det}")  # -2.0

# 反矩陣 (inverse)
A_inv = np.linalg.inv(A)
print(A_inv)

# 驗證 A * A_inv = I
identity = A @ A_inv
print(np.round(identity))  # 四捨五入以消除浮點誤差

# 特徵值與特徵向量
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"特徵值: {eigenvalues}")
print(f"特徵向量:\n{eigenvectors}")
```

### 7.3 解線性方程組

```python
# 解 Ax = b
# 範例：2x + 3y = 8
#       4x + 5y = 14

A = np.array([[2, 3],
              [4, 5]])
b = np.array([8, 14])

x = np.linalg.solve(A, b)
print(f"解: x = {x}")  # [1. 2.]

# 驗證
print(f"驗證: Ax = {A @ x}")  # [8. 14.]

# 化工應用：物料平衡計算
# 假設有兩個單元操作，建立物料平衡方程式
# 0.5*F1 + 0.3*F2 = 100  (組成A)
# 0.5*F1 + 0.7*F2 = 140  (組成B)

A_balance = np.array([[0.5, 0.3],
                      [0.5, 0.7]])
b_balance = np.array([100, 140])

flows = np.linalg.solve(A_balance, b_balance)
print(f"流量 F1: {flows[0]:.2f} kg/h")
print(f"流量 F2: {flows[1]:.2f} kg/h")
```

---

## 8. 廣播 (Broadcasting)

廣播是 NumPy 的強大功能，允許不同形狀的陣列進行運算。

### 8.1 廣播規則

當兩個陣列進行運算時，NumPy 會比較它們的形狀，從最後一個維度開始比較：

1. 如果維度相同，或其中一個維度為 1，則可以廣播
2. 如果維度不同且都不是 1，則會產生錯誤

```python
# 範例 1：純量與陣列
arr = np.array([1, 2, 3, 4])
result = arr + 10  # 10 被廣播到每個元素
print(result)  # [11 12 13 14]

# 範例 2：一維陣列與二維陣列
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6]])
arr_1d = np.array([10, 20, 30])

result = arr_2d + arr_1d  # arr_1d 被廣播到每一列
print(result)
# [[11 22 33]
#  [14 25 36]]

# 範例 3：使用 reshape 進行廣播
row = np.array([1, 2, 3])
col = np.array([10, 20, 30]).reshape(3, 1)

result = row + col  # 產生 3x3 矩陣
print(result)
# [[11 12 13]
#  [21 22 23]
#  [31 32 33]]
```

### 8.2 化工應用：標準化數據

```python
# 假設有多個感測器的讀數（列：時間點，欄：感測器）
data = np.array([[100, 25, 1.2],
                 [110, 26, 1.3],
                 [105, 24, 1.1],
                 [115, 27, 1.4]])

# 計算每個感測器的平均值與標準差
mean = data.mean(axis=0)  # shape: (3,)
std = data.std(axis=0)    # shape: (3,)

# 標準化（利用廣播）
standardized = (data - mean) / std

print(f"原始數據:\n{data}")
print(f"\n平均值: {mean}")
print(f"標準差: {std}")
print(f"\n標準化數據:\n{standardized}")

# 驗證標準化後的平均值應接近 0，標準差應接近 1
print(f"\n標準化後平均值: {standardized.mean(axis=0)}")
print(f"標準化後標準差: {standardized.std(axis=0)}")
```

---

## 9. 實務應用案例

### 9.1 化工數據分析：反應動力學參數估算

```python
# 一階反應動力學：ln(C) = ln(C0) - k*t
# 已知不同時間點的濃度數據，估算速率常數 k

time = np.array([0, 10, 20, 30, 40, 50])  # 時間 (min)
concentration = np.array([1.0, 0.82, 0.67, 0.55, 0.45, 0.37])  # 濃度 (mol/L)

# 線性化處理
ln_C = np.log(concentration)
ln_C0 = ln_C[0]

# 使用線性回歸估算斜率（-k）
# 最小平方法: k = -Σ(t * ln(C)) / Σ(t^2)
k = -np.sum(time * ln_C) / np.sum(time ** 2)

print(f"反應速率常數 k: {k:.6f} min⁻¹")
print(f"半生期 t₁/₂: {np.log(2) / k:.2f} min")

# 計算預測值與誤差
ln_C_pred = ln_C0 - k * time
C_pred = np.exp(ln_C_pred)
error = concentration - C_pred
mse = np.mean(error ** 2)

print(f"均方誤差 (MSE): {mse:.6f}")
```

### 9.2 溫度分布模擬

```python
# 一維熱傳導：穩態溫度分布
# 假設管壁兩端溫度固定，計算內部溫度分布

# 參數設定
n_points = 11
T_left = 100   # 左端溫度 (°C)
T_right = 20   # 右端溫度 (°C)

# 線性溫度分布（簡化模型）
position = np.linspace(0, 1, n_points)
temperature = T_left + (T_right - T_left) * position

print(f"位置: {position}")
print(f"溫度: {temperature}")

# 計算溫度梯度
gradient = np.gradient(temperature, position)
print(f"溫度梯度: {gradient}")
```

### 9.3 質量平衡計算

```python
# 連續攪拌槽反應器 (CSTR) 質量平衡
# 假設有 3 個串聯的 CSTR，計算穩態濃度分布

# 參數
C_in = 10.0      # 進料濃度 (mol/L)
k = 0.5          # 反應速率常數 (1/min)
tau = 2.0        # 滯留時間 (min)
n_reactors = 3   # 反應器數量

# 初始化濃度陣列
C = np.zeros(n_reactors + 1)
C[0] = C_in

# 計算每個反應器的出口濃度
for i in range(n_reactors):
    C[i+1] = C[i] / (1 + k * tau)

print(f"進料濃度: {C[0]:.2f} mol/L")
for i in range(n_reactors):
    print(f"反應器 {i+1} 出口濃度: {C[i+1]:.2f} mol/L")

# 計算總轉化率
conversion = (C[0] - C[-1]) / C[0] * 100
print(f"\n總轉化率: {conversion:.2f}%")
```

---

## 10. 效能優化技巧

### 10.1 向量化 vs 迴圈

```python
import time

# 使用迴圈（慢）
n = 1000000
arr = np.arange(n)

start = time.time()
result_loop = []
for x in arr:
    result_loop.append(x ** 2)
time_loop = time.time() - start

# 使用向量化（快）
start = time.time()
result_vectorized = arr ** 2
time_vectorized = time.time() - start

print(f"迴圈時間: {time_loop:.4f} 秒")
print(f"向量化時間: {time_vectorized:.6f} 秒")
print(f"加速比: {time_loop / time_vectorized:.1f}x")
```

### 10.2 記憶體管理

```python
# 使用 view 而非 copy 節省記憶體
arr = np.arange(1000000)

# 創建 view（不複製數據）
arr_view = arr[::2]  # 取偶數索引

# 創建 copy（複製數據）
arr_copy = arr[::2].copy()

# 檢查是否共享記憶體
print(f"View 共享記憶體: {np.shares_memory(arr, arr_view)}")
print(f"Copy 共享記憶體: {np.shares_memory(arr, arr_copy)}")

# 注意：修改 view 會影響原陣列
arr_view[0] = -999
print(f"原陣列第一個元素: {arr[0]}")  # -999
```

### 10.3 選擇適當的資料型態

```python
# 預設 dtype 通常是 float64 或 int64
arr_default = np.array([1, 2, 3])
print(f"預設型態: {arr_default.dtype}")  # int64 (8 bytes)

# 如果數值範圍較小，可以使用較小的型態節省記憶體
arr_small = np.array([1, 2, 3], dtype=np.int8)  # 1 byte
print(f"記憶體使用: {arr_small.nbytes} bytes")

# 化工應用：感測器數據通常不需要 float64 精度
sensor_data = np.random.randn(10000).astype(np.float32)
print(f"Float32 記憶體: {sensor_data.nbytes / 1024:.2f} KB")

sensor_data_64 = sensor_data.astype(np.float64)
print(f"Float64 記憶體: {sensor_data_64.nbytes / 1024:.2f} KB")
```

---

## 11. 常見錯誤與除錯

### 11.1 形狀不匹配錯誤

```python
# 錯誤範例
try:
    A = np.array([[1, 2, 3]])  # shape: (1, 3)
    B = np.array([[1], [2]])   # shape: (2, 1)
    result = A + B
except ValueError as e:
    print(f"錯誤: {e}")

# 正確做法：確保形狀相容
A = np.array([[1, 2, 3]])  # (1, 3)
B = np.array([[1], [2], [3]])  # (3, 1)
result = A + B  # 廣播為 (3, 3)
print(result)
```

### 11.2 整數除法陷阱

```python
# Python 3 中已解決，但仍需注意 NumPy 行為
arr_int = np.array([1, 2, 3], dtype=int)

# 整數陣列除法會截斷小數部分
result_int = arr_int / 2
print(result_int)  # [0.5 1.  1.5]（自動轉為 float）

# 確保使用浮點數運算
arr_float = arr_int.astype(float)
result_float = arr_float / 2
print(result_float)
```

### 11.3 複製 vs 視圖

```python
# 陷阱：意外修改原陣列
original = np.array([1, 2, 3, 4, 5])
subset = original[1:4]  # 這是 view，不是 copy

subset[0] = 999
print(original)  # [1 999 3 4 5]（原陣列被修改！）

# 安全做法：明確複製
original = np.array([1, 2, 3, 4, 5])
subset = original[1:4].copy()

subset[0] = 999
print(original)  # [1 2 3 4 5]（原陣列未變）
```

---

## 12. 總結與最佳實踐

### 12.1 核心概念回顧

1. **陣列特性**：
   - 同質性（所有元素同型態）
   - 高效能（C 語言實作）
   - 固定大小（建立後形狀可改但大小不變）

2. **關鍵技巧**：
   - 優先使用向量化運算，避免明確迴圈
   - 善用廣播機制處理不同形狀的陣列
   - 理解 view 與 copy 的差異
   - 選擇適當的資料型態節省記憶體

3. **化工應用**：
   - 數值模擬與計算
   - 製程數據分析
   - 反應動力學參數估算
   - 物料與能量平衡

### 12.2 學習資源

- **官方文件**：[numpy.org/doc](https://numpy.org/doc/)
- **教學資源**：NumPy User Guide
- **進階主題**：NumPy for MATLAB Users

### 12.3 下一步學習

完成 NumPy 基礎後，建議繼續學習：

- **Pandas**：資料處理與分析（建立在 NumPy 之上）
- **Matplotlib**：資料視覺化
- **SciPy**：科學計算與工程應用
- **Scikit-learn**：機器學習（大量使用 NumPy 陣列）

---

## 練習題

### 基礎練習

1. 建立一個 5×5 的隨機矩陣，並計算其平均值、標準差與最大值
2. 使用 `linspace` 建立 0 到 2π 的 100 個點，計算 sin 與 cos 值
3. 建立一個 4×4 矩陣，提取其對角線元素

### 進階練習

4. 模擬 1000 次擲骰子，統計每個點數出現的次數
5. 給定溫度陣列，使用向量化運算將攝氏轉換為華氏與克氏溫標
6. 實作移動平均濾波器，對含有雜訊的訊號進行平滑處理

### 化工應用練習

7. 計算不同溫度下的 Antoine 方程式蒸氣壓
8. 使用 NumPy 解三元物料平衡方程組
9. 模擬批次反應器的濃度變化曲線（一階反應）

---

**恭喜您完成 Unit03 NumPy 數值計算基礎！**

掌握 NumPy 是進行資料科學與機器學習的重要基石。接下來，我們將學習 Pandas，它建立在 NumPy 之上，提供更高階的資料處理功能。

