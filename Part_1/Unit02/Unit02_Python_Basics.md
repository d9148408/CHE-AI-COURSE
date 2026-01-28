# Unit02 Python程式語言基礎

## 課程簡介

Python 是目前資料科學、機器學習與人工智慧領域最受歡迎的程式語言之一。本單元將帶領您從零開始學習 Python 程式語言的核心概念，為後續的機器學習課程打下堅實的基礎。

### 為什麼選擇 Python？

1. **語法簡潔易讀**：Python 的語法接近自然語言，易於學習和理解
2. **豐富的生態系統**：擁有大量的科學計算與機器學習套件（NumPy、Pandas、Scikit-learn、TensorFlow 等）
3. **廣泛的社群支持**：龐大的開發者社群，遇到問題容易找到解決方案
4. **跨平台特性**：可在 Windows、macOS、Linux 等不同作業系統上執行
5. **工業界廣泛應用**：許多科技公司與研究機構採用 Python 進行資料分析與機器學習

### 學習目標

完成本單元後，您將能夠：

- ✓ 理解 Python 的基本語法結構
- ✓ 掌握變數、資料型態與運算子的使用
- ✓ 運用條件語句與迴圈進行流程控制
- ✓ 定義與使用函式來組織程式碼
- ✓ 處理例外狀況，撰寫更穩健的程式
- ✓ 理解 Python 在資料科學中的應用優勢

---

## 1. Python 基本語法

### 1.1 註解 (Comments)

註解是程式碼中不會被執行的文字，用於說明程式碼的功能。良好的註解習慣能讓您和其他人更容易理解程式碼。

```python
# 這是單行註解
# Python 使用 # 符號來標示註解

"""
這是多行註解
可以跨越多行
通常用於函式或模組的說明文件
"""

# 計算圓的面積
radius = 5
area = 3.14159 * radius ** 2  # 使用半徑計算面積
print(f"圓的面積為： {area}")
```

### 1.2 縮排 (Indentation)

Python 使用縮排來定義程式碼區塊，這是 Python 的一大特色。**縮排必須保持一致**，通常使用 4 個空格。

```python
# 正確的縮排
if temperature > 100:
    print("水已沸騰")
    print("請小心燙傷")

# 錯誤的縮排會導致語法錯誤
if temperature > 100:
print("這會產生錯誤")  # IndentationError
```

### 1.3 程式碼執行順序

Python 程式碼從上到下、從左到右依序執行。

```python
# 依序執行
print("第一行")
print("第二行")
print("第三行")

# 輸出：
# 第一行
# 第二行
# 第三行
```

---

## 2. 變數與資料型態

### 2.1 變數 (Variables)

變數是用來儲存資料的容器。Python 是動態型別語言，不需要事先宣告變數的型態。

```python
# 變數命名規則
temperature = 25.5        # 使用有意義的變數名稱
pressure = 101.3          # 數字、字母、底線組合
flow_rate = 100           # 使用底線分隔單字（snake_case）

# 不合法的變數名稱
# 1st_variable = 10      # 不能以數字開頭
# my-variable = 20       # 不能使用連字號
# class = 30             # 不能使用 Python 關鍵字
```

**變數命名最佳實踐**：

- 使用小寫字母和底線（snake_case）
- 變數名稱應該具有描述性
- 避免使用 Python 關鍵字
- 常數使用全大寫（CONSTANT_NAME）

### 2.2 基本資料型態

Python 有多種內建的資料型態，以下是最常用的幾種：

#### 2.2.1 數值型態 (Numeric Types)

```python
# 整數 (int)
count = 100
negative_num = -50

# 浮點數 (float)
temperature = 25.5
pressure = 1.013e5  # 科學記號表示法

# 複數 (complex)
impedance = 3 + 4j

# 型態檢查
print(type(count))        # <class 'int'>
print(type(temperature))  # <class 'float'>
print(type(impedance))    # <class 'complex'>
```

#### 2.2.2 字串 (String)

```python
# 字串定義
name = "Chemical Engineering"
university = 'Feng Chia University'
multi_line = """這是一個
多行字串"""

# 字串操作
greeting = "Hello, " + name  # 字串連接
repeat = "AI " * 3           # 字串重複
length = len(name)           # 字串長度

# 字串索引與切片
text = "Python"
first_char = text[0]         # 'P' (索引從 0 開始)
last_char = text[-1]         # 'n' (負索引從末尾開始)
substring = text[0:3]        # 'Pyt' (切片不包含結束索引)

# 常用字串方法
upper_text = text.upper()    # 'PYTHON'
lower_text = text.lower()    # 'python'
replaced = text.replace('P', 'J')  # 'Jython'
```

#### 2.2.3 布林值 (Boolean)

```python
# 布林值只有兩個： True 和 False
is_running = True
is_stopped = False

# 布林運算
result = (10 > 5)      # True
result = (10 == 5)     # False
result = (10 != 5)     # True

# 邏輯運算子
print(True and False)  # False
print(True or False)   # True
print(not True)        # False
```

### 2.3 資料結構

#### 2.3.1 串列 (List)

串列是可變的有序集合，可以包含不同型態的元素。

```python
# 建立串列
temperatures = [25.5, 26.0, 24.8, 25.2]
mixed_list = [1, "two", 3.0, True]
empty_list = []

# 串列操作
temperatures.append(27.0)       # 新增元素
temperatures.insert(0, 24.0)    # 在指定位置插入
temperatures.remove(26.0)       # 移除特定值
last_temp = temperatures.pop()  # 移除並返回最後一個元素

# 串列索引與切片
first_temp = temperatures[0]
last_temp = temperatures[-1]
sub_list = temperatures[1:3]

# 串列常用方法
length = len(temperatures)      # 串列長度
max_temp = max(temperatures)    # 最大值
min_temp = min(temperatures)    # 最小值
avg_temp = sum(temperatures) / len(temperatures)  # 平均值
```

#### 2.3.2 元組 (Tuple)

元組是不可變的有序集合，常用於表示固定的資料組合。

```python
# 建立元組
coordinates = (25.0, 120.5)
single_element = (10,)  # 單一元素元組需要逗號
reactor_config = ("CSTR", 100, 350.0)

# 元組解包
x, y = coordinates
reactor_type, volume, temperature = reactor_config

# 元組不可變
# coordinates[0] = 30.0  # 會產生 TypeError
```

#### 2.3.3 字典 (Dictionary)

字典是鍵值對的集合，用於儲存相關聯的資料。

```python
# 建立字典
reactor = {
    "type": "CSTR",
    "volume": 100,
    "temperature": 350.0,
    "pressure": 10.0
}

# 存取字典元素
reactor_type = reactor["type"]
temperature = reactor.get("temperature", 25.0)  # 使用 get 避免 KeyError

# 修改字典
reactor["temperature"] = 360.0
reactor["catalyst"] = "Platinum"

# 字典操作
keys = reactor.keys()      # 取得所有鍵
values = reactor.values()  # 取得所有值
items = reactor.items()    # 取得所有鍵值對

# 檢查鍵是否存在
if "pressure" in reactor:
    print(f"壓力： {reactor['pressure']} bar")
```

#### 2.3.4 集合 (Set)

集合是無序且不重複的元素集合。

```python
# 建立集合
elements = {"H", "O", "N", "C"}
numbers = {1, 2, 3, 4, 5}

# 集合操作
elements.add("S")        # 新增元素
elements.remove("H")     # 移除元素（若不存在會報錯）
elements.discard("X")    # 移除元素（若不存在不會報錯）

# 集合運算
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

union = set1 | set2          # 聯集： {1, 2, 3, 4, 5, 6}
intersection = set1 & set2   # 交集： {3, 4}
difference = set1 - set2     # 差集： {1, 2}
```

### 2.4 型態轉換

```python
# 數值轉換
x = int(3.14)        # 3
y = float("2.5")     # 2.5
z = str(100)         # "100"

# 字串轉數值
num_str = "123"
num_int = int(num_str)
num_float = float(num_str)

# 串列轉元組、集合
my_list = [1, 2, 3, 2, 1]
my_tuple = tuple(my_list)    # (1, 2, 3, 2, 1)
my_set = set(my_list)        # {1, 2, 3}

# 字串與串列轉換
text = "H2O"
char_list = list(text)       # ['H', '2', 'O']
joined = "".join(char_list)  # 'H2O'
```

---

## 3. 運算子 (Operators)

### 3.1 算術運算子

```python
# 基本運算
a = 10
b = 3

addition = a + b        # 13 (加法)
subtraction = a - b     # 7  (減法)
multiplication = a * b  # 30 (乘法)
division = a / b        # 3.333... (除法，結果為浮點數)
floor_division = a // b # 3  (整數除法，向下取整)
modulus = a % b         # 1  (取餘數)
exponent = a ** b       # 1000 (次方)

# 化工應用範例
# 理想氣體方程式： PV = nRT
pressure = 1.0        # atm
volume = 22.4         # L
R = 0.0821            # L·atm/(mol·K)
temperature = 273.15  # K

moles = (pressure * volume) / (R * temperature)
print(f"莫耳數： {moles:.2f} mol")
```

### 3.2 比較運算子

```python
# 比較運算
x = 10
y = 5

equal = (x == y)          # False (等於)
not_equal = (x != y)      # True  (不等於)
greater = (x > y)         # True  (大於)
less = (x < y)            # False (小於)
greater_equal = (x >= y)  # True  (大於等於)
less_equal = (x <= y)     # False (小於等於)

# 化工應用：安全範圍檢查
reactor_temp = 360.0
MAX_TEMP = 400.0
MIN_TEMP = 300.0

is_safe = (MIN_TEMP <= reactor_temp <= MAX_TEMP)
print(f"反應器溫度是否在安全範圍： {is_safe}")
```

### 3.3 邏輯運算子

```python
# 邏輯運算
temp_ok = True
pressure_ok = False

# and：所有條件都為 True 時才為 True
all_safe = temp_ok and pressure_ok  # False

# or：至少一個條件為 True 時就為 True
any_safe = temp_ok or pressure_ok   # True

# not：反轉布林值
not_safe = not temp_ok              # False

# 化工應用：製程安全檢查
temperature = 350.0
pressure = 9.5
flow_rate = 95.0

# 檢查所有參數是否在正常範圍
temp_normal = 300.0 <= temperature <= 400.0
pressure_normal = 8.0 <= pressure <= 12.0
flow_normal = 80.0 <= flow_rate <= 120.0

all_normal = temp_normal and pressure_normal and flow_normal
print(f"製程狀態正常： {all_normal}")
```

### 3.4 賦值運算子

```python
# 基本賦值
x = 10

# 複合賦值運算子
x += 5   # x = x + 5  (15)
x -= 3   # x = x - 3  (12)
x *= 2   # x = x * 2  (24)
x /= 4   # x = x / 4  (6.0)
x //= 2  # x = x // 2 (3.0)
x %= 2   # x = x % 2  (1.0)
x **= 3  # x = x ** 3 (1.0)

# 多重賦值
a, b, c = 1, 2, 3
x = y = z = 0

# 交換變數
a, b = b, a
```

---

## 4. 控制流程

### 4.1 條件語句 (if-elif-else)

條件語句用於根據不同的條件執行不同的程式碼區塊。

```python
# 基本 if 語句
temperature = 105.0

if temperature > 100.0:
    print("水已沸騰")

# if-else 語句
if temperature > 100.0:
    print("溫度過高")
else:
    print("溫度正常")

# if-elif-else 語句
if temperature > 100.0:
    print("溫度過高")
elif temperature < 0.0:
    print("溫度過低")
else:
    print("溫度正常")

# 巢狀條件
pressure = 10.5
if temperature > 100.0:
    if pressure > 10.0:
        print("高溫高壓，危險！")
    else:
        print("高溫但壓力正常")
else:
    print("溫度正常")
```

**化工應用案例：反應器操作監控**

```python
# 反應器參數
reactor_temp = 375.0      # °C
reactor_pressure = 11.0   # bar
conversion_rate = 0.85    # 轉化率

# 操作狀態判斷
TEMP_OPTIMAL = (350.0, 400.0)
PRESSURE_OPTIMAL = (9.0, 12.0)
CONVERSION_TARGET = 0.80

# 溫度檢查
if reactor_temp < TEMP_OPTIMAL[0]:
    temp_status = "偏低"
elif reactor_temp > TEMP_OPTIMAL[1]:
    temp_status = "偏高"
else:
    temp_status = "正常"

# 壓力檢查
if reactor_pressure < PRESSURE_OPTIMAL[0]:
    pressure_status = "偏低"
elif reactor_pressure > PRESSURE_OPTIMAL[1]:
    pressure_status = "偏高"
else:
    pressure_status = "正常"

# 綜合判斷
print(f"反應器溫度： {reactor_temp}°C ({temp_status})")
print(f"反應器壓力： {reactor_pressure} bar ({pressure_status})")

if temp_status == "正常" and pressure_status == "正常":
    if conversion_rate >= CONVERSION_TARGET:
        print("✓ 反應器運作正常，轉化率達標")
    else:
        print("△ 反應器參數正常，但轉化率未達標")
else:
    print("⚠ 反應器參數異常，請檢查！")
```

### 4.2 迴圈 (Loops)

迴圈用於重複執行程式碼區塊。

#### 4.2.1 for 迴圈

```python
# 基本 for 迴圈
temperatures = [25.5, 26.0, 24.8, 25.2, 26.5]

for temp in temperatures:
    print(f"溫度： {temp}°C")

# 使用 range() 函式
for i in range(5):
    print(f"第 {i+1} 次迭代")

# range(start, stop, step)
for i in range(0, 10, 2):  # 0, 2, 4, 6, 8
    print(i)

# 迭代字典
reactor = {
    "溫度": 350.0,
    "壓力": 10.0,
    "流量": 100.0
}

for key, value in reactor.items():
    print(f"{key}： {value}")

# enumerate() 取得索引和值
for index, temp in enumerate(temperatures):
    print(f"第 {index+1} 筆資料： {temp}°C")
```

#### 4.2.2 while 迴圈

```python
# 基本 while 迴圈
count = 0
while count < 5:
    print(f"計數： {count}")
    count += 1

# 化工應用：批次反應器模擬
time = 0.0        # 時間 (小時)
concentration = 1.0  # 濃度 (mol/L)
k = 0.1           # 反應速率常數 (1/hr)

print("時間(hr)  濃度(mol/L)")
print("-" * 25)

while concentration > 0.1:
    print(f"{time:6.1f}    {concentration:8.4f}")
    # 一階反應動力學： C(t) = C0 * exp(-kt)
    time += 0.5
    concentration = 1.0 * (2.71828 ** (-k * time))

print(f"{time:6.1f}    {concentration:8.4f}")
print(f"\n反應達到目標濃度所需時間： {time:.1f} 小時")
```

#### 4.2.3 迴圈控制語句

```python
# break：跳出迴圈
for i in range(10):
    if i == 5:
        break  # 當 i 等於 5 時跳出迴圈
    print(i)  # 輸出： 0, 1, 2, 3, 4

# continue：跳過當前迭代
for i in range(5):
    if i == 2:
        continue  # 跳過 i = 2
    print(i)  # 輸出： 0, 1, 3, 4

# else 子句：迴圈正常結束時執行
for i in range(5):
    print(i)
else:
    print("迴圈正常結束")

# 化工應用：感測器資料品質檢查
sensor_data = [25.2, 25.5, 999.9, 25.8, 26.0]  # 999.9 是異常值
MAX_VALID_TEMP = 100.0

valid_data = []
for temp in sensor_data:
    if temp > MAX_VALID_TEMP:
        print(f"⚠ 偵測到異常值： {temp}°C，已跳過")
        continue
    valid_data.append(temp)

print(f"有效資料： {valid_data}")
```

### 4.3 串列生成式 (List Comprehension)

串列生成式是 Python 的強大特性，可以用簡潔的語法建立新串列。

```python
# 基本語法： [expression for item in iterable]
squares = [x**2 for x in range(10)]
# 等同於：
# squares = []
# for x in range(10):
#     squares.append(x**2)

# 加入條件： [expression for item in iterable if condition]
even_squares = [x**2 for x in range(10) if x % 2 == 0]

# 化工應用：溫度單位轉換
temps_celsius = [25.0, 30.0, 35.0, 40.0]
temps_fahrenheit = [(temp * 9/5) + 32 for temp in temps_celsius]
print(f"攝氏溫度： {temps_celsius}")
print(f"華氏溫度： {temps_fahrenheit}")

# 過濾異常數據
sensor_readings = [25.2, 25.5, -999.0, 25.8, 26.0, 999.9]
valid_readings = [temp for temp in sensor_readings if -50 < temp < 100]
print(f"有效讀數： {valid_readings}")
```

---

## 5. 函式 (Functions)

函式是可重複使用的程式碼區塊，用於執行特定任務。良好的函式設計能讓程式碼更易於維護和測試。

### 5.1 定義函式

```python
# 基本函式定義
def greet():
    print("Hello, Chemical Engineering!")

# 呼叫函式
greet()

# 帶參數的函式
def greet_person(name):
    print(f"Hello, {name}!")

greet_person("Alice")

# 帶預設參數的函式
def calculate_area(length, width=1.0):
    return length * width

area1 = calculate_area(5.0, 3.0)  # 15.0
area2 = calculate_area(5.0)        # 5.0 (使用預設 width=1.0)
```

### 5.2 返回值

```python
# 返回單一值
def celsius_to_fahrenheit(celsius):
    fahrenheit = (celsius * 9/5) + 32
    return fahrenheit

temp_f = celsius_to_fahrenheit(25.0)
print(f"25°C = {temp_f}°F")

# 返回多個值（元組）
def get_reactor_status():
    temperature = 350.0
    pressure = 10.0
    flow_rate = 100.0
    return temperature, pressure, flow_rate

temp, press, flow = get_reactor_status()

# 無返回值（預設返回 None）
def log_message(message):
    print(f"[LOG] {message}")

result = log_message("System started")  # result 為 None
```

### 5.3 參數類型

```python
# 位置參數
def power(base, exponent):
    return base ** exponent

result = power(2, 3)  # 8

# 關鍵字參數
result = power(base=2, exponent=3)
result = power(exponent=3, base=2)  # 順序可以不同

# 任意數量的位置參數 (*args)
def sum_all(*numbers):
    total = 0
    for num in numbers:
        total += num
    return total

result = sum_all(1, 2, 3, 4, 5)  # 15

# 任意數量的關鍵字參數 (**kwargs)
def print_reactor_info(**params):
    for key, value in params.items():
        print(f"{key}： {value}")

print_reactor_info(temperature=350.0, pressure=10.0, type="CSTR")
```

### 5.4 化工應用函式範例

```python
# 理想氣體方程式計算
def ideal_gas_law(P=None, V=None, n=None, T=None, R=0.0821):
    """
    理想氣體方程式： PV = nRT
    
    Parameters:
    -----------
    P : float, optional
        壓力 (atm)
    V : float, optional
        體積 (L)
    n : float, optional
        莫耳數 (mol)
    T : float, optional
        溫度 (K)
    R : float, default=0.0821
        氣體常數 (L·atm/(mol·K))
    
    Returns:
    --------
    float
        未知的變數值
    """
    if P is None:
        return (n * R * T) / V
    elif V is None:
        return (n * R * T) / P
    elif n is None:
        return (P * V) / (R * T)
    elif T is None:
        return (P * V) / (n * R)
    else:
        raise ValueError("必須有一個參數為 None")

# 使用範例
pressure = ideal_gas_law(V=22.4, n=1.0, T=273.15)
print(f"壓力： {pressure:.4f} atm")

# 反應轉化率計算
def calculate_conversion(C0, C):
    """
    計算反應轉化率
    
    Parameters:
    -----------
    C0 : float
        初始濃度 (mol/L)
    C : float
        當前濃度 (mol/L)
    
    Returns:
    --------
    float
        轉化率 (0-1)
    """
    if C0 <= 0:
        raise ValueError("初始濃度必須大於 0")
    if C < 0:
        raise ValueError("濃度不能為負值")
    
    conversion = (C0 - C) / C0
    return conversion

# 使用範例
initial_conc = 1.0
final_conc = 0.15
conversion = calculate_conversion(initial_conc, final_conc)
print(f"轉化率： {conversion:.2%}")

# 熱容計算（多項式擬合）
def heat_capacity(T, a, b, c, d=0):
    """
    計算熱容 Cp = a + bT + cT² + dT³
    
    Parameters:
    -----------
    T : float
        溫度 (K)
    a, b, c, d : float
        多項式係數
    
    Returns:
    --------
    float
        熱容 (J/(mol·K))
    """
    Cp = a + b*T + c*T**2 + d*T**3
    return Cp

# 水的熱容（簡化範例）
T = 373.15  # K
Cp = heat_capacity(T, a=75.3, b=0.0, c=0.0)
print(f"水在 {T} K 的熱容： {Cp:.2f} J/(mol·K)")
```

### 5.5 Lambda 函式

Lambda 函式是匿名函式，通常用於簡單的一次性操作。

```python
# 基本 lambda 函式
square = lambda x: x**2
print(square(5))  # 25

# 用於排序
temperatures = [
    {"time": "08:00", "value": 25.5},
    {"time": "09:00", "value": 26.2},
    {"time": "10:00", "value": 24.8}
]

# 根據溫度值排序
sorted_temps = sorted(temperatures, key=lambda x: x["value"])
print(sorted_temps)

# 用於 map() 和 filter()
celsius_temps = [0, 25, 50, 75, 100]
fahrenheit_temps = list(map(lambda c: (c * 9/5) + 32, celsius_temps))
print(f"華氏溫度： {fahrenheit_temps}")

# 過濾高溫數據
high_temps = list(filter(lambda t: t["value"] > 25.0, temperatures))
print(f"高溫記錄： {high_temps}")
```

---

## 6. 模組與套件

模組是包含 Python 程式碼的檔案，套件是多個模組的集合。使用模組可以組織程式碼並重複使用功能。

### 6.1 導入模組

```python
# 導入整個模組
import math
result = math.sqrt(16)  # 4.0
pi = math.pi            # 3.14159...

# 導入特定函式
from math import sqrt, pi
result = sqrt(16)

# 導入並重新命名
import numpy as np
array = np.array([1, 2, 3])

# 導入所有內容（不建議）
from math import *

# 化工常用模組
import numpy as np          # 數值運算
import pandas as pd         # 資料處理
import matplotlib.pyplot as plt  # 資料視覺化
```

### 6.2 常用內建模組

```python
# math：數學運算
import math
result = math.exp(2)        # e^2
result = math.log(10)       # ln(10)
result = math.sin(math.pi)  # sin(π)

# random：隨機數生成
import random
rand_num = random.random()           # 0-1 之間的隨機數
rand_int = random.randint(1, 10)     # 1-10 之間的隨機整數
choice = random.choice([1, 2, 3])    # 隨機選擇

# datetime：日期時間處理
import datetime
now = datetime.datetime.now()
print(f"當前時間： {now}")

date = datetime.date(2025, 1, 27)
print(f"指定日期： {date}")

# os：作業系統介面
import os
current_dir = os.getcwd()            # 取得當前工作目錄
os.makedirs("data", exist_ok=True)   # 建立資料夾
```

### 6.3 建立自己的模組

假設我們建立一個名為 `reactor_utils.py` 的模組：

```python
# reactor_utils.py
"""
反應器工具函式模組
"""

def calculate_conversion(C0, C):
    """計算轉化率"""
    return (C0 - C) / C0

def calculate_yield(product_formed, reactant_consumed):
    """計算產率"""
    return product_formed / reactant_consumed

def calculate_selectivity(target_product, total_products):
    """計算選擇性"""
    return target_product / total_products

# 模組級常數
R_GAS = 8.314  # J/(mol·K)
AVOGADRO = 6.022e23  # 1/mol
```

使用自定義模組：

```python
# 在另一個檔案中使用
import reactor_utils

conversion = reactor_utils.calculate_conversion(1.0, 0.2)
print(f"轉化率： {conversion:.2%}")

# 或者
from reactor_utils import calculate_conversion, R_GAS

conversion = calculate_conversion(1.0, 0.2)
print(f"氣體常數： {R_GAS} J/(mol·K)")
```

---

## 7. 例外處理 (Exception Handling)

例外處理用於處理程式執行時可能發生的錯誤，讓程式更加穩健。

### 7.1 基本例外處理

```python
# try-except 基本語法
try:
    result = 10 / 0
except ZeroDivisionError:
    print("錯誤：不能除以零")

# 捕獲多種例外
try:
    number = int("abc")
except ValueError:
    print("錯誤：無法轉換為整數")
except ZeroDivisionError:
    print("錯誤：除以零")

# 捕獲所有例外（不建議過度使用）
try:
    # 可能發生錯誤的程式碼
    result = risky_operation()
except Exception as e:
    print(f"發生錯誤： {e}")
```

### 7.2 else 和 finally

```python
# else：沒有例外時執行
try:
    result = 10 / 2
except ZeroDivisionError:
    print("除以零錯誤")
else:
    print(f"計算成功： {result}")

# finally：無論是否發生例外都會執行
try:
    file = open("data.txt", "r")
    data = file.read()
except FileNotFoundError:
    print("檔案不存在")
finally:
    file.close()  # 確保檔案被關閉
```

### 7.3 拋出例外

```python
# 使用 raise 拋出例外
def check_temperature(temp):
    if temp < -273.15:
        raise ValueError("溫度不能低於絕對零度")
    if temp > 1000:
        raise ValueError("溫度超出測量範圍")
    return True

try:
    check_temperature(-300)
except ValueError as e:
    print(f"溫度檢查失敗： {e}")
```

### 7.4 化工應用：數據驗證

```python
def validate_reactor_parameters(temperature, pressure, flow_rate):
    """
    驗證反應器參數是否在安全範圍內
    
    Parameters:
    -----------
    temperature : float
        溫度 (°C)
    pressure : float
        壓力 (bar)
    flow_rate : float
        流量 (L/min)
    
    Raises:
    -------
    ValueError
        當參數超出安全範圍時
    """
    errors = []
    
    # 溫度檢查
    if not (250 <= temperature <= 450):
        errors.append(f"溫度 {temperature}°C 超出範圍 [250, 450]")
    
    # 壓力檢查
    if not (5 <= pressure <= 15):
        errors.append(f"壓力 {pressure} bar 超出範圍 [5, 15]")
    
    # 流量檢查
    if not (50 <= flow_rate <= 150):
        errors.append(f"流量 {flow_rate} L/min 超出範圍 [50, 150]")
    
    if errors:
        raise ValueError("\n".join(errors))
    
    return True

# 使用範例
try:
    validate_reactor_parameters(
        temperature=500,  # 超出範圍
        pressure=10,
        flow_rate=100
    )
    print("✓ 參數驗證通過")
except ValueError as e:
    print(f"⚠ 參數驗證失敗：\n{e}")

# 讀取數據檔案的穩健處理
def read_sensor_data(filename):
    """
    讀取感測器數據檔案
    
    Parameters:
    -----------
    filename : str
        檔案名稱
    
    Returns:
    --------
    list
        數據列表，若讀取失敗則返回空列表
    """
    try:
        with open(filename, 'r') as file:
            data = [float(line.strip()) for line in file]
        print(f"✓ 成功讀取 {len(data)} 筆數據")
        return data
    except FileNotFoundError:
        print(f"⚠ 檔案 {filename} 不存在")
        return []
    except ValueError:
        print(f"⚠ 檔案包含無效的數值格式")
        return []
    except Exception as e:
        print(f"⚠ 讀取檔案時發生錯誤： {e}")
        return []

# 使用範例
data = read_sensor_data("temperatures.txt")
if data:
    avg_temp = sum(data) / len(data)
    print(f"平均溫度： {avg_temp:.2f}°C")
```

---

## 8. Python 在資料科學與機器學習中的優勢

### 8.1 為什麼 Python 是資料科學的首選語言？

Python 之所以成為資料科學與機器學習領域的主流語言，主要有以下原因：

#### 1. 豐富的科學計算生態系統

```python
# NumPy：高效能數值運算
import numpy as np
data = np.array([[1, 2, 3], [4, 5, 6]])
mean_value = np.mean(data)
std_value = np.std(data)

# Pandas：強大的資料處理
import pandas as pd
df = pd.DataFrame({
    'Temperature': [25.5, 26.0, 24.8],
    'Pressure': [10.1, 10.3, 10.0]
})
summary = df.describe()

# Matplotlib/Seaborn：資料視覺化
import matplotlib.pyplot as plt
plt.plot(df['Temperature'])
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Trend')
plt.show()
```

#### 2. 機器學習框架

```python
# Scikit-learn：傳統機器學習
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# TensorFlow/Keras：深度學習
import tensorflow as tf
from tensorflow import keras

# PyTorch：深度學習研究
import torch
import torch.nn as nn
```

#### 3. 化工領域專用套件

```python
# CoolProp：熱力學性質計算
from CoolProp.CoolProp import PropsSI
density = PropsSI('D', 'T', 298.15, 'P', 101325, 'Water')

# Cantera：化學反應動力學
import cantera as ct
gas = ct.Solution('gri30.yaml')

# DWSIM：化工製程模擬（Python API）
# Aspen Plus：可透過 Python 自動化
```

### 8.2 資料科學工作流程

Python 貫穿整個資料科學工作流程：

```python
# 1. 資料收集與載入
import pandas as pd
data = pd.read_csv('reactor_data.csv')

# 2. 資料清理與前處理
data = data.dropna()  # 移除缺失值
data['Temperature_K'] = data['Temperature_C'] + 273.15  # 單位轉換

# 3. 探索性資料分析 (EDA)
print(data.describe())
print(data.corr())

# 4. 特徵工程
data['Temp_Pressure_Ratio'] = data['Temperature'] / data['Pressure']

# 5. 模型建立與訓練
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X = data[['Temperature', 'Pressure', 'Flow_Rate']]
y = data['Conversion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 6. 模型評估
from sklearn.metrics import r2_score, mean_squared_error
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# 7. 結果視覺化
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Conversion')
plt.ylabel('Predicted Conversion')
plt.title(f'Model Performance (R² = {r2:.3f})')
plt.show()
```

### 8.3 化工領域的 Python 應用案例

#### 案例 1：反應動力學參數擬合

```python
import numpy as np
from scipy.optimize import curve_fit

# Arrhenius 方程式： k = A * exp(-Ea / (R*T))
def arrhenius(T, A, Ea):
    R = 8.314  # J/(mol·K)
    return A * np.exp(-Ea / (R * T))

# 實驗數據
temperatures = np.array([300, 320, 340, 360, 380])  # K
rate_constants = np.array([0.01, 0.05, 0.15, 0.35, 0.70])  # 1/s

# 參數擬合
params, covariance = curve_fit(arrhenius, temperatures, rate_constants)
A_fitted, Ea_fitted = params

print(f"擬合結果：")
print(f"頻率因子 A = {A_fitted:.2e} 1/s")
print(f"活化能 Ea = {Ea_fitted:.2f} J/mol")
```

#### 案例 2：製程數據異常檢測

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# 模擬製程數據
np.random.seed(42)
normal_data = np.random.normal(100, 5, 950)
anomalies = np.random.normal(100, 5, 50) + 30  # 異常值
data = np.concatenate([normal_data, anomalies]).reshape(-1, 1)

# 異常檢測模型
model = IsolationForest(contamination=0.05, random_state=42)
predictions = model.fit_predict(data)

# 結果分析
anomaly_indices = np.where(predictions == -1)[0]
print(f"偵測到 {len(anomaly_indices)} 個異常值")
```

#### 案例 3：蒸餾塔模擬計算

```python
import numpy as np

def mccabe_thiele_stages(xF, xD, xB, R, q, alpha):
    """
    McCabe-Thiele 方法計算理論板數
    
    Parameters:
    -----------
    xF : float
        進料組成（摩爾分率）
    xD : float
        塔頂產物組成
    xB : float
        塔底產物組成
    R : float
        回流比
    q : float
        進料熱狀態參數
    alpha : float
        相對揮發度
    
    Returns:
    --------
    int
        理論板數
    """
    # 操作線方程
    def operating_line_rectifying(x):
        return (R / (R + 1)) * x + xD / (R + 1)
    
    def operating_line_stripping(x):
        slope = (R + 1) / (R + 1 - q)
        intercept = (xB - slope * xF)
        return slope * x + intercept
    
    # 平衡線方程
    def equilibrium_line(x):
        return (alpha * x) / (1 + (alpha - 1) * x)
    
    # 逐級計算（簡化版）
    stages = 0
    x = xD
    
    while x > xB and stages < 100:
        # 從操作線到平衡線
        if x > xF:
            y = operating_line_rectifying(x)
        else:
            y = operating_line_stripping(x)
        
        # 從平衡線到操作線
        x = y / (alpha - (alpha - 1) * y)
        stages += 1
    
    return stages

# 使用範例
stages = mccabe_thiele_stages(
    xF=0.5,    # 進料組成
    xD=0.95,   # 塔頂組成
    xB=0.05,   # 塔底組成
    R=2.0,     # 回流比
    q=1.0,     # 飽和液體進料
    alpha=2.5  # 相對揮發度
)

print(f"所需理論板數： {stages}")
```

### 8.4 Python 學習資源

#### 官方文件與教學
- Python 官方教學： https://docs.python.org/zh-tw/3/tutorial/
- NumPy 官方文件： https://numpy.org/doc/
- Pandas 官方文件： https://pandas.pydata.org/docs/
- Scikit-learn 官方教學： https://scikit-learn.org/stable/tutorial/

#### 化工專用資源
- ChemE Code：化工計算範例
- Process Control：製程控制模擬
- Python for Chemical Engineers：專業書籍

#### 線上練習平台
- LeetCode：程式設計練習
- HackerRank：Python 挑戰題
- Kaggle：資料科學競賽與學習

---

## 9. 實務建議與最佳實踐

### 9.1 程式碼風格指南

遵循 PEP 8 風格指南，讓程式碼更易讀：

```python
# 好的命名方式
reactor_temperature = 350.0
MAX_PRESSURE = 15.0
calculate_conversion_rate()

# 不好的命名方式
rt = 350.0
MAXP = 15.0
calcConvRate()

# 適當的空白與換行
def calculate_yield(product, reactant):
    """計算產率"""
    if reactant == 0:
        raise ValueError("反應物不能為零")
    
    yield_value = product / reactant
    return yield_value

# 適當的註解
# 計算 Arrhenius 方程式
k = A * np.exp(-Ea / (R * T))  # k: 速率常數 (1/s)
```

### 9.2 除錯技巧

```python
# 使用 print() 除錯
temperature = 350.0
print(f"Debug: temperature = {temperature}")

# 使用 assert 檢查假設
def calculate_conversion(C0, C):
    assert C0 > 0, "初始濃度必須大於 0"
    assert C >= 0, "濃度不能為負"
    return (C0 - C) / C0

# 使用 Python 除錯器 (pdb)
import pdb
pdb.set_trace()  # 設定中斷點

# 使用 try-except 捕獲詳細錯誤
try:
    result = complex_calculation()
except Exception as e:
    import traceback
    traceback.print_exc()  # 印出完整錯誤追蹤
```

### 9.3 效能優化

```python
# 使用列表生成式（更快）
squares = [x**2 for x in range(1000)]

# 避免使用迴圈（較慢）
squares = []
for x in range(1000):
    squares.append(x**2)

# 使用 NumPy 進行向量化運算
import numpy as np
temperatures = np.array([25.0, 30.0, 35.0, 40.0])
fahrenheit = temperatures * 9/5 + 32  # 向量化運算

# 避免逐元素運算（較慢）
fahrenheit = []
for temp in temperatures:
    fahrenheit.append(temp * 9/5 + 32)
```

### 9.4 程式碼組織

```python
# 良好的程式結構
"""
reactor_simulation.py
反應器模擬模組
"""

# 1. 匯入套件
import numpy as np
import pandas as pd

# 2. 常數定義
R_GAS = 8.314  # J/(mol·K)
AVOGADRO = 6.022e23

# 3. 函式定義
def calculate_conversion(C0, C):
    """計算轉化率"""
    return (C0 - C) / C0

def simulate_reactor(params):
    """模擬反應器"""
    pass

# 4. 類別定義
class Reactor:
    """反應器類別"""
    def __init__(self, volume, temperature):
        self.volume = volume
        self.temperature = temperature

# 5. 主程式
if __name__ == "__main__":
    # 當直接執行此檔案時才會執行
    reactor = Reactor(volume=100, temperature=350)
    print(f"反應器體積： {reactor.volume} L")
```

---

## 10. 課程總結

### 10.1 本單元學習重點

在本單元中，我們學習了 Python 程式語言的核心概念：

1. **基本語法**：註解、縮排、程式碼執行順序
2. **變數與資料型態**：數值、字串、布林值、串列、元組、字典、集合
3. **運算子**：算術、比較、邏輯、賦值運算子
4. **控制流程**：條件語句（if-elif-else）、迴圈（for、while）
5. **函式**：函式定義、參數、返回值、lambda 函式
6. **模組**：導入模組、使用內建模組、建立自定義模組
7. **例外處理**：try-except、raise、finally
8. **Python 優勢**：豐富的生態系統、化工領域應用

### 10.2 關鍵要點

- ✓ Python 語法簡潔易讀，適合快速開發
- ✓ 豐富的科學計算套件支援資料分析與機器學習
- ✓ 函式與模組有助於程式碼重用與組織
- ✓ 例外處理讓程式更加穩健
- ✓ 良好的編程習慣提高程式碼品質

### 10.3 下一步學習

完成本單元後，建議您：

1. **練習程式碼**：透過 [Unit02_Python_Basics.ipynb](Unit02_Python_Basics.ipynb) 進行實作練習
2. **學習 NumPy**：前往 Unit03 學習數值運算與陣列操作
3. **學習 Pandas**：前往 Unit03 學習資料處理與分析
4. **學習視覺化**：前往 Unit04 學習 Matplotlib 與 Seaborn
5. **持續實作**：多寫程式碼，從錯誤中學習

### 10.4 化工應用展望

Python 在化工領域有廣泛的應用前景：

- 製程數據分析與視覺化
- 反應動力學參數擬合
- 製程控制與最佳化
- 機器學習預測模型
- 異常檢測與故障診斷
- 製程模擬與設計

掌握 Python 基礎後，您將能夠運用這些技術解決實際的化工問題，提升工程效率與創新能力。

---

## 11. 延伸閱讀

### 11.1 推薦書籍

- **Python Crash Course** - Eric Matthes
- **Automate the Boring Stuff with Python** - Al Sweigart
- **Python for Data Analysis** - Wes McKinney
- **Python for Chemical Engineers** - 化工專業教材

### 11.2 線上資源

- Python 官方文件： https://docs.python.org/
- Real Python 教學網站： https://realpython.com/
- W3Schools Python 教學： https://www.w3schools.com/python/
- GitHub 化工專案： 搜尋 "chemical engineering python"

### 11.3 練習專案建議

1. **溫度單位轉換器**：建立攝氏、華氏、克氏溫度轉換程式
2. **理想氣體計算器**：輸入三個參數計算第四個參數
3. **反應動力學模擬**：模擬一階反應的濃度變化
4. **製程數據分析器**：讀取 CSV 檔案並產生統計報表
5. **簡易配方計算**：根據配方比例計算所需原料量

---

**恭喜您完成 Unit02 Python 程式語言基礎！**

請繼續前往 [Unit02_Python_Basics.ipynb](Unit02_Python_Basics.ipynb) 進行實作練習，鞏固您所學的知識。

**下一單元預告**：Unit03 將深入學習 NumPy 與 Pandas，這是資料科學與機器學習的重要工具。

---

**課程資訊**
- 課程名稱：AI在化工上之應用
- 課程單元：Unit 02 Python 程式語言基礎
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-01-28

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---
