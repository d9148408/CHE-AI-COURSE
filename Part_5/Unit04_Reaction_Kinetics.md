# [Unit 04] 應用二：反應動力學模擬 (Reaction Kinetics Simulation)

**課程名稱**：化工資料科學與機器學習實務（CHE-AI-101）  
**本單元目標**：
- 理解批式反應器 (Batch Reactor) 的質量守恆原理  
- 掌握連串反應 (Consecutive Reaction) 的動力學特性  
- 使用 Python `scipy.integrate.odeint` 求解常微分方程組 (ODEs)  
- 透過模擬結果進行製程優化：尋找最佳反應時間 ($t_{opt}$)  
- 驗證數值模擬與理論解析解的一致性

---

## 1. 反應動力學理論基礎 (Theoretical Background)

在進行程式模擬前，我們必須先建立正確的物理模型。

### 1.1 批式反應器質量守恆
對於一個理想混合的批式反應器 (Batch Reactor)，假設體積 $V$ 固定，根據質量守恆定律：
$$ \text{Accumulation} = \text{In} - \text{Out} + \text{Generation} - \text{Consumption} $$

由於是批式操作（無進出料），$\text{In} = \text{Out} = 0$，因此：
$$ \frac{dN_i}{dt} = r_i V $$
其中 $N_i$ 為成份 $i$ 的莫耳數，$r_i$ 為反應速率。若體積恆定，可寫成濃度形式 ($C_i = N_i/V$)：
$$ \frac{dC_i}{dt} = r_i $$

### 1.2 反應速率定律 (Rate Laws)
對於基元反應 (Elementary Reaction)，反應速率通常與濃度呈冪次關係。
若反應為 $aA + bB \to P$，則：
$$ -r_A = k C_A^a C_B^b $$
其中 $k$ 為反應速率常數 (Rate Constant)，通常依循 **Arrhenius Equation**：
$$ k = A \exp\left(\frac{-E_a}{RT}\right) $$
這意味著溫度 $T$ 是控制反應速率的關鍵變數（本單元暫假設為恆溫反應）。

---

## 2. 案例研究：連串反應 (Consecutive Reactions)

本單元探討經典的連串一級反應，這在化工製程（如部分氧化反應、藥物合成）中非常常見：

$$ A \xrightarrow{k_1} B \xrightarrow{k_2} C $$

- **A**：原料 (Reactant)
- **B**：中間產物 (Intermediate Product, 目標產物)
- **C**：副產物 (By-product/Waste)

### 2.1 微分方程組 (ODEs)
根據質量作用定律 (Mass Action Law)，我們可以寫出各物種的速率方程：

1.  **A 的消耗**：
    $$ \frac{dC_A}{dt} = -k_1 C_A $$
2.  **B 的生成與消耗**：
    $$ \frac{dC_B}{dt} = k_1 C_A - k_2 C_B $$
3.  **C 的生成**：
    $$ \frac{dC_C}{dt} = k_2 C_B $$

### 2.2 解析解 vs 數值解
對於這個特定系統，其實存在**解析解 (Analytical Solution)**：
$$ C_A(t) = C_{A0} e^{-k_1 t} $$
$$ C_B(t) = C_{A0} \frac{k_1}{k_2 - k_1} (e^{-k_1 t} - e^{-k_2 t}) $$

然而，一旦反應變複雜（例如 $k$ 隨溫度變化、反應級數非整數、或有更多副反應），解析解將變得極難推導甚至不存在。這就是為什麼我們需要學習 **數值模擬 (Numerical Simulation)**。

---

## 3. Python 實作：數值求解 ODEs

我們使用 `scipy.integrate.odeint`，其底層使用 **LSODA** 演算法，能自動在非剛性 (Non-stiff, Adams方法) 與剛性 (Stiff, BDF方法) 求解器間切換，非常適合化學動力學問題。

### 3.1 定義模型函數

```python
def reaction_model(C, t, k1, k2):
    Ca, Cb, Cc = C  # Unpack concentrations
    
    dCa_dt = -k1 * Ca
    dCb_dt = k1 * Ca - k2 * Cb
    dCc_dt = k2 * Cb
    
    return [dCa_dt, dCb_dt, dCc_dt]
```

### 3.2 執行模擬

```python
from scipy.integrate import odeint
import numpy as np

# 參數設定
k1, k2 = 0.3, 0.1
C0 = [1.0, 0.0, 0.0] # 初始濃度
t = np.linspace(0, 50, 100)

# 求解
solution = odeint(reaction_model, C0, t, args=(k1, k2))
```

---

## 4. 模擬結果深度分析 (Result Analysis)

執行程式後，我們得到如下的濃度變化圖。這張圖蘊含了豐富的製程資訊。

![Reaction Kinetics Simulation](../Jupyter_Scripts/Unit04_Results/01_reaction_kinetics.png)

### 4.1 曲線特徵解讀

1.  **反應物 A (紅線)**：
    - 呈現單調的指數衰減 (Exponential Decay)。
    - 這是典型的一級反應特徵，濃度越高消耗越快，隨時間趨緩。
2.  **中間產物 B (藍線)**：
    - **上升段**：初期 $C_A$ 高，生成速率 $r_{form} = k_1 C_A$ 大於消耗速率 $r_{cons} = k_2 C_B$（因 $C_B$ 尚低），導致 B 累積。
    - **峰值 (Peak)**：當生成速率等於消耗速率時 ($k_1 C_A = k_2 C_B$)，濃度達到最大值。
    - **下降段**：隨著 A 耗盡，生成變慢，而 B 繼續轉化為 C，導致濃度下降。
3.  **產物 C (綠線)**：
    - 呈現 S 型曲線 (Sigmoid-like)。初期有延遲 (Lag Phase)，因為需要先生成 B 才能生成 C。

### 4.2 最佳反應時間的理論驗證

我們能否用數學驗證模擬結果的正確性？
欲求 $C_B$ 的最大值，令其對時間導數為 0：
$$ \frac{dC_B}{dt} = k_1 C_A - k_2 C_B = 0 \implies C_B = \frac{k_1}{k_2} C_A $$

代入解析解公式推導，可得理論最佳時間 $t_{opt}$：
$$ t_{opt} = \frac{\ln(k_1 / k_2)}{k_1 - k_2} $$

**驗證本範例 ($k_1=0.3, k_2=0.1$)：**
$$ t_{opt} = \frac{\ln(0.3 / 0.1)}{0.3 - 0.1} = \frac{\ln(3)}{0.2} \approx \frac{1.0986}{0.2} = 5.493 \text{ sec} $$

**模擬結果**：程式輸出的最佳時間約為 **5.5 sec**。
$\Rightarrow$ **數值模擬與理論計算高度吻合！** 這證明了我們的模型與求解過程是正確的。

### 4.3 質量守恆檢查

在任何時刻 $t$，總莫耳濃度應守恆：
$$ C_A(t) + C_B(t) + C_C(t) = C_{A0} $$
程式碼中的 `total_C` 檢查即是為了確保數值積分過程中沒有發生發散或誤差累積。

---

## 5. 實戰演練：`Part_5/Unit04_Reaction_Kinetics.ipynb`

本單元程式碼檔案包含完整的實作流程：

1.  **參數定義**：設定 $k$ 值與初始條件。
2.  **ODE 建模**：將化學方程式轉譯為 Python 函數。
3.  **數值求解**：利用 `odeint` 算出濃度曲線。
4.  **數據分析**：
    - 使用 `np.argmax` 自動尋找 $t_{opt}$。
    - 驗證質量守恆。
5.  **視覺化**：繪製專業的動力學曲線圖。

### 建議練習
試著將 $k_2$ 改為 $1.0$ (即 $k_2 \gg k_1$)，重新執行模擬。
- **預期結果**：B 的峰值會變得非常低且非常早出現。
- **物理意義**：中間產物極不穩定，一生成就馬上分解。這在製程上代表很難獲得高純度的 B，可能需要改變反應路徑或使用催化劑。

---

## 6. 進階動力學：溫度效應 (Temperature Effect)

在真實化工製程中，反應速率常數 $k$ 並非定值，而是強烈依賴於溫度。

### 6.1 理論背景：Arrhenius 方程式
反應速率常數 $k$ 與溫度的關係遵循 **Arrhenius Equation**：
$$ k = A \exp\left(\frac{-E_a}{RT}\right) $$

取對數微分可得其敏感度：
$$ \frac{d \ln k}{dT} = \frac{E_a}{RT^2} $$
這條式子告訴我們：**活化能 $E_a$ 越大的反應，其速率常數對溫度越敏感。**

### 6.2 競爭反應分析
對於連串反應 $A \xrightarrow{k_1} B \xrightarrow{k_2} C$：
- 生成速率 $r_1 = k_1 C_A$
- 消耗速率 $r_2 = k_2 C_B$

若 $E_{a1} > E_{a2}$ (如本範例設定 $80 > 60$ kJ/mol)：
當溫度 $T$ 上升時，$k_1$ 的增加幅度會大於 $k_2$。這意味著 $k_1/k_2$ 的比值會變大，有利於中間產物 B 的累積。

### 6.3 模擬結果解析 (Industrial Scale)
![Temperature Effect](../Jupyter_Scripts/Unit04_Results/02_temperature_effect.png)

觀察模擬結果（藍線 vs 紅線）：
1.  **時間軸壓縮 (Kinetics)**：
    - 低溫 ($360K \approx 87^\circ C$) 時，反應緩慢，峰值出現在約 **60 分鐘** 處。
    - 高溫 ($400K \approx 127^\circ C$) 時，反應加速，峰值提早至約 **10 分鐘** 處。
    - 這符合真實工廠的操作邏輯：升溫可以大幅縮短批次時間 (Batch Time)，提升產能。
2.  **峰值高度提升 (Thermodynamics/Selectivity)**：
    - 紅線的峰值高於藍線。這是因為高溫下 $k_1$ 增長得比 $k_2$ 多，使得 B 在分解前能累積到更高的濃度。
3.  **結論**：對於 $E_{a1} > E_{a2}$ 的系統，**高溫操作**是雙贏策略（時間短、產量高）。

---

## 7. 化工關鍵指標 (Key Performance Indicators)

除了濃度對時間的圖形，化工工程師更習慣使用無因次群 (Dimensionless Groups) 來評估製程效能。

### 7.1 定義與公式

1.  **轉化率 (Conversion, $X_A$)**：
    $$ X_A = \frac{C_{A0} - C_A}{C_{A0}} $$
    代表原料的使用程度。

2.  **產率 (Yield, $Y_B$)**：
    $$ Y_B = \frac{C_B}{C_{A0}} $$
    代表投入的每一莫耳原料，最終回收了多少莫耳的目標產物。這是計算工廠營收最直接的指標。

3.  **選擇率 (Selectivity, $S$)**：
    $$ S = \frac{C_B}{C_{A0} - C_A} = \frac{Y_B}{X_A} $$
    代表「消耗掉的原料」中，有多少比例走到了正確的路徑。這反映了化學反應的效率與副產物廢棄物處理的成本。

### 7.2 模擬結果解析：Yield-Conversion 軌跡
![Yield vs Selectivity](../Jupyter_Scripts/Unit04_Results/03_yield_selectivity.png)

這張圖是反應器操作的核心決策圖：
1.  **選擇率 (綠虛線) 的衰退**：
    - 在反應初期 ($X_A \approx 0$)，由於 $C_B \approx 0$，副反應 $B \to C$ 幾乎不發生，因此選擇率接近 100%。
    - 隨著反應進行，B 濃度累積，分解速率加快，選擇率開始單調下降。
2.  **產率 (藍實線) 的極值**：
    - 產率呈現拋物線形狀。
    - 在 $X_A$ 較低時，產率隨轉化率上升。
    - 在 $X_A$ 過高時，雖然原料用掉了，但大部分都變成了副產物 C，導致 B 的產率反而下降。
3.  **操作策略 (Trade-off)**：
    - **最大產量策略**：操作在藍線最高點對應的轉化率。
    - **高純度/低廢棄物策略**：若副產物 C 很難分離或有毒，我們可能會選擇在較低的轉化率（例如 $X_A = 0.6$）就停止反應，此时選擇率較高，未反應的 A 可以回收再利用 (Recycle)。

---

## 8. 機器學習應用：反應器代理模型 (Surrogate Modeling)

在工業 4.0 與智慧化工廠的架構下，將傳統的「物理模型 (First-principles Model)」與「機器學習 (Data-driven Model)」結合是重要趨勢。

### 8.1 理論背景：為什麼需要代理模型？
在化工製程中，某些物理模型的計算成本極高：
1.  **複雜動力學**：如裂解爐反應包含數千個基元反應，求解 ODE 系統極慢。
2.  **計算流體力學 (CFD)**：結合流場與反應的模擬，單次計算可能耗時數小時。
3.  **即時控制 (Real-time Control)**：工廠控制系統 (DCS) 需要在毫秒級做出決策，物理模型往往來不及計算。

**代理模型 (Surrogate Model)** 的核心概念是用一個計算快速的函數 $\hat{f}$ 來逼近昂貴的物理函數 $f$：
$$ y = f(\mathbf{x}) \approx \hat{f}(\mathbf{x}; \mathbf{w}) $$
其中 $\mathbf{w}$ 是機器學習模型的權重。一旦訓練完成，$\hat{f}$ 的推論速度通常比 $f$ 快數千倍，使即時優化成為可能。

### 8.2 化工反應器類比：從 Batch 到 PFR
本單元雖然模擬 **批式反應器 (Batch Reactor)**，但其數學結論完全適用於連續式的 **平推流反應器 (PFR)**。

*   **Batch Reactor (時間域)**：
    $$ \frac{dC_i}{dt} = r_i $$
    濃度隨 **反應時間 $t$** 變化。

*   **PFR Reactor (空間域)**：
    取一微小體積 $dV$，質量守恆為：
    $$ F_{i,in} - F_{i,out} + r_i dV = 0 \implies \frac{dF_i}{dV} = r_i $$
    由於 $F_i = v_0 C_i$ (假設體積流率 $v_0$ 恆定)：
    $$ v_0 \frac{dC_i}{dV} = r_i \implies \frac{dC_i}{d(V/v_0)} = r_i $$
    定義 **空間停留時間 (Space Time)** $\tau = V/v_0$，則方程式變為：
    $$ \frac{dC_i}{d\tau} = r_i $$

**結論**：Batch 反應器的「反應時間 $t$」在數學上完全等同於 PFR 的「停留時間 $\tau$」。因此，我們用 ML 優化出的最佳時間 $t_{opt}$，可以直接用來設計 PFR 的體積 ($V = v_0 \times t_{opt}$)。

### 8.3 模擬結果解析：操作地圖 (Operating Map)

我們利用 Random Forest 建立的代理模型，繪製了反應器的全域操作地圖。
*(註：為了使結果更具工程意義，我們調整了 Arrhenius 參數，模擬真實化工製程的溫度與時間尺度)*

![ML Optimization](../Jupyter_Scripts/Unit04_Results/04_ml_optimization.png)

**圖表深度解析**：
1.  **山脊形狀 (The Ridge)**：
    - 黃色亮區代表高產率 ($Y_B$) 的操作區間。
    - 地圖呈現一條從「左上 (低溫、長時間)」延伸到「右下 (高溫、短時間)」的山脊。
    - **物理意義**：這反映了動力學的補償效應。溫度低 ($350K$) 時反應慢，需要約 60 分鐘才能達到峰值；溫度高 ($420K$) 時反應快，僅需約 5-10 分鐘。

2.  **最佳操作點 (The Optimum)**：
    - 白色星號標示了全域最高產率點。
    - 位置通常落在 **高溫 ($T \approx 420K$)** 與 **短時間 ($t \approx 5 \text{ min}$)** 的區域。
    - **為何選高溫？**：因為我們設定 $E_{a1} (80) > E_{a2} (60)$，高溫能顯著提升 $k_1/k_2$ 的比值（選擇率），從而獲得更高的最大產率。

3.  **工程決策 (Engineering Decision)**：
    - **PFR 設計建議**：選擇右下角的操作點 ($420K, 5 \text{ min}$)。
    - **優勢**：雖然高溫需要加熱成本，但極短的停留時間意味著我們可以設計一個 **體積非常小** 的反應器來達到相同的產能，大幅降低設備投資成本 (CapEx)。這展示了 AI 如何輔助工程師在多個變數間找到最佳的經濟平衡點。

---

**[Next Unit]**  
掌握了單一條件下的模擬後，**Unit 05** 將進入 **製程最佳化 (Optimization)**。
我們將不再只是「觀察」結果，而是讓電腦自動調整操作參數（如溫度、時間），尋找利潤最大化的操作點。
