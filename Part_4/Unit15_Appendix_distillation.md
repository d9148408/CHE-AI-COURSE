# Unit15_Appendix_distillation | Distillation Column: DNN predicts ethanol molar concentration

本附錄提供一個蒸餾塔（distillation column）的工業資料案例，示範如何用 **DNN (MLP) 回歸**建立軟測器（soft sensor），由塔內溫度分佈與流量等可量測訊號，預測產品端 **ethanol molar concentration**（資料欄位：`Ethanol concentration`）。

---

## 1. 案例背景：為什麼要預測乙醇莫耳濃度？

蒸餾塔的產品組成是品質與經濟效益的核心指標，但在實務上「即時、連續」量測組成通常成本高、維護複雜或頻率不足。因此常見做法是用已部署的感測器（壓力、塔板溫度、流量）建立 **資料驅動軟測器**：

- 即時品質估測（quality inference）
- 操作調整與控制（例如維持產品規格、降低能耗）
- 異常預警（如塔板液泛、負載變動造成的品質漂移）

本案例為監督式回歸：

$$
\hat{y} = f(\mathbf{x}),\quad \mathbf{x}\in \mathbb{R}^{20},\ y\in \mathbb{R}
$$

其中 $y$ 是乙醇莫耳濃度（`Ethanol concentration`），$\mathbf{x}$ 是壓力、塔板溫度與流量組成的特徵向量。

---

## 2. 資料集與欄位

資料檔案：

- `Part_4/data/distillation_column/dataset_distill.csv`

### 2.1 欄位說明

- `Pressure`：塔壓 (bar)
- `T1` ~ `T14`：各塔板溫度 (K)
- `L`：Liquid flowrate (Kg mol/hour)
- `V`：Vapor flowrate (Kg mol/hour)
- `D`：Distillate flowrate (Kg mol/hour)
- `B`：Bottoms flowrate (Kg mol/hour)
- `F`：Feed flowrate (Kg mol/hour)
- `Ethanol concentration`：乙醇莫耳濃度（模型輸出/目標）

### 2.2 資料格式注意事項（很重要）

- CSV 以分號 `;` 作為分隔符號（不是逗號 `,`）。
- `L`/`V` 欄位中包含小數逗號科學記號（例如 `1,23E+08`），需先將小數逗號轉為小數點再轉成數值。

---

## 3. EDA 與特徵直覺（含範例輸出）

### 3.1 範例執行結果：目標分佈

Notebook 會先畫出 `Ethanol concentration` 分佈，建立尺度直覺（本資料範圍約 `0.538` 到 `0.892`）：

![Target distribution](outputs/P4_Unit15_Appendix_Distillation/figs/target_hist.png)

### 3.2 範例執行結果：相關性（Correlation）

使用 Pearson correlation 快速檢查線性關係（提醒：相關不等於因果）。

$$
\rho_{X,Y}=\frac{\mathrm{cov}(X,Y)}{\sigma_X \sigma_Y}
$$

本資料的 `Pressure` 為常數 `1.01`（無變化），因此與目標的相關係數會是 NaN（不具資訊量）。溫度特徵（尤其 `T1`）對目標呈現強烈負相關，代表在此資料分佈下，溫度越高時乙醇濃度傾向越低（可能反映操作狀態/負載對溫度剖面與分離效果的共同影響）。

本次資料集中（轉換 `L`/`V` 為數值後）與目標的線性相關性重點如下（四捨五入到小數第 3 位，`Pressure` 因為為常數而略過）：

| 類型 | 代表特徵 | corr(特徵, Ethanol concentration) |
|---|---|---:|
| 強負相關（溫度剖面） | `T1` | -0.981 |
|  | `T2` | -0.928 |
|  | `T3` | -0.886 |
| 正相關（流量/分配） | `B` | 0.321 |
|  | `F` | 0.272 |
| 弱相關（接近 0） | `L` / `V` | 約 -0.009 |

![Correlation bar](outputs/P4_Unit15_Appendix_Distillation/figs/corr_bar.png)

---

## 4. 建模流程：切分、清理、標準化（避免 data leakage）

### 4.1 切分策略

本案例採用：

- train / valid / test = 70% / 15% / 15%

valid 用於 early stopping 監控泛化誤差；test 作為最終評估集。

### 4.2 L/V 極端值處理（示範）

資料中 `L`/`V` 會出現非常大的數值（例如 `1.23e8`, `1.23e9`），容易造成尺度主導與訓練不穩定。

本範例做法：

- 只用 **訓練集** 統計量，對 `L`/`V` 設定上限並做 clipping
- 套用同一上限到 valid/test（避免洩漏）

本次範例執行得到的上限為：

- `L` cap = `22500`
- `V` cap = `22650`

在本資料集中，約有 `258/4408 ≈ 5.9%` 的樣本 `L` 與 `V` 會超過該上限而被 clipping（這些值多為科學記號型態的極大值）。此步驟的目的不是「修正真實物理」，而是避免極端尺度在標準化前就主導訓練，讓模型把學習能力用在更有資訊的區域。

### 4.3 標準化（StandardScaler）

神經網路與梯度型方法通常對特徵尺度敏感，因此使用 Z-score：

$$
x' = \frac{x-\mu_{\text{train}}}{\sigma_{\text{train}}}
$$

注意：$\mu_{\text{train}},\sigma_{\text{train}}$ 必須只由訓練集估計，才能避免 data leakage。

---

## 5. Baseline: Ridge Regression vs DNN (MLP)

### 5.1 Ridge Regression（baseline）

Ridge 是線性回歸加上 L2 正則化：

$$
\min_{\mathbf{w}, b}\ \frac{1}{n}\sum_{i=1}^{n}\left(y_i-(\mathbf{w}^\top \mathbf{x}_i + b)\right)^2 + \lambda \|\mathbf{w}\|_2^2
$$

優點：訓練快、可解釋性好、適合作為第一個可用基線。

### 5.2 DNN (MLP) 回歸模型

MLP（多層感知器）可以表示為多層非線性映射：

$$
\mathbf{h}^{(0)}=\mathbf{x},\quad
\mathbf{h}^{(l)}=\phi\left(\mathbf{W}^{(l)}\mathbf{h}^{(l-1)}+\mathbf{b}^{(l)}\right)
$$

$$
\hat{y}=\mathbf{W}^{(L)}\mathbf{h}^{(L-1)}+\mathbf{b}^{(L)}
$$

本範例使用 MSE 作為訓練損失：

$$
\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

並以 MAE / RMSE / $R^2$ 做評估：

$$
\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|,\quad
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

$$
R^2=1-\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}
$$

---

## 6. 範例執行結果與分析（包含圖片）

### 6.1 訓練曲線（loss curve）

訓練曲線用於判讀是否過擬合：

- 若 train loss 持續下降但 valid loss 上升：常見過擬合訊號
- 若兩者下降後趨於平穩：通常代表收斂且泛化較佳

本次實際執行（Keras backend）跑滿 `500` epochs，最後一個 epoch 的記錄約為 `val_loss ≈ 4.17e-06`、`val_mae ≈ 0.0017`，且在 ReduceLROnPlateau 作用下學習率降至 `1e-06`；整體曲線呈現穩定收斂。

![Loss curve](outputs/P4_Unit15_Appendix_Distillation/figs/loss_curve.png)

### 6.2 性能比較：Ridge vs DNN（MLP）

本次 Notebook 執行 backend 為 `tensorflow/keras`，資料切分筆數為 train `3085` / valid `661` / test `662`（同一切分下比較 Ridge 與 DNN）。

| 模型 | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Ridge baseline | 0.004795 | 0.007037 | 0.992161 |
| DNN (MLP) | 0.001837 | 0.003255 | 0.998323 |

解讀（本案例的重要教學點）：

- 兩者 $R^2$ 都很高，代表此資料在目前特徵下「可預測性強」，且線性關係已能解釋大部分變異（Ridge 已達 0.992）。
- DNN 仍能進一步降低誤差：RMSE 約下降 `53.7%`、MAE 約下降 `61.7%`，顯示非線性/交互作用仍有可學習的空間。
- 若你的實務需求偏向可解釋、維護簡單，Ridge 常是很強的第一版；若需求偏向更高精度，DNN 可作為進階版本。

#### 重要提醒：資料特性可能讓評估偏樂觀

本資料集中存在相當比例的「完全重複特徵列」（約 `945/4408 ≈ 21.4%`）。若採用隨機切分，重複樣本可能同時出現在 train 與 test，導致測試表現看起來非常好但偏向「記憶相同工況」的能力。

建議在課堂上延伸討論/練習：

- 先做去重（deduplicate）再切分，或
- 使用分組切分（group split）：以操作工況（例如 `D/B/F` 或更完整的條件組合）作為 group，確保同一工況不會同時出現在 train 與 test。

### 6.3 Parity plot 與 Residual plot（診斷）

Parity plot（真值 vs 預測）越貼近對角線 $y=\hat{y}$ 越好；Residual plot 用於檢查殘差是否存在結構（若殘差呈現曲線或分段，常暗示仍缺少關鍵特徵或需要分段建模）。

![Parity plot](outputs/P4_Unit15_Appendix_Distillation/figs/parity_plot.png)

![Residual plot](outputs/P4_Unit15_Appendix_Distillation/figs/residual_plot.png)

### 6.4 預測分佈與特徵關係（sanity check）

這裡使用 holdout test set 檢查：

- `y_test` 的真實分佈 vs `y_pred` 的預測分佈（若差異巨大，可能表示系統性偏差或某些工況預測不穩）
- `y_pred` 與關鍵特徵（`Pressure`、`T14`）的關係是否合理

![Distribution: true vs predicted](outputs/P4_Unit15_Appendix_Distillation/figs/test_true_vs_pred_dist.png)

![test: y_pred vs Pressure/T14](outputs/P4_Unit15_Appendix_Distillation/figs/test_pred_scatter.png)

---

## 7. 交付物與如何重現

### 7.1 對應程式碼（Notebook）

- `Part_4/Unit15_Appendix_distillation.ipynb`

### 7.2 輸出檔案（自動產生）

Notebook 執行後會輸出到：

- 指標檔：`Part_4/outputs/P4_Unit15_Appendix_Distillation/metrics.json`
- 圖片：`Part_4/outputs/P4_Unit15_Appendix_Distillation/figs/*.png`
- 模型與 scaler：`Part_4/outputs/P4_Unit15_Appendix_Distillation/models/`

### 7.3 執行建議

建議從 `Part_4` 目錄執行 Notebook（路徑假設以 `Part_4` 為工作目錄）。
