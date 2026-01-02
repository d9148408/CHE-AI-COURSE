# Unit15｜深度學習工作流程與軟測器（MLP）：從工程問題到可部署系統

> **Part_4 深度學習系列開篇**：本單元不是「又一個 MNIST 教學」，而是回答化工工程師最關心的問題：**如何把深度學習模型從實驗室帶到現場？**我們將通過「軟測器」這個經典化工AI應用，學習完整的深度學習工程流程：數據洩漏、基線對照、模型監控、交付文檔。

---

## 📚 本單元學習地圖

### 核心目標（Learning Objectives）

完成本單元後，你將能夠：

✅ **理解深度學習在化工的定位**：何時需要 DL？何時不需要？  
✅ **避免數據洩漏**：掌握時間序列切分的正確方法  
✅ **建立基線對照**：Ridge → RandomForest → MLP 的比較策略  
✅ **評估工業可行性**：不只看準確率，更看可維護性、可解釋性、部署成本  
✅ **交付完整系統**：模型卡、OOD 檢測、重訓策略

### 課程定位

| 面向 | 內容 |
|------|------|
| **前置知識** | Part 2 機器學習基礎、Python 數據處理 |
| **技術棧** | TensorFlow/Keras, scikit-learn, pandas |
| **應用場景** | 化工軟測器（Soft Sensor） |
| **後續銜接** | Unit16 CNN（影像）、Unit18 RNN（時間序列）、Unit19 Autoencoder（異常檢測） |

### 為什麼從「軟測器」開始？

1. **化工最常見的 AI 應用**：品質預測、過程監控、優化控制
2. **暴露所有工程挑戰**：時間對齊、工況切換、模型漂移
3. **容易評估價值**：減少化驗次數 = 直接省成本
4. **適合對照 ML vs DL**：什麼時候用線性模型就夠了？什麼時候需要神經網絡？

---

## 1. 工業情境：你是製程工程師，品質量測很貴、很慢

### 1.1 軟測器的數學建模

**問題定義**：建立映射函數 $f: \mathbb{R}^p \to \mathbb{R}$

$$
\hat{y}_t = f(\mathbf{x}_t) + \epsilon_t
$$

其中：
- $\mathbf{x}_t = [x_1, x_2, \ldots, x_p]^T$：即時量測的製程變數（溫度、壓力、流量、回流比…）
- $y_t$：昂貴/慢的品質指標（GC 組成、黏度、純度、水分…）
- $\epsilon_t$：量測雜訊與模型誤差
- $f(\cdot)$：待學習的映射函數

**典型化工軟測器應用**：

| 產業 | 輸入變數（X） | 目標品質（y） | 量測成本對比 |
|------|--------------|-------------|-------------|
| **蒸餾塔** | 溫度、壓力、回流比、進料組成 | GC 組成、純度 | 即時 vs 30-120 分鐘 |
| **聚合反應** | 溫度、壓力、單體流量、催化劑濃度 | 分子量分佈、黏度 | 即時 vs 數小時 |
| **廢水處理** | pH、DO、流量、污泥濃度 | COD、BOD | 即時 vs 24-48 小時 |
| **晶圓製造** | 溫度、壓力、氣體流量、RF 功率 | 薄膜厚度、電阻率 | 即時 vs 數小時 |

### 1.2 工業價值量化

**成本效益分析範例**（蒸餾塔組成預測）：

假設：
- GC 分析成本：$50/次（含人力、耗材、設備折舊）
- 分析頻率：每小時 2 次
- 軟測器準確度：MAE < 0.5%（可接受）

**年節省成本**：
$$
\text{Savings} = (\$50/\text{次}) \times (2 \times 24 \times 365 - 24 \times 365 \times 0.2) \approx \$700,000/\text{年}
$$

（假設仍保留 20% 的化驗做校驗）

**投資回收期（ROI）**：
$$
\text{ROI} = \frac{\text{年節省} - \text{維護成本}}{\text{模型開發 + 硬體}} = \frac{\$700K - \$50K}{\$100K} = 6.5
$$

→ 約 2 個月回本（若考慮品質改善帶來的額外收益，ROI 更高）

### 1.3 你真正要交付的是什麼？

**不只是「分數很高」，而是**：

✅ **技術交付物**：
- 可解釋的數據切分與評估方式（讓產線願意信）
- 可維護的模型版本與監控策略（讓資訊/儀控願意上）
- OOD（Out-of-Distribution）檢測機制（知道何時不能信）

✅ **文檔交付物**：
- **模型卡（Model Card）**：適用範圍、準確度、限制條件
- **部署手冊**：輸入規格、輸出解讀、異常處置
- **維護計畫**：重訓觸發條件、性能監控指標

✅ **組織交付物**：
- **培訓方案**：操作員如何使用？工程師如何維護？
- **回滾計畫**：新模型失效時如何快速復原？
- **持續改進**：如何蒐集新數據？如何更新模型？

---

## 2. 時間序列的致命陷阱：數據洩漏（Data Leakage）

### 2.1 為什麼隨機切分會失敗？

**數學證明**：假設時間序列存在自相關性

$$
\text{Corr}(y_t, y_{t+k}) = \rho_k \neq 0, \quad k = 1, 2, \ldots
$$

若使用隨機切分：
- 訓練集包含 $t, t+1, t+3, \ldots$
- 測試集包含 $t+2, t+4, \ldots$

則測試樣本 $y_{t+2}$ 可從訓練集中的 $y_{t+1}, y_{t+3}$ **插值**得出，模型學到的是：

$$
\hat{y}_{t+2} \approx \frac{y_{t+1} + y_{t+3}}{2} + f(\mathbf{x}_{t+2})
$$

這不是真正的預測能力，而是**記憶相鄰樣本**！

### 2.2 實測結果對比

**本 Notebook 執行結果**：

| 模型 | Random Split RMSE | Time Split RMSE | 洩漏誇大倍數 |
|------|------------------|----------------|-------------|
| **Ridge** | 0.00515 | 0.00506 | 1.02x |
| **RandomForest** | 0.00532 | 0.00535 | 0.99x |
| **MLP (sklearn)** | 0.01485 | 0.01411 | 1.05x |

**關鍵觀察**：

1. **線性模型（Ridge）影響小**：因為自相關性主要是非線性的
2. **樹模型（RandomForest）幾乎無差**：樹模型對時間順序不敏感
3. **MLP 差異明顯**：神經網絡更容易學到時間模式

**洩漏程度取決於**：
- 數據的自相關程度（$\rho_k$ 越大，洩漏越嚴重）
- 模型的記憶能力（容量越大，洩漏越明顯）
- 工況變化頻率（穩態多 → 洩漏嚴重；切換多 → 洩漏減輕）

### 2.3 正確的時間切分策略

**方法 1：簡單時間切分**（本 Notebook 使用）

```python
cut = int(0.8 * len(df))  # 前 80% 訓練，後 20% 測試
X_train, X_test = X[:cut], X[cut:]
y_train, y_test = y[:cut], y[cut:]
```

**優點**：簡單、接近上線情境  
**缺點**：若數據有季節性/長期趨勢，測試集可能 OOD

**方法 2：滾動窗口（Rolling Window）**

$$
\text{Train: } [t-w, t-1], \quad \text{Test: } [t, t+h]
$$

適用於需要「逐步預測」的場景（如每日更新模型）。

**方法 3：時間 K-Fold**

```
訓練    |測試|訓練    |測試|訓練    |測試|
--------|----|---------|----|--------|----|→ 時間
```

保證測試集**永遠在訓練集之後**，適合評估模型在不同時間段的穩定性。

### 2.4 其他常見洩漏來源

**群組洩漏**：同一批次/同一設備的數據被切到兩邊

$$
\text{Train: batch}_i \cap \text{Test: batch}_i \neq \emptyset \quad \text{(錯誤!)}
$$

**解決方法**：使用 `GroupKFold`，確保同一批次完全在訓練集或測試集。

**特徵洩漏**：使用「未來才知道」的特徵

- ❌ 錯誤：用「當天收盤價」預測「當天最高價」
- ✅ 正確：用「前一天收盤價」預測「當天最高價」

---

## 3. 模型選擇策略：階梯式對照法（Progressive Baseline）

### 3.1 為什麼需要 Baseline？

**Occam's Razor 原則**：「如無必要，勿增實體」

在軟測器應用中：
- **簡單模型優勢**：可解釋、易維護、部署成本低、不易過擬合
- **複雜模型優勢**：可學習非線性、交互作用、動態模式

**決策準則**：

$$
\text{選擇 DL} \iff \frac{\text{Gain}_{\text{DL}} - \text{Gain}_{\text{baseline}}}{\text{Cost}_{\text{DL}} - \text{Cost}_{\text{baseline}}} > \text{Threshold}
$$

其中 Gain 包含：準確度提升、品質改善、成本節省  
Cost 包含：開發時間、維護成本、計算資源、風險

### 3.2 三層對照架構

#### **Level 1：Ridge Regression（線性基線）**

$$
\hat{y} = \mathbf{w}^T \mathbf{x} + b, \quad \min_{\mathbf{w}, b} \sum_{i=1}^{N} (y_i - \mathbf{w}^T \mathbf{x}_i - b)^2 + \lambda \|\mathbf{w}\|_2^2
$$

**優勢**：
- 可解釋：每個變數的貢獻 = $w_j \cdot x_j$
- 穩定：L2 正則化避免過擬合
- 快速：毫秒級訓練與推論

**本 Notebook 結果**：
- Time Split RMSE: **0.00506**
- Time Split MAE: **0.00407**

#### **Level 2：Random Forest（非線性基線）**

$$
\hat{y} = \frac{1}{M} \sum_{m=1}^{M} T_m(\mathbf{x})
$$

其中 $T_m$ 為第 $m$ 棵決策樹。

**優勢**：
- 自動學習交互作用：$x_1 \times x_2$
- 對異常值魯棒
- Feature Importance 可解釋

**本 Notebook 結果**：
- Time Split RMSE: **0.00535**
- Time Split MAE: **0.00427**

#### **Level 3：MLP（深度學習）**

**scikit-learn MLP**：
$$
\begin{aligned}
\mathbf{h}_1 &= \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \quad &\text{(Hidden Layer 1: 64 neurons)} \\
\mathbf{h}_2 &= \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2) \quad &\text{(Hidden Layer 2: 32 neurons)} \\
\hat{y} &= \mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3 \quad &\text{(Output Layer: 1 neuron)}
\end{aligned}
$$

**本 Notebook 結果（sklearn MLP）**：
- Time Split RMSE: **0.01411**
- Time Split MAE: **0.01090**

**Keras/TensorFlow MLP**（更好的優化）：
- Time Split RMSE: **0.02173**
- Time Split MAE: **0.01693**

### 3.3 實測結果分析

**性能排名**（Time Split，越小越好）：

1. 🥇 **Ridge**: 0.00506 RMSE（最佳！）
2. 🥈 **RandomForest**: 0.00535 RMSE
3. 🥉 **MLP (sklearn)**: 0.01411 RMSE
4. **MLP (Keras)**: 0.02173 RMSE

**驚人發現**：**線性模型最準！**

**為什麼 MLP 反而更差？**

1. **數據量不足**：5000 樣本對 MLP 來說偏少（通常需 50K-100K+）
2. **特徵已充分**：7 個製程變數已包含關鍵信息，非線性增益有限
3. **過擬合風險**：MLP 參數量大（~3000 參數），容易過擬合小數據
4. **優化困難**：sklearn MLP 未收斂警告（需更多 epoch 或調整學習率）

**學習曲線證據**（Keras MLP）：

![Training Curve](outputs/P4_Unit15_Results/05_keras_training_curve.png)

觀察：
- 訓練 Loss 快速下降至 ~0.005
- 驗證 Loss 在 Epoch 5 後趨平
- Early Stopping 於 Epoch 7 停止（patience=5）
- 最終驗證 Loss ≈ 0.002（但測試 Loss 更高 → 過擬合）

### 3.4 什麼時候該用 DL？

**適合 DL 的場景**：
- ✅ **數據量大**：> 10 萬樣本
- ✅ **高維輸入**：> 50 個特徵（或影像、文本）
- ✅ **複雜非線性**：已知物理關係高度非線性（如反應動力學）
- ✅ **時間依賴**：需要 RNN/LSTM 捕捉時間模式
- ✅ **遷移學習**：可利用預訓練模型（如 Transformer）

**不適合 DL 的場景**（如本例）：
- ❌ **小數據**：< 1 萬樣本
- ❌ **低維輸入**：< 10 個特徵
- ❌ **需可解釋**：監管要求或工程師需理解
- ❌ **嵌入式部署**：資源受限（MCU、PLC）

### 3.5 評估指標的工程意義

**MAE（Mean Absolute Error）**：
$$
\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
$$

- **優點**：單位與 $y$ 相同，易解釋（平均差 0.004 → 0.4% 純度誤差）
- **適用**：與規格/容許誤差對照（如純度規格 ≥ 95%，MAE 0.4% 可接受）

**RMSE（Root Mean Squared Error）**：
$$
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}
$$

- **優點**：對大誤差懲罰更重（$\text{error}^2$）
- **適用**：關注極端事件（如超標/工安事件）

**殘差分佈分析**：

$$
\text{Residuals} = y_i - \hat{y}_i, \quad i = 1, \ldots, N
$$

應檢查：
- **均值 ≈ 0**：無系統性偏差
- **常態分佈**：無異常模式
- **與工況無關**：殘差不應集中在特定工況段

---

## 4. 深度學習實作：Keras MLP 完整流程

### 4.1 MLP 前向傳播的數學原理

**第一層（輸入 → 隱藏層1）**：
$$
\mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}, \quad \mathbf{a}^{(1)} = \text{ReLU}(\mathbf{z}^{(1)})
$$

其中：
- $\mathbf{x} \in \mathbb{R}^{7}$：標準化後的輸入特徵
- $\mathbf{W}^{(1)} \in \mathbb{R}^{64 \times 7}$：權重矩陣（448 參數）
- $\mathbf{b}^{(1)} \in \mathbb{R}^{64}$：偏置向量（64 參數）
- $\text{ReLU}(z) = \max(0, z)$：激活函數

**第二層（隱藏層1 → 隱藏層2）**：
$$
\mathbf{z}^{(2)} = \mathbf{W}^{(2)} \mathbf{a}^{(1)} + \mathbf{b}^{(2)}, \quad \mathbf{a}^{(2)} = \text{ReLU}(\mathbf{z}^{(2)})
$$

- $\mathbf{W}^{(2)} \in \mathbb{R}^{32 \times 64}$：2048 參數
- $\mathbf{b}^{(2)} \in \mathbb{R}^{32}$：32 參數

**輸出層（隱藏層2 → 預測值）**：
$$
\hat{y} = \mathbf{w}^{(3)T} \mathbf{a}^{(2)} + b^{(3)}
$$

- $\mathbf{w}^{(3)} \in \mathbb{R}^{32}$：32 參數
- $b^{(3)} \in \mathbb{R}$：1 參數

**總參數量**：
$$
\text{Total Params} = (7 \times 64 + 64) + (64 \times 32 + 32) + (32 + 1) = 2,625
$$

### 4.2 損失函數與優化

**Mean Squared Error (MSE)**：
$$
\mathcal{L}(\mathbf{W}, \mathbf{b}) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

**反向傳播（Backpropagation）**：

輸出層梯度：
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}^{(3)}} = -\frac{2}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i) \mathbf{a}_i^{(2)}
$$

隱藏層梯度（鏈式法則）：
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(2)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(2)}} \cdot \frac{\partial \mathbf{z}^{(2)}}{\partial \mathbf{W}^{(2)}} = \delta^{(2)} (\mathbf{a}^{(1)})^T
$$

其中 $\delta^{(2)} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \mathbf{w}^{(3)} \odot \text{ReLU}'(\mathbf{z}^{(2)})$

**Adam 優化器**：
$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad &\text{(一階動量)} \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad &\text{(二階動量)} \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad &\text{(偏差修正)} \\
\mathbf{W}_t &= \mathbf{W}_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \quad &\text{(參數更新)}
\end{aligned}
$$

本 Notebook 設定：$\eta = 0.001, \beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-7}$

### 4.3 訓練配置與 Early Stopping

**Keras 模型定義**：
```python
model = Sequential([
    Dense(64, activation='relu', input_dim=7, name='hidden1'),
    Dense(32, activation='relu', name='hidden2'),
    Dense(1, name='output')  # 回歸：線性輸出
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='mse',
    metrics=['mae']
)
```

**Early Stopping 機制**：
$$
\text{Stop if } \mathcal{L}_{\text{val}}^{(t)} > \min_{\tau < t - p} \mathcal{L}_{\text{val}}^{(\tau)}
$$

- Patience = 5：容忍 5 個 epoch 無改善
- Restore best weights：恢復最佳 checkpoint

**實測訓練過程**：

| Epoch | Train Loss | Val Loss | 狀態 |
|-------|-----------|----------|-----|
| 1 | 0.0243 | 0.0021 | ⬇️ 快速下降 |
| 2 | 0.0089 | 0.0019 | ⬇️ 持續改善 |
| 3 | 0.0051 | 0.0018 | ⬇️ 最佳點 |
| 4 | 0.0042 | 0.0019 | ⚠️ Val 開始上升 |
| 5 | 0.0037 | 0.0021 | ⚠️ 過擬合訊號 |
| 6-7 | ... | ... | ⏸️ Patience 計數 |
| 8 | - | - | ⛔ Early Stop |

**最終測試結果**：
- Test MAE: **0.01693** (1.69% 純度誤差)
- Test RMSE: **0.02173**

### 4.4 與 sklearn MLP 的詳細對比

| 面向 | sklearn MLPRegressor | Keras/TensorFlow |
|------|---------------------|------------------|
| **架構彈性** | 固定 MLP | 任意（CNN/RNN/Transformer）|
| **優化器** | LBFGS/SGD/Adam | Adam/SGD/RMSprop/AdaGrad/⋯ |
| **學習率調度** | 固定或自適應 | Callbacks (ReduceLROnPlateau) |
| **正則化** | L2 only | L1/L2/Dropout/BatchNorm |
| **GPU 加速** | ❌ CPU only | ✅ CUDA/ROCm/TPU |
| **批次訓練** | 全批次或 mini-batch | 完整 DataLoader 支援 |
| **部署** | pickle (50KB) | SavedModel (500KB-2MB) |
| **推論速度** | ~0.1ms | ~0.5ms (CPU), ~0.05ms (GPU) |
| **可視化** | ❌ 需手動 | ✅ TensorBoard |
| **適用場景** | 快速原型、小數據 | 生產環境、大規模 |

**本例結論**：
- sklearn MLP 收斂失敗（max_iter=200 不足）
- Keras MLP 正常收斂但仍不如 Ridge
- 建議：**數據量 < 10K 時優先使用 sklearn**

### 4.5 工業部署考量

**時間對齊與量測延遲**：

軟測器最常見的落地失敗原因：**你的 y 不是即時量測**。

典型情況：
- GC/化驗：每 30–120 分鐘一筆，結果是「取樣後一段時間」才知道
- 取樣/傳輸/分析有 **dead time**（死時間）與 **time lag**
- 部分輸入也有延遲（實驗室調配的原料性質、手動記錄的狀態）

**工程處理流程**：
1. 定義「y 對應的時間點」：取樣時間？分析完成時間？還是代表上一段時間的平均？
2. 用 cross-correlation 或經驗知識估計合理的延遲範圍（0–60 分鐘）
3. 用 lag features（$X_{t-k}$）建模，並用 time split 評估（避免洩漏）
4. 交付時寫清楚：模型輸出代表哪個時間窗口的品質

---

## 5. Out-of-Distribution (OOD) 偵測：模型的「我不知道」機制

### 5.1 為什麼模型不會說「我不知道」？

**數學本質**：神經網路是萬能逼近器（Universal Approximator）

$$
\hat{y} = f(\mathbf{x}; \mathbf{W}) \quad \forall \mathbf{x} \in \mathbb{R}^p
$$

問題：**任意輸入都會產生輸出**，即使 $\mathbf{x}$ 完全不在訓練分佈內！

**工業風險實例**：

| 情境 | 輸入異常 | 模型行為 | 後果 |
|------|---------|---------|-----|
| **設備故障** | 溫度感測器漂移 +50°C | 仍給出預測值 | 錯誤控制決策 |
| **工況切換** | 切換到新產品配方 | 使用舊模型預測 | 品質失控 |
| **感測器故障** | 壓力讀值 = 0 | 外推預測 | 嚴重誤判 |
| **維護期間** | 部分變數缺失/凍結 | 用歷史值預測 | 失去預警能力 |

### 5.2 三層防護機制

#### **Level 1：範圍閘門（Range Gate）**

$$
\text{OOD}_{\text{range}} = \bigvee_{j=1}^{p} \left( x_j < \min(X_{\text{train}, j}) \lor x_j > \max(X_{\text{train}, j}) \right)
$$

**優點**：
- 極快（微秒級）
- 可解釋性強
- 適合硬體實現（PLC/DCS）

**缺點**：
- 無法檢測「合法範圍內的異常組合」
- 例：溫度、壓力都在範圍內，但組合從未見過

**本 Notebook 實測**：

```
OOD Gate 計數（測試集）：
  超出範圍樣本數：3 / 1000 (0.3%)
  觸發變數：feed_rate (2次), pressure (1次)
```

#### **Level 2：統計距離（Statistical Distance）**

**Mahalanobis Distance**：考慮特徵間相關性的距離

$$
D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}
$$

其中：
- $\boldsymbol{\mu} = \mathbb{E}[\mathbf{X}_{\text{train}}]$：訓練集均值向量
- $\boldsymbol{\Sigma} = \text{Cov}(\mathbf{X}_{\text{train}})$：訓練集協方差矩陣

**判定準則**：
$$
\text{OOD}_{\text{Maha}} = \mathbb{1}\left[ D_M(\mathbf{x}) > \chi^2_{p, 0.95} \right]
$$

使用卡方分佈 95% 分位數作為閾值（假設數據近似高斯分佈）。

**優點**：
- 考慮變數間相關性（Ridge 迴歸的幾何基礎）
- 對橢圓形分佈有效

**缺點**：
- 需要計算協方差逆矩陣（$O(p^3)$ 複雜度）
- 對非高斯分佈效果差
- 高維度時協方差矩陣可能奇異

#### **Level 3：模型不確定度（Model Uncertainty）**

**Monte Carlo Dropout**：訓練時使用 Dropout，推論時保留 Dropout 並多次採樣

$$
\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_T \sim f(\mathbf{x}; \mathbf{W}_{\text{dropout}}), \quad \sigma_{\text{pred}} = \text{std}(\{\hat{y}_t\}_{t=1}^{T})
$$

判定：
$$
\text{OOD}_{\text{uncertainty}} = \mathbb{1}\left[ \sigma_{\text{pred}} > \text{threshold} \right]
$$

**Ensemble 方法**：訓練多個模型，計算預測分歧度

$$
\sigma_{\text{ensemble}} = \sqrt{\frac{1}{M} \sum_{m=1}^{M} (\hat{y}_m - \bar{y})^2}
$$

### 5.3 工業部署建議

**三層並聯策略**：

```python
def predict_with_safety(x_new, model):
    # Level 1: 範圍檢查（必須）
    if range_check(x_new):
        return None, "OOD: Range violation"
    
    # Level 2: Mahalanobis（推薦）
    if mahalanobis_check(x_new) > threshold_maha:
        return None, "OOD: Statistical anomaly"
    
    # Level 3: 模型預測
    y_pred = model.predict(x_new)
    uncertainty = estimate_uncertainty(x_new, model)
    
    if uncertainty > threshold_unc:
        return y_pred, "Warning: High uncertainty"
    else:
        return y_pred, "OK"
```

**閾值設定原則**：

| 應用場景 | False Positive 容忍度 | False Negative 容忍度 | 建議閾值 |
|---------|---------------------|---------------------|----------|
| **安全關鍵** | 高（寧可誤報） | 極低 | 90% 分位數 |
| **品質監控** | 中等 | 低 | 95% 分位數 |
| **成本優化** | 低 | 中等 | 99% 分位數 |

### 5.4 OOD 後的處置流程

```
檢測到 OOD
    ↓
記錄事件（時間、變數、距離）
    ↓
├─ 立即處置：
│   ├─ 切換到備用模型（保守預測）
│   ├─ 觸發人工採樣/化驗
│   └─ 發送警報給工程師
│
└─ 長期處置：
    ├─ 累積 OOD 樣本 → 擴展訓練集
    ├─ 分析 OOD 原因 → 改善感測器/工況設計
    └─ 重新訓練模型 → 擴大適用範圍
```

### 5.5 工況切換要分段評估

軟測器的平均分數可能很漂亮，但一到切換段就爆掉。交付時至少要回答：
- 模型在「穩態段」vs「切換段」的誤差各是多少？
- 是否需要分工況模型、或加入工況指標（mode/recipe/phase）做條件化？

---

## 6. 實戰演練：Notebook 完整實作

### 6.1 數據集與問題設定

**合成數據設計**（`Part_4/Unit15_DL_Workflow_SoftSensor_MLP.ipynb`）：

- **樣本數**：5,000 筆時間序列數據
- **特徵**（7 個）：`feed_rate`, `reflux`, `boilup`, `pressure`, `feed_comp`, `tray_temp`, `top_temp`
- **目標**：`purity`（產品純度，範圍 0.80-0.995）
- **模擬效果**：
  - 長期趨勢：0.15 斜率漂移
  - 週期性波動：sin/cos 項
  - 量測噪音：Gaussian noise

### 6.2 實作重點

你會完成：
1. **數據切分**：Random Split vs Time Split 對比
2. **Baseline 建立**：Ridge → RandomForest → MLP 階梯式對照
3. **Keras MLP**：完整訓練流程 + Early Stopping
4. **OOD 檢測**：Range Gate 實作
5. **結果可視化**：
   - 訓練曲線（Training/Validation Loss）
   - 預測 vs 真值散佈圖
   - 殘差分佈圖
   - Random vs Time Split 性能對比

### 6.3 最低交付物（交作業標準）

✅ **1 張圖：Time Split 測試預測曲線**
   - X 軸：樣本索引或時間
   - Y 軸：Purity（真值 vs 預測）
   - 標註：MAE, RMSE

✅ **1 張圖：Random vs Time Split 指標差異**
   - 柱狀圖對比 RMSE
   - 用來講解數據洩漏的影響

✅ **1 份表：模型選擇理由**
   
| 模型 | RMSE | 優勢 | 劣勢 | 最終選擇 |
|------|------|------|------|---------|
| Ridge | 0.0051 | 快速、可解釋、穩定 | 無法學習非線性 | ✅ **推薦** |
| RandomForest | 0.0054 | 自動特徵交互 | 黑盒、部署較大 | 備選 |
| MLP (Keras) | 0.0217 | 理論上能學複雜模式 | 需大數據、易過擬合 | ❌ 不適合本案例 |

✅ **1 段模型卡（Model Card）**：

```markdown
### 蒸餾塔純度軟測器 - 模型卡

**模型版本**: v1.0  
**訓練日期**: 2024-12-XX  
**適用範圍**: 
  - 進料流量: 50-150 kg/h
  - 回流比: 1.5-4.0
  - 塔頂溫度: 60-85°C
  
**性能指標** (Time Split):
  - MAE: 0.407% (純度)
  - RMSE: 0.506%
  - 95% 預測區間: ±1.0%

**限制條件**:
  - ⚠️ 量測延遲: GC 結果代表取樣前 15 分鐘的平均純度
  - ⚠️ OOD 處置: 超出訓練範圍時切換回實驗室化驗
  - ⚠️ 工況切換: 切換期間（±30 分鐘）預測不可靠

**維護計畫**:
  - 每季度重訓（累積新數據）
  - 儀表校正後需重新驗證
  - 原料供應商變更時需評估模型適用性
```

---

## 7. 上線觀點：三個必答問題

### 7.1 何時不信模型？

**信心度評估機制**：
- ✅ 輸入範圍內 + Mahalanobis < 閾值 → **信任**
- ⚠️ 輸入範圍內 + Mahalanobis > 閾值 → **低信心，建議複檢**
- ❌ 輸入超出範圍 → **不信任，強制化驗**

**適用工況範圍**：
- 明確定義訓練數據涵蓋的工況（產品等級、進料組成、操作模式）
- 新產品/新配方上線前必須收集數據重訓

**OOD 觸發條件**：
```python
if any(x < x_train_min) or any(x > x_train_max):
    action = "REJECT: Out of range"
elif mahalanobis_distance(x) > chi2_95:
    action = "WARNING: Statistical outlier"
elif model_uncertainty(x) > threshold:
    action = "CAUTION: High uncertainty"
else:
    action = "OK"
```

### 7.2 何時重訓？

**觸發條件**：
1. **定期重訓**：每季度或累積 N 筆新化驗數據後
2. **性能衰退**：線上 MAE 連續 7 天超過訓練期 1.5 倍
3. **工況變更**：
   - 原料供應商切換
   - 儀表校正/更換
   - 操作策略改變（如新的回流控制邏輯）
4. **季節性**：若存在季節效應，每年同期重訓

**重訓流程**：
```
1. 收集新數據（確保時間對齊、標註 OOD 事件）
   ↓
2. 合併舊訓練集（保留歷史工況知識）
   ↓
3. 重新切分（Time Split，最新 20% 做測試）
   ↓
4. 重訓所有 Baseline（確認 Ridge 仍最優）
   ↓
5. A/B 測試（新模型 vs 舊模型並行運行 1-2 週）
   ↓
6. 正式上線（保留舊模型作為 fallback）
```

### 7.3 如何回滾？

**部署策略**：
- **Blue-Green Deployment**：新模型（Green）與舊模型（Blue）同時運行
  - 監控期（1-2 週）：僅記錄新模型預測，不影響控制
  - 驗證通過：切換流量到新模型
  - 發現問題：立即切回舊模型

- **金絲雀發布（Canary）**：
  - 先讓 10% 流量使用新模型
  - 逐步增加到 50% → 100%
  - 任何階段發現異常可快速回退

**回滾觸發條件**：
```python
if new_model_mae > old_model_mae * 1.3:  # 性能惡化 30%
    rollback()
elif ood_rate_new > ood_rate_old * 2:    # OOD 率翻倍
    rollback()
elif critical_incident:                   # 品質/安全事件
    rollback()
```

**回滾執行時間**：
- 目標：< 5 分鐘完成切換
- 需預先測試回滾流程（每月演練）

---

## 8. 總結：深度學習工程思維

### 8.1 核心要點回顧

✅ **時間洩漏是頭號陷阱**
   - Random Split → 樂觀偏差
   - Time Split → 真實性能
   - 群組洩漏同樣致命

✅ **Baseline 決定 DL 價值**
   - 先做 Ridge（1 分鐘）
   - 再做 RandomForest（5 分鐘）
   - 最後考慮 MLP（1 小時）
   - **本例：Ridge 勝出！**

✅ **OOD 檢測不可或缺**
   - Range Gate（必須）
   - Mahalanobis（推薦）
   - Uncertainty（進階）

✅ **工程交付 ≠ 高分模型**
   - 模型卡：適用範圍、限制、維護
   - 部署手冊：OOD 處置、回滾計畫
   - 培訓方案：操作員、工程師

### 8.2 本單元與後續課程的銜接

**前置知識**（Part_1-3）：
- ✅ Python 數據處理
- ✅ 時間序列清洗
- ✅ 機器學習 Baseline

**本單元技能**（Part_4 Unit15）：
- ✅ Keras/TensorFlow 基礎
- ✅ MLP 數學原理
- ✅ 工程工作流（數據切分、OOD、部署）

**後續銜接**（Part_4 Unit16-18）：
- Unit16：CNN（影像數據）
- Unit17：RNN/LSTM（複雜時間序列）
- Unit18：Transformer（序列建模、遷移學習）

### 8.3 實務建議

**小數據情境**（< 10K 樣本）：
1. 優先嘗試 Ridge、RandomForest
2. 若需 DL，考慮 **遷移學習**（後續課程）
3. 積極收集數據，而非調參

**中等數據**（10K-100K）：
1. MLP 開始有意義
2. 可嘗試 Dropout、BatchNorm
3. 重點：特徵工程 + 數據增強

**大數據**（> 100K）：
1. DL 優勢明顯
2. 可上 CNN/RNN/Transformer
3. GPU 訓練必要性增加

**最終準則**：
> **能用 Ridge 解決的問題，不要用 MLP。  
> 能用 MLP 解決的問題，不要用 CNN。  
> 能用監督學習解決的問題，不要用強化學習。**
>
> — Occam's Razor for AI Engineers
