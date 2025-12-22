# Unit18 進階案例補充：反應器溫度與空氣品質預測

> 本文檔為 Unit18 的補充教材，提供額外的時間序列預測案例。
> 主檔 `Unit18_LSTM_Forecasting_Process_TimeSeries.md` 以鍋爐數據為核心教學案例。

---

## 案例一：反應器溫度預測（模擬數據）

### 1.1 工業情境

固定床反應器的出口溫度受多種因素影響：
- **基準溫度**：約 150°C（穩態操作條件）
- **日夜循環**：24 小時正弦波（模擬環境溫差或批次循環）
- **觸媒失活趨勢**：反應活性緩慢下降導致溫度微降
- **測量雜訊**：測溫元件與訊號干擾

### 1.2 數據模擬

模擬 100 小時操作，每分鐘一筆，共 6000 筆資料：

```python
import numpy as np
import pandas as pd

# 時間軸
minutes = np.arange(0, 6000)
time = minutes / 60  # 轉換為小時

# 組合訊號
base_temp = 150.0
seasonal = 3 * np.sin(2 * np.pi * time / 24)  # 24h週期
trend = -0.01 * time  # 緩慢下降趨勢
noise = np.random.normal(0, 0.5, len(time))  # 測量雜訊

temperature = base_temp + seasonal + trend + noise
```

**視覺化特徵**：
- 清晰的週期性波動（振幅約 3°C）
- 疊加在緩慢下降趨勢上（100 小時約降 1°C）
- 高頻雜訊（σ ≈ 0.5°C）

### 1.3 LSTM 模型設定

**滑動視窗**：
- Look-back = 60 分鐘
- 預測目標：第 61 分鐘溫度（1 步預測）

**模型架構**：
```python
model = Sequential([
    LSTM(50, input_shape=(60, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
```

### 1.4 結果分析

**效能指標**：
- LSTM 測試集 RMSE：**0.65°C**
- Naive baseline（$T_{t+1} = T_t$）：**0.72°C**
- Moving Average (k=5)：**0.56°C**

**關鍵觀察**：

1. **LSTM 優於 naive baseline**（0.65 < 0.72）  
   → 模型不只是複製前一時刻的值

2. **但移動平均仍稍優**（0.56 < 0.65）  
   → 在高度自迴歸且平穩的模擬場景中，簡單平滑已足夠

3. **預測曲線呈現兩大特徵**：
   - **平滑化 (Smoothing)**：LSTM 濾除高頻雜訊，學習平均趨勢
   - **相位延遲 (Phase Lag)**：波峰/波谷位置略有滯後

**改善方向**：
- 加入操作特徵（進料流量、冷卻水溫度、催化劑年齡）
- 使用更進階架構（Seq2Seq, Attention, Transformer）
- 針對不同工況訓練分段模型

### 1.5 教學重點

**對學生的啟示**：
> **深度模型不一定永遠勝過簡單方法**  
> 必須與 baseline 比較才能判斷價值

**實務建議**：
- 先建立簡單 baseline（移動平均、AR、ARIMA）
- 再引入 LSTM，比較 RMSE 是否顯著改善
- 評估計算成本與效能提升的權衡

---

## 案例二：空氣污染 PM2.5 預測

### 2.1 數據來源

北京 PM2.5 監測站的歷史數據（`pollution.csv`），包含：
- **目標變數**：PM2.5 濃度（μg/m³）
- **氣象特徵**：
  - `DEWP`：露點溫度
  - `TEMP`：氣溫
  - `PRES`：氣壓
  - `Iws`：累計風速

### 2.2 預處理流程

**時間彙整**：
- 原始數據：逐小時紀錄
- 彙整為：每日平均值（降低高頻波動）
- 最終資料：(1789 天, 5 特徵)

**特徵工程**：
```python
# 保留主要欄位
features = ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws']
df = df[features].resample('D').mean()

# 移除缺失值
df = df.dropna()
```

### 2.3 多變量時間序列建模

**目標**：預測第 31 天的 PM2.5 日平均濃度

**輸入特徵（Look-back = 30 天）**：
$$
y_{t+1} = f\\big(
  y_t, y_{t-1}, \dots, y_{t-29},
  \\mathbf{u}_t, \\mathbf{u}_{t-1}, \dots, \\mathbf{u}_{t-29}
\\big)
$$

其中：
- $y_t$：PM2.5 歷史值
- $\\mathbf{u}_t = [\\text{DEWP}_t, \\text{TEMP}_t, \\text{PRES}_t, \\text{Iws}_t]^\\top$

**LSTM 架構**：
```python
model = Sequential([
    LSTM(64, input_shape=(30, 5)),  # 30天, 5特徵
    Dense(1)
])
```

### 2.4 訓練結果

**Loss 曲線**：
- Train Loss：從 0.024 降至 0.014
- Val Loss：從 0.015 降至 0.009
- 兩者同步下降，無明顯 overfitting

**RMSE 指標**：
- 訓練集：**62.33 μg/m³**
- 測試集：**63.26 μg/m³**

### 2.5 預測表現分析

**數值解讀**：
- 絕對 RMSE 看似很大（~63）
- 但 PM2.5 日平均範圍為 10 ~ 300+ μg/m³
- 模型能抓住「濃度等級與季節性」

**視覺化觀察（測試集最後 180 天）**：

| 場景 | 真實值 | 預測值 | 表現 |
|------|--------|--------|------|
| 清潔天氣（夏季） | 20-50 | 25-55 | ✅ 貼近真實 |
| 中度污染（春秋） | 80-120 | 75-110 | ✅ 趨勢正確 |
| 嚴重污染尖峰 | 250+ | 150-180 | ⚠️ 低估峰值 |

**尖峰低估原因**：
1. **特徵不完整**：缺少區域排放源、交通量、暖氣使用等關鍵資訊
2. **極端事件稀少**：訓練集中嚴重污染樣本佔比低
3. **MSE Loss 特性**：平均誤差最小化導致模型偏好預測均值

### 2.6 改善策略

**特徵增強**：
- 加入風向（N/S/E/W 編碼）
- 納入區域排放源操作數據
- 考慮其他污染物（NO₂, SO₂, O₃）

**模型調整**：
- 使用 **Quantile Regression** 預測不確定性區間
- 分別訓練「清潔」與「污染」工況模型
- 引入 **Attention Mechanism** 識別關鍵時間步

**損失函數改進**：
```python
# 自定義損失：對高濃度樣本加權
def weighted_mse(y_true, y_pred):
    weights = 1 + (y_true / 100)  # 濃度越高權重越大
    return K.mean(weights * K.square(y_true - y_pred))
```

### 2.7 環境監測應用

**實務場景**：
- **焚化爐排放預測**：提前調整燃燒參數避免超標
- **石化園區監控**：預警鄰近居民區污染風險
- **製程優化**：調整操作減少 PM 生成

**決策邏輯**：
```
IF 預測 PM2.5 > 150 μg/m³（紅色警報）
  → 觸發緊急應變程序（減產、加強洗滌）

IF 預測 PM2.5 > 100 μg/m³（橙色警報）
  → 提前通知環保部門與周邊社區

IF 預測 PM2.5 持續上升（斜率 > 10 μg/m³/day）
  → 檢查排放源操作與氣象預報
```

---

## 案例三：多變量時間序列的完整討論

### 3.1 單變量 vs 多變量

**單變量時間序列**：
- 僅考慮一個變數（如單一反應器溫度）
- 優點：模型簡單、易於解釋
- 缺點：忽略相關變數的影響

**多變量時間序列**：
- 同時使用多個相關變數
- 優點：捕捉變數間的相互作用
- 挑戰：維度災難、共線性、特徵選擇

### 3.2 化工製程中的多變量依賴

**蒸餾塔案例**：

預測塔頂組成時，相關變數包括：
- 進料流量與組成
- 回流比
- 再沸器加熱蒸汽流量
- 塔壓與塔溫分佈
- 環境溫度（影響冷凝器效能）

**建模策略**：
```python
# 多變量輸入
features = ['feed_flow', 'feed_comp', 'reflux_ratio', 
            'reboiler_duty', 'column_pressure', 'top_temp']

# 滑動視窗
X, y = create_dataset(df[features], df['top_comp'], 
                      look_back=120)  # 2小時歷史

# LSTM 架構
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(120, 6)),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1)
])
```

### 3.3 特徵選擇方法

**1. 領域知識驅動**：
- 根據化工原理選擇關鍵變數
- 優先納入控制變數（manipulated variables）

**2. 統計方法**：
```python
from sklearn.feature_selection import mutual_info_regression

# 計算互資訊
mi_scores = mutual_info_regression(X_lagged, y)
top_features = features[np.argsort(mi_scores)[-10:]]
```

**3. 嵌入式方法**：
- LSTM + Attention：學習不同時間步與特徵的重要性
- Tree-based baseline：觀察特徵重要性排名

### 3.4 多步預測的兩種策略

**Recursive (遞迴式)**：

```python
# 預測未來 10 步
predictions = []
current_input = X_test[0]  # 初始窗口

for i in range(10):
    pred = model.predict(current_input[np.newaxis, :, :])
    predictions.append(pred[0, 0])
    
    # 更新窗口：移除最舊、加入預測值
    current_input = np.roll(current_input, -1, axis=0)
    current_input[-1, 0] = pred[0, 0]  # 假設第一個特徵是目標
```

**優點**：只需一個模型  
**缺點**：誤差累積（預測越遠越不準）

---

**Direct (直接式)**：

```python
# 為每個 horizon 訓練獨立模型
models = {}
for h in range(1, 11):  # 預測 1~10 步
    y_h = df['target'].shift(-h)  # 目標為 t+h
    models[h] = train_model(X, y_h)

# 預測時
predictions = [models[h].predict(X_test) for h in range(1, 11)]
```

**優點**：每步誤差獨立，不累積  
**缺點**：需訓練多個模型，計算成本高

---

**Seq2Seq (序列到序列)**：

```python
# Encoder-Decoder 架構
encoder_inputs = Input(shape=(60, 6))
encoder = LSTM(64, return_state=True)
_, state_h, state_c = encoder(encoder_inputs)

decoder_inputs = Input(shape=(10, 1))  # 未來10步的外生變數
decoder_lstm = LSTM(64, return_sequences=True)
decoder_outputs = decoder_lstm(decoder_inputs, 
                                initial_state=[state_h, state_c])
decoder_dense = Dense(1)
predictions = decoder_dense(decoder_outputs)
```

**優點**：同時預測多步，學習內部依賴  
**應用**：製程控制 MPC 需要完整軌跡預測

### 3.5 實務部署考量

**1. 滾動更新 (Rolling Forecast)**：

```python
# 每天重新預測
for day in test_days:
    # 使用前 30 天數據
    X_window = df[day-30:day][features]
    
    # 預測未來 10 天
    y_pred = model.predict(X_window)
    
    # 儲存結果並前進到下一天
    predictions[day] = y_pred
```

**2. 在線學習 (Online Learning)**：

```python
# 每週用新數據微調模型
if week % 4 == 0:  # 每月一次
    new_data = fetch_latest_data()
    model.fit(new_data, epochs=5, verbose=0)
```

**3. 漂移偵測**：

```python
# 監控預測誤差趨勢
recent_mae = np.mean(np.abs(y_true[-100:] - y_pred[-100:]))

if recent_mae > threshold * baseline_mae:
    trigger_retraining()
    send_alert("Model performance degraded!")
```

---

## 學習路徑建議

### 初學者（完成 Unit18 主檔後）
1. 先理解反應器溫度案例（模擬數據，概念清晰）
2. 比較 LSTM vs baseline，建立「評估思維」
3. 了解平滑化與相位延遲的物理意義

### 進階學習
1. 實作 PM2.5 案例，體會多變量時序挑戰
2. 嘗試特徵工程（風向編碼、滯後項選擇）
3. 比較 Recursive vs Direct 多步預測

### 專題應用
1. 選擇實驗室/工廠的真實數據
2. 建立完整 pipeline（前處理 → baseline → LSTM → 部署）
3. 設計決策規則（何時預警、何時調參）

---

## 延伸閱讀

**學術論文**：
- Hochreiter & Schmidhuber (1997) - LSTM 原始論文
- Cho et al. (2014) - GRU（簡化版 LSTM）
- Vaswani et al. (2017) - Transformer（替代 LSTM）

**工業應用**：
- Google Flu Trends（時序預測失敗案例分析）
- DeepMind 資料中心冷卻優化（LSTM 控制）

**開源數據集**：
- UCI Machine Learning Repository - Air Quality
- Kaggle - Time Series Forecasting competitions

---

**最後更新**：2025/12/19  
**狀態**：✅ Phase 2 整合完成  
**建議教學時數**：3-4 小時（選讀補充）
