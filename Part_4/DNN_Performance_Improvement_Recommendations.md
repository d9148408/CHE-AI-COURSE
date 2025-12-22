# DNN 性能改進建議

## 當前三個案例的性能分析

### 案例 1：葡萄酒質量預測
- **Test R²**: Ridge=0.309, RF=0.407, **DNN=0.329**
- **問題**: DNN略好於Ridge但不如RandomForest
- **訓練集過擬合**: RandomForest Train_R²=0.918 → Test_R²=0.407（嚴重過擬合）

### 案例 2：SRU H₂S濃度預測
- **Test R²**: Ridge=0.939, RF=0.902, **DNN=0.929**
- **問題**: DNN表現接近Ridge但仍略差
- **泛化能力**: DNN的Train-Val-Test R²較穩定（0.932→0.946→0.929）

### 案例 3：Debutanizer蒸餾塔
- **Test R²**: Ridge=0.999, RF=0.996, **DNN=0.997**
- **問題**: 所有模型都很好，但Ridge最優
- **潛力**: DNN已達到很高性能，提升空間有限

---

## 🔴 主要問題診斷

### 1. **Sigmoid激活函數不適合回歸任務**
**當前問題**：
- Sigmoid輸出範圍限制在 [0, 1]
- 數據已經z-score標準化，範圍遠超[0,1]（例如：案例2的y訓練集範圍[-1.51, 18.09]）
- 導致梯度消失和表達能力受限

**影響**：
```
案例2的標準化y範圍：[-1.5124, 18.0917]
Sigmoid輸出範圍：[0, 1]
→ 嚴重的表達能力限制！
```

### 2. **網絡架構可能不夠深/寬**
**當前架構**：
- 案例1: 16 → 128 → 64 → 32 → 1
- 案例2: 8 → 128 → 64 → 32 → 1
- 案例3: 11 → 128 → 64 → 32 → 1

**問題**：
- 對於複雜的化工過程，可能需要更深的網絡
- 第一層隱藏層可能需要更寬以捕捉特徵交互

### 3. **正則化可能過強**
**當前設置**：
- L2正則化：0.01
- Dropout：0.3 → 0.2 → 0.1
- 可能導致欠擬合，尤其是案例1

---

## ✅ 優先改進建議（按重要性排序）

### 🥇 優先級1：更換激活函數（立即見效）

#### 建議1.1：使用 ReLU 系列
```python
# 適用於所有案例
def create_improved_dnn():
    model = keras.Sequential([
        Input(shape=(n_features,)),
        layers.Dense(256, activation='relu'),  # 改用ReLU
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),  # ReLU
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),   # ReLU
        layers.Dropout(0.1),
        layers.Dense(1, activation='linear')   # 輸出層保持linear
    ])
    return model
```

**為什麼有效**：
- ✅ ReLU不會限制輸出範圍
- ✅ 避免梯度消失問題
- ✅ 計算效率高
- ✅ 適合深度網絡

#### 建議1.2：使用 ELU（更平滑的選擇）
```python
# ELU允許負值輸出，更適合標準化數據
layers.Dense(128, activation='elu')  # alpha=1.0
```

**優點**：
- ✅ 比ReLU更平滑
- ✅ 允許負值輸出
- ✅ 減少偏移問題

#### 建議1.3：使用 Swish/SiLU（最新研究）
```python
# Google研究發現對DNN特別有效
layers.Dense(128, activation='swish')
```

---

### 🥈 優先級2：優化網絡架構

#### 建議2.1：增加網絡深度（適合案例2、3）
```python
# 適用於時序數據（案例2、3）
def create_deeper_dnn():
    model = keras.Sequential([
        Input(shape=(n_features,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(256, activation='relu'),  # 新增層
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.1),
        
        layers.Dense(32, activation='relu'),   # 新增層
        layers.Dropout(0.1),
        
        layers.Dense(1, activation='linear')
    ])
    return model
```

#### 建議2.2：使用殘差連接（ResNet風格）
```python
# 適用於深度網絡
from tensorflow.keras.layers import Add

def residual_block(x, units):
    # 主路徑
    y = layers.Dense(units, activation='relu')(x)
    y = layers.BatchNormalization()(y)
    y = layers.Dense(units, activation='relu')(y)
    y = layers.BatchNormalization()(y)
    
    # 殘差連接
    x_shortcut = layers.Dense(units)(x) if x.shape[-1] != units else x
    return Add()([x_shortcut, y])
```

#### 建議2.3：調整第一層寬度（案例1特別適用）
```python
# 案例1有16個特徵，第一層可以更寬
layers.Dense(512, activation='relu'),  # 從128增加到512
```

---

### 🥉 優先級3：調整正則化策略

#### 建議3.1：降低L2正則化
```python
# 當前：0.01 → 建議：0.001 或 0.0001
layers.Dense(128, activation='relu', 
             kernel_regularizer=keras.regularizers.l2(0.001))
```

#### 建議3.2：降低Dropout率
```python
# 當前：0.3 → 0.2 → 0.1
# 建議：0.2 → 0.15 → 0.1（或更低）
layers.Dropout(0.15)  # 降低dropout
```

#### 建議3.3：使用早停（已有，調整參數）
```python
EarlyStopping(
    monitor='val_loss', 
    patience=50,        # 從35增加到50
    restore_best_weights=True,
    min_delta=1e-5      # 新增：避免過早停止
)
```

---

### 🎯 優先級4：優化訓練策略

#### 建議4.1：使用學習率調度
```python
# Cosine Annealing
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    alpha=0.0001
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
```

#### 建議4.2：使用更大的Batch Size
```python
# 當前：batch_size=32（案例1）或128（案例2、3）
# 建議：增加batch_size以穩定訓練
history = model.fit(
    X_train, y_train,
    batch_size=256,  # 增加到256
    ...
)
```

#### 建議4.3：增加訓練Epochs
```python
# 當前：300 epochs
# 建議：500-1000 epochs（配合早停）
history = model.fit(
    X_train, y_train,
    epochs=500,  # 增加epochs
    ...
)
```

#### 建議4.4：使用Warmup策略
```python
# 先用小學習率預熱
import numpy as np

class WarmUpSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, target_lr):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
    
    def __call__(self, step):
        if step < self.warmup_steps:
            return self.initial_lr + (self.target_lr - self.initial_lr) * step / self.warmup_steps
        return self.target_lr

lr_schedule = WarmUpSchedule(
    initial_lr=1e-5,
    warmup_steps=100,
    target_lr=1e-3
)
```

---

### 💡 優先級5：集成學習策略

#### 建議5.1：多模型集成
```python
# 訓練多個DNN模型（不同初始化）
def train_ensemble(n_models=5):
    models = []
    for i in range(n_models):
        model = create_improved_dnn()
        model.compile(...)
        model.fit(...)
        models.append(model)
    return models

def ensemble_predict(models, X):
    predictions = [model.predict(X) for model in models]
    return np.mean(predictions, axis=0)  # 平均預測
```

#### 建議5.2：混合模型（DNN + Traditional ML）
```python
# Stacking: DNN作為元特徵
from sklearn.linear_model import Ridge

# Level 1: 基礎預測
dnn_pred = dnn_model.predict(X_val)
rf_pred = rf_model.predict(X_val)

# Level 2: 元模型
meta_X = np.column_stack([dnn_pred, rf_pred])
meta_model = Ridge(alpha=0.1)
meta_model.fit(meta_X, y_val)
```

---

### 🔬 優先級6：數據增強策略

#### 建議6.1：添加噪聲增強（適合案例1）
```python
# 訓練時添加小量噪聲
def add_noise_augmentation(X, y, noise_level=0.01):
    X_noisy = X + np.random.normal(0, noise_level, X.shape)
    return np.vstack([X, X_noisy]), np.concatenate([y, y])
```

#### 建議6.2：時序數據增強（適合案例2、3）
```python
# 滑動窗口增強
def sliding_window_augmentation(X, y, window_shift=1):
    X_aug, y_aug = [], []
    for shift in range(1, window_shift+1):
        X_shifted = np.roll(X, shift, axis=0)
        X_aug.append(X_shifted[shift:])
        y_aug.append(y[shift:])
    return np.vstack(X_aug), np.concatenate(y_aug)
```

---

## 📊 針對各案例的具體建議

### 案例1：葡萄酒質量預測

**當前問題**：
- DNN表現一般（Test R²=0.329）
- RandomForest嚴重過擬合

**改進方案**：
1. **更換激活函數**：sigmoid → ReLU ⭐⭐⭐⭐⭐
2. **增加第一層寬度**：128 → 512（有16個特徵）⭐⭐⭐⭐
3. **降低正則化**：L2=0.01 → 0.001 ⭐⭐⭐⭐
4. **數據增強**：添加噪聲增強 ⭐⭐⭐
5. **多模型集成**：訓練5個模型取平均 ⭐⭐⭐

**預期改進**：Test R² 0.329 → 0.45-0.50

---

### 案例2：SRU H₂S濃度預測

**當前問題**：
- DNN接近Ridge但略差（Test R²=0.929 vs 0.939）
- 有時序特性但未充分利用

**改進方案**：
1. **更換激活函數**：sigmoid → ELU（允許負值）⭐⭐⭐⭐⭐
2. **增加網絡深度**：4層 → 6-7層 ⭐⭐⭐⭐
3. **使用LSTM/GRU層**：捕捉長期依賴 ⭐⭐⭐⭐
4. **學習率調度**：Cosine Annealing ⭐⭐⭐
5. **增加訓練Epochs**：300 → 500 ⭐⭐⭐

**預期改進**：Test R² 0.929 → 0.94-0.95

**進階方案：混合架構**
```python
def create_hybrid_model():
    # 特徵提取層（DNN）
    dense_input = Input(shape=(8,))
    x = layers.Dense(256, activation='elu')(dense_input)
    x = layers.Dense(128, activation='elu')(x)
    
    # 時序處理層（LSTM）
    # 將特徵重塑為時序格式
    x = layers.Reshape((8, 1))(dense_input)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32)(x)
    
    # 融合與輸出
    x = layers.Dense(64, activation='elu')(x)
    output = layers.Dense(1, activation='linear')(x)
    
    return keras.Model(inputs=dense_input, outputs=output)
```

---

### 案例3：Debutanizer蒸餾塔

**當前問題**：
- 所有模型都已達到極高性能（R²>0.99）
- Ridge略優於DNN（Test R²=0.999 vs 0.997）

**改進方案**：
1. **微調優化**：sigmoid → ReLU ⭐⭐⭐⭐
2. **殘差連接**：幫助極深網絡訓練 ⭐⭐⭐
3. **集成學習**：多模型平均 ⭐⭐⭐
4. **更精細的早停**：min_delta=1e-6 ⭐⭐

**預期改進**：Test R² 0.997 → 0.998-0.999（提升空間有限）

**注意**：此案例已達到很高性能，進一步優化的邊際效益較低。

---

## 🚀 快速實施方案（最小改動、最大收益）

### 方案A：僅更換激活函數（5分鐘實施）
```python
# 將所有模型的 activation='sigmoid' 改為 activation='relu'
layers.Dense(128, activation='relu')  # 改這裡
layers.Dense(64, activation='relu')   # 改這裡
layers.Dense(32, activation='relu')   # 改這裡
layers.Dense(1, activation='linear')  # 保持linear
```
**預期提升**：10-30%

### 方案B：激活函數 + 架構優化（15分鐘實施）
```python
# 1. 改activation為relu
# 2. 增加第一層寬度：128 → 256
# 3. 降低L2正則化：0.01 → 0.001
# 4. 降低Dropout：0.3 → 0.2
```
**預期提升**：20-40%

### 方案C：完整優化（30分鐘實施）
```python
# 1. 改activation為relu/elu
# 2. 增加網絡深度和寬度
# 3. 調整正則化
# 4. 優化學習率調度
# 5. 增加訓練epochs
```
**預期提升**：30-50%

---

## 📈 預期改進對照表

| 案例 | 當前Test R² | 方案A | 方案B | 方案C | 理論上限 |
|------|------------|-------|-------|-------|----------|
| 案例1 | 0.329 | 0.38 | 0.43 | 0.48 | 0.50 |
| 案例2 | 0.929 | 0.94 | 0.945 | 0.95 | 0.955 |
| 案例3 | 0.997 | 0.998 | 0.998 | 0.999 | 0.999 |

---

## ⚠️ 注意事項

### 1. 過擬合風險
- 增加模型複雜度時，密切監控Train vs Val性能差距
- 如果Train R² >> Val R²，需要增加正則化

### 2. 訓練時間
- 更深更寬的網絡需要更長訓練時間
- 建議使用GPU加速（如果可用）

### 3. 數據規模
- 案例1僅有1143樣本，過深網絡可能過擬合
- 案例2、3樣本較多，可以使用更複雜模型

### 4. 實驗記錄
- 每次修改記錄超參數和結果
- 使用TensorBoard或MLflow追蹤實驗

---

## 🎓 總結

**核心建議**：
1. ⭐⭐⭐⭐⭐ **立即更換激活函數**（sigmoid → ReLU/ELU）
2. ⭐⭐⭐⭐ 增加網絡容量（更深/更寬）
3. ⭐⭐⭐⭐ 降低正則化強度
4. ⭐⭐⭐ 優化學習率調度
5. ⭐⭐⭐ 使用集成學習

**實施順序**：
1. 先做方案A（最小改動）
2. 評估效果，如果不夠再做方案B
3. 如需進一步提升，實施方案C

**預期結果**：
- 案例1：最大提升空間，預期改進30-50%
- 案例2：中等提升空間，預期改進5-10%
- 案例3：最小提升空間，預期改進1-2%

---

## 🔗 相關資源

1. **Activation Functions**：
   - ReLU: Nair & Hinton (2010)
   - ELU: Clevert et al. (2015)
   - Swish: Ramachandran et al. (2017)

2. **Network Architecture**：
   - ResNet: He et al. (2016)
   - DenseNet: Huang et al. (2017)

3. **Learning Rate Schedules**：
   - Cosine Annealing: Loshchilov & Hutter (2016)
   - Warmup: Goyal et al. (2017)

4. **Ensemble Methods**：
   - Stacking: Wolpert (1992)
   - Snapshot Ensemble: Huang et al. (2017)
