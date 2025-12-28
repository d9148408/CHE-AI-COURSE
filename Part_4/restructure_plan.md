# Unit15_DNN_MLP_Overview.md 重構計劃

## 發現的問題

### 1. **損失函數嚴重重複** ⚠️ 

**第一次出現：第1.5節 (82-174行)**
- 位置：基礎理論章節
- 內容：詳細的數學公式、適用場景、優缺點
- 包含：MSE, MAE, Huber, BCE, CCE, SCCE
- 特點：理論導向，解釋「為什麼」

**第二次出現：第6.3節 (1063-1228行)**
- 位置：模型編譯章節
- 內容：幾乎完全相同的數學公式
- 包含：MSE, MAE, Huber, MSLE, BCE, CCE, SCCE
- 特點：實作導向，強調 Keras 使用方式
- **額外內容**：MSLE（均方對數誤差）、Keras 程式碼範例

**重複程度**：約 90% 重複，數學公式完全相同

**解決方案**：
- **保留第1.5節**：作為理論基礎，包含所有損失函數的數學定義和理論解釋
- **簡化第6.3節**：
  - 刪除數學公式（引用第1.5節）
  - 保留 Keras 使用方式（程式碼範例）
  - 添加引用：「損失函數的數學原理請參考第1.5節」
  - 保留快速參考表格

### 2. **激活函數重複**

**第一次出現：第2章 (206-291行)**
- 完整介紹：ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax, Linear
- 包含：公式、優缺點、適用場景
- 有選擇指南表格

**第二次出現：第5.1.3章 ReLU深入理論 (約650-750行)**
- 詳細解釋 ReLU 的三大理由
- 梯度消失問題、計算效率、稀疏激發
- 死亡 ReLU 問題及解決方案

**重複程度**：約 30% 重複（ReLU 基本介紹）

**解決方案**：
- **保留第2章**：所有激活函數的基本介紹
- **保留第5.1.3的深入理論**：但移到新的第3章
- 在第2.1.1節 ReLU 添加引用：「關於 ReLU 的深入理論分析，請參考第3.4節」
- 在新第3.4節添加引用：「ReLU 的基本定義請參考第2.1.1節」

### 3. **章節編號混亂**

後面有多個重複的章節編號（如第15章出現多次）

### 4. **教學流程可優化**

當前流程：
```
1. 基礎理論 (數學模型、前向傳播、損失函數、反向傳播)
2. 激活函數
3. 應用場景
4. TensorFlow/Keras介紹
5. 建立模型
   5.1 Sequential vs Functional API
   5.1.3 DNN架構設計理論 (新增的詳細理論)
   5.2 常用層
6. 模型編譯
   6.3 損失函數 (與1.5重複)
   ...
```

**問題分析**：
- 架構設計理論埋在「建立模型」章節中，不夠突出
- 損失函數在第1章和第6章重複
- 學生在實作前缺乏完整的設計指導

## 重構策略

### 階段1：重新組織章節結構

**建議的新結構**：

```
1. DNN與MLP基礎理論
   1.1 什麼是神經網路
   1.2 歷史發展
   1.3 神經元數學模型
   1.4 前向傳播
   1.5 損失函數 ⭐ (保留，作為理論基礎)
   1.6 反向傳播
   1.7 梯度下降與參數更新

2. 激活函數
   2.1 常用激活函數 (ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax, Linear)
   2.2 激活函數選擇指南

3. DNN架構設計理論 ⭐ (從5.1.3提升為獨立章節)
   3.1 漏斗型架構 (Funnel Architecture)
   3.2 第一層節點數的決定
   3.3 層數深度選擇
   3.4 ReLU深入理論 (為何隱藏層全用ReLU)

4. DNN/MLP應用場景
   4.1 適合使用DNN/MLP的情境
   4.2 化工領域應用案例
   4.3 DNN的優勢與限制

5. TensorFlow/Keras框架介紹
   5.1 TensorFlow與Keras簡介
   5.2 環境安裝
   5.3 基本導入

6. 使用Keras建立DNN模型
   6.1 模型架構: Sequential vs Functional API
   6.2 常用層 (Dense, Dropout, BatchNormalization, Activation)
   6.3 權重初始化策略
   6.4 正則化

7. 模型編譯
   7.1 model.compile() 方法
   7.2 優化器 (Adam, SGD, RMSprop)
   7.3 損失函數 ⭐ (簡化，只保留Keras使用方式)
   7.4 評估指標
   7.5 模型摘要與視覺化

8. 模型訓練
   8.1 model.fit() 方法
   8.2 重要參數說明
   8.3 Callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)

9. 訓練過程視覺化
   9.1 繪製訓練曲線
   9.2 解讀訓練曲線

10. 模型評估
    10.1 model.evaluate() 方法
    10.2 詳細評估指標計算

11. 模型預測
    11.1 model.predict() 方法
    11.2 預測結果分析

12. 模型保存與載入
    12.1 保存模型
    12.2 載入模型

13. TensorFlow/Keras vs sklearn MLPRegressor/MLPClassifier
    13.1 功能比較
    13.2 使用場景建議

14. 完整工作流程範例
    14.1 數據準備
    14.2 模型建立
    14.3 訓練與評估
    14.4 結果分析

15. 最佳實踐與建議
    15.1 數據預處理
    15.2 模型設計
    15.3 訓練策略
    15.4 常見問題與解決

16. 課堂作業
    作業一: 完整DNN建模流程
    作業二: 超參數探討

17. 總結

18. 參考資料
```

### 階段2：消除重複內容

#### 2.1 損失函數重複處理

**第1.5節 (保留完整內容)**：
```markdown
### 1.5 損失函數(Loss Function)

損失函數用於衡量模型預測值與真實值之間的差異。

**回歸問題常用損失函數**:
1. MSE - 完整數學公式和理論
2. MAE - 完整數學公式和理論
3. Huber Loss - 完整數學公式和理論

**分類問題常用損失函數**:
1. Binary Crossentropy - 完整數學公式和理論
2. Categorical Crossentropy - 完整數學公式和理論
3. Sparse Categorical Crossentropy - 完整數學公式和理論

### 損失函數選擇指南
[保留表格]
```

**第7.3節 (大幅簡化)**：
```markdown
### 7.3 損失函數在Keras中的使用

> [!NOTE]
> 損失函數的數學原理和理論基礎請參考 **第1.5節**。本節僅說明在Keras中的使用方式。

#### 7.3.1 回歸問題

**MSE (均方誤差)**:
```python
model.compile(optimizer='adam', loss='mse')
# 或
from tensorflow.keras.losses import MeanSquaredError
model.compile(optimizer='adam', loss=MeanSquaredError())
```

**MAE (平均絕對誤差)**:
```python
model.compile(optimizer='adam', loss='mae')
```

**Huber Loss**:
```python
from tensorflow.keras.losses import Huber
model.compile(optimizer='adam', loss=Huber(delta=1.0))
```

**MSLE (均方對數誤差)** - 適合目標值範圍很大的情況:
```python
model.compile(optimizer='adam', loss='msle')
```

#### 7.3.2 分類問題

**Binary Crossentropy** (二元分類):
```python
model.compile(optimizer='adam', loss='binary_crossentropy')
```

**Categorical Crossentropy** (多類別，one-hot標籤):
```python
model.compile(optimizer='adam', loss='categorical_crossentropy')
```

**Sparse Categorical Crossentropy** (多類別，整數標籤):
```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
```

#### 7.3.3 快速參考

| 問題類型 | Keras損失函數 | 輸出層激活函數 |
|---------|--------------|---------------|
| 回歸(一般) | `'mse'` | `'linear'` |
| 回歸(有異常值) | `'mae'` 或 `Huber()` | `'linear'` |
| 回歸(大範圍) | `'msle'` | `'linear'` |
| 二元分類 | `'binary_crossentropy'` | `'sigmoid'` |
| 多類別(one-hot) | `'categorical_crossentropy'` | `'softmax'` |
| 多類別(整數) | `'sparse_categorical_crossentropy'` | `'softmax'` |
```

**節省的篇幅**：約 120 行 → 60 行

#### 2.2 激活函數重複處理

**第2.1.1節 ReLU (添加引用)**：
```markdown
#### 2.1.1 ReLU (Rectified Linear Unit)
$$
f(x) = \max(0, x)
$$

**優點**:
- 計算簡單、速度快
- 有效緩解梯度消失問題
- 使網路具有稀疏性

**缺點**:
- 可能出現"神經元死亡"問題(dying ReLU)

**適用場景**: 隱藏層的首選激活函數

> [!TIP]
> 關於 ReLU 為何成為深度學習標準配備的深入理論分析，請參考 **第3.4節**。
```

**第3.4節 (從5.1.3移過來，添加引用)**：
```markdown
#### 3.4 為何 DNN 隱藏層全都使用 ReLU？

> [!NOTE]
> ReLU 的基本定義和公式請參考 **第2.1.1節**。本節深入探討 ReLU 的理論優勢。

在現代 DNN 的隱藏層中，ReLU 幾乎就是「標準配備」...
[保留完整的深入理論內容]
```

### 階段3：修正章節編號

- 重新編號所有章節，確保從1到18連續
- 更新所有內部交叉引用
- 更新目錄

### 階段4：優化教學流程

**新流程的優勢**：

1. **理論完整性**：第1-3章建立完整的理論基礎
2. **設計指導明確**：第3章獨立講解架構設計，學生在實作前就了解設計原則
3. **實作循序漸進**：第5-8章從建立到訓練，流程清晰
4. **消除重複**：損失函數和激活函數各有明確定位
5. **便於查閱**：理論在前面章節，實作在後面章節

## 預期效果

### 內容精簡
- **損失函數**：從 ~250 行 → ~150 行（節省 40%）
- **激活函數**：消除 ReLU 基礎部分的重複
- **總體**：預計精簡約 100-150 行

### 結構優化
- 章節編號連續（1-18）
- 理論→設計→應用→實作，邏輯清晰
- 交叉引用明確，方便學生查閱

### 學習體驗提升
1. **第1-2章**：學習數學基礎（神經元、損失函數、激活函數）
2. **第3章**：學習設計原則（如何設計網路架構）
3. **第4章**：了解應用場景
4. **第5-8章**：動手實作
5. **第9-15章**：進階技巧和最佳實踐

## 驗證計劃

### 手動驗證
1. **內容完整性檢查**：
   - 確認所有損失函數都有說明（理論在1.5，Keras用法在7.3）
   - 確認所有激活函數都有說明（基礎在2.1，ReLU深入在3.4）
   
2. **交叉引用檢查**：
   - 檢查所有「參考第X節」的引用是否正確
   - 確認章節編號連續

3. **教學流程檢查**：
   - 從頭到尾閱讀，確認邏輯順暢
   - 確認沒有「前面引用後面章節」的情況

4. **程式碼範例檢查**：
   - 確認所有 Keras 程式碼範例正確
   - 確認損失函數使用方式完整

### 用戶確認
- 請用戶審閱重構後的文件
- 確認教學流程是否符合預期
- 收集反饋並調整
