# Unit 10: 智慧實驗設計 (Bayesian Optimization) - 使用指南

## 檔案結構

```
Part_5/
  ├── Unit10_Bayesian_Optimization.md          # 主要講義（已擴充）
  ├── Unit10_Bayesian_Optimization.ipynb       # 程式實作（已擴充）
  ├── Unit10_Enhancement_Summary.md            # 改進總結文件（講師備註）
  └── Unit10_README.md                         # 本使用指南（講師備註）
Jupyter_Scripts/Unit10_Results/                # 教材示例圖片（已產生）
outputs/Unit10_Results/                        # 你自己執行 Notebook 的新輸出（會自動建立）
```

## 快速開始

### 1. 環境準備

```bash
# 安裝必要套件
pip install numpy matplotlib scipy scikit-learn jupyter

# 確認版本
python -c "import sklearn; print(sklearn.__version__)"  # 應 >= 0.24
python -c "import scipy; print(scipy.__version__)"      # 應 >= 1.7
```

### 2. 執行 Jupyter Notebook

```bash
# 在專案根目錄啟動 Jupyter（或在 IDE 直接開啟 .ipynb）
jupyter notebook Part_5/Unit10_Bayesian_Optimization.ipynb
```

### 3. 執行順序

**建議按以下順序執行 Cells**：

1. **第 1-2 節**：匯入套件 + 定義黑盒子函數（1D 簡單案例）
2. **第 3 節**：貝葉斯最佳化迭代（1D）
3. **第 4 節**：結果總結
4. **第 5 節**：生質柴油 2D 真實案例（新增）
5. **第 6 節**：獲得函數比較實驗（新增）

## 學習路徑

### 初學者路徑（2-3 小時）

1. 閱讀講義章節 1-2（實驗室現實 + BO 基本概念）
2. 執行 Notebook 第 1-4 節（1D 簡單案例）
3. 觀察迭代過程的視覺化圖片
4. 理解 EI 獲得函數的作用

**學習目標**：理解 BO 的基本原理與工作流程

### 進階路徑（4-6 小時）

1. 閱讀講義章節 3（高斯過程與獲得函數數學原理）
2. 執行 Notebook 第 5 節（2D 真實案例）
3. 執行 Notebook 第 6 節（獲得函數比較）
4. 閱讀講義章節 6（真實化工案例分析）

**學習目標**：掌握多維度優化與不同獲得函數的選擇策略

### 實務應用路徑（6-8 小時）

1. 閱讀講義章節 8（實驗室應用實務指南）
2. 修改 Notebook 參數，觀察對結果的影響
3. 嘗試將 BO 應用到自己的研究問題
4. 設計包含約束條件的優化方案

**學習目標**：能夠在真實實驗室環境中導入 BO

## 重點內容導覽

### 理論精華

- **高斯過程 (GP)**：講義 3.1 + Notebook 第 3 節
- **Expected Improvement (EI)**：講義 3.2.1 + Notebook 第 6 節
- **多維度優化**：講義 4 + Notebook 第 5 節

### 真實案例

| 案例 | 位置 | 產業 | 優化變數 |
|:-----|:-----|:-----|:---------|
| Fischer-Tropsch 合成 | 講義 6.1 | 石化 | 4D |
| 藥物晶型篩選 | 講義 6.2 | 製藥 | 3D |
| 聚丙烯生產 | 講義 6.3 | 高分子 | 3D |
| 生質柴油製程 | 講義 6.4 + Notebook 5 | 生質能源 | 2D 或 4D |

### 程式實作

| 功能 | Notebook 位置 | 程式碼行數 |
|:-----|:-------------|:-----------|
| 1D 基礎案例 | 第 1-4 節 | ~150 行 |
| 2D 真實案例 | 第 5 節 | ~200 行 |
| 獲得函數比較 | 第 6 節 | ~150 行 |

## 常見問題

### Q1: 執行時出現 "module not found" 錯誤

**解決方案**：
```bash
pip install --upgrade numpy matplotlib scipy scikit-learn
```

### Q2: 圖片無法正常顯示中文

**解決方案**：
```python
# 在 Notebook 第一個 cell 加入
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
```

### Q3: 2D 案例執行很慢

**原因**：網格計算較密集（100×100 = 10,000 個點）

**解決方案**：
```python
# 降低網格密度
methanol_grid = np.linspace(6, 12, 50)  # 原本 100 改成 50
temp_grid = np.linspace(50, 65, 50)
```

### Q4: 如何應用到自己的問題？

**步驟**：
1. 定義你的目標函數（實驗結果）
2. 確定優化變數的範圍
3. 選擇合適的 kernel（參考講義 3.3）
4. 選擇合適的獲得函數（參考講義 3.2.2）
5. 設定停止準則（參考講義 8.2 Q4）

## 預期輸出

### 視覺化圖表

執行完成後，`Unit10_Results/` 資料夾應包含：

- `iteration_1.png` ~ `iteration_5.png`：1D 案例迭代過程
- `final_result.png`：1D 案例最終結果
- `biodiesel_true_surface.png`：2D 案例真實曲面
- `biodiesel_final_comparison.png`：2D 案例最終對比
- `acquisition_function_comparison.png`：三種獲得函數比較

### 數值結果

**1D 案例**：
- 總實驗次數：7 次（2 初始 + 5 AI）
- 預期找到的最佳值：y ≈ 2.9-3.0
- 理論全域最佳值：y ≈ 3.0

**2D 案例（生質柴油）**：
- 總實驗次數：20 次（5 初始 + 15 AI）
- 預期找到的最佳產率：96-97%
- 理論全域最佳值：~97%

## 延伸學習資源

### 線上課程
- [Coursera: Bayesian Methods for Machine Learning](https://www.coursera.org/learn/bayesian-methods-in-machine-learning)
- [YouTube: Nando de Freitas - Gaussian Processes](https://www.youtube.com/watch?v=4vGiHC35j9s)

### Python 套件
- `scikit-optimize` (skopt)：適合入門 [文檔](https://scikit-optimize.github.io/)
- `GPyOpt`：功能完整 [文檔](https://sheffieldml.github.io/GPyOpt/)
- `BoTorch`：最先進 [文檔](https://botorch.org/)

### 學術論文
- Shahriari et al. (2016), "Taking the Human Out of the Loop", *Proceedings of the IEEE*
- Garnett (2023), *Bayesian Optimization* [免費線上書](https://bayesoptbook.com/)

## 回饋與建議

如有任何問題或建議，歡迎透過以下方式聯繫：

- 課程討論區
- Email: [請填入課程聯絡信箱]
- Office Hour: [請填入時間地點]

---

**最後更新**：2025-11-30
**版本**：v2.0
**適用課程**：化工數據科學與機器學習實務（CHE-AI-101）
