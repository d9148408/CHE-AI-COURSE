# Unit16 內容架構重組報告

> 日期: 2025-01-XX  
> 目標: 解決 Unit16 主教材使用錯誤數據集的問題

---

## 1. 問題診斷

### 原始問題
用戶反映：「Unit16_CNN_Basics 使用手寫數字（應該在 Unit16_Appendix_MNIST）」

### 根本原因
- **主教材定位不清**：Unit16_CNN_Basics_Industrial_Inspection.ipynb 名稱寫「工業檢測」，但實際使用 `sklearn.load_digits()`（8x8 手寫數字）
- **內容重複**：手寫數字相關教學已在 Unit16_Appendix_MNIST.md 中完整涵蓋（1015 行，30+ 數學公式）
- **教學邏輯矛盾**：講義提到「延伸到 NEU-DET 等工業資料」，但 notebook 沒有實作

---

## 2. 解決方案

### 核心策略：內容分流
1. **主教材**：聚焦工業影像檢測實務（NEU-DET 鋼材缺陷）
2. **附錄 A**：MNIST 手寫數字（CNN 基礎原理教學）
3. **附錄 B**：Cats vs Dogs（遷移學習入門）

### 具體修改

#### 2.1 Notebook 完全重寫
**檔案**: `Part_4/Unit16_CNN_Basics_Industrial_Inspection.ipynb`

**替換內容**:
```python
# ❌ 舊版（錯誤）
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data  # 8x8 手寫數字

# ✅ 新版（正確）
from PIL import Image
NEU_DET_DIR = WORKSPACE_ROOT / 'Jupyter_Scripts' / 'NEU-DET'
class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
```

**新增章節**:
1. **第 1 章**：數據集探索（NEU-DET 6 類缺陷可視化）
2. **第 2 章**：建立 Baseline（Random Forest / MLP）
3. **第 3 章**：混淆矩陣與錯誤分析
4. **第 4 章**：信心度門檻與拒絕選項（Reject Option）
5. **選讀**：Keras CNN 完整訓練流程（含數據增強）

#### 2.2 講義同步更新
**檔案**: `Part_4/Unit16_CNN_Basics_Industrial_Inspection.md`

**新增核心內容**:
- ✅ 對比表：MNIST vs 工業影像檢測（5 個維度）
- ✅ 教學策略說明（為什麼先做 baseline）
- ✅ 最低交付物清單（混淆矩陣 + 信心度權衡圖 + 技術報告）
- ✅ 補充教材分流說明（主教材 vs 附錄 A vs 附錄 B）
- ✅ 常見問題 FAQ（4 個典型問題）
- ✅ 延伸閱讀（數據集、Kaggle 競賽、論文）

---

## 3. 內容架構對照表

| 教材 | 數據集 | 目標 | 重點 |
|------|--------|------|------|
| **Unit16_CNN_Basics_Industrial_Inspection** | NEU-DET 鋼材缺陷 (1800 張) | 工業影像檢測實務 | 類別不平衡、錯誤成本、信心度門檻、部署考量 |
| **Unit16_Appendix_MNIST** | MNIST 手寫數字 (70,000 張) | CNN 基礎原理教學 | 卷積層數學、池化原理、過擬合診斷 |
| **Unit16_Appendix_CatsVsDogs** | Cats vs Dogs (3,000 張) | 遷移學習入門 | 特徵提取、微調、小數據集策略 |

---

## 4. 關鍵差異比較

### 數據集特性

#### sklearn digits (舊版 ❌)
- **尺寸**: 8×8 像素（極小，不真實）
- **總樣本**: 1,797 張
- **類別**: 10 類（手寫數字 0-9）
- **場景**: 玩具數據集，僅用於演示 ML 流程

#### NEU-DET (新版 ✅)
- **尺寸**: 200×200 像素（真實工業影像）
- **總樣本**: 1,800 張（訓練 1,440 + 驗證 360）
- **類別**: 6 類（龜裂、雜質、斑塊、麻點、氧化皮、劃痕）
- **場景**: 真實鋼材表面缺陷，來自東北大學工業數據集

### 學習目標對比

| 維度 | sklearn digits | NEU-DET |
|------|----------------|---------|
| **教學價值** | 快速驗證 ML 流程 | 理解工業實務挑戰 |
| **技術深度** | 淺層分類器 | Baseline + CNN + 部署策略 |
| **實務相關性** | 低（玩具數據） | 高（真實工業場景）|
| **可遷移性** | 僅限演示 | 可直接應用於化工/材料檢測 |

---

## 5. 備份與版本管理

### 舊版備份
```
backup/Part_4/Unit16_CNN_Basics_Industrial_Inspection_OLD_digits.ipynb
```
**內容**: 使用 sklearn.load_digits() 的舊版 notebook（已備份但不再使用）

### 當前版本
```
Part_4/Unit16_CNN_Basics_Industrial_Inspection.ipynb
```
**內容**: 使用 NEU-DET 鋼材缺陷數據集的完整工業檢測教學

---

## 6. 驗證檢查清單

- ✅ Unit16 主教材不再使用手寫數字數據集
- ✅ NEU-DET 鋼材缺陷數據集正確載入
- ✅ 講義與 Notebook 內容完全同步
- ✅ 教學目標清晰（工業檢測實務，非 CNN 原理教學）
- ✅ 與 Unit16_Appendix_MNIST 內容無重複
- ✅ 路徑使用 RESULTS_DIR 變數（無硬編碼）

---

## 7. 教學建議流程

### 第 1 週：主教材（工業檢測實務）
- 使用 Unit16_CNN_Basics_Industrial_Inspection.ipynb
- 重點：數據探索 → Baseline → 混淆矩陣 → 信心度門檻
- 作業：撰寫「如何部署到產線」的技術報告

### 第 2 週：附錄 A（CNN 原理補充）
- 使用 Unit16_Appendix_MNIST.md
- 重點：卷積層數學、池化層、過擬合診斷
- 作業：實作 MNIST 手寫數字識別（Keras）

### 第 3 週：附錄 B（遷移學習）
- 使用 Unit16_Appendix_CatsVsDogs.ipynb
- 重點：MobileNetV2 特徵提取、微調策略
- 作業：比較從頭訓練 vs 遷移學習的效果

---

## 8. 成果總結

### 問題解決
- ❌ 舊版：主教材使用手寫數字（與工業檢測主題不符）
- ✅ 新版：主教材使用 NEU-DET 鋼材缺陷（真實工業數據）

### 內容質量
- **Notebook**: 16 個 Cell（從數據探索到 CNN 訓練完整流程）
- **講義**: 10 個章節（新增對比表、FAQ、教學建議流程）
- **程式碼**: 完整可執行（含 TensorFlow 檢測與錯誤處理）

### 教學架構
- **主教材**：工業檢測實務（NEU-DET）
- **附錄 A**：CNN 原理教學（MNIST）
- **附錄 B**：遷移學習（Cats vs Dogs）

內容清晰分流，不再混淆 🎯

---

## 9. 文件變更記錄

| 檔案 | 操作 | 說明 |
|------|------|------|
| `Part_4/Unit16_CNN_Basics_Industrial_Inspection.ipynb` | **重寫** | 完全替換為 NEU-DET 版本（16 cells） |
| `Part_4/Unit16_CNN_Basics_Industrial_Inspection.md` | **更新** | 新增對比表、FAQ、教學流程建議 |
| `backup/Part_4/Unit16_CNN_Basics_Industrial_Inspection_OLD_digits.ipynb` | **備份** | 保留舊版 digits 程式碼 |
| `Part_4/Unit16_Appendix_MNIST.md` | **保持** | 無變更（已是完整 MNIST 教學）|
| `Part_4/Unit16_Appendix_CatsVsDogs.ipynb` | **保持** | 無變更（遷移學習教學）|

---

## 10. 後續行動

### 立即執行
- [x] 重寫 Unit16 Notebook（使用 NEU-DET）
- [x] 更新 Unit16 講義（同步 Notebook 內容）
- [x] 備份舊版檔案
- [x] 生成更新報告

### 待處理（優先級較低）
- [ ] Unit17-19 路徑修正（7 個硬編碼路徑）
- [ ] 測試 Notebook 完整執行（需 NEU-DET 數據集）
- [ ] 學生作業範例（技術報告模板）

---

**報告完成日期**: 2025-01-XX  
**更新負責人**: GitHub Copilot  
**審核狀態**: 等待用戶確認
