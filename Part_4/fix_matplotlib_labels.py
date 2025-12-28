import json
import re

# 讀取 notebook
with open(r'd:\MyGit\CHE-AI-COURSE\Part_4\Unit16_CNN_Overview.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# 中文到英文的翻譯映射
translations = {
    '數字': 'Digit',
    '數量': 'Count',
    '訓練集標籤分布': 'Training Set Label Distribution',
    '測試集標籤分布': 'Test Set Label Distribution',
    '標籤': 'Label',
    'MNIST 訓練集樣本 (前 25 張)': 'MNIST Training Samples (First 25)',
    '影像 (標籤:': 'Image (Label:',
    '像素值熱圖': 'Pixel Value Heatmap',
    '像素值': 'Pixel Value',
    '頻率': 'Frequency',
    '像素值分布': 'Pixel Value Distribution',
    '訓練損失': 'Training Loss',
    '驗證損失': 'Validation Loss',
    '訓練與驗證損失': 'Training and Validation Loss',
    '訓練準確率': 'Training Accuracy',
    '驗證準確率': 'Validation Accuracy',
    '訓練與驗證準確率': 'Training and Validation Accuracy',
    '預測標籤': 'Predicted Label',
    '真實標籤': 'True Label',
    '混淆矩陣': 'Confusion Matrix',
    '預測:': 'Pred:',
    '信心度:': 'Conf:',
    '正確預測的樣本': 'Correctly Predicted Samples',
    '錯誤預測的樣本': 'Incorrectly Predicted Samples',
    '真實:': 'True:',
    '測試樣本': 'Test Sample',
    '數字類別': 'Digit Class',
    '預測機率': 'Prediction Probability',
    '預測類別': 'Predicted Class',
    '真實類別 (錯誤預測)': 'True Class (Incorrect)',
    '其他類別': 'Other Classes',
    '使用載入模型的預測結果': 'Predictions Using Loaded Model',
}

# 修正所有代碼單元格中的中文標籤
modified_count = 0
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = cell.get('source', [])
        new_source = []
        for line in source:
            new_line = line
            # 替換所有翻譯
            for chinese, english in translations.items():
                if chinese in new_line:
                    new_line = new_line.replace(chinese, english)
                    modified_count += 1
            new_source.append(new_line)
        cell['source'] = new_source

# 保存修改後的 notebook
with open(r'd:\MyGit\CHE-AI-COURSE\Part_4\Unit16_CNN_Overview.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=4)

print(f"修正完成! 共修改了 {modified_count} 處中文標籤")
