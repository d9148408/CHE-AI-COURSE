"""
重建 Unit11_Advanced_Topics_V2.ipynb
從原始檔案讀取所有 cells 並按正確順序寫入新檔案
"""
import json
import re
from pathlib import Path

# 檔案路徑
source_file = Path("Unit11_Advanced_Topics.ipynb")
target_file = Path("Unit11_Advanced_Topics_V2.ipynb")
backup_file = Path("Unit11_Advanced_Topics_backup_rebuild.ipynb")

print("="*80)
print("開始重建 Notebook")
print("="*80)

# 1. 備份原始檔案
if source_file.exists():
    import shutil
    shutil.copy(source_file, backup_file)
    print(f"✓ 已備份原始檔案至：{backup_file}")

# 2. 讀取原始 notebook
with open(source_file, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

cells = notebook['cells']
print(f"✓ 讀取原始 notebook：{len(cells)} cells")

# 3. 定義正確的 cell 順序 (基於 Sections)
# Section 0-6: 已執行成功的部分 (Cells 1-67)
# Section 7-8: 優化實驗部分 (Cells 68-76) - 需要重新排序
# Section 9: 總結部分 (Cells 77-89)

# 找出關鍵分界點
section_7_start = None  # "Weighted Ensemble 閾值優化分析" 的下一個 cell
section_8_start = None  # "實驗 5 分析" 標題
section_9_start = None  # "進一步優化方向"

for i, cell in enumerate(cells):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        
        # 找到 Section 7 的結束 (Weighted 閾值優化分析之後)
        if '**Weighted Ensemble 閾值優化分析**：' in source and section_7_start is None:
            section_7_start = i + 1
            print(f"✓ Section 7 結束於 Cell {i+1} (1-based)")
        
        # 找到 Section 8 的開始 (實驗 5 分析)
        if '**實驗 5 分析**' in source and section_8_start is None:
            section_8_start = i
            print(f"✓ Section 8 開始於 Cell {i+1} (1-based)")
        
        # 找到 Section 9 的開始 (進一步優化方向)
        if '## 進一步優化方向' in source and section_9_start is None:
            section_9_start = i
            print(f"✓ Section 9 開始於 Cell {i+1} (1-based)")

# 4. 建立新的 cells 列表
new_cells = []

# 保留 Sections 0-6 (cells 0 到 section_7_start-1)
if section_7_start:
    new_cells.extend(cells[:section_7_start])
    print(f"✓ 保留 Sections 0-6: Cells 1-{section_7_start} ({section_7_start} cells)")

# Section 8 重建：實驗 5-7 (需要確保正確順序)
if section_8_start and section_9_start:
    section_8_cells = cells[section_8_start:section_9_start]
    print(f"\n--- Section 8 重建 (Cells {section_8_start+1}-{section_9_start}) ---")
    
    # 分析 Section 8 的內容
    exp5_cells = []
    exp6_cells = []
    exp7_cells = []
    merged_viz_cell = None
    
    for i, cell in enumerate(section_8_cells):
        source = ''.join(cell['source'])
        
        # 實驗 5 相關
        if '**實驗 5 分析**' in source or '#### **實驗 5：' in source:
            exp5_cells.append(cell)
            print(f"  [Exp 5] Cell {section_8_start+i+1}: {cell['cell_type']}")
        elif cell['cell_type'] == 'code' and 'percentiles_high' in source and 'df_weighted_high' in source:
            exp5_cells.append(cell)
            print(f"  [Exp 5] Cell {section_8_start+i+1}: Code (建立 df_weighted_high)")
        
        # 實驗 6 相關
        elif '#### **實驗 6：' in source:
            exp6_cells.append(cell)
            print(f"  [Exp 6] Cell {section_8_start+i+1}: Title")
        elif cell['cell_type'] == 'code' and 'cost_scenarios' in source:
            exp6_cells.append(cell)
            print(f"  [Exp 6] Cell {section_8_start+i+1}: Code (cost-sensitive)")
        elif cell['cell_type'] == 'code' and 'df_cost_sensitive' in source and 'axes[0, 0].bar' in source:
            exp6_cells.append(cell)
            print(f"  [Exp 6] Cell {section_8_start+i+1}: Code (visualization)")
        elif '**實驗 6 分析**' in source:
            exp6_cells.append(cell)
            print(f"  [Exp 6] Cell {section_8_start+i+1}: Analysis")
        
        # 合併視覺化 cell
        elif cell['cell_type'] == 'code' and 'all_percentiles = percentiles_weighted + percentiles_high' in source:
            merged_viz_cell = cell
            print(f"  [Merged Viz] Cell {section_8_start+i+1}: 合併視覺化")
        
        # 實驗 7 相關 (所有剩餘 cells)
        elif '#### **實驗 7：' in source or 'add_statistical_features' in source:
            exp7_cells.append(cell)
            print(f"  [Exp 7] Cell {section_8_start+i+1}: {cell['cell_type']}")
    
    # 重新排序：Exp5 Title → Exp5 Code → Exp6 Title → Exp6 Code → Exp6 Viz → Exp6 Analysis → Merged Viz → Exp7...
    print("\n✓ Section 8 重建順序：")
    
    # 實驗 5
    if len(exp5_cells) >= 2:
        # Title, Code
        new_cells.extend(exp5_cells[:2])
        print(f"  • Exp 5: {len(exp5_cells[:2])} cells (Title + Code)")
    
    # 實驗 6
    if len(exp6_cells) == 4:
        # Title, Code, Viz, Analysis
        new_cells.extend(exp6_cells)
        print(f"  • Exp 6: {len(exp6_cells)} cells (Title → Code → Viz → Analysis)")
    
    # 合併視覺化
    if merged_viz_cell:
        new_cells.append(merged_viz_cell)
        print(f"  • Merged Viz: 1 cell")
    
    # 實驗 7
    if exp7_cells:
        new_cells.extend(exp7_cells)
        print(f"  • Exp 7: {len(exp7_cells)} cells")

# Section 9: 總結部分
if section_9_start:
    new_cells.extend(cells[section_9_start:])
    print(f"\n✓ Section 9 (總結): {len(cells[section_9_start:])} cells")

# 5. 建立新 notebook
new_notebook = {
    "cells": new_cells,
    "metadata": notebook['metadata'],
    "nbformat": notebook['nbformat'],
    "nbformat_minor": notebook['nbformat_minor']
}

# 6. 寫入檔案
with open(target_file, 'w', encoding='utf-8') as f:
    json.dump(new_notebook, f, ensure_ascii=False, indent=1)

print("\n" + "="*80)
print(f"✅ 重建完成！")
print(f"✓ 新檔案：{target_file} ({len(new_cells)} cells)")
print(f"✓ 備份檔案：{backup_file}")
print("="*80)
print("\n請在 VS Code 中開啟新檔案並執行 Cells 68-76 (Section 8 的優化實驗)")
