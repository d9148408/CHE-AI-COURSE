"""
生成 Water Potability 示例數據集

由於原始 Kaggle 數據集需要手動下載，此腳本生成結構相同的示例數據供測試。
實際使用時請從 Kaggle 下載真實數據。
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_sample_data(n_samples=3276):
    """
    生成與真實 Water Potability 數據集結構相同的示例數據
    
    特徵範圍參考 WHO/EPA 飲用水標準
    """
    np.random.seed(42)
    
    # 生成特徵數據（參考真實數據的統計分布）
    data = {
        'ph': np.random.normal(7.0, 1.5, n_samples).clip(0, 14),
        'Hardness': np.random.normal(200, 50, n_samples).clip(0, 350),
        'Solids': np.random.normal(22000, 12000, n_samples).clip(0, 60000),
        'Chloramines': np.random.normal(7.0, 2.0, n_samples).clip(0, 15),
        'Sulfate': np.random.normal(330, 80, n_samples).clip(0, 600),
        'Conductivity': np.random.normal(425, 90, n_samples).clip(0, 750),
        'Organic_carbon': np.random.normal(14, 4, n_samples).clip(0, 30),
        'Trihalomethanes': np.random.normal(66, 16, n_samples).clip(0, 120),
        'Turbidity': np.random.normal(4.0, 1.5, n_samples).clip(0, 10),
    }
    
    df = pd.DataFrame(data)
    
    # 引入缺失值（約 20-30%）
    for col in df.columns:
        missing_rate = np.random.uniform(0.15, 0.30)
        mask = np.random.random(n_samples) < missing_rate
        df.loc[mask, col] = np.nan
    
    # 生成目標變數（基於簡單規則 + 隨機性）
    # 簡化規則：多數特徵接近標準值 → 可飲用
    potability_score = (
        ((df['ph'] > 6.5) & (df['ph'] < 8.5)).astype(int) * 0.2 +
        (df['Solids'] < 30000).astype(int) * 0.2 +
        (df['Chloramines'] < 5).astype(int) * 0.15 +
        (df['Sulfate'] < 400).astype(int) * 0.15 +
        (df['Conductivity'] < 500).astype(int) * 0.15 +
        (df['Trihalomethanes'] < 80).astype(int) * 0.15
    )
    
    # 加入隨機性（模擬真實複雜性）
    potability_score += np.random.normal(0, 0.2, n_samples)
    df['Potability'] = (potability_score > 0.5).astype(int)
    
    # 調整類別平衡（約 39% 可飲用，61% 不可飲用，接近真實數據）
    target_positive_rate = 0.39
    current_positive_rate = df['Potability'].mean()
    
    if current_positive_rate > target_positive_rate:
        # 隨機將一些 1 改為 0
        n_to_flip = int((current_positive_rate - target_positive_rate) * n_samples)
        positive_indices = df[df['Potability'] == 1].index
        flip_indices = np.random.choice(positive_indices, n_to_flip, replace=False)
        df.loc[flip_indices, 'Potability'] = 0
    elif current_positive_rate < target_positive_rate:
        # 隨機將一些 0 改為 1
        n_to_flip = int((target_positive_rate - current_positive_rate) * n_samples)
        negative_indices = df[df['Potability'] == 0].index
        flip_indices = np.random.choice(negative_indices, n_to_flip, replace=False)
        df.loc[flip_indices, 'Potability'] = 1
    
    return df

def main():
    """主函數"""
    output_file = Path(__file__).parent / 'water_potability.csv'
    
    if output_file.exists():
        print(f"⚠️ 檔案已存在: {output_file}")
        response = input("是否覆蓋? (y/n): ")
        if response.lower() != 'y':
            print("已取消")
            return
    
    print("生成示例數據集...")
    df = generate_sample_data(n_samples=3276)
    
    print(f"\n數據集資訊:")
    print(f"  樣本數: {len(df)}")
    print(f"  特徵數: {df.shape[1] - 1}")
    print(f"  可飲用比例: {df['Potability'].mean():.2%}")
    print(f"  缺失值:")
    for col in df.columns:
        missing_pct = df[col].isnull().mean()
        if missing_pct > 0:
            print(f"    {col}: {missing_pct:.1%}")
    
    # 儲存
    df.to_csv(output_file, index=False)
    print(f"\n✓ 示例數據已儲存至: {output_file}")
    print("\n⚠️ 注意：這是示例數據，僅供測試使用")
    print("請從 Kaggle 下載真實數據以進行正式分析:")
    print("https://www.kaggle.com/datasets/adityakadiwal/water-potability/")

if __name__ == "__main__":
    main()
