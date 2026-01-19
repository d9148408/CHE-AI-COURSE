"""
Water Potability 數據集下載腳本

使用方法：
python download_data.py

或在 Notebook 中執行：
!python data/waterquality/download_data.py
"""

import requests
from pathlib import Path

def download_water_potability():
    """下載 Water Potability 數據集"""
    
    # 數據集備份連結（從多個來源嘗試）
    urls = [
        # OpenML 鏡像
        "https://www.openml.org/data/get_csv/22044501/water_potability.csv",
        # GitHub 鏡像
        "https://github.com/MainakRepositor/Datasets/raw/master/Water%20Quality/water_potability.csv",
    ]
    
    output_file = Path(__file__).parent / 'water_potability.csv'
    
    if output_file.exists():
        print(f"✓ 檔案已存在: {output_file}")
        return True
    
    print("開始下載 Water Potability 數據集...")
    
    for i, url in enumerate(urls, 1):
        try:
            print(f"\n嘗試來源 {i}/{len(urls)}: {url[:50]}...")
            response = requests.get(url, timeout=30, stream=True)
            
            if response.status_code == 200:
                # 檢查檔案大小（應該約 100-200 KB）
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) < 10000:
                    print(f"  ⚠️ 檔案過小 ({content_length} bytes)，可能不是正確的數據集")
                    continue
                
                # 儲存檔案
                with open(output_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                file_size = output_file.stat().st_size
                print(f"  ✓ 下載成功！檔案大小: {file_size:,} bytes")
                print(f"  ✓ 已儲存至: {output_file}")
                return True
            else:
                print(f"  ✗ HTTP {response.status_code}")
        
        except Exception as e:
            print(f"  ✗ 錯誤: {e}")
    
    print("\n❌ 所有下載來源都失敗")
    print("\n請手動下載：")
    print("1. 訪問: https://www.kaggle.com/datasets/adityakadiwal/water-potability/")
    print("2. 點擊 'Download' 按鈕下載 water_potability.csv")
    print(f"3. 將檔案放置於: {output_file}")
    return False

if __name__ == "__main__":
    download_water_potability()
