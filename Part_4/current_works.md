# 逢甲大學 化學工程學系AI課程
## 課程名稱: AI在化工上之應用
**課程代碼: CHE-AI-114**
**課程教師: 莊曜禎 助理教授**
**授課時間: 114學年度第2學期 每周四上午09:10-12:00**
**授課地點: 學思樓 705**

### 任務描述:
進行課程內容的重構整理

### 課程大綱:
專為化學工程學系的學生,開設有關"AI在化工上應用"的課程, 共分為五大部分:
- Part 0: Google Colab環境教學
- Part 1：Python 基礎 + EDA (numpy, pandas, matplotlib, seaborn, statsmodels)
- Part 2：非監督式學習(sklearn模組)
- Part 3：監督式學習(sklearn模組)
- Part 4：深度學習(TensorFlow/Keras模組)
- Part 5：強化式學習

### 課程內容大綱
### **Part 0(尚未開始)**
### **Part 1(尚未開始)**
### **Part 2(尚未開始)**
### **Part 3(尚未開始)**
### **Part 4: Deep Learning (已完成)**
#### **Unit15 (已完成)**
Unit15
 - Unit15_DNN_MLP_Overview_restructure.md (教學講義):
    - DNN(MLP)模型介紹, 詳細背景理論, 數學公式說明, DNN(MLP)模型適合應用場景, DNN(MLP)模型化工領域應用案例.
    - 詳細介紹如何使用Tensorflow/Keras模組建立DNN(MLP)模型, DNN(MLP)模型所需用到的Layer有哪一些, 如何import Layer, 如何堆疊Layer建立模型, 各種Layer的功能差異, 如何設定各種Layer層的參數, 說明各種activation function的功能差異.
    - 如何編譯模型 model.compile(), 如何設定optimizer, loss function, metrics, model.summary()觀察模型結構, 各種參數功能與詳細設定指令說明
    - 如何訓練模型 model.fit(), 如何設定epochs, batch_size, validation_split, validation_data, callbacks,  各種參數功能與詳細設定指令說明.
    - history=model.fit()紀錄訓練過程, 訓練完成後如何可視化history物件中的各種指標. 
    - 進階使用Tensorboard紀錄與觀察訓練過程教學
    - 如何評估模型, model.evaluate()
    - 如何預測模型, model.predict()
    - 如何保存模型, model.save()
    - 如何載入模型, model.load()
    - 最後簡單說明與與sklearn模組中MLPRegressor, MLPClassifier的差異, 
 - Unit15_DNN_MLP_Overview.ipynb (程式演練):
    - 完整流程的程式碼演練範例, 包含資料前處理(設計使用模擬數據), 模型建立, 模型訓練, 模型評估, 模型預測, 模型保存, 模型載入等流程. 配合講義內容, 讓學生可學會如何建立與訓練DNN(MLP)模型.
 - Unit15_DNN_MLP_Homework.ipynb (作業):
    - 學生課堂練習題, 包含資料前處理(設計使用模擬數據), 模型建立, 模型訓練, 模型評估, 模型預測, 模型保存, 模型載入等流程.
 - DNN(MLP)模型在各種工業案例上的應用案例
   - Unit15_Example_FuelGasEmission (.md講義檔, .ipynb程式演練), 燃料氣體排放預測為主題
   - Unit15_Example_distillation (.md講義檔, .ipynb程式演練), 蒸餾塔操作控制為主題
   - Unit15_Example_RedWine (.md講義檔, .ipynb程式演練), 紅酒品質預測為主題
   - Unit15_Example_Mining (.md講義檔, .ipynb程式演練), 礦業浮選過程矽石濃度預測為主題


#### **Unit16 (已完成)**
Unit16
 - Unit16_CNN_Overview_restructure.md (教學講義):
    - CNN詳細背景理論, CNN常見且被廣泛使用的模型有哪一些(詳細列出), CNN模型適合應用場景, CNN模型化工領域應用案例.
    - 詳細介紹如何使用Tensorflow/Keras模組建立CNN模型, CNN模型所需用到的Layer有哪一些, 如何import Layer, 如何堆疊Layer建立模型, 各種Layer的功能差異, 如何設定各種Layer層的參數, 說明會用到的activation function的功能差異.
    - 如何編譯模型 model.compile(), 如何設定optimizer, loss function, metrics, model.summary()觀察模型結構, 各種參數功能與詳細設定指令說明
    - 如何訓練模型 model.fit(), 如何設定epochs, batch_size, validation_split, validation_data, callbacks,  各種參數功能與詳細設定指令說明.
    - history=model.fit()紀錄訓練過程, 訓練完成後如何可視化history物件中的各種指標. 
    - 進階使用Tensorboard紀錄與觀察訓練過程教學
    - 如何評估模型, model.evaluate()
    - 如何預測模型, model.predict()
    - 如何保存模型, model.save()
    - 如何載入模型, model.load()
 - Unit16_CNN_Overview.ipynb (程式演練):
    - 完整流程的程式碼演練範例, 包含資料前處理(設計使用模擬數據), 模型建立, 模型訓練, 模型評估, 模型預測, 模型保存, 模型載入等流程. 配合講義內容, 讓學生可學會如何建立與訓練CNN模型.
 - Unit16_CNN_Homework.ipynb (作業):
    - 學生課堂練習題, 包含資料前處理(設計使用模擬數據), 模型建立, 模型訓練, 模型評估, 模型預測, 模型保存, 模型載入等流程.
 - CNN模型在各種工業案例上的應用案例
    - Unit16_CNN_Basics_Industrial_Inspection (.md講義檔, .ipynb程式演練), 鋼材表面缺陷辨識
    - Unit16_Transfer_Learning_Defect_Detection (.md講義檔, .ipynb程式演練), 鋼材表面缺陷辨識 +遷移學習
    - Unit16_Example_ChineseMNIST (.md講義檔, .ipynb程式演練), 中文手寫數字辨識
    - Unit16_Eample_Colorectal_Cancer (.md講義檔, .ipynb程式演練), 醫學影像分類

#### **Unit17 (已完成)**
Unit17
 - Unit17_RNN_Overview.md (教學講義):
    - RNN詳細背景理論(LSTM和GRU詳細介紹), RNN常見且被廣泛使用的模型架構, RNN模型適合應用場景, RNN模型化工領域應用案例.
    - 詳細介紹如何使用Tensorflow/Keras模組建立RNN模型, RNN模型所需用到的Layer有哪一些, 如何import Layer, 如何堆疊Layer建立模型, 各種Layer的功能差異, 如何設定各種Layer層的參數, 說明會用到的activation function的功能差異.
    - 如何編譯模型 model.compile(), 如何設定optimizer, loss function, metrics, model.summary()觀察模型結構, 各種參數功能與詳細設定指令說明
    - 如何訓練模型 model.fit(), 如何設定epochs, batch_size, validation_split, validation_data, callbacks,  各種參數功能與詳細設定指令說明.
    - history=model.fit()紀錄訓練過程, 訓練完成後如何可視化history物件中的各種指標. 
    - 進階使用Tensorboard紀錄與觀察訓練過程教學
    - 如何評估模型, model.evaluate()
    - 如何預測模型, model.predict()
    - 如何保存模型, model.save()
    - 如何載入模型, model.load()
 - Unit17_RNN_Overview.ipynb (程式演練):
    - 完整流程的程式碼演練範例, 包含資料前處理(設計使用模擬數據), 模型建立, 模型訓練, 模型評估, 模型預測, 模型保存, 模型載入等流程. 配合講義內容, 讓學生可學會如何建立與訓練CNN模型.
 - Unit17_RNN_Advanced.md (進階內容教學講義):
    - 包含Bidirectional RNN, Sequence-to-Sequence (Encoder-Decoder), Attention等詳細背景理論, 模型架構等內容
    - Unit17_RNN_Advanced_Seq2Seq_v3.ipynb (Sequence-to-Sequence模型應用)
    - Unit17_RNN_Advanced_BiRNN_v11.ipynb (Bidirectional模型應用)
    - Unit17_RNN_Advanced_Attention.ipynb (Attention應用)
    - Unit17_Example_Boiler (.md講義檔, .ipynb程式演練), 鍋爐溫度預測
    - Unit17_Example_debutanized_column (.md講義檔, .ipynb程式演練), 去丁烷塔操作預測
    - Unit17_Example_NASA_Turbofan (.md講義檔, .ipynb程式演練), NASA渦扇引擎剩餘壽命預測