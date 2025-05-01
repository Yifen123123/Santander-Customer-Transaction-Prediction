## 📊 數據來源

本專案所使用的數據來自 [Kaggle - Santander Customer Transaction Prediction]

([https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data?select=sample_submission.csv))

資料檔案未包含於本專案中，請至 Kaggle 官方下載。

## 🎯 學習目的

- 進行資料前處理、探索性資料分析（EDA）
- 應用資料標準化技巧
- 使用 XGBoost 進行二元分類模型訓練
- 評估模型效果（AUC、ROC、Confusion Matrix）
- 產出預測結果並與範例提交檔案比較
- 繪製模型特徵重要性與 ROC 曲線

## 📚 使用模型
### 📌 XGBoost（二元分類）
一種基於梯度提升決策樹（Gradient Boosting Decision Tree, GBDT）的機器學習演算法

優點：
- 能處理數值型資料與特徵重要性評估  
- 支援 early stopping，加速訓練過程  
- 對於非線性資料表現良好 


