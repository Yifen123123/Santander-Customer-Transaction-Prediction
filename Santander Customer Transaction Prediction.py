# 匯入所需的庫
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# 1. 數據讀取
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 2. 數據探索
print(train_data.head())
print(train_data.info())
print(train_data.describe())

# 3. 特徵和標籤分離
X = train_data.drop(['ID_code', 'target'], axis=1)
y = train_data['target']
X_test = test_data.drop(['ID_code'], axis=1)

# 4. 數據標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 5. 訓練集和驗證集拆分
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. 準備 DMatrix 格式的數據
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
dtest = xgb.DMatrix(X_test_scaled)

# 7. 設置 XGBoost 參數
params = {
    'objective': 'binary:logistic',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'auc'
}

# 8. 設置評估數據集
evals = [(dtrain, 'train'), (dvalid, 'valid')]

# 9. 使用 xgboost.train 進行模型訓練，啟用 early stopping
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=True
)

# 10. 預測驗證集的概率
y_valid_pred = model.predict(dvalid)
auc_score = roc_auc_score(y_valid, y_valid_pred)
print(f'Validation AUC: {auc_score:.4f}')

# 11. 對測試集進行預測
y_test_pred = model.predict(dtest)

# 12. 生成提交結果
submission = pd.DataFrame({'ID_code': test_data['ID_code'], 'target': y_test_pred})
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

import matplotlib.pyplot as plt
import seaborn as sns

# 設置繪圖風格
sns.set(style="whitegrid")

# 繪製數據的直方圖（以 var_0 為例）
plt.figure(figsize=(14, 7))
sns.histplot(train_data['var_0'], bins=50, kde=True)
plt.title('Distribution of var_0')
plt.xlabel('var_0')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# 繪製特徵重要性
plt.figure(figsize=(10, 8))
xgb.plot_importance(model, importance_type='weight', max_num_features=10)
plt.title('Feature Importance')
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 生成預測結果
y_valid_pred_binary = (y_valid_pred > 0.5).astype(int)

# 計算混淆矩陣
cm = confusion_matrix(y_valid, y_valid_pred_binary)

# 繪製混淆矩陣
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import roc_curve, auc

# 計算 ROC 曲線數據
fpr, tpr, thresholds = roc_curve(y_valid, y_valid_pred)
roc_auc = auc(fpr, tpr)

# 繪製 ROC 曲線
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc), color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

from sklearn.metrics import mean_absolute_error

# 讀取檔案
submission = pd.read_csv('submission.csv')
sample = pd.read_csv('sample_submission.csv')

# 確認兩個檔案的排序一致
submission = submission.sort_values('ID_code').reset_index(drop=True)
sample = sample.sort_values('ID_code').reset_index(drop=True)

# 計算 MAE
mae = mean_absolute_error(sample['target'], submission['target'])
print(f'Mean Absolute Error (MAE)：{mae:.5f}')