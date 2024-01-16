import pandas as pd
import numpy as np
from pandas import DataFrame
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
from sklearn.inspection import permutation_importance

# Load your data
input_data = pd.read_csv('input3.csv')

y = input_data.iloc[:, 8]
x = input_data.iloc[:, [0,1,2,3,5,6,7]]

# Normalize all data
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
x = min_max_scaler.fit_transform(x)

# Split the data into 80% train and 20% test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33)

xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate=0.35)
# 训练模型
xg_reg.fit(x_train, y_train)

# 对测试集进行预测
y_pred = xg_reg.predict(x_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_percentage_error
mae_score = mean_absolute_error(y_test, y_pred)
rmse_score = np.sqrt(mean_squared_error(y_test, y_pred))
r_2_score = r2_score(y_test, y_pred)

print(mae_score,rmse_score,r_2_score)
plt.scatter(y_test, y_test, s=11)
plt.scatter(y_test, y_pred, s=11)
plt.show()

# # Importance analysis
# for xx in xg_reg.feature_importances_:
#     print('%.3f'%xx)
# print('---------------------')
# result = permutation_importance(xg_reg, x, y, n_repeats=10) #random_state=42,n_jobs=-1
# for xx in result.importances_mean:
#     print('%.3f'%xx)
