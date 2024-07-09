import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib  # 导入joblib库

# 加载数据
input_data = pd.read_csv('diedai.csv')

y = input_data.iloc[:, 8]
x = input_data.iloc[:, [0,1,2,3,5,6,7]]

# 数据归一化
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
x = min_max_scaler.fit_transform(x)

# 数据划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=27)

# 训练模型
rtr = RandomForestRegressor(n_estimators=100)
rtr.fit(x_train, y_train)

# 预测
y_pred = rtr.predict(x_test)

# 评估
mae_score = mean_absolute_error(y_test, y_pred)
mse_score = mean_squared_error(y_test, y_pred)
r_2_score = r2_score(y_test, y_pred)
rmse = np.sqrt(mse_score)

print(mae_score, rmse, r_2_score)

# 绘图
plt.scatter(y_test, y_test, s=11)
plt.scatter(y_test, y_pred, s=11)
# plt.show()

# 保存模型
joblib.dump(rtr, 'random_forest_regressor_model2.joblib')