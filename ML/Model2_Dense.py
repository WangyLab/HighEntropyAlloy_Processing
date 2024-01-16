import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout

# 加载数据
input_data = pd.read_csv('input3.csv')
# input_data = input_data.drop(input_data.index[113])
# input_data = input_data.drop(input_data.index[27])
# input_data = input_data.drop(input_data.index[26])
# 定义输入输出
x = input_data.iloc[:, [0,1,2,3,5,6,7]]
y = input_data.iloc[:, 8]

# 数据划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=97)

min_max_scaler = MinMaxScaler(feature_range=(0, 1))
x_train=min_max_scaler.fit_transform(x_train)
x_test = min_max_scaler.fit_transform(x_test)

# 构建模型（与之前相同）
model = Sequential([
    Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
from tensorflow.keras.optimizers import Adam, SGD
learning_rate = 0.003  # 你可以设置为你希望的值
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('best_model2.h5', save_best_only=True, monitor='val_loss', mode='min')
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
model.fit(x_train, y_train, epochs=500, validation_split=0.2, callbacks=[checkpoint])
best_model = tf.keras.models.load_model('Dense22.h5')

# 预测
y_pred = best_model.predict(x_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_percentage_error
mae_score = mean_absolute_error(y_test, y_pred)
mse_score = mean_squared_error(y_test, y_pred)
r_2_score = r2_score(y_test, y_pred)
rmse = np.sqrt(mse_score)

print(mae_score,rmse,r_2_score)
plt.scatter(y_test, y_test, s=11)
plt.scatter(y_test, y_pred, s=11)
plt.show()

