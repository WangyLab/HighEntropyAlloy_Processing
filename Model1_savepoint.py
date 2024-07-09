import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 加载数据
input_data = pd.read_csv('D:\OneDrive - USTC\Code\Solid_Battery_with_YanHuang\ML\data\dataset.csv')

# 定义输入输出
x = input_data.iloc[:, :5]
y = input_data.iloc[:, 5:]

# 数据划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 初始化MinMaxScaler
scaler = MinMaxScaler()

# 归一化y
y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.transform(y_test)

# 构建模型（与之前相同）
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_train_scaled.shape[1])
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('D:\OneDrive - USTC\Code\Solid_Battery_with_YanHuang\ML\\1%\\test.h5', save_best_only=True, monitor='val_loss', mode='min')
model.fit(x_train, y_train_scaled, epochs=400, batch_size=32, validation_split=0.2, callbacks=[checkpoint])
best_model = tf.keras.models.load_model('D:\OneDrive - USTC\Code\Solid_Battery_with_YanHuang\ML\\1%\\test.h5')

# 预测
y_pred_scaled = best_model.predict(x_test)

# 将预测结果逆归一化
y_pred = scaler.inverse_transform(y_pred_scaled)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

# 假设 etr_y_pred 是模型预测的输出，y_test 是实际的输出
# y_test 和 etr_y_pred 都是 DataFrame 或者相似的数据结构，每一列代表一个任务的输出

num_of_tasks = y_test.shape[1]  # 任务的数量

# 为每个任务创建图表
for i in range(num_of_tasks):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test.iloc[:, i], y_pred[:, i], alpha=0.5)
    plt.plot([y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], 'k--', lw=4)
    plt.title(f'Task {i+1}: Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    # plt.show()

    # 计算误差指标
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
    rmse = np.sqrt(mse)  # 计算RMSE
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])

    print(f"Task {i+1}:")
    print(f"Mean Absolute Error: {mae}")
    print(f"MSE: {mse}")
    print(f"Root Mean Squared Error: {rmse}")  # 显示RMSE
    print(f"R^2 Score: {r2}")
    print("\n")