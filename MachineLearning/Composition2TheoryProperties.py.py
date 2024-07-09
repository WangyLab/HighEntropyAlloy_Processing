import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

input_data = pd.read_csv('dataset.csv')

x = input_data.iloc[:, :5]
y = input_data.iloc[:, 5:]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

scaler = MinMaxScaler()
y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.transform(y_test)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_train_scaled.shape[1])
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('test.h5', save_best_only=True, monitor='val_loss', mode='min')
model.fit(x_train, y_train_scaled, epochs=400, batch_size=32, validation_split=0.2, callbacks=[checkpoint])
best_model = tf.keras.models.load_model('test.h5')

y_pred_scaled = best_model.predict(x_test)
y_pred = scaler.inverse_transform(y_pred_scaled)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

num_of_tasks = y_test.shape[1]
for i in range(num_of_tasks):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test.iloc[:, i], y_pred[:, i], alpha=0.5)
    plt.plot([y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], 'k--', lw=4)
    plt.title(f'Task {i+1}: Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    # plt.show()

    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])

    print(f"Task {i+1}:")
    print(f"Mean Absolute Error: {mae}")
    print(f"MSE: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R^2 Score: {r2}")
    print("\n")
