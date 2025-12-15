# -*- coding: utf-8 -*-
"""
LSTM Time Series Forecasting
Modified for portability, robustness, and safe execution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# -------------------------------------------------
# Dataset loading (portable path)
# -------------------------------------------------
data = pd.read_csv(
    "international-airline-passengers.csv",
    skipfooter=5,
    engine="python"
)

print(data.head())

# -------------------------------------------------
# Data visualization
# -------------------------------------------------
dataset = data.iloc[:, 1].values
plt.plot(dataset)
plt.xlabel("Time")
plt.ylabel("Number of Passengers")
plt.title("International Airline Passengers")
plt.show()

# -------------------------------------------------
# Data preprocessing
# -------------------------------------------------
dataset = dataset.reshape(-1, 1).astype("float32")

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.75)
test_size = len(dataset) - train_size

train = dataset[:train_size, :]
test = dataset[train_size:, :]

print(f"Train size: {len(train)}, Test size: {len(test)}")

# -------------------------------------------------
# Create time-series samples
# -------------------------------------------------
time_stamp = 10

def create_dataset(data, time_stamp):
    dataX, dataY = [], []
    for i in range(len(data) - time_stamp - 1):
        dataX.append(data[i:(i + time_stamp), 0])
        dataY.append(data[i + time_stamp, 0])
    return np.array(dataX), np.array(dataY)

trainX, trainY = create_dataset(train, time_stamp)
testX, testY = create_dataset(test, time_stamp)

# Reshape for LSTM [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# -------------------------------------------------
# Model definition
# -------------------------------------------------
model = Sequential()
model.add(LSTM(10, input_shape=(1, time_stamp)))
model.add(Dense(1))

model.compile(
    loss="mean_squared_error",
    optimizer="adam"
)

# -------------------------------------------------
# Model training (with validation)
# -------------------------------------------------
model.fit(
    trainX,
    trainY,
    epochs=30,
    batch_size=1,
    validation_split=0.1,
    verbose=1
)

# -------------------------------------------------
# Model summary
# -------------------------------------------------
model.summary()

# -------------------------------------------------
# Model visualization (OPTIONAL – SAFE FIX)
# -------------------------------------------------
try:
    plot_model(
        model,
        to_file="model_plot.png",
        show_shapes=True,
        show_layer_names=True
    )
except Exception:
    print("Model plot skipped (Graphviz not available).")

# -------------------------------------------------
# Predictions
# -------------------------------------------------
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Inverse scaling
trainPredict = scaler.inverse_transform(trainPredict)
trainY_inv = scaler.inverse_transform(trainY.reshape(-1, 1))
testPredict = scaler.inverse_transform(testPredict)
testY_inv = scaler.inverse_transform(testY.reshape(-1, 1))

# -------------------------------------------------
# Evaluation
# -------------------------------------------------
trainScore = math.sqrt(mean_squared_error(trainY_inv[:, 0], trainPredict[:, 0]))
print(f"Train Score: {trainScore:.2f} RMSE")

testScore = math.sqrt(mean_squared_error(testY_inv[:, 0], testPredict[:, 0]))
print(f"Test Score: {testScore:.2f} RMSE")

# -------------------------------------------------
# Plot predictions
# -------------------------------------------------
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_stamp:len(trainPredict) + time_stamp, :] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (time_stamp * 2) + 1:len(dataset) - 1, :] = testPredict

plt.plot(scaler.inverse_transform(dataset), label="Real Values")
plt.plot(trainPredictPlot, label="Train Predictions")
plt.plot(testPredictPlot, label="Test Predictions")
plt.legend()
plt.show()
