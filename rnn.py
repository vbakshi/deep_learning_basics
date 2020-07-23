import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.keras.layers import LSTM, Dense, Dropout 
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam

EPOCHS = 100
BATCH_SIZE = 32
DROPOUT_RATE = 0.2

WINDOW_SIZE_IN_DAYS = 60

# importing the data
train_df = pd.read_csv("./data/rnn/Google_Stock_Price_Train.csv")
X = train_df.iloc[:,1:2].values

# Scaling data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

#Creating data with n timestamps. Here n=60

X_train = []
y_train = []

for i in range(60, X_scaled.shape[0]):
    X_train.append(X_scaled[i-60:i,0])
    y_train.append(X_scaled[i,0])


X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], -1, 1))

# Building the RNN
regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(rate=0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(rate=0.2))

regressor.add(Dense(units=1))

# Compiling and fitting the RNN

regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(X_train, y_train, epochs=100, batch_size=32)


# Predicting on the test data
test_df = pd.read_csv("./data/rnn/Google_Stock_Price_Test.csv")
real_stock_price = test_df.iloc[:,1:2].values

dataset_total = pd.concat((train_df['Open'], test_df['Open']), axis=0)

print("Shape of training dataset: {}".format(train_df.shape[0]))
print("Shape of test dataset: {}".format(test_df.shape[0]))
print("Shape of concatenated dataset: {}".format(dataset_total.shape[0]))

inputs_test = dataset_total[dataset_total.shape[0] - test_df.shape[0] - 60:].values
inputs_test = inputs_test.reshape(-1,1)
input_test = scaler.transform(inputs_test)

X_test = []
for i in range(60,80):
    X_test.append(input_test[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)


# Plotting results
plt.plot(real_stock_price, color='red', label='Actual Stock Price')
plt.plot(predicted_stock_price, color='blue', linestyle='-.', label='Predicted Stock Price')
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

#Saving data

plt.savefig("./output/rnn_prediction.png")




















