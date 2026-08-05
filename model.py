import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_model(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=3, batch_size=32, verbose=0)

    return model, scaler, scaled_data


def predict_future(model, scaler, scaled_data, days=7):
    last_60_days = scaled_data[-60:]
    future_predictions = []

    current_input = last_60_days

    for _ in range(days):
        current_input_reshaped = current_input.reshape(1, 60, 1)
        next_pred = model.predict(current_input_reshaped, verbose=0)[0]
        future_predictions.append(next_pred)
        current_input = np.append(current_input[1:], [next_pred], axis=0)

    future_predictions = scaler.inverse_transform(future_predictions)
    return future_predictions