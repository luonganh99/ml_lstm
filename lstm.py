import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential  # Deeplearing API
from keras.layers import LSTM, Dropout, Dense


def train(comp, numOfTrain):
    scaler = MinMaxScaler(feature_range=(0, 1))

    df = pd.read_csv("stock_data.csv")
    df = df.loc[df["Stock"] == comp]

    df["Date"] = pd.to_datetime(df.Date, format="%m/%d/%Y")
    df.index = df["Date"]  # Set index dataframe = date

    data = df.sort_index(ascending=True, axis=0)  # Sort index (date) ascending
    new_dataset = pd.DataFrame(
        index=range(0, len(df)), columns=["Date", "Close", "Rate"]
    )  # Create new dataframe with Date and Close columns

    # Copy from data to new_dataset (Date, Close)
    for i in range(0, len(data)):
        new_dataset["Date"][i] = data["Date"][i]
        new_dataset["Close"][i] = data["Close"][i]
        new_dataset["Rate"][i] = data["Rate"][i]

    new_dataset.index = new_dataset.Date  # Set index = Date
    new_dataset.drop(
        "Date", axis=1, inplace=True
    )  # Delete comlumn Date, axis = 1 means column, inplace = true means do operation inplace and return None
    roc_dataset = new_dataset[:]
    new_dataset.drop("Rate", axis=1, inplace=True)

    final_dataset = new_dataset.values  # [[0],[1]]

    train_data = final_dataset[0:numOfTrain, :]
    valid_data = final_dataset[numOfTrain:, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(final_dataset)

    x_train_data, y_train_data = [], []

    for i in range(60, len(train_data)):
        x_train_data.append(scaled_data[i - 60 : i, 0])
        y_train_data.append(scaled_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
    x_train_data = np.reshape(
        x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1)
    )

    lstm_model = Sequential()
    lstm_model.add(
        LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1))
    )
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))

    lstm_model.compile(loss="mean_squared_error", optimizer="adam")
    lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)
    # epochs: So lan quet

    inputs_data = new_dataset[len(new_dataset) - len(valid_data) - 60 :].values
    inputs_data = inputs_data.reshape(-1, 1)
    inputs_data = scaler.transform(inputs_data)

    X_test = []
    for i in range(60, inputs_data.shape[0]):
        X_test.append(inputs_data[i - 60 : i, 0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    closing_price = lstm_model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)

    train_data = new_dataset[:numOfTrain]
    valid_data = new_dataset[numOfTrain:]
    valid_data["Predictions"] = closing_price

    # Price Rate of Change
    train_roc_dataset = roc_dataset[:numOfTrain]
    valid_roc_dataset = roc_dataset[numOfTrain - 1 : len(roc_dataset) - 1]
    rateOfChanges = []
    for i in range(0, len(closing_price)):
        rateOfChanges.append(
            (closing_price[i][0] - closing_price[i - 1][0])
            / closing_price[i - 1][0]
            * 100
        )

    valid_roc_dataset["Predictions"] = rateOfChanges
    valid_roc_dataset.drop(index=valid_roc_dataset.index[0], axis=0, inplace=True)

    return [train_data, valid_data, train_roc_dataset, valid_roc_dataset]
