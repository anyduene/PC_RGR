import os

import pandas as pd
import numpy as np
import sys

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.src.layers import InputLayer, LayerNormalization
import tensorflow as tf

# os.environ['PYTHONHASHSEED'] = str(42)
# tf.keras.utils.set_random_seed(42)
# tf.config.experimental.enable_op_determinism()
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATA_PATH = '/Users/illiamatsko/My/University/paralel/HyperparameterTuning/'
DATA_FILENAME = 'data.csv'
PARAMS_PATH = sys.argv[1]
# PARAMS_PATH = '/Users/illiamatsko/My/University/paralel/HyperparameterTuning/params/params.txt'


TRAIN_START_DATE = '2000-02-08'
TRAIN_END_DATE= '2024-09-20'
TEST_END_DATE= '2025-04-20'

MODEL_VERSION = 'v4.1'

def read_hyperparams():
    params = {}
    with open(PARAMS_PATH, "r") as file:
        possible_indicators = file.readline()
        indicators = file.readline()
        key, value = possible_indicators.strip().split("=")
        params['possible_indicators'] = []
        for i in value.split(' '):
            params['possible_indicators'].append(i)

        key, value = indicators.strip().split("=")
        params['indicators'] = []
        for i in value.split(' '):
            params['indicators'].append(i)

        for line in file:
            key, value = line.strip().split("=")
            if key == 'dropout':
                value = float(value)
            else:
                value = int(value)

            params[key] = value

    return params

def get_data(possible_indicators, indicators, file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

    for col in possible_indicators:
        if col not in indicators:
            df.drop(columns=[col], inplace=True)

    start_date = pd.to_datetime(TRAIN_START_DATE)
    df = df[df.index >= start_date]

    return df

def get_test_and_train_data(data, window_size):
    train_data = data[data.index < TRAIN_END_DATE]
    test_data = data[(data.index >= pd.to_datetime(TRAIN_END_DATE) - pd.Timedelta(days=window_size)) & (data.index < TEST_END_DATE)]

    def create_xy(dataset):
        x, y = [], []
        values = dataset.values
        for i in range(window_size, len(dataset)):
            x_window = values[i - window_size:i, :]
            x.append(x_window)
            y.append(values[i, 0:2])

        return np.array(x), np.array(y)

    x_train, y_train = create_xy(train_data)
    x_test, y_test = create_xy(test_data)

    return x_train, y_train, x_test, y_test

def build_model(params, x_train, y_train):
    layers = params['layers']
    units = params['units']
    dropout = params['dropout']
    epochs = params['epochs']
    batch_size = params['batch_size']

    model = Sequential()
    model.add(InputLayer(shape=(x_train.shape[1], x_train.shape[2])))

    for _ in range(layers - 1):
        model.add(LSTM(units, return_sequences=True, recurrent_dropout=dropout))
        model.add(LayerNormalization())
        model.add(Dropout(dropout))

    model.add(LSTM(units, return_sequences=False, recurrent_dropout=dropout))
    model.add(Dropout(dropout))

    model.add(Dense(2))

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    return model

def main():
    params = read_hyperparams()
    df = get_data(params['possible_indicators'], params['indicators'], DATA_PATH + DATA_FILENAME)
    ticker = DATA_FILENAME.split('_')[0]

    x_train, y_train, x_test, y_test = get_test_and_train_data(df, window_size=params['window_size'])

    x_scaler = MinMaxScaler()
    x_train = x_scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    x_test = x_scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
    # joblib.dump(x_scaler, MODEL_VERSION + '/scalers/' + ticker + '_scaler_x.save')

    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)
    # joblib.dump(y_scaler, MODEL_VERSION + '/scalers/' + ticker + '_scaler_y.save')

    model = build_model(params, x_train, y_train)
    model_name = ticker + '_' + TRAIN_START_DATE[:4] + '_' + TRAIN_END_DATE[:4] + '_model.keras'
    # model.save(MODEL_VERSION + '/' + model_name)

    predicted = model.predict(x_test, verbose=0)
    error = np.mean(np.abs(predicted[0] - y_test[0]) / y_test[0]) * 100
    print(error, end='')

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        exit(1)