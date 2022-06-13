import keras
import tensorflow
import tensorflow
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
from keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dataTools import get_csv_data, get_csv_data_FULL, get_tech_ind, get_csv_date_future, assign_dates

history_points = 50  # number of days to use when making predictions


# create model that predicts the next 10 days
def create_extended_model(file_path):
    np.random.seed(4)
    tensorflow.random.set_seed(4)

    # create dataset
    ohlcv_histories, next_day_open_values, unscaled_y, y_normalizer, history_dates, next_day_dates, base_values, base_dates = get_csv_date_future(
        'csv/GE_daily.csv')

    test_split = 0.9
    n = int(ohlcv_histories.shape[0] * test_split)

    ohlcv_train = ohlcv_histories[:n]
    next_day_train = next_day_open_values[:n]
    history_dates_train = history_dates[:n]
    next_day_dates_train = next_day_dates[:n]

    ohlcv_test = ohlcv_histories[n:]
    next_day_test = next_day_open_values[n:]
    history_dates_test = history_dates[n:]
    next_day_dates_test = next_day_dates[n:]

    unscaled_y_test = unscaled_y[n:]

    print(ohlcv_train.shape)
    print(ohlcv_test.shape)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(history_points, 5)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=10))
    model.add(Activation('sigmoid'))
    model.compile(optimizer='adam', loss='mean_squared_error')

    results = model.fit(ohlcv_train, next_day_train.reshape(next_day_train.shape[0], 10), epochs=50, batch_size=50)

    # evaluation
    next_day_test_predicted = model.predict(ohlcv_test)
    next_day_test_predicted = y_normalizer.inverse_transform(next_day_test_predicted)
    next_day_predicted = model.predict(ohlcv_histories)
    next_day_predicted = y_normalizer.inverse_transform(next_day_predicted)

    assert unscaled_y_test.shape == next_day_test_predicted.shape
    real_mse = np.mean(np.square(unscaled_y_test - next_day_test_predicted))
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
    print(real_mse)
    print(scaled_mse)

    plt.gcf().set_size_inches(22, 15, forward=True)

    futures, future_dates = assign_dates(next_day_dates_test[:], next_day_test_predicted[:])

    # real = plt.plot(base_dates[:], base_values[:, 0], label='real')
    # pred = plt.plot(next_day_dates_test[:], next_day_test_predicted[:, 0], label='predicted')
    real = plt.plot(next_day_dates_test[:], unscaled_y_test[:, 0], label='real')
    pred = plt.plot(future_dates, futures, label='predicted')

    # real = plt.plot(unscaled_y[:], label='real')
    # pred = plt.plot(y_predicted[start:end], label='predicted')

    plt.legend(['Real', 'Predicted'])

    plt.show()

    model.save(f'basic_model.h5')


# create stock model with technical indicators incorporated
def create_tech_model(file_path):
    np.random.seed(4)
    tensorflow.random.set_seed(4)

    # create dataset
    ohlcv_histories, next_day_open_values, unscaled_y, y_normalizer, history_dates, next_day_dates = get_csv_data(
        'csv/GE_daily.csv', False)
    technical_indicators = get_tech_ind(ohlcv_histories)

    test_split = 0.9
    n = int(ohlcv_histories.shape[0] * test_split)

    ohlcv_train = ohlcv_histories[:n]
    next_day_train = next_day_open_values[:n]
    history_dates_train = history_dates[:n]
    next_day_dates_train = next_day_dates[:n]
    technical_indicators_train = technical_indicators[:n]

    ohlcv_test = ohlcv_histories[n:]
    next_day_test = next_day_open_values[n:]
    history_dates_test = history_dates[n:]
    next_day_dates_test = next_day_dates[n:]
    technical_indicators_test = technical_indicators[n:]

    unscaled_y_test = unscaled_y[n:]

    print(ohlcv_train.shape)
    print(ohlcv_test.shape)

    # define two sets of inputs
    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    dense_input = Input(shape=(technical_indicators.shape[1],), name='tech_input')

    # the first branch operates on the first input
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    lstm_branch = Model(inputs=lstm_input, outputs=x)

    # the second branch opreates on the second input
    y = Dense(20, name='tech_dense_0')(dense_input)
    y = Dense(20, name='tech_dense_1')(dense_input)
    y = Activation("relu", name='tech_relu_0')(y)
    y = Dropout(0.2, name='tech_dropout_0')(y)
    technical_indicators_branch = Model(inputs=dense_input, outputs=y)

    # combine the output of the two branches
    combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')

    z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
    z = Dense(1, activation="linear", name='dense_out')(z)

    # our model will accept the inputs of the two branches and
    # then output a single value
    model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)
    adam = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse')
    model.fit(x=[ohlcv_train, technical_indicators_train], y=next_day_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)

    # evaluation
    next_day_test_predicted = model.predict([ohlcv_test, technical_indicators_test])
    next_day_test_predicted = y_normalizer.inverse_transform(next_day_test_predicted)
    next_day_predicted = model.predict([ohlcv_histories, technical_indicators])
    next_day_predicted = y_normalizer.inverse_transform(next_day_predicted)

    assert unscaled_y_test.shape == next_day_test_predicted.shape
    real_mse = np.mean(np.square(unscaled_y_test - next_day_test_predicted))
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
    print(scaled_mse)

    plt.gcf().set_size_inches(22, 15, forward=True)

    real = plt.plot(next_day_dates_test[:], unscaled_y_test[:], label='real')
    pred = plt.plot(next_day_dates_test[:], next_day_test_predicted[:], label='predicted')

    # real = plt.plot(unscaled_y[start:end], label='real')
    # pred = plt.plot(y_predicted[start:end], label='predicted')

    plt.legend(['Real', 'Predicted'])

    plt.show()

    model.save(f'tech_model.h5')


# create basic stock model
def create_basic_model(file_path):
    np.random.seed(4)
    tensorflow.random.set_seed(4)

    # create dataset
    ohlcv_histories, next_day_open_values, unscaled_y, y_normalizer, history_dates, next_day_dates = get_csv_data(
        'csv/GE_daily.csv', False)

    test_split = 0.9
    n = int(ohlcv_histories.shape[0] * test_split)

    ohlcv_train = ohlcv_histories[:n]
    next_day_train = next_day_open_values[:n]
    history_dates_train = history_dates[:n]
    next_day_dates_train = next_day_dates[:n]

    ohlcv_test = ohlcv_histories[n:]
    next_day_test = next_day_open_values[n:]
    history_dates_test = history_dates[n:]
    next_day_dates_test = next_day_dates[n:]

    unscaled_y_test = unscaled_y[n:]

    print(ohlcv_train.shape)
    print(ohlcv_test.shape)

    # create the model
    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    x = Dense(64, name='dense_0')(x)
    x = Dense(64, name='dense_1')(x)
    x = Activation('sigmoid', name='sigmoid_0')(x)
    x = Dense(1, name='dense_2')(x)
    output = Activation('linear', name='linear_output')(x)

    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse')
    model.fit(x=ohlcv_train, y=next_day_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)

    # evaluation
    next_day_test_predicted = model.predict(ohlcv_test)
    next_day_test_predicted = y_normalizer.inverse_transform(next_day_test_predicted)
    next_day_predicted = model.predict(ohlcv_histories)
    next_day_predicted = y_normalizer.inverse_transform(next_day_predicted)

    assert unscaled_y_test.shape == next_day_test_predicted.shape
    real_mse = np.mean(np.square(unscaled_y_test - next_day_test_predicted))
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
    print(scaled_mse)

    plt.gcf().set_size_inches(22, 15, forward=True)

    real = plt.plot(next_day_dates_test[:], unscaled_y_test[:], label='real')
    pred = plt.plot(next_day_dates_test[:], next_day_test_predicted[:], label='predicted')

    # real = plt.plot(unscaled_y[start:end], label='real')
    # pred = plt.plot(y_predicted[start:end], label='predicted')

    plt.legend(['Real', 'Predicted'])

    plt.show()

    model.save(f'basic_model.h5')


def create_basic_model_FULL(file_path):
    np.random.seed(4)
    tensorflow.random.set_seed(4)

    # create dataset
    ohlcv_histories, next_day_normalized, next_day_values, y_normalizer, dates = get_csv_data_FULL(file_path, False)

    # train model based on historical data
    # find the index to split data into test/training sets
    test_split = 0.9
    n = int(ohlcv_histories.shape[0] * test_split)

    # create training set
    ohlcv_train = ohlcv_histories[:n]
    next_day_train = next_day_values[:n]
    dates_train = dates[:n]

    # create testing set
    ohlcv_test = ohlcv_histories[n:]
    next_day_test = next_day_values[n:]
    dates_test = dates[n:]

    unscaled_next_day_test = next_day_values[n:]

    print(ohlcv_train.shape)
    print(ohlcv_test.shape)

    # create the model
    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    x = Dense(64, name='dense_0')(x)
    x = Activation('sigmoid', name='sigmoid_0')(x)
    x = Dense(5, name='dense_1')(x)
    output = Activation('linear', name='lstm_output')(x)

    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse')
    model.fit(x=ohlcv_train, y=next_day_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)

    # evaluation
    next_day_test_predicted = model.predict(ohlcv_test)
    next_day_test_predicted = y_normalizer.inverse_transform(next_day_test_predicted)
    next_day_predicted = model.predict(ohlcv_histories)
    next_day_predicted = y_normalizer.inverse_transform(next_day_predicted)

    assert unscaled_next_day_test.shape == next_day_test_predicted.shape

    plt.gcf().set_size_inches(22, 15, forward=True)
    plot1 = unscaled_next_day_test[:, 0]
    plot2 = next_day_test_predicted[:, 0]

    real = plt.plot(unscaled_next_day_test[:, 0], label='real')
    pred = plt.plot(next_day_test_predicted[:, 0], label='predicted')

    plt.legend(['Real', 'Predicted'])

    plt.show()

    model.save(f'basic_model.h5')


if __name__ == "__main__":
    create_extended_model('csv/MSFT_daily.csv')
