from alpha_vantage.timeseries import TimeSeries
import pandas
from sklearn import preprocessing
import numpy
from datetime import datetime
from pprint import pprint

history_points = 50  # number of days to use when making predictions


# get timeseries data on stock hisotry for symbol
def save_dataset(stock_symbol):
    # todo: dynamically read credentials
    api_key = "S6Y6BDREVRU0RUO0"

    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=stock_symbol, outputsize='full')

    # save data as csv
    data.to_csv(f'./{stock_symbol}_daily.csv')


# clean up csv data
def csv_to_dataset(csv_path, young):
    data = pandas.read_csv(csv_path)
    data = data.iloc[::-1]
    # split dates from data
    dates = data.date
    data = data.drop('date', axis=1)
    # if company is younger than 20 years discount the first day as it's data is unreliable
    if young:
        data = data.drop(0, axis=0)
        dates = dates[1:]

    # convert dates from strings to date objects
    newDates = []
    for day in dates:
        tempDay = datetime.strptime(day, '%Y-%m-%d')
        newDates.append(tempDay)

    dates = newDates

    data = data.values

    # normalize the data (translates data to value between 1 and 0
    data_normalizer = preprocessing.MinMaxScaler()
    data_normalized = data_normalizer.fit_transform(data)

    # using the last {history_points} open close high low volume data points, predict the next open value
    ohlcv_histories_normalized = numpy.array(
        [data_normalized[i:i + history_points].copy() for i in range(len(data_normalized) - history_points)])
    next_day_open_values_normalized = numpy.array(
        [data_normalized[:, 0][i + history_points].copy() for i in range(len(data_normalized) - history_points)])
    next_day_open_values_normalized = numpy.expand_dims(next_day_open_values_normalized, -1)

    next_day_open_values = numpy.array(
        [data[:, 0][i + history_points].copy() for i in range(len(data) - history_points)])
    next_day_open_values = numpy.expand_dims(next_day_open_values, -1)

    y_normalizer = preprocessing.MinMaxScaler()
    y_normalizer.fit(next_day_open_values)

    technical_indicators = []
    for his in ohlcv_histories_normalized:
        # note since we are using his[3] we are taking the SMA of the closing price
        sma = numpy.mean(his[:, 3])
        macd = calc_ema(his, 12) - calc_ema(his, 26)
        technical_indicators.append(numpy.array([sma]))
        # technical_indicators.append(np.array([sma,macd,]))

    technical_indicators = numpy.array(technical_indicators)

    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalized = tech_ind_scaler.fit_transform(technical_indicators)

    assert ohlcv_histories_normalized.shape[0] == next_day_open_values_normalized.shape[0] == technical_indicators_normalized.shape[0]
    return ohlcv_histories_normalized, technical_indicators_normalized, next_day_open_values_normalized, next_day_open_values, y_normalizer, dates


def calc_ema(values, time_period):
    # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
    sma = numpy.mean(values[:, 3])
    ema_values = [sma]
    k = 2 / (1 + time_period)
    for i in range(len(values) - time_period, len(values)):
        close = values[i][3]
        ema_values.append(close * k + ema_values[-1] * (1 - k))
    return ema_values[-1]


if __name__ == "__main__":
    ohlcv_histories, next_day_open_values, unscaled_y, y_normalizer = csv_to_dataset('../csv/MSFT_daily.csv', False)

    test_split = 0.9  # the percent of data to be used for testing
    n = int(ohlcv_histories.shape[0] * test_split)

    # splitting the dataset up into train and test sets
    ohlcv_train = ohlcv_histories[:n]
    y_train = next_day_open_values[:n]

    ohlcv_test = ohlcv_histories[n:]
    y_test = next_day_open_values[n:]

    unscaled_y_test = unscaled_y[n:]
