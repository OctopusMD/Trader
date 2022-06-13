from alpha_vantage.timeseries import TimeSeries
import pandas
from sklearn import preprocessing
import numpy
from datetime import datetime, timedelta
import yfinance as yf
from bs4 import BeautifulSoup
import requests

history_points = 50     # number of days to use when making predictions
future_points = 10      # number of days to predict
api_key = "S6Y6BDREVRU0RUO0"  # todo: dynamically read credentials


# get tickers from the s&p 500
def get_SP500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    assets = soup.find_all('a', attrs={"class": "external"})

    symbols = []
    for asset in assets:
        if 'https://www.nyse.com/quote/' in asset.attrs['href'] or 'http://www.nasdaq.com/symbol/' in asset.attrs['href']:
            symbols.append(asset.text)

    return symbols


# get current data on stock
def get_yahoo_quote(stock_symbol):
    stock = yf.Ticker(stock_symbol)

    return stock.info


# get intraday data on stock history for symbol
def get_yahoo_intraday(stock_symbol, time_interval="1m"):
    data = yf.download(tickers=stock_symbol, period="5d", auto_adjust=True, interval=time_interval, threads=True)
    return data


# get time series data on stock history for symbol
def get_yahoo_csv(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    data = stock.history(period="max", interval="1d")

    # save data as csv
    try:
        data.to_csv(f'./csv/{stock_symbol}_daily.csv')
    except FileNotFoundError:
        data.to_csv(f'../csv/{stock_symbol}_daily.csv')

    return data


# get intraday data for stock symbol from yahoo finance
def get_yahoo_intraday_csv(stock_symbol):
    data = yf.download(tickers=stock_symbol, period="5d", auto_adjust=True, interval="1m", threads=True)

    # save data as csv
    data.to_csv(f'./csv/{stock_symbol}_intraday.csv')
    return data


# get top gainers
def get_yahoo_gainers():
    url = "https://finance.yahoo.com/gainers"
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    assets = soup.find_all('a', attrs={"class": "Fw(600) C($linkColor)"})

    symbols = []
    for asset in assets:
        symbols.append(asset.text)

    return symbols


# get time series data on stock history for symbol
def get_alpha_vantage_csv(stock_symbol):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=stock_symbol, outputsize='full')

    # save data as csv
    data.to_csv(f'./csv/{stock_symbol}_daily.csv')


# get current stock price for symbol
def get_alpha_vantage_quote(stock_symbol):
    ts = TimeSeries(key=api_key)
    quote = ts.get_quote_endpoint(symbol=stock_symbol)

    return quote


# get data from csv formatted to predict 10 days in the future
def get_csv_date_future(file_path):
    data = pandas.read_csv(file_path)
    data = data[-8000:]

    # split dates from data
    dates = data.Date
    data = data.drop('Date', axis=1)

    # drop unused stats
    data = data.drop('Dividends', axis=1)
    data = data.drop('Stock Splits', axis=1)

    # convert dates from strings to date objects
    new_dates = []
    for day in dates:
        temp_day = datetime.strptime(day, '%Y-%m-%d')
        new_dates.append(temp_day)

    dates = new_dates

    data = data.values

    # normalize the data (translates data to value between 1 and 0
    data_normalizer = preprocessing.MinMaxScaler()
    data_normalized = data_normalizer.fit_transform(data)

    # using the last {history_points} open close high low volume data points, predict the next entry
    ohlcv_histories_normalized = numpy.array(
        [data_normalized[i - history_points:i].copy() for i in range(history_points, len(data_normalized) - future_points)])
    history_dates = numpy.array([dates[i] for i in range(history_points, len(dates) - future_points)])
    next_day_open_values_normalized = numpy.array(
        [data_normalized[i + 1:i + 1 + future_points, 0].copy() for i in range(history_points, len(data_normalized) - future_points)])
    next_day_open_values_normalized = numpy.expand_dims(next_day_open_values_normalized, -1)

    next_day_open_values = numpy.array(
        [data[i + 1:i + 1 + future_points, 0].copy() for i in range(history_points, len(data) - future_points)])
    # next_day_open_values = numpy.expand_dims(next_day_open_values, -1)
    next_day_dates = numpy.array([dates[i + 1] for i in range(history_points, len(data) - future_points)])

    y_normalizer = preprocessing.MinMaxScaler()
    y_normalizer.fit(next_day_open_values)

    # check to make sure x and y normalized data match
    assert ohlcv_histories_normalized.shape[0] == next_day_open_values_normalized.shape[0]
    return ohlcv_histories_normalized, next_day_open_values_normalized, next_day_open_values, y_normalizer, history_dates, next_day_dates, data, dates


# read data from csv file
def get_csv_data(file_path, young):
    data = pandas.read_csv(file_path)
    data = data[-8000:]
    # split dates from data
    dates = data.Date
    data = data.drop('Date', axis=1)
    data = data.drop('Dividends', axis=1)
    data = data.drop('Stock Splits', axis=1)

    # if company is younger than 20 years discount the first day as it's data is unreliable
    if young:
        data = data.drop(0, axis=0)
        dates = dates[1:]

    # convert dates from strings to date objects
    new_dates = []
    for day in dates:
        temp_day = datetime.strptime(day, '%Y-%m-%d')
        new_dates.append(temp_day)

    dates = new_dates

    data = data.values

    # normalize the data (translates data to value between 1 and 0
    data_normalizer = preprocessing.MinMaxScaler()
    data_normalized = data_normalizer.fit_transform(data)

    # using the last {history_points} open close high low volume data points, predict the next entry
    ohlcv_histories_normalized = numpy.array(
        [data_normalized[i - history_points:i].copy() for i in range(history_points, len(data_normalized) - 1)])
    history_dates = numpy.array([dates[i] for i in range(history_points, len(dates) - 1)])
    next_day_open_values_normalized = numpy.array(
        [data_normalized[i, 0].copy() for i in range(history_points + 1, len(data_normalized))])
    next_day_open_values_normalized = numpy.expand_dims(next_day_open_values_normalized, -1)

    next_day_open_values = numpy.array(
        [data[i, 0].copy() for i in range(history_points + 1, len(data))])
    next_day_open_values = numpy.expand_dims(next_day_open_values, -1)
    next_day_dates = numpy.array([dates[i] for i in range(history_points + 1, len(dates))])

    y_normalizer = preprocessing.MinMaxScaler()
    y_normalizer.fit(next_day_open_values)

    # check to make sure x and y normalized data match
    assert ohlcv_histories_normalized.shape[0] == next_day_open_values_normalized.shape[0]
    return ohlcv_histories_normalized, next_day_open_values_normalized, next_day_open_values, y_normalizer, history_dates, next_day_dates


# read data from csv file
def get_csv_data_old(file_path, young):
    data = pandas.read_csv(file_path)
    data = data.iloc[::-1]
    # split dates from data
    dates = data.date
    data = data.drop('date', axis=1)

    # if company is younger than 20 years discount the first day as it's data is unreliable
    if young:
        data = data.drop(0, axis=0)
        dates = dates[1:]

    # convert dates from strings to date objects
    new_dates = []
    for day in dates:
        temp_day = datetime.strptime(day, '%Y-%m-%d')
        new_dates.append(temp_day)

    dates = new_dates

    data = data.values

    # normalize the data (translates data to value between 1 and 0
    data_normalizer = preprocessing.MinMaxScaler()
    data_normalized = data_normalizer.fit_transform(data)

    # using the last {history_points} open close high low volume data points, predict the next entry
    ohlcv_histories_normalized = numpy.array(
        [data_normalized[i - history_points:i].copy() for i in range(history_points, len(data_normalized) - 1)])
    history_dates = numpy.array([dates[i] for i in range(history_points, len(dates) - 1)])
    next_day_open_values_normalized = numpy.array(
        [data_normalized[i, 0].copy() for i in range(history_points + 1, len(data_normalized))])
    next_day_open_values_normalized = numpy.expand_dims(next_day_open_values_normalized, -1)

    next_day_open_values = numpy.array(
        [data[i, 0].copy() for i in range(history_points + 1, len(data))])
    next_day_open_values = numpy.expand_dims(next_day_open_values, -1)
    next_day_dates = numpy.array([dates[i] for i in range(history_points + 1, len(dates))])

    y_normalizer = preprocessing.MinMaxScaler()
    y_normalizer.fit(next_day_open_values)

    # check to make sure x and y normalized data match
    assert ohlcv_histories_normalized.shape[0] == next_day_open_values_normalized.shape[0]
    return ohlcv_histories_normalized, next_day_open_values_normalized, next_day_open_values, y_normalizer, history_dates, next_day_dates


# read data from csv file
def get_csv_data_FULL(file_path, young):
    data = pandas.read_csv(file_path)
    data = data.iloc[::-1]
    # split dates from data
    dates = data.date
    data = data.drop('date', axis=1)

    # if company is younger than 20 years discount the first day as it's data is unreliable
    if young:
        data = data.drop(0, axis=0)
        dates = dates[1:]

    # convert dates from strings to date objects
    new_dates = []
    for day in dates:
        temp_day = datetime.strptime(day, '%Y-%m-%d')
        new_dates.append(temp_day)

    dates = new_dates

    data = data.values

    # normalize the data (translates data to value between 1 and 0
    data_normalizer = preprocessing.MinMaxScaler()
    data_normalized = data_normalizer.fit_transform(data)

    # using the last {history_points} open close high low volume data points, predict the next entry
    ohlcv_histories_normalized = numpy.array(
        [data_normalized[i - history_points:i].copy() for i in range(history_points, len(data_normalized) - 1)])
    next_day_normalized = numpy.array(
        [data_normalized[i].copy() for i in range(history_points + 1, len(data_normalized))])

    next_day_values = numpy.array(
        [data[i].copy() for i in range(history_points + 1, len(data_normalized))])

    y_normalizer = preprocessing.MinMaxScaler()
    y_normalizer.fit(next_day_values)

    assert ohlcv_histories_normalized.shape[0] == next_day_normalized.shape[0]
    return ohlcv_histories_normalized, next_day_normalized, next_day_values, y_normalizer, dates


# calculate EMA (Exponential Moving Average)
def calc_ema(values, time_period):
    # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
    sma = numpy.mean(values[:, 3])
    ema_values = [sma]
    k = 2 / (1 + time_period)
    for i in range(len(values) - time_period, len(values)):
        close = values[i][3]
        ema_values.append(close * k + ema_values[-1] * (1 - k))
    return ema_values[-1]


# calculate EMA (Exponential Moving Average)
def calc_ema_in_range(values, start, stop):
    # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
    sma = numpy.mean(values[start:stop])
    ema_values = [sma]
    k = 2 / (1 + stop - start)
    for i in range(start, stop):
        close = values[i]
        ema_values.append(close * k + ema_values[-1] * (1 - k))
    return ema_values[-1]


# calculate moving average convergence / divergence values
def calc_macd_values(values):
    macd = []
    signal = []
    size = values.shape[0]
    for i in range(26, values.shape[0]):
        macd.append(calc_ema_in_range(values, i - 26, i) - calc_ema_in_range(values, i - 12, i))
        signal.append(calc_ema_in_range(values, i - 9, i))

    return macd, signal


# calculate a moving average convergence / divergence
def calc_macd(values, position):
    assert position < (len(values) - 26)
    return calc_ema(values, position + 26) - calc_ema(values, position + 12)


# calculate when to buy and sell stock
def calc_buy_sell_test(values, dates, macd, signal):
    # list of date of sell/buy and sell/buy value
    buy_value = []
    buy_dates = []
    sell_value = []
    sell_dates = []
    own = False         # do we already own this stock

    for i in range(values.shape[0]):
        if own:
            # stock is on the decline
            if macd[i] < signal[i]:
                sell_value.append(values.Close.values[i])
                sell_dates.append(dates[i])
                own = False
        else:
            # stock is on the rise
            if macd[i] > signal[i]:
                buy_value.append(values.Close.values[i])
                buy_dates.append(dates[i])
                own = True

    calc_wealth(buy_value, sell_value)
    return buy_value, buy_dates, sell_value, sell_dates


# calculate when to buy and sell stock
def calc_short_test(values, dates, ma1, ma2):
    # list of date of sell/buy and sell/buy value
    buy_value = []
    buy_dates = []
    sell_value = []
    sell_dates = []
    own = False         # do we already own this stock

    for i in range(values.shape[0]):
        if own:
            # stock is on the decline
            if ma1[i] < ma2[i]:
                buy_value.append(values.Close.values[i])
                buy_dates.append(dates[i])
                own = False
        else:
            # stock is on the rise
            if ma1[i] > ma2[i]:
                sell_value.append(values.Close.values[i])
                sell_dates.append(dates[i])
                own = True

    calc_short_wealth(buy_value, sell_value)
    return buy_value, buy_dates, sell_value, sell_dates


# calculate average deviation of bollinger upper and lower bands
def bollinger_deviation(boll_up, boll_low):
    sum = 0
    for i in range(1, len(boll_up)):
        sum += boll_up[i] - boll_low[i]

    print(str(sum / len(boll_up)))
    return sum / len(boll_up)


# calculate when to buy and sell based on bollinger bands
def bollinger_buy_sells(data, dates, boll, boll_up, boll_low):
    # list of date of sell/buy and sell/buy value
    buy_value = []
    buy_dates = []
    sell_value = []
    sell_dates = []
    own = True  # do we already own this stock
    average_dev = 0.1
    i = 0
    while i < len(data):
        if own:
            # stock is at peak
            # if len(buy_value) != 0 and data[i] < data[i - 1] - average_dev:
            #     sell_value.append(data[i])
            #     sell_dates.append(dates[i])
            #     own = False
            #     i += 59
            if data[i] > boll_up[i]:
                sell_value.append(data[i])
                sell_dates.append(dates[i])
                own = False
        else:
            # stock is at trouph
            if data[i] < boll_low[i]:
                buy_value.append(data[i])
                buy_dates.append(dates[i])
                own = True

        i += 1

    if own:
        print("You own one stock")

    calc_wealth(buy_value, sell_value)
    return buy_value, buy_dates, sell_value, sell_dates


# calculate when to buy and sell based on rsi
def rsi_scalp_buy_sells(data, dates, rsi, scalp):
    # list of date of sell/buy and sell/buy value
    buy_value = []
    buy_dates = []
    sell_value = []
    sell_dates = []
    own = False  # do we already own this stock
    waiting = False
    # scalp = numpy.std(data)/10
    i = 0
    while i < len(rsi):
        if waiting:
            if rsi[i] >= 50:
                buy_value.append(data[i])
                buy_dates.append(dates[i])
                price = data[i]
                own = True
                waiting = False
        elif own:
            if data[i] >= price + scalp:
                sell_value.append(data[i])
                sell_dates.append(dates[i])
                own = False
            elif data[i] < price - (scalp * 3):
                sell_value.append(data[i])
                sell_dates.append(dates[i])
                own = False
                waiting = True
        else:
            # stock is at trouph
            if rsi[i] <= 33:
                buy_value.append(data[i])
                buy_dates.append(dates[i])
                price = data[i]
                own = True

        i += 1

    if own:
        print("You own stocks worth " + str(data.values[-1]))

    calc_wealth(buy_value, sell_value)
    return buy_value, buy_dates, sell_value, sell_dates


# calculate when to buy and sell based on rsi
def rsi_buy_sells(data, dates, rsi):
    # list of date of sell/buy and sell/buy value
    buy_value = []
    buy_dates = []
    sell_value = []
    sell_dates = []
    own = False  # do we already own this stock
    average_dev = 0.1
    i = 0
    while i < len(rsi):
        if own:
            if rsi[i] >= 66:
                sell_value.append(data[i])
                sell_dates.append(dates[i])
                own = False
        else:
            # stock is at trouph
            if rsi[i] <= 33:
                buy_value.append(data[i])
                buy_dates.append(dates[i])
                price = data[i]
                own = True

        i += 1

    if own:
        print("You own one stock")

    calc_wealth(buy_value, sell_value)
    return buy_value, buy_dates, sell_value, sell_dates


def calc_wealth(buys, sells):
    money = 100.00
    own = False
    num_owned = 0
    extra = 100.00

    for i in range(len(sells)):
        num_owned = money // buys[i]
        # update local data
        extra = money - (num_owned * buys[i])

        # update local data
        money = extra + (sells[i]*num_owned)
        extra = money
        num_owned = 0

    if len(buys) > len(sells):
        num_owned = money // buys[-1]
        print("You own " + str(money // buys[-1]) + "stock")
        money = money - (num_owned * buys[-1])

    print("Money: " + str(money))


def calc_short_wealth(buys, sells):
    money = 100.00
    own = False
    num_owned = 0
    extra = 100.00

    for i in range(len(sells)):
        num_owned = money // buys[i]
        # update local data
        extra = money + (num_owned * buys[i])

        # update local data
        money = extra - (sells[i]*num_owned)
        extra = money
        num_owned = 0

    if len(buys) > len(sells):
        num_owned = money // buys[-1]
        print("You own " + str(money // buys[-1]) + "stock")
        money = money - (num_owned * buys[-1])

    print("Money: " + str(money))


# get technical indicators for current data
def get_tech_ind(histories):
    technical_indicators = []
    for his in histories:
        # note since we are using his[3] we are taking the SMA of the closing price
        sma = numpy.mean(his[:, 3])
        macd = calc_ema(his, 26) - calc_ema(his, 12)
        technical_indicators.append(numpy.array([sma]))

    technical_indicators = numpy.array(technical_indicators)

    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalized = tech_ind_scaler.fit_transform(technical_indicators)
    assert histories.shape[0] == technical_indicators_normalized.shape[0]
    return technical_indicators_normalized


# add dates to values predicted
def assign_dates(dates, values):
    new_values = []
    new_dates = []
    counter = 0
    value_spot = 0  # spot in values array currently being used

    # loop until you are have the correct number of dates
    while value_spot < future_points:
        temp_day = dates[-1] + timedelta(days=counter)

        # stock market is closed on saturday and sunday
        if temp_day.weekday() == 5 or temp_day.weekday() == 6:
            counter += 1
        else:
            new_values.append(values[-1, value_spot])
            new_dates.append(dates[-1] + timedelta(days=counter))
            counter += 1
            value_spot += 1

    return new_values, new_dates


# calculate the RSI
def get_RSI(prices, period):
    delta = prices['Close'].diff()
    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the EWMA
    roll_up = up.ewm(span=period).mean()
    roll_down = down.abs().ewm(span=period).mean()

    rs = roll_up / roll_down
    rsi = 100 - (100/(1 + rs))
    return rsi.values
