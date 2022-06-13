import matplotlib.pyplot as plt
import pandas
import dataTools
import numpy
from datetime import datetime
from stockstats import StockDataFrame


# graph intraday data
def intraday_graph(file_path):
    stock = StockDataFrame.retype(pandas.read_csv(file_path))
    data = pandas.read_csv(file_path)

    # convert dates from strings to date objects
    dates = data.Datetime
    new_dates = []
    for day in dates:
        temp_day = datetime.strptime(day[0:-6], '%Y-%m-%d %H:%M:%S')
        new_dates.append(temp_day)

    dates = new_dates

    # plot results
    plt.plot(dates, data.Close)

    plt.show()

    return data


# create RSI scalp graph
def rsi_scalp_graph(file_path):
    stock = StockDataFrame.retype(pandas.read_csv(file_path))
    data = pandas.read_csv(file_path)

    # get indicator information
    rsi = stock['rsi_6']

    # convert dates from strings to date objects
    dates = data.Datetime
    new_dates = []
    for day in dates:
        temp_day = datetime.strptime(day[0:-6], '%Y-%m-%d %H:%M:%S')
        new_dates.append(temp_day)

    dates = new_dates

    buy_value, buy_dates, sell_value, sell_dates = dataTools.rsi_scalp_buy_sells(data.Close[:], dates, rsi, 0.01)

    # plot results
    plt.plot(dates, data.Close)
    plt.plot(buy_dates, buy_value, 'ro')
    plt.plot(sell_dates, sell_value, 'go')

    plt.legend(['Close', 'Buy', 'Sell'])

    plt.show()

    return data


# create RSI graph
def rsi_graph(file_path):
    stock = StockDataFrame.retype(pandas.read_csv(file_path))
    data = pandas.read_csv(file_path)

    # get indicator information
    rsi = stock['rsi_6']

    # convert dates from strings to date objects
    dates = data.Datetime
    new_dates = []
    for day in dates:
        temp_day = datetime.strptime(day[0:-6], '%Y-%m-%d %H:%M:%S')
        new_dates.append(temp_day)

    dates = new_dates

    buy_value, buy_dates, sell_value, sell_dates = dataTools.rsi_buy_sells(data.Close[:], dates, rsi)

    # plot results
    plt.plot(dates, data.Close)
    plt.plot(buy_dates, buy_value, 'ro')
    plt.plot(sell_dates, sell_value, 'go')

    plt.legend(['Close', 'Buy', 'Sell'])

    plt.show()

    return data


# create bollinger bands graph
def bollinger_bands_graph(file_path):
    stock = StockDataFrame.retype(pandas.read_csv(file_path))
    data = pandas.read_csv(file_path)

    # get indicator information
    boll = stock['boll']
    boll_up = stock['boll_ub']
    boll_low = stock['boll_lb']

    # convert dates from strings to date objects
    dates = data.Datetime
    new_dates = []
    for day in dates:
        temp_day = datetime.strptime(day[0:-6], '%Y-%m-%d %H:%M:%S')
        new_dates.append(temp_day)

    dates = new_dates

    buy_value, buy_dates, sell_value, sell_dates = dataTools.bollinger_buy_sells(data.Close[:], dates, boll, boll_up, boll_low)

    # plot results
    plt.plot(dates, data.Close)
    plt.plot(dates, boll)
    plt.plot(dates, boll_up)
    plt.plot(dates, boll_low)
    plt.plot(buy_dates, buy_value, 'ro')
    plt.plot(sell_dates, sell_value, 'go')

    plt.legend(['Close', 'Bollinger Band', 'Bollinger Band Upper', 'Bollinger Band Lower', 'Buy', 'Sell'])

    plt.show()

    return data


# graph algorithm for MACD trading
def macd_graph(file_path):
    stock = StockDataFrame.retype(pandas.read_csv(file_path))
    data = pandas.read_csv(file_path)

    # get indicator information
    # macd, signal = dataTools.calc_macd_values(data)
    macd = stock["macd"]
    signal = stock["macds"]

    # convert dates from strings to date objects
    dates = data.Date
    new_dates = []
    for day in dates:
        temp_day = datetime.strptime(day, '%Y-%m-%d')
        new_dates.append(temp_day)

    dates = new_dates

    print("Starting Value: " + str(data.Close.values[-350]))

    # get buys and sells
    buy_value, buy_dates, sell_value, sell_dates = dataTools.calc_buy_sell_test(data[-350:], dates[-350:], macd[-350:], signal[-350:])

    # plot results
    plt.plot(dates, data.Close)
    plt.plot(dates, macd)
    plt.plot(dates, signal)
    plt.plot(buy_dates, buy_value, 'ro')
    plt.plot(sell_dates, sell_value, 'go')

    plt.legend(['Close', 'MACD', 'Signal', 'Buy', 'Sell'])

    plt.show()

    return data


# graph algorithm for Trend MA Cross
def trend_MA_cross_graph(file_path):
    data = pandas.read_csv(file_path)

    # get indicator information
    ma20 = []
    ma50 = []

    # get indicator information
    for i in range(20, len(data.Close.values[:])):
        ma20.append(numpy.mean(data.Close.values[i - 20:i]))
    for i in range(50, len(data.Close.values[:])):
        ma50.append(numpy.mean(data.Close.values[i - 50:i]))

    # convert dates from strings to date objects
    dates = data.Date
    new_dates = []
    for day in dates:
        temp_day = datetime.strptime(day, '%Y-%m-%d')
        new_dates.append(temp_day)

    dates = new_dates

    print("Starting Value: " + str(data.Close.values[-350]))

    # get buys and sells
    # buy_value, buy_dates, sell_value, sell_dates = dataTools.calc_buy_sell_test(data[-350:], dates[-350:], ma20[-350:], ma50[-350:])
    buy_value, buy_dates, sell_value, sell_dates = dataTools.calc_short_test(data[-350:], dates[-350:], ma20[-350:], ma50[-350:])
    # buy_value, buy_dates, sell_value, sell_dates = dataTools.calc_buy_sell_test(data, dates, macd, signal)

    # plot results
    plt.plot(dates, data.Close)
    plt.plot(dates[-len(ma20):], ma20)
    plt.plot(dates[-len(ma50):], ma50)
    plt.plot(buy_dates, buy_value, 'ro')
    plt.plot(sell_dates, sell_value, 'go')

    plt.legend(['Close', 'MA 20', 'MA50', 'Buy', 'Sell'])

    plt.show()

    return data


def graph(dates, data):
    plt.plot(dates, data)

    plt.show()


if __name__ == "__main__":
    symbol = "GE"

    microsoft = dataTools.get_yahoo_csv(symbol)

    trend_MA_cross_graph("csv/" + symbol + "_daily.csv")
