import AlpacaTrader
import yfinance as yf
from stocklist import NasdaqController
import dataTools
import numpy as np
import threading
import datetime
import time
import sys


# mutex lock for threading
mutex = threading.Lock()
# stock shared resource list
scalp_stocks = []
buy_prices = []


# thread object
class StockThread (threading.Thread):
    def __init__(self, threadID, symbol, trader):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.symbol = symbol
        self.trader = trader

    # function ran when thread started
    def run(self):
        print("Starting thread: " + self.threadID)

        # buy stock
        mutex.acquire()

        self.trader.order(self.symbol, 10, "buy", "market")
        scalp_stocks.append(self.symbol)
        buy_prices.append(self.trader.getOrders(self.symbol)[0].price)

        mutex.release()

        # wait until profit or stop loss


# get stock info for symbol
def get_stock_info(symbol):
    return yf.download(tickers=symbol, period="3h", auto_adjust=True, interval="1m", threads=True)


# find stock to scalp
def get_potential_stocks(stock_list):
    up_stock = []   # stocks in uptrend
    for stock in stock_list:
        data = get_stock_info(stock)

        # get ma and rsi
        ma20 = np.mean(data.Close[-20:])
        rsi = dataTools.get_RSI(data, 14)
        close = data.Close[-1]

        if close > ma20 and 45 <= rsi <= 55:
            up_stock.append(stock)

    return up_stock


def scalper(price_min=5, price_max=10, scalp=0.01):
    # get stocks to search for
    StocksController = NasdaqController(True)
    list_of_tickers = StocksController.getList()
    stocks = []

    for ticker in list_of_tickers:
        data = yf.download(tickers=ticker, period="5m", auto_adjust=True, interval="1m", threads=True)
        if price_min <= data.Close[-1] <= price_max:
            stocks.append(ticker)

    while True:
        potential = get_potential_stocks(stocks)

        mutex.acquire()




