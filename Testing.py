import numpy as np
import pandas as pd
import dataTools
import Strategies.MA_Strategies

if __name__ == "__main__":
    ticker='GOOGL'
    print(ticker)
    data = dataTools.get_yahoo_csv(ticker)

    percentage, MA, money, drawback, sharpe = Strategies.MA_Strategies.find_best_MA(data.Close[-252:])
    #temp = dataTools.get_yahoo_gainers()

    # symbol = "FIT"
    #
    # key, secret = AlpacaTrader.getFileCred("./data/LiveAlpaca.txt")
    #
    # trader = AlpacaTrader.AlpacaTrader("https://api.alpaca.markets", key, secret)
    #
    # assets = trader.getOrders("GE")
    # print(str(trader.api.get_position("BJ")))
