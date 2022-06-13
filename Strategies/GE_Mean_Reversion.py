import AlpacaTrader
import dataTools
import numpy as np
import datetime
import time
import sys


def mean_rev(money, buy_price=0, sell_price=0):
    # get api key, secret, and endpoint
    key, secret = AlpacaTrader.getFileCred('../data/LiveAlpaca.txt')
    endpoint = "https://api.alpaca.markets"

    trader = AlpacaTrader.AlpacaTrader(endpoint, key, secret)

    # get number of stocks held
    try:
        num_owned = int(trader.api.get_position("GE").qty)
    except:
        num_owned = 0

    while True:
        data = dataTools.get_yahoo_intraday("GE", "30m")
        price = data.Close[-1]
        max_close = max(data.Close[-100:])
        min_close = min(data.Close[-150:])
        MA = np.average(data.Close[-20:])

        # own stocks
        if num_owned > 0:
            # made good money
            if price >= buy_price + .50:
                trader.order("GE", num_owned, "sell", "limit", price)
                print("Sold " + str(num_owned) + " shares for profit.")
                money += price * num_owned
                num_owned = 0
            # too much loss
            if price < buy_price - .20:
                trader.order("GE", num_owned, "sell", "limit", price)
                print("Sold " + str(num_owned) + " shares for loss")
                money += price * num_owned
                num_owned = 0
        # don't own stocks
        elif num_owned == 0:
            # buy stocks
            if price >= MA and price >= max_close:
                num_owned = money // price
                trader.order("GE", num_owned, "buy", "limit", price)
                print("Buy " + str(num_owned) + " shares @ " + str(price) + ".")
                money -= num_owned * price
                buy_price = price
            # short stocks
            if price < MA and price < min_close:
                num_owned = 0 - (money // price)
                trader.order("GE", num_owned, "sell", "limit", price)
                print("Short sold " + str(num_owned) + " shares @ " + str(price) + ".")
                money += (0 - num_owned) * price
                sell_price = price
        # owns shorts
        else:
            # close short for profit
            if price < sell_price - .50:
                trader.order("GE", (0 - num_owned), "buy", "limit", price)
                print("Close " + str(num_owned) + " short shares.")
                money -= num_owned * price
                num_owned = 0
            # close short for loss
            if price >= sell_price - .20:
                trader.order("GE", (0 - num_owned), "buy", "limit", price)
                print("Close " + str(num_owned) + " short shares.")
                money += num_owned * price
                num_owned = 0

        time.sleep(120)     # sleep a little before checking time to make sure you are after 30/60 min mark
        now = datetime.datetime.now()
        if now.minute >= 30:
            wait_time = (60 - now.minute - 1) * 60
        else:
            wait_time = (30 - now.minute - 1) * 60

        print("----------------------------------------------")
        print("Time completed: " + str(now))
        print("Stocks owned: " + str(num_owned))
        print("Current price: " + str(price))
        print("Current money: " + str(money))
        print("Buy price: " + str(buy_price))
        print("Sell price: " + str(buy_price))
        print("----------------------------------------------")

        time.sleep(wait_time)


if __name__ == "__main__":
    money_available = 10000.00

    if len(sys.argv) >= 2:
        money_available = sys.argv[1]

    mean_rev(money_available)
