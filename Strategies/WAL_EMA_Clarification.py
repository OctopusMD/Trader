import sys
import AlpacaTrader
import dataTools
import numpy as np
import datetime
import time


# get the triple ema of the stock
def triple_ema(values, length):
    ema1 = []
    ema2 = []
    ema3 = []

    for i in range(len(values) - length):
        ema1.append(dataTools.calc_ema_in_range(values, i, i + length))
    for i in range(len(ema1) - length):
        ema2.append(dataTools.calc_ema_in_range(ema1, i, i + length))
    for i in range(len(ema2) - length):
        ema3.append(dataTools.calc_ema_in_range(ema2, i, i + length))

    return 3 * (ema1[-1] - ema2[-1]) + ema3[-1]


def mean_rev(money):
    # get api key, secret, and endpoint
    key, secret = AlpacaTrader.getFileCred('../data/LiveAlpaca.txt')
    endpoint = "https://api.alpaca.markets"

    trader = AlpacaTrader.AlpacaTrader(endpoint, key, secret)

    # get number of stocks held
    try:
        num_owned = int(trader.api.get_position("WAL").qty)
    except:
        num_owned = 0

    while True:
        data = dataTools.get_yahoo_intraday("WAL", "5m")
        price = data.Close[-1]
        MA = np.average(data.Close[-50:])
        tripleEMA = triple_ema(data.Close[-100:], 9)

        # get number of stocks held
        try:
            curr_num_owned = int(trader.api.get_position("WAL").qty)
        except:
            curr_num_owned = 0

        if curr_num_owned != num_owned:
            print("Order not filled")
            exit(0)

        # own stocks
        if num_owned > 0:
            # close long
            if price < tripleEMA:
                trader.order("WAL", num_owned, "sell", "limit", price - 0.03)
                print("Sold " + str(num_owned) + " shares for profit.")
                money += price * num_owned
                num_owned = 0
        # don't own stocks
        elif num_owned == 0:
            # buy stocks
            if price >= MA and price >= tripleEMA:
                num_owned = money // price
                trader.order("WAL", num_owned, "buy", "limit", price + 0.03)
                print("Buy " + str(num_owned) + " shares @ " + str(price) + ".")
                money -= num_owned * price
            # short stocks
            if price < MA and price < tripleEMA:
                num_owned = 0 - (money // price)
                trader.order("WAL", 0 - num_owned, "sell", "limit", price - 0.03)
                print("Short sold " + str(num_owned) + " shares @ " + str(price) + ".")
                money += (0 - num_owned) * price
        # owns shorts
        else:
            # close short
            if price >= tripleEMA:
                trader.order("WAL", (0 - num_owned), "buy", "limit", price + 0.03)
                print("Close " + str(num_owned) + " short shares.")
                money += num_owned * price
                num_owned = 0

        time.sleep(60)  # sleep a little before checking time to make sure you are after 5 min mark
        now = datetime.datetime.now()
        wait_time = (5 - (now.minute % 5)) * 60
        wait_time -= 60

        print("----------------------------------------------")
        print("Time completed: " + str(now))
        print("Stocks owned: " + str(num_owned))
        print("Current price: " + str(price))
        print("Current money: " + str(money))
        print("Moving Average: " + str(MA))
        print("Triple EMA: " + str(tripleEMA))
        print("----------------------------------------------")

        time.sleep(wait_time)


if __name__ == "__main__":
    money_available = 10000.00

    if len(sys.argv) >= 2:
        money_available = int(sys.argv[1])

    mean_rev(money_available)
