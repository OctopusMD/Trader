import pandas
import numpy as np
import dataTools
import os
import time
import sys
import TradingTools


# perform trades based on rsi
def rsi_trader(symbol, money):
    running = True
    total = money
    starting = money
    minimum = money * 0.9
    extra = total
    count = 0
    sells = 0
    buys = 0
    cancels = 0

    # connect to robin hood
    TradingTools.login("pcwright113@gmail.com", "Plasma13sword!")

    # see if we already hold stocks
    num_owned = TradingTools.get_quantity(symbol)

    # retrieve stock quote from yahoo finance
    data = dataTools.get_yahoo_csv()
    high = max(data.High[-30:])

    # try catch sees if program is stopped (CTRL + C)
    try:
        # run until told to stop
        while running:

            # check if we need to login again
            if count == 100:
                TradingTools.login("pcwright113@gmail.com", "Plasma13sword!")
                count = 0

            # retrieve stock quote from yahoo finance
            data = dataTools.get_yahoo_intraday(symbol)

            # check if order is pending
            actual_owned = TradingTools.get_quantity(symbol)
            # if order is pending it's better just to cancel order
            if actual_owned != num_owned:
                print("Canceling order")
                TradingTools.cancel_order(num_owned, symbol)
                cancels += 1
                extra = pre_extra
                num_owned = actual_owned

            if actual_owned > 0:
                own = True
            else:
                own = False

            # calculate total holdings
            total = num_owned * data.Close[-1] + extra
            pre_extra = extra

            # market too low
            if total < minimum:
                # sell
                if not TradingTools.sell(num_owned, symbol):
                    return extra, num_owned
                sells += 1
                print("Current holdings dropped below initial value of $" + str(
                    minimum) + ". Application has stopped with holdings at $" + str(total) + ".")
                return

            # calculate RSI value
            rsi = get_RSI(data, 6)

            # decide whether or not to trade
            if own:
                # overbought so sell stocks
                if rsi[-1] >= 60:
                    # sell
                    if not TradingTools.sell(num_owned, symbol):
                        return extra, num_owned
                    sells += 1
                    # update local data
                    extra = total
                    num_owned = 0
                    own = False

            else:
                # stock is at low so buy
                if 40 >= rsi[-1]:
                    num_owned = total // data.Close[-1]
                    # buy
                    if not TradingTools.buy(num_owned, symbol):
                        return extra, num_owned
                    buys += 1
                    # update local data
                    extra = total - (num_owned * data.Close[-1])
                    own = True

            # output current status
            clear()
            print("Current Status of " + symbol + " @" + str(data.Close.index[-1]))
            print("---------------------")
            print("Starting Value: " + str(starting))
            print("Total Value: " + str(total))
            print("---------------------")
            print("Stock Price: " + str(data.Close[-1]))
            print("RSI Value: " + str(rsi[-1]))
            print("Stocks Owned: " + str(num_owned))
            print("Buys: " + str(buys) + " Sells: " + str(sells) + " Cancels: " + str(cancels))

            # wait one minute
            count += 1
            time.sleep(60)
    except KeyboardInterrupt:
        return


# perform trades based on rsi
def rsi_trader_limit(symbol, money):
    running = True
    total = money
    starting = money
    minimum = money * 0.95
    maximum = money * 1.02
    extra = total
    count = 0

    # connect to robin hood
    TradingTools.login("pcwright113@gmail.com", "Plasma13sword!")

    # see if we already hold stocks
    num_owned = TradingTools.get_quantity(symbol)

    # try catch sees if program is stopped (CTRL + C)
    try:
        # run until told to stop
        while running:

            # check if we need to login again
            if count == 100:
                TradingTools.login("pcwright113@gmail.com", "Plasma13sword!")
                count = 0

            # retrieve stock quote from yahoo finance
            data = dataTools.get_yahoo_intraday(symbol)

            # check if order is pending
            actual_owned = TradingTools.get_quantity(symbol)
            # if order is pending it's better just to cancel order
            while actual_owned != num_owned:
                print("Waiting on limit order...")
                time.sleep(15)

            if actual_owned > 0:
                own = True
            else:
                own = False

            # calculate total holdings
            total = num_owned * round(data.Close[-1], 2) + extra
            pre_extra = extra

            # market too low
            if total < minimum:
                # sell
                if not TradingTools.sell_limit(num_owned, symbol, round(data.Close[-1], 2)):
                    return extra, num_owned
                print("Current holdings dropped below initial value of $" + str(
                    minimum) + ". Application has stopped with holdings at $" + str(total) + ".")
                return

            # calculate RSI value
            rsi = get_RSI(data, 6)

            # decide whether or not to trade
            if own:
                # overbought so sell stocks
                if rsi[-1] >= 60:
                    # sell
                    if not TradingTools.sell_limit(num_owned, symbol, round(data.Close[-1], 2)):
                        return extra, num_owned
                    # update local data
                    extra = total
                    num_owned = 0
                    own = False

            else:
                # stock is at low so buy
                if 40 >= rsi[-1] > 20:
                    num_owned = total // round(data.Close[-1], 2)
                    # buy
                    if not TradingTools.buy_limit(num_owned, symbol, round(data.Close[-1], 2)):
                        return extra, num_owned
                    # update local data
                    extra = total - (num_owned * round(data.Close[-1], 2))
                    own = True

            # output current status
            clear()
            print("Current Status @" + str(data.Close.index[-1]))
            print("---------------------")
            print("Starting Value: " + str(starting))
            print("Total Value: " + str(total))
            print("---------------------")
            print("Stock Price: " + str(round(data.Close[-1], 2)))
            print("RSI Value: " + str(rsi[-1]))
            print("Stocks Owned: " + str(num_owned))

            # wait one minute
            count += 1
            time.sleep(60)
    except KeyboardInterrupt:
        return


# perform scalps based on rsi
def rsi_scalper(symbol, total, scalp, cont=False, cont_extra=0.0):
    waiting = False
    buying = False
    starting = total
    minimum = total * 0.9
    maximum = total * 1.1
    extra = total
    count = 0
    sells = 0
    buys = 0
    cancels = 0
    price = 0.00

    # connect to robin hood
    TradingTools.login("pcwright113@gmail.com", "Plasma13sword!")

    # see if we already hold stocks
    num_owned = TradingTools.get_quantity(symbol)
    if num_owned == 0:
        own = False
    else:
        own = True

    # retrieve stock info from yahoo finance to find high
    data = dataTools.get_yahoo_csv(symbol)
    high = max(data.High[-30:])

    if cont:
        extra = cont_extra

    # try catch sees if program is stopped (CTRL + C)
    try:
        # run until told to stop
        while True:
            # check if we need to login again
            if count == 100:
                TradingTools.login("pcwright113@gmail.com", "Plasma13sword!")
                count = 0

            # retrieve stock quote from yahoo finance
            data = dataTools.get_yahoo_intraday(symbol)

            # check if order is pending
            actual_owned = TradingTools.get_quantity(symbol)

            # if order is pending it's better just to cancel order
            if actual_owned != num_owned:
                if actual_owned == 0:
                    own = False
                else:
                    own = True

                print("Canceling order")
                TradingTools.cancel_order(symbol)
                cancels += 1

                if buying:
                    extra = extra + ((num_owned - actual_owned) * pre_price)
                else:
                    extra = extra - (actual_owned * pre_price)
                num_owned = actual_owned

            # calculate total holdings
            total = num_owned * data.Close[-1] + extra

            # save current price and extra money in case we need to cancel
            pre_extra = extra
            pre_price = data.Close[-1]

            # market too low
            if total < minimum:
                # sell
                if not TradingTools.sell(num_owned, symbol):
                    return extra, num_owned
                sells += 1
                print("Pre-extra: " + str(pre_extra))
                print("Total: " + str(total))
                print("Minimum: " + str(minimum))
                print("Min Exit condition reached.")
                return
            # leave while the gettings good
            elif total > maximum:
                # sell
                if not TradingTools.sell(num_owned, symbol):
                    return extra, num_owned
                sells += 1
                print("Pre-extra: " + str(pre_extra))
                print("Total: " + str(total))
                print("Maximum: " + str(maximum))
                print("Max Exit condition reached.")
                return

            # calculate RSI value
            rsi = get_RSI(data, 6)

            # decide whether or not to trade
            if waiting:
                if rsi[-1] >= 50:
                    if not TradingTools.buy(num_owned, symbol):
                        return extra, num_owned
                    buys += 1
                    price = data.Close[-1]
                    extra = total - (num_owned * data.Close[-1])
                    own = True
                    waiting = False
                    buying = True
            elif own:
                # overbought so sell stocks
                if data.Close[-1] >= price + scalp:
                    # sell
                    if not TradingTools.sell(num_owned, symbol):
                        return extra, num_owned
                    sells += 1
                    # update local data
                    extra = total
                    num_owned = 0
                    own = False
                    buying = False
                elif data.Close[-1] < price - (scalp * 3):
                    # sell
                    if not TradingTools.sell(num_owned, symbol):
                        return extra, num_owned
                    sells += 1
                    # update local data
                    extra = total
                    num_owned = 0
                    own = False
                    waiting = True
                    buying = False
            else:
                # stock is at low so buy
                if rsi[-1] <= 33:
                    num_owned = total // data.Close[-1]
                    # buy
                    if not TradingTools.buy(num_owned, symbol):
                        return extra, num_owned
                    buys += 1
                    price = data.Close[-1]
                    # update local data
                    extra = total - (num_owned * data.Close[-1])
                    own = True
                    buying = True

            # output current status
            clear()
            print("Current Status of " + symbol + " @" + str(data.Close.index[-1]))
            print("---------------------")
            print("Starting Value: " + str(starting))
            print("Total Value: " + str(total))
            print("---------------------")
            print("Stock Price: " + str(data.Close[-1]))
            print("RSI Value: " + str(rsi[-1]))
            print("Stocks Owned: " + str(num_owned))
            print("Buys: " + str(buys) + " Sells: " + str(sells) + " Cancels: " + str(cancels))

            # wait one minute
            count += 1
            time.sleep(60)
    except KeyboardInterrupt:
        print("++++++++++++++++++++++++++++++++++++")
        print("Extra: " + str(extra))
        print("Stock Price: " + str(data.Close[-1]))
        print("Stocks Owned: " + str(num_owned))
        print("Total: " + str(total))
        print("++++++++++++++++++++++++++++++++++++")
        return


def clear():
    os.system('cls')


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


if __name__ == "__main__":
    stock = "FIT"
    money_available = 100.00
    scalp = 0.01

    if len(sys.argv) >= 2:
        stock = sys.argv[1]
    if len(sys.argv) >= 3:
        money_available = float(sys.argv[2])
    if len(sys.argv) >= 4:
        scalp = float(sys.argv[3])

    rsi_scalper(stock, money_available, scalp)
