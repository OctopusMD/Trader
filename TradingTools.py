import robin_stocks
import time


# login to robinhood
def login(username, password):
    robin_stocks.login(username, password)


# check how many of stock is currently owned
def get_quantity(symbol):
    # see if we already hold stocks
    my_stocks = robin_stocks.build_holdings()
    try:
        return float(my_stocks[symbol]['quantity'])
    except:
        return 0.0


# sell stocks
def sell(num, symbol):
    if num == 0:
        print("nothing to sell")
        return True

    try:
        robin_stocks.order_sell_market(symbol, num)
        return True
    except:
        print("Unable to sell stock " + symbol)
        return False


def sell_limit(num, symbol, price):
    if num == 0:
        print("nothing to sell")
        return True

    try:
        robin_stocks.order_sell_limit(symbol, num, price)
        return True
    except:
        print("Unable to sell stock " + symbol)
        return False


# buy stocks
def buy(num, symbol):
    if num == 0:
        print("nothing to buy")
        return True

    try:
        robin_stocks.order_buy_market(symbol, num)
        return True
    except:
        print("Unable to buy stock " + symbol)
        return False


def buy_limit(num, symbol, price):
    if num == 0:
        print("nothing to sell")
        return True

    try:
        robin_stocks.order_buy_limit(symbol, num, price)
        return True
    except:
        print("Unable to sell stock " + symbol)
        return False


def cancel_order(symbol):
    orders = robin_stocks.find_stock_orders(symbol=symbol)

    for order in orders:
        robin_stocks.cancel_stock_order(order["id"])