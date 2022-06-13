import alpaca_trade_api as tradeapi


# read a file to get api_key and secret
def getFileCred(path):
    file = open(path, "r+")

    key = file.readline()[:-1]
    secret = file.readline()

    return key, secret


# trader used to trade stocks on alpaca
class AlpacaTrader:
    api_key = ""
    secret = ""
    endpoint = ""
    api = 0

    # constructor for trader object
    def __init__(self, end, key, secr):
        self.api_key = key
        self.secret = secr
        self.endpoint = end

        self.api = tradeapi.REST(
            base_url=self.endpoint,
            key_id=self.api_key,
            secret_key=self.secret
        )

    # get list of all tradable equities
    def getAssets(self):
        return self.api.list_assets(status='active')

    # see if asset is tradable for a specific symbol
    def getAsset(self, symbol):
        return self.api.get_asset(symbol)

    # place an order
    def order(self, symbol, num, buy_sell, order_type, limit_price=0.0):
        if order_type == "market":
            self.api.submit_order(
                symbol=symbol,
                qty=num,
                side=buy_sell,
                type=order_type,
                time_in_force='gtc'
            )
        elif order_type == "limit":
            self.api.submit_order(
                symbol=symbol,
                qty=num,
                side=buy_sell,
                type=order_type,
                time_in_force='gtc',
                limit_price=limit_price
            )
        else:
            print("Order type must be market or limit.")

    # get orders list for symbol
    def getOrders(self, symbol, is_open=False):
        if not is_open:
            orders = self.api.list_orders(
                status='all',
                limit=100,
                nested=True  # show nested multi-leg orders
            )

            return [o for o in orders if o.symbol == symbol]
        else:
            orders = self.api.list_orders(
                status='open',
                limit=100,
                nested=True  # show nested multi-leg orders
            )

            return [o for o in orders if o.symbol == symbol]