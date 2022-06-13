import time
import dataTools
import datetime
import pandas
import TechnicalIndicators
import dataTools
import numpy
from stocklist import NasdaqController
import csv

# find the best length for MA
def find_best_MA(stockData):
    percentMax = 5
    MAMax = 50
    # per run variables
    money = 10000.00
    own = False
    numOwned = 0
    buyPrice = 0
    wealthArr = []
    returns = []
    # best variables
    bestMoney = 0
    bestMA = 0
    bestPercent = 0
    bestDrawback = 0
    bestSharpe = 0

    percentages = []
    for i in range(1, percentMax*10+1):
        percentages.append(i * 0.1)

    for MALength in range(1, MAMax):
        ma = TechnicalIndicators.get_MA(stockData, MALength)    # moving average for each day
        tempStockData = stockData[MALength:]                    # stock data with matching days to ma
        for percent in percentages:
            for i in range(0, tempStockData.size):
                wealthArr.append(money + numOwned * tempStockData)

                if own:
                    # sell the stock
                    if tempStockData[i] >= buyPrice*percent:
                        money += numOwned*tempStockData[i]
                        numOwned = 0
                        own = False
                else:
                    if tempStockData[i] <= ma[i][1]:
                        numOwned = money//tempStockData[i]
                        money -= numOwned*tempStockData[i]
                        own = True
                        buyPrice = tempStockData[i]

            # collect share's worth at the end
            if own:
                money += numOwned*tempStockData[-1]

            # check if this run did the best
            if money > bestMoney:
                bestMoney = money
                bestMA = MALength
                bestPercent = percent
                bestDrawback = numpy.min(wealthArr)
                bestSharpe = (money/10000 - 0.02)/(numpy.std(wealthArr) / 10000)
                print("New best of $" + str(bestMoney) + " with an MA length of " + str(bestMA) + " and a sell off percentage of " + str(bestPercent) + ".")
                print("This algorithm has a max drawback of " + str(bestDrawback) + " and a sharpe ratio of " + str(bestSharpe))

            # reset per run variables
            money = 10000.00
            own = False
            numOwned = 0
            buyPrice = 0
            wealthArr = []
            returns = []

    print("Best Algorithm ended with $" + str(bestMoney) + " with an MA length of " + str(bestMA) + " and a sell off percentage of " + str(bestPercent) + ".")
    return bestPercent, bestMA, bestMoney, bestDrawback, bestSharpe


def combine_CSV():
    data = []
    for i in range(1,6):
        with open('../csv/BestMA_' + str(i) + '.csv', 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                data.append(row)

    with open('../csv/BestMA.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Stock', 'Percentage', 'Moving Average', 'Final Money', 'Max Drawback', 'Sharpe Ratio'])
        for row in data:
            writer.writerow(row)


def MA_trader(stock, length, percent, shares, buy_price, money):
    data = []
    with open('../log/MA_log_' + stock + "_" + str(datetime.datetime.now) + '.txt', 'w', encoding='UTF8', newline='') as f:
        while true:
            now = datetime.datetime.now()

            if now.hour == 4 and now.minute == 0:
                data = dataTools.get_yahoo_csv()[:-50]
                ma = TechnicalIndicators.get_MA(data, length)[-1]
                print("Moving average is now " + str(ma))
                f.write("Moving average is now " + str(ma))

                return

            price = dataTools.get_yahoo_quote(stock)
            print("Current price is " + str(price))
            f.write("Current price is " + str(price))

            if shares > 0:
                if buy_price * (percent + 1) >= price:
                    # sell
                    print("Sold " + str(shares) + " shares at $" +str(price) + " each")
                    f.write("Sold " + str(shares) + " shares at $" +str(price) + " each")

                    money += shares * price
                    shares = 0

                    print("Current cash is $" + str(money))
                    f.write("Current cash is $" + str(money))
            else:
                if price <= ma:
                    # buy
                    shares = money // price
                    money -= shares * price
                    buyPrice = price

                    print("Buy " + str(shares) + " shares at $" + str(price) + " each")
                    f.write("Buy " + str(shares) + " shares at $" + str(price) + " each")

                    print("Current cash is $" + str(money))
                    f.write("Current cash is $" + str(money))

            time.sleep(60)



if __name__ == "__main__":
    # list_of_tickers = dataTools.get_SP500()[400:]
    # allData = []
    #
    # for ticker in list_of_tickers:
    #     print(ticker)
    #     data = dataTools.get_yahoo_csv(ticker)
    #     if data.size == 0:
    #         continue
    #
    #     percentage, MA, money, drawback, sharpe = find_best_MA(data.Close[-252:])
    #     allData.append([ticker, percentage, MA, money, drawback, sharpe])
    #
    # with open('../csv/BestMA_5.csv', 'w', encoding='UTF8', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['Stock', 'Percentage', 'Moving Average', 'Final Money', 'Max Drawback', 'Sharpe Ratio'])
    #     for data in allData:
    #         writer.writerow(data)


    combine_CSV()

