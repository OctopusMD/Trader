import os
import time
import yfinance as yf
import dateutil.relativedelta
from datetime import date
import datetime
import numpy as np
import pandas as pd
import sys
from stocklist import NasdaqController
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend
import multiprocessing
from datetime import datetime
from dateutil.parser import parse

# Change variables to your liking then run the script
MONTH_CUTTOFF = 36


class Gravity:
    def __init__(self):
        StocksController = NasdaqController(True)
        self.list_of_tickers = StocksController.getList()
        # self.list_of_tickers = ['GME', 'NIO']
        self.stocks = []

        for ticker in self.list_of_tickers:
            self.stocks.append(self.getData(ticker))

        self.ups = 0
        self.downs = 0
        self.average = 0

        self.csv = pd.Series(self.getStats(self.list_of_tickers, self.stocks))
        self.csv.to_csv(f'./csv/gravity.csv')

        if self.ups + self.downs != 0:
            print("Stocks went up " + str((self.ups/(self.ups + self.downs)) * 100) + "% of the time after larger change.")
            print("Stocks went down " + str((self.downs / (self.ups + self.downs)) * 100) + "% of the time after larger change.")

    # get volume data
    def getData(self, ticker):
        global MONTH_CUTOFF

        # set data parameters
        currentDate = datetime.datetime.strptime(date.today().strftime("%Y-%m-%d"), "%Y-%m-%d")
        pastDate = currentDate - \
            dateutil.relativedelta.relativedelta(months=MONTH_CUTTOFF)
        sys.stdout = open(os.devnull, "w")

        # retrieve and return data
        data = yf.download(ticker, pastDate, currentDate)
        sys.stdout = sys.__stdout__
        print("Data retrieved for stock " + ticker)

        return data.axes[0], data.to_numpy()

    def getStats(self, tickers, stocks):
        record = [["first change", "second change", "stock price 1", "stock price 2", "stock price 3"]]
        for i in range(0, len(tickers)):
            dates = stocks[i][0]
            tempArr = stocks[i][1]
            # 3 = close &
            for j in range(1, len(tempArr) - 1):
                change1 = tempArr[j][3]/tempArr[j-1][3]

                if change1 >= 1.2:
                    change2 = tempArr[j+1][3]/tempArr[j][3]

                    if change2 > 1:
                        self.ups += 1
                        print("[" + tickers[i] + "] up " + str(((change2 - 1) * 100)) + "%. Stock went from $" + str(tempArr[j-1][3]) + " to $" + str(tempArr[j][3]) + ".")
                    elif change2 <= 1:
                        self.downs += 1
                        print("[" + tickers[i] + "] down " + str(((1 - change2) * 100)) + "%. Stock went from $" + str(tempArr[j-1][3]) + " to $" + str(tempArr[j][3]) + ".")

                    record.append([dates[j], change1, change2, tempArr[j-1][3], tempArr[j][3], tempArr[j+1][3]])

        return record

if __name__ == '__main__':
    # test = Gravity()

    data = pd.read_csv('../csv/gravity.csv')

    count = 0
    ups = 0
    downs = 0
    sum = 0
    sumInc = 0
    sumDec = 0
    multi = 1
    for row in data.to_numpy():
        date = datetime.strptime(row[0], '%m/%d/%Y')
        if date.year == 2020:
            continue
        if 100 <= row[3] < 200:
            if row[2] > 1:
                ups += 1
                sumInc += row[2]
            else:
                downs += 1
                sumDec += row[2]

            multi *= row[2]
            sum += row[2]
            count += 1

    print("Number of increases: " + str(ups))
    print("Number of decreases: " + str(downs))
    print("Average Increase: " + str(sumInc/ups))
    print("Average Decrease: " + str(sumDec/downs))
    print("Total Average: " + str(sum/count))
    print("Portfolio Growth: " + str(multi))

    # plt.scatter(data['stock price 1'].values, data['second change'].values)
    # plt.show()
    #
    # print("here")