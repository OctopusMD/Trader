import os
import time
import yfinance as yf
import dateutil.relativedelta
from datetime import date
import datetime
import numpy as np
import sys
from stocklist import NasdaqController
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend
import multiprocessing

###########################
# THIS IS THE MAIN SCRIPT #
###########################

# Change variables to your liking then run the script
MONTH_CUTTOFF = 5
DAY_CUTTOFF = 3
STD_CUTTOFF = 9


class InsiderOut:

    def __init__(self):
        pass

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
        return data[["Volume"]]

    # find anomalous data
    def find_anomalies(self, data):
        global STD_CUTTOFF

        indexes = []
        outliers = []
        data_std = np.std(data['Volume'])           # standard deviation
        data_mean = np.mean(data['Volume'])         # mean
        anomaly_cut_off = data_std * STD_CUTTOFF
        upper_limit = data_mean + anomaly_cut_off

        data.reset_index(level=0, inplace=True)
        for i in range(len(data)):
            temp = data['Volume'].iloc[i]
            if temp > upper_limit:
                indexes.append(str(data['Date'].iloc[i])[:-9])
                outliers.append(temp)
        anomalies = {'Dates': indexes, 'Volume': outliers}
        return anomalies

    # print data in standard format
    def customPrint(self, d, tick):
        print("\n\n\n*******  " + tick.upper() + "  *******")
        print("Ticker is: "+tick.upper())
        for i in range(len(d['Dates'])):
            str1 = str(d['Dates'][i])
            str2 = str(d['Volume'][i])
            print(str1 + " - " + str2)
        print("*********************\n\n\n")

    def days_between(self, d1, d2):
        d1 = datetime.datetime.strptime(d1, "%Y-%m-%d")
        d2 = datetime.datetime.strptime(d2, "%Y-%m-%d")
        return abs((d2 - d1).days)

    def parallel_wrapper(self, x, currentDate, positive_scans):
        global DAY_CUTTOFF
        d = (self.find_anomalies(self.getData(x)))
        if d['Dates']:
            for i in range(len(d['Dates'])):
                if self.days_between(str(currentDate)[:-9], str(d['Dates'][i])) <= DAY_CUTTOFF:
                    self.customPrint(d, x)
                    stonk = dict()
                    stonk['Ticker'] = x
                    stonk['TargetDate'] = d['Dates'][0]
                    stonk['TargetVolume'] = str(
                        '{:,.2f}'.format(d['Volume'][0]))[:-3]
                    positive_scans.append(stonk)

    def main_func(self):
        StocksController = NasdaqController(True)
        list_of_tickers = StocksController.getList()
        currentDate = datetime.datetime.strptime(
            date.today().strftime("%Y-%m-%d"), "%Y-%m-%d")
        start_time = time.time()

        manager = multiprocessing.Manager()
        positive_scans = manager.list()

        with parallel_backend('loky', n_jobs=multiprocessing.cpu_count()):
            Parallel()(delayed(self.parallel_wrapper)(x, currentDate, positive_scans)
                       for x in tqdm(list_of_tickers))

        print("\n\n\n\n--- this took %s seconds to run ---" %
              (time.time() - start_time))

        return positive_scans


if __name__ == '__main__':
    list = InsiderOut().main_func()
    print(str(list))
