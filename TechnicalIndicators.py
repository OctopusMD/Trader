import numpy as np
import dataTools
import grapher


def get_MA(data, length):
    ma = []
    for i in range(length, data.size):
        date = data.axes[0][i]
        average = np.average(data[i-length:i])
        ma.append([date, average])

    return ma


if __name__ == "__main__":
    data = dataTools.get_yahoo_csv("GE")
    MA = np.array(get_MA(data.Close, 10))

    grapher.graph(MA[:, 0], MA[:, 1])
