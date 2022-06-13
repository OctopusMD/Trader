import datetime
import pandas
import dataTools
import numpy

if __name__ == "__main__":
    gainers = dataTools.get_yahoo_gainers()
    data = []

    for gainer in gainers:
        stock = dataTools.get_yahoo_intraday(gainer)

        # get max volume
        max = stock.Volume.values.max()

        # get average volume
        average = numpy.average(stock.Volume.values[0:-2])

        # get current volume
        current = stock.Volume.values[-2]

        # compare current volume to max and average
        maxCompare = max/current
        averageCompare = average/current

        data.append([gainer, max, average, current, maxCompare, averageCompare])

    data = pandas.DataFrame(numpy.array(data), columns=["stock", "max", "average", "current", "maxComparison", "averageComparison"])
    data.to_csv(f'.././csv/{datetime.date.today()}_low_volume_pullback.csv')
