from old.getStockData import save_dataset

if __name__ == "__main__":
    symbol = "MSFT"
    print("Symbol: " + symbol)

    save_dataset(symbol)
