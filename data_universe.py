import os
import datetime
import numpy as np
import pandas as pd
import quandl


__data_dir__ = "\\data\\"
__price_dir__ = "\\data\\prices\\"

__cwd__ = os.getcwd()
__data_path__ = __cwd__ + __data_dir__
__price_path__ = __cwd__ + __price_dir__

def create_universe_from_json():
    """Creates the listfile of tickers to use in the rest of the app provided a given source"""
    snp_list = []

    if not os.path.exists(__data_path__):
        os.makedirs(__data_path__)

    with open('S&P500.json', 'rb') as f:
        json = pd.read_json(f)
        snp_list = ['WIKI/' + x['ticker'] for x in json['constituents']]

    snp_list.sort()
    with open(__data_path__ + 'universe.txt', 'wt', encoding='utf-8') as f:
        for x in snp_list:
            f.write(x + '\n')


def update_all_price_caches():
    """ Go through each ticker in the universe and either get beginning data or update latest data """
    if not os.path.exists(__price_path__):
        os.makedirs(__price_path__)

    snp_list = []
    with open(__data_path__ + 'universe.txt', 'rt', encoding='utf-8') as f:
        snp_list = [x.strip('\n') for x in f]
    print("Got tickers... Top 3: ")
    print(snp_list[:3])

    # TODO: only working with the first 3 tickers while debugging
    for ticker in snp_list[:3]:
        update_price_cache(ticker)


def update_price_cache(ticker):
    is_any, last_date = __any_ticker_files(ticker)
    if not is_any:
        data = quandl.get(ticker)
        print(" --- ")
        print("New data for: " + ticker)
        print(data.head())
        unique_months = {(x.year, x.month) for x in data.index}
        print("got " + len(unique_months) + " months worth of data.")
        for (year, month) in unique_months:
            sub_data = data.index[datetime.date(year, month, 1):datetime.date(year, month, 31)]
            with open(__price_path__ + ticker + "." + )
            #sub_data.to_json()


def __any_ticker_files(ticker):
    last_date = datetime.date(1900, 1, 1)
    old_date = last_date
    for file_found in os.listdir(__price_path__):
        if file_found.startswith(ticker):
            file_split = file_found[len(ticker):].split('.')
            file_date = datetime.date(int(file_split[0]), int(file_split[1]), 1)
            last_date = max([last_date, file_date])
    if old_date == last_date:
        return False, last_date
    return True, last_date


if __name__ == '__main__':
    # create_universe_from_json()
    update_all_price_caches()
