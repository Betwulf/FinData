import os
import datetime
import calendar
import numpy as np
import pandas as pd
import quandl


_data_dir = "\\data\\"
_price_dir = "\\data\\prices\\"

_cwd = os.getcwd()
_data_path = _cwd + _data_dir
_price_path = _cwd + _price_dir


def create_universe_from_json():
    """Creates the listfile of tickers to use in the rest of the app provided a given source"""
    snp_list = []

    if not os.path.exists(_data_path):
        os.makedirs(_data_path)

    with open('S&P500.json', 'rb') as f:
        json = pd.read_json(f)
        snp_list = ['WIKI/' + x['ticker'] for x in json['constituents']]

    snp_list.sort()
    with open(_data_path + 'universe.txt', 'wt', encoding='utf-8') as f:
        for x in snp_list:
            f.write(x + '\n')


def update_all_price_caches():
    """ Go through each ticker in the universe and either get beginning data or update latest data """
    if not os.path.exists(_price_path):
        os.makedirs(_price_path)

    snp_list = []
    with open(_data_path + 'universe.txt', 'rt', encoding='utf-8') as f:
        snp_list = [x.strip('\n') for x in f]
    print("Got tickers... Top 3: ")
    print(snp_list[:3])

    # TODO: only working with the first 3 tickers while debugging
    for ticker in snp_list[:3]:
        update_price_cache(ticker)


def update_price_cache(ticker):
    """ Creates a set of monthly price files for the ticker """
    print(" --- ")
    is_any, last_date = _any_ticker_files(ticker)
    if not is_any:
        print("New data for: " + ticker)
        data = quandl.get(ticker)
    else:
        print("Update data for: " + ticker)
        data = quandl.get(ticker, start_date=last_date)

    unique_months = {(x.year, x.month) for x in data.index}
    print(data.head())
    print("got {} months worth of data.".format(len(unique_months)))
    # Add ticker to the dataframe
    data['ticker'] = [ticker for _ in range(len(data[data.columns[0]]))]
    for (year, month) in unique_months:
        month_first, month_last = _get_month_day_range(year, month)
        sub_data = data[month_first:month_last]
        # not sure if this date formatting will be reversible on load...
        sub_data.index = [x.strftime('%Y-%m-%d') for x in sub_data.index]
        with open(_create_filename(ticker, year, month), 'wt') as f:
            f.write(sub_data.to_json())

# TODO: Create a method to combine and store all prices for speed later on...
def get_all_prices():
    """ This gets every single price for all securities in the universe - warning this could take some time... """
    ttl_data = pd.DataFrame()
    for file_found in os.listdir(_price_path):
        with open(_price_path + file_found, 'rt') as f:
            current_data = pd.read_json(f)
            ttl_data = pd.concat([current_data, ttl_data])
    return ttl_data


def _create_filename(ticker, year, month):
    # ticker may have odd symbols that cannot be in a filename like a forward slash
    ticker = ticker.replace('/', '+')
    return _price_path + ticker + ".%04d.%02d.json" % (year, month)


def _parse_filename(filename):
    pieces = filename.split('.')
    ticker = pieces[0].replace('+', '/')
    year = int(pieces[1])
    month = int(pieces[2])
    return ticker, year, month


def _get_month_day_range(year, month):
    first_day = datetime.date(year, month, 1)
    last_day = first_day.replace(day=calendar.monthrange(first_day.year, first_day.month)[1])
    return first_day, last_day


def _any_ticker_files(ticker):
    """ Check to see if there are any ticker files already saved to disk.
    This saves us from having to go to the source and reload data. """
    last_date = datetime.date(1900, 1, 1)
    old_date = last_date
    for file_found in os.listdir(_price_path):
        file_ticker, file_year, file_month = _parse_filename(file_found)
        if file_ticker == ticker:
            file_date = datetime.date(file_year, file_month, 1)
            last_date = max([last_date, file_date])
    if old_date == last_date:
        return False, last_date
    return True, last_date


if __name__ == '__main__':
    # create_universe_from_json()
    # update_all_price_caches()

    data = get_all_prices()
    print('Total number of rows: {}'.format(len(data)))
    print('Got data for the following tickers:')
    for x in {t for t in data['ticker']}:
        print(x)
