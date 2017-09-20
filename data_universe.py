import sys
import os
import datetime
import calendar
import numpy as np
import pandas as pd
import io
import requests
import quandl
from utils import timing

_data_dir = "/data/"
_price_dir = "/data/prices/"
_fundamental_dir = "/data/fundamental/"
_combined_filename = "__all.csv"
_wiki_prefix = 'WIKI/'

_cwd = os.getcwd()
_data_path = _cwd + _data_dir
_price_path = _cwd + _price_dir
_fundamental_path = _cwd + _fundamental_dir

# ensure paths are there...
if not os.path.exists(_data_path):
    os.makedirs(_data_path)
if not os.path.exists(_price_path):
    os.makedirs(_price_path)
if not os.path.exists(_fundamental_path):
    os.makedirs(_fundamental_path)


def create_universe_from_json():
    """Creates the list file of tickers to use in the rest of the app provided a given source"""
    # snp_list = []

    with open('S&P500.json', 'rb') as f:
        json = pd.read_json(f)
        snp_list = [_wiki_prefix + x['ticker'] for x in json['constituents']]

    snp_list.sort()
    with open(_data_path + 'universe.txt', 'wt', encoding='utf-8') as f:
        for x in snp_list:
            f.write(x + '\n')


def update_all_price_caches():
    """ Go through each ticker in the universe and either get beginning data or update latest data """

    snp_list = _get_tickerlist()
    for ticker in snp_list:
        update_price_cache(ticker)


def _get_tickerlist(remove_wiki=False, num_tickers=510):
    # snp_list = []
    with open(_data_path + 'universe.txt', 'rt', encoding='utf-8') as f:
        snp_list = [x.strip('\n') for x in f]
    print("Got tickers... Top {}: ".format(num_tickers))
    if remove_wiki:
        snp_list = [x[len(_wiki_prefix):] for x in snp_list]
    print(snp_list[:num_tickers])
    return snp_list[:num_tickers]


def update_price_cache(ticker):
    """ Creates a set of monthly price files for the ticker """

    print(" --- ")
    # TODO: Check to see if the file was updated today, if so then skip (unless a force param was sent)
    is_any, last_date, file_timestamp = _any_ticker_files(ticker)
    fin_data = None
    curr_day = datetime.datetime.today().replace(hour=0, minute=0, second=0)
    try:
        if not is_any:
            print("New data for: " + ticker)
            fin_data = quandl.get(ticker)
        elif file_timestamp >= curr_day:
            print("data for {} is up to date.".format(ticker))
        else:
            print("Update data for: " + ticker)
            fin_data = quandl.get(ticker, start_date=last_date)
    except:
        pass

    if fin_data is not None:
        fin_data.columns = [x.lower() for x in fin_data.columns]
        unique_months = {(x.year, x.month) for x in fin_data.index}
        print(fin_data.head())
        print("got {} months worth of data.".format(len(unique_months)))
        # Add ticker to the data frame
        fin_data['ticker'] = [ticker for _ in range(len(fin_data[fin_data.columns[0]]))]
        for (year, month) in unique_months:
            month_first, month_last = _get_day_range_for_month(year, month)
            sub_data = fin_data[month_first:month_last]
            # not sure if this date formatting will be reversible on load...
            sub_data.index = [x.strftime('%Y-%m-%d') for x in sub_data.index]
            sub_data.index.name = 'date'
            # Don't want data before 1970 - timestamps don't work before then...
            # But really... I don't want anything older than 2006 i reckon... vastly different trading behaviors
            if year > 2006:
                with open(_create_filename(ticker, year, month), 'wt') as f:
                        f.write(sub_data.to_csv())


# TODO: Do we need this, or just use __all.csv ?
def get_ticker_prices():
    """ returns a dictionary of tickers to their 'pandas' dataframes of prices"""
    snp_list = _get_tickerlist()
    df_dict = {t: pd.DataFrame() for t in snp_list}
    for file_found in os.listdir(_price_path):
        if file_found.find(_combined_filename) == -1:
            file_ticker, file_year, file_month = _parse_filename(file_found)
            with open(_price_path + file_found, 'rt') as f:
                current_data = pd.read_csv(f)
                df_dict[file_ticker] = pd.concat([current_data, df_dict[file_ticker]])
    return df_dict


# TODO: Create a method to combine and store all prices for speed later on...
@timing
def get_all_prices():
    """ This gets every single price for all securities in the universe - warning this could take some time... """
    ttl_data = pd.DataFrame()
    file_list = [_price_path + a_file for a_file in os.listdir(_price_path)]
    latest_file = max(file_list, key=os.path.getmtime)
    print('latest file found: {}'.format(latest_file))
    if latest_file.find(_combined_filename) > -1:
        print('Reading cached file: {}'.format(_combined_filename))
        with open(_price_path + _combined_filename, 'rt') as f:
            all_data = pd.read_csv(f, index_col=0)
            return all_data
    print('Reading raw price files...')
    counter = 0
    total_count = len(file_list)
    for file_found in file_list:
        if (file_found != _price_path + _combined_filename) & file_found.endswith('.csv'):
            if counter % int(total_count/20) == 0:
                print("   {0:.0f}% done...".format((counter/total_count)*100))
            counter += 1
            with open(file_found, 'rt') as f:
                current_data = pd.read_csv(f)
                ttl_data = pd.concat([current_data, ttl_data])

    # process munged data
    # ttl_data['date'] = [index_date.strftime('%Y-%m-%d') for index_date in ttl_data.index]
    ttl_data.sort_values('date', inplace=True)
    ttl_data.reset_index(drop=True, inplace=True)

    # Save the file...
    with open(_price_path + _combined_filename, 'wt') as f:
        f.write(ttl_data.to_csv())
    return ttl_data


def _create_filename(ticker, year, month, extension=".csv"):
    # ticker may have odd symbols that cannot be in a filename like a forward slash
    ticker = ticker.replace('/', '+')
    return (_price_path + ticker + ".{:04d}.{:02d}" + extension).format(year, month)


def _parse_filename(filename):
    pieces = filename.split('.')
    ticker = pieces[0].replace('+', '/')
    year = int(pieces[1])
    month = int(pieces[2])
    return ticker, year, month


def _get_day_range_for_month(year, month):
    first_day = datetime.date(year, month, 1)
    last_day = first_day.replace(day=calendar.monthrange(first_day.year, first_day.month)[1])
    return first_day, last_day


def _any_ticker_files(ticker):
    """ Check to see if there are any ticker files already saved to disk.
    This saves us from having to go to the source and reload data. """
    # Generate a list of all current price files...
    file_list = []
    for file_found in os.listdir(_price_path):
        file_list.append(file_found)

    last_date = datetime.date(1900, 1, 1)
    old_date = last_date
    file_timestamp = datetime.datetime(1900, 1, 1)
    for file_found in file_list:
        file_timestamp = max(file_timestamp,
                             datetime.datetime.fromtimestamp(os.path.getmtime(_price_path + file_found)))
        if (file_found.find(_combined_filename) == -1) & file_found.endswith('.csv'):
            file_ticker, file_year, file_month = _parse_filename(file_found)
            if file_ticker == ticker:
                file_date = datetime.date(file_year, file_month, 1)
                last_date = max([last_date, file_date])
    if old_date == last_date:
        return False, last_date, file_timestamp
    return True, last_date, file_timestamp


def get_all_fundamental_data():
    """ This gets every single price for all securities in the universe - warning this could take some time... """
    ttl_data = pd.DataFrame()
    fundamental_file_list = [_fundamental_path + a_file for a_file in os.listdir(_fundamental_path)]
    latest_file = max(fundamental_file_list, key=os.path.getmtime)
    print('latest file found: {}'.format(latest_file))
    if latest_file.find(_combined_filename) > -1:
        print('Reading cached file: {}'.format(_combined_filename))
        with open(_fundamental_path + _combined_filename, 'rt') as f:
            all_data = pd.read_csv(f, index_col=0)
            return all_data
    print('Reading raw fundamental files...')
    counter = 0
    total_count = len(fundamental_file_list)
    for file_found in fundamental_file_list:
        if (file_found != _fundamental_path + _combined_filename) & file_found.endswith('.csv'):
            if counter % int(total_count / 10) == 0:
                print("   {0:.0f}% done...".format((counter / total_count) * 100))
            counter += 1
            with open(file_found, 'rt') as f:
                current_data = pd.read_csv(f, index_col=0)
                ttl_data = pd.concat([current_data, ttl_data])

    # process munged data
    # ttl_data['date'] = [index_date.strftime('%Y-%m-%d') for index_date in ttl_data.index]
    ttl_data = ttl_data.rename(columns={'quarter end': 'date'})
    ttl_data.sort_values('date', inplace=True)
    ttl_data.reset_index(drop=True, inplace=True)

    # Save the file...
    with open(_fundamental_path + _combined_filename, 'wt') as f:
        f.write(ttl_data.to_csv())
    return ttl_data


def update_all_fundamental_data():
    print("Updating Fundamental Data...")
    snp_list = _get_tickerlist(remove_wiki=True)
    fundamental_file_list = []
    for fundamental_file_found in os.listdir(_fundamental_path):
        fundamental_file_list.append(fundamental_file_found)
    for ticker in snp_list:
        update_fundamental_data(ticker, fundamental_file_list)


def update_fundamental_data(ticker, fundamental_file_list):
    try:
        ticker_filename = ticker + '.csv'
        ticker_exists = len([c for c in fundamental_file_list if c == ticker_filename])
        if ticker_exists == 1:
            print("Found {}".format(ticker_filename))
        else:
            print("Getting fundamental data for {}".format(ticker))
            url = "http://www.stockpup.com/data/" + ticker + "_quarterly_financial_data.csv"
            s = requests.get(url).content
            csv_df = pd.read_csv(io.StringIO(s.decode('utf-8')), error_bad_lines=False)
            if (len(csv_df.columns)) == 1:
                raise TypeError("got HTML instead of CSV")
            csv_df['ticker'] = pd.Series(_wiki_prefix + ticker, index=csv_df.index)
            csv_df.columns = [x.lower() for x in csv_df.columns]
            # Clean the crappy data - where there are 'None' Strings, try to stale data
            csv_df = csv_df.replace('None', np.nan)
            csv_df = csv_df.fillna(method='backfill')
            csv_df = csv_df.dropna(axis=0)
            if csv_df.shape[0] < 2:
                raise ValueError("Not enough data in CSV to use.")
            # remove unnecessary data
            csv_df.drop(['shares', 'split factor', 'assets', 'current assets', 'liabilities', 'current liabilities',
                         'shareholders equity', 'non-controlling interest', 'preferred equity',
                         'goodwill & intangibles', 'long-term debt', 'earnings available for common stockholders',
                         'eps diluted', 'dividend per share', 'cash from operating activities',
                         'cash from investing activities', 'cash from financing activities',
                         'cash change during period', 'capital expenditures', 'price', 'price high', 'price low',
                         'roa', 'p/b ratio', 'p/e ratio', 'cumulative dividends per share', 'dividend payout ratio',
                         'long-term debt to equity ratio', 'equity to assets ratio', 'asset turnover',
                         'free cash flow per share', 'current ratio'], axis=1, inplace=True)

            with open(_fundamental_path + ticker_filename, 'wt') as f:
                f.write(csv_df.to_csv())
    except (ValueError, RuntimeError, NameError, TypeError) as err:
        print("failed getting fundamental for {} - {}".format(ticker, err))
        pass


if __name__ == '__main__':
    # update_all_fundamental_data()
    get_all_fundamental_data()
    # create_universe_from_json()
    api_key = ""
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        print("Please paste in your quandl api key:")
        api_key = sys.stdin.readline().replace('\n', '')
    quandl.ApiConfig.api_key = api_key
    update_all_price_caches()
    fin_data = get_all_prices()
    print(fin_data.describe())
    print('Total number of rows: {}'.format(len(fin_data)))
    print('Got data for the following tickers:')
    print({t for t in fin_data['ticker']})
