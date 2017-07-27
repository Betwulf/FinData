import sys
import os
import datetime
import calendar
import pandas as pd
import quandl
from utils import timing

_data_dir = "\\data\\"
_price_dir = "\\data\\prices\\"
_combined_price_filename = "__all.json"

_cwd = os.getcwd()
_data_path = _cwd + _data_dir
_price_path = _cwd + _price_dir

# ensure paths are there...
if not os.path.exists(_data_path):
    os.makedirs(_data_path)
if not os.path.exists(_price_path):
    os.makedirs(_price_path)

# Generate a list of all current price files...
file_list = []
for file_found in os.listdir(_price_path):
    file_list.append(file_found)


def create_universe_from_json():
    """Creates the list file of tickers to use in the rest of the app provided a given source"""
    # snp_list = []

    with open('S&P500.json', 'rb') as f:
        json = pd.read_json(f)
        snp_list = ['WIKI/' + x['ticker'] for x in json['constituents']]

    snp_list.sort()
    with open(_data_path + 'universe.txt', 'wt', encoding='utf-8') as f:
        for x in snp_list:
            f.write(x + '\n')


def update_all_price_caches():
    """ Go through each ticker in the universe and either get beginning data or update latest data """

    # TODO: only working with the first 3 tickers while debugging
    snp_list = _get_tickerlist()
    for ticker in snp_list:
        update_price_cache(ticker)


def _get_tickerlist():
    # snp_list = []
    with open(_data_path + 'universe.txt', 'rt', encoding='utf-8') as f:
        snp_list = [x.strip('\n') for x in f]
    print("Got tickers... Top 200: ")
    print(snp_list[:200])
    return snp_list[:200]


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
        # Add ticker to the dataframe
        fin_data['ticker'] = [ticker for _ in range(len(fin_data[fin_data.columns[0]]))]
        for (year, month) in unique_months:
            month_first, month_last = _get_day_range_for_month(year, month)
            sub_data = fin_data[month_first:month_last]
            # not sure if this date formatting will be reversible on load...
            sub_data.index = [x.strftime('%Y-%m-%d') for x in sub_data.index]
            # Don't want data before 1970 - timestamps don't work before then...
            # But really... I don't want anything older than 1998 i reckon... vastly different trading behaviors
            if year > 1998:
                with open(_create_filename(ticker, year, month), 'wt') as f:
                        f.write(sub_data.to_json())


# TODO: Do we need this, or just use __all.json ?
def get_ticker_prices():
    """ returns a dictionary of tickers to their 'pandas' dataframes of prices"""
    snp_list = _get_tickerlist()
    df_dict = {t: pd.DataFrame() for t in snp_list}
    for file_found in os.listdir(_price_path):
        if file_found.find(_combined_price_filename) == -1:
            file_ticker, file_year, file_month = _parse_filename(file_found)
            with open(_price_path + file_found, 'rt') as f:
                current_data = pd.read_json(f)
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
    if latest_file.find(_combined_price_filename) > -1:
        print('Reading cached file: {}'.format(_combined_price_filename))
        with open(_price_path + _combined_price_filename, 'rt') as f:
            all_data = pd.read_json(f)
            return all_data
    print('Reading raw price files...')
    counter = 0
    total_count = len(file_list)
    for file_found in file_list:
        if (file_found != _price_path + _combined_price_filename) & file_found.endswith('.json'):
            if counter % int(total_count/20) == 0:
                print("   {0:.0f}% done...".format((counter/total_count)*100))
            counter += 1
            with open(file_found, 'rt') as f:
                current_data = pd.read_json(f)
                ttl_data = pd.concat([current_data, ttl_data])

    # process munged data
    ttl_data['date'] = [index_date.strftime('%Y-%m-%d') for index_date in ttl_data.index]
    ttl_data.sort_values('date', inplace=True)
    ttl_data.reset_index(drop=True, inplace=True)

    # Save the file...
    with open(_price_path + _combined_price_filename, 'wt') as f:
        f.write(ttl_data.to_json())
    return ttl_data


def _create_filename(ticker, year, month, extension=".json"):
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
    last_date = datetime.date(1900, 1, 1)
    old_date = last_date
    file_timestamp = datetime.datetime(1900, 1, 1)
    for file_found in file_list:
        file_timestamp = max(file_timestamp,
                             datetime.datetime.fromtimestamp(os.path.getmtime(_price_path + file_found)))
        if (file_found.find(_combined_price_filename) == -1) & file_found.endswith('.json'):
            file_ticker, file_year, file_month = _parse_filename(file_found)
            if file_ticker == ticker:
                file_date = datetime.date(file_year, file_month, 1)
                last_date = max([last_date, file_date])
    if old_date == last_date:
        return False, last_date, file_timestamp
    return True, last_date, file_timestamp


if __name__ == '__main__':
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
