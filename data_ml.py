import os
import pandas as pd
import numpy as np
import data_universe as du
import datetime
from utils import timing


_label_dir = "/data/labels/"
_feature_dir = "/data/features/"
_combined_features_filename = "features.csv"
_combined_labels_filename = "labels.csv"
_cwd = "C:/Temp/"
_feature_path = _cwd + _feature_dir
_label_path = _cwd + _label_dir
_business_days_in_a_year = 252  # according to NYSE
_forecast_days = 31  # numbers of days in the future to train on
_forecast_buy_threshold = 2  # train for positive results above this percent return
_forecast_sell_threshold = -2  # train for positive results below this percent return
_forecast_slope = 0.2  # the steep climb from 0 to 1 as x approaches the threshold precentage


# ensure paths are there...
if not os.path.exists(_feature_path):
    os.makedirs(_feature_path)

if not os.path.exists(_label_path):
    os.makedirs(_label_path)


def norm(data_list, min_val=0.5, max_val=1.5):
    return [(v - min_val) / (max_val - min_val) for v in data_list]


def min_max_scale(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def adjusted_double_sigmoid(x, target_value, slope):
    """ This function creates a 2 step smooth transition. Where x = -target_value, y = -1, then as x goes to 0
    y is zero, and as x approaches target_value, y approaches 1.
    Slope controls how quick the transitions are. """
    if x < -50:
        x = -50  # prevents overflow in exp.
    return (1 / (1 + np.exp((-4.0 / slope) * (x - target_value)))) + \
           (1 / (1 + np.exp((-4.0 / slope) * (x + target_value)))) - 1


def step(x, target_value, greater_than):
    if greater_than:
        return (0, 1)[x >= target_value]
    return (0, 1)[x <= target_value]


def sigmoid(x, target_value, slope):
    if x < -50:
        x = -50  # prevents overflow in exp.
    return 1 / (1 + np.exp((-4.0 / slope) * (x - target_value)))


def ticker_data():
    """ Iterator to get the next ticker and its corresponding data_frame of prices """

    prices_df = du.get_all_prices()
    fundamental_df = du.get_all_fundamental_data()
    print(" --- FUNDAMENTAL --- ")
    fundamental_df.describe()

    ticker_set = {t for t in prices_df['ticker']}
    fundamental_ticker_set = {t for t in fundamental_df['ticker']}
    ticker_set = [val for val in fundamental_ticker_set if val in ticker_set]
    ticker_count = len(ticker_set)
    tickers = iter(ticker_set)
    counter = 0

    # for each ticker, sort and process calculated data for ml
    sub_df = None
    while sub_df is None:
        counter += 1
        ticker = next(tickers)
        if ticker is None:
            break
        sub_df = prices_df[prices_df.ticker == ticker]
        sub_df = pd.merge(sub_df, fundamental_df, how='left', on=['date', 'ticker'])
        sub_df = sub_df.sort_values(by='date')
        sub_df = sub_df.fillna(method='pad')
        sub_df = sub_df[np.isfinite(sub_df['roe'])]
        if len(sub_df) < 2:
            sub_df = None
            print("--- CANNOT merge data for ticker {}".format(ticker))
        else:
            print('ticker: {} - rows: {}'.format(ticker, len(sub_df)))
            start_date = sub_df.head(1)['date'].iloc[0]
            end_date = sub_df.tail(1)['date'].iloc[0]
            print('   date_range: {} - {}'.format(start_date, end_date))
            percent_done = counter / ticker_count
            yield ticker, sub_df, percent_done
            sub_df = None


def _get_aggregated_data(a_path, a_filename):
    ttl_data = pd.DataFrame()
    file_list = [a_path + a_file for a_file in os.listdir(a_path)]
    latest_file = ""
    if len(file_list) > 0:
        latest_file = max(file_list, key=os.path.getmtime)
    if latest_file.find(a_filename) > -1:
        print('Reading cached file: {}'.format(a_filename))
        with open(a_path + a_filename, 'rt') as f:
            all_data = pd.read_csv(f, index_col=0)
            # convert datetime column
            all_data['date'].apply(pd.to_datetime)  # TODO: THIS returns the modified column.. test the fix
            return all_data
    print('latest file found: {}'.format(latest_file))
    print('Reading raw files... count: {}'.format(len(file_list)))
    if len(file_list) == 0:
        print("NO FILES FOUND IN: {}".format(a_path))
        raise ValueError("NO FILES FOUND IN: {}".format(a_path))
    for file_found in file_list:
        if (file_found != a_path + a_filename) & file_found.endswith('.csv'):
            with open(file_found, 'rt') as f:
                current_data = pd.read_csv(f, index_col=0)
                if current_data['date'].dtype == np.int64:
                    print("file: {} - type: {} ".format(file_found, current_data['date'].dtype))
                ttl_data = pd.concat([current_data, ttl_data])

    # convert datetime column
    ttl_data['date'].apply(pd.to_datetime)  # TODO: THIS returns the modified column.. test the fix

    # process munged data
    ttl_data.reset_index(drop=True, inplace=True)

    with open(a_path + a_filename, 'wt', encoding='utf-8') as f:
        f.write(ttl_data.to_csv())
    return ttl_data


@timing
def get_all_feature_data():
    """ Returns a dataframe with all calculated data for ml to consume """
    return _get_aggregated_data(_feature_path, _combined_features_filename)


@timing
def get_all_label_data():
    """ Returns a dataframe with all label data for ml to consume """
    return _get_aggregated_data(_label_path, _combined_labels_filename)


def get_all_ml_data():
    df_feature = get_all_feature_data()
    df_label = get_all_label_data()
    df_merged = pd.merge(df_feature, df_label, how='inner', on=['date', 'ticker'])
    if df_merged.empty:
        print('merge created empty dataframe.')
    df_merged.sort_values('date', inplace=True)
    df_merged.reset_index(drop=True, inplace=True)
    df_merged['date'] = pd.to_datetime(df_merged['date'])
    return df_merged


def get_all_predictable_data():
    df_feature = get_all_feature_data()
    df_merged = df_feature.assign(future_return_label=list(range(len(df_feature))))
    df_merged.sort_values('date', inplace=True)
    df_merged.reset_index(drop=True, inplace=True)
    df_merged['date'] = pd.to_datetime(df_merged['date'])
    return df_merged

@timing
def calc_label_data():
    """ Generates ml label data by calculating future returns off of daily prices """
    # for each ticker, sort and process label data for ml training
    for ticker, sub_df, percent_done in ticker_data():

        # Let people know how long this might take...
        if int(percent_done*100) % 5 == 0:
            print("   {0:.0f}% done...".format(percent_done * 100))

        # Check data size....
        if len(sub_df) < _forecast_days:
            print("{} does not have enough data to forecast training data".format(ticker))
        else:
            new_data_size = len(sub_df) - _forecast_days
            new_df = _create_training_data_frame(new_data_size)
            # for each forecast-able training data
            for i in range(new_data_size):
                curr_close = sub_df['adj. close'].iloc[i]
                curr_date = sub_df['date'].iloc[i]
                future_close = sub_df['adj. close'].iloc[i+_forecast_days]
                future_return = (future_close/curr_close - 1)*100

                future_return_label = future_close/curr_close

                # Try shaping the label more smoothly
                buy_label = step(future_return, _forecast_buy_threshold, True)
                sell_label = step(future_return, _forecast_sell_threshold, False)

                label_row_values = [ticker, curr_date, future_return_label, future_return]
                new_df.loc[i] = label_row_values

                if i > (new_data_size - 2):
                    print('date: {} adj close: {:4.3f} - future return: {:3.2f}'.format(
                        curr_date, curr_close, future_return))
                    print('   buy_label: {:1.3f} sell_label: {:1.3f} '.format(buy_label, sell_label))
            # Now normalize-ish the data by clipping it to acceptable ranges for ML
            for col in get_label_columns():
                new_df[col] = norm(new_df[col])
                new_df[col] = np.clip(new_df[col], 0., 1.)
            new_df.dropna(how='any')
            # SAVE
            with open(_label_path + _get_calc_filename(ticker), 'wt', encoding='utf-8') as f:
                f.write(new_df.to_csv())


def get_sector_features(sectors_df):
    sector_list = sectors_df.sector.unique().tolist()
    try:
        sector_list.remove('Failed')
    except:
        pass
    sector_list.sort()
    sector_dict = {k: i for i, k in enumerate(sector_list)}
    return sector_list, sector_dict


@timing
def calc_feature_data():
    """ Generates ml data by calculating specific values off of daily prices """
    print(" --------- CALCULATE FEATURE DATA ------------------")
    sectors_df = du.get_sectors()
    sector_list, sector_dict = get_sector_features(sectors_df)
    # for each ticker, sort and process calculated data for ml
    for ticker, sub_df, percent_done in ticker_data():
        try:
            # add EMA to the ticker data
            sectors_features = np.zeros(len(sector_list))
            ticker_sector_str = sectors_df[sectors_df.ticker == ticker.replace(du.wiki_prefix, '')].sector.tolist()[0]
            sectors_features[sector_dict[ticker_sector_str]] = 1

            ema_12_column_name = "ema_12"
            ema_26_column_name = "ema_26"
            ema_12 = sub_df['adj. close'].ewm(span=12, adjust=True, min_periods=0, ignore_na=False).mean()
            ema_26 = sub_df['adj. close'].ewm(span=26, adjust=True, min_periods=0, ignore_na=False).mean()
            sub_df[ema_12_column_name] = ema_12
            sub_df[ema_26_column_name] = ema_26

            # Let people know how long this might take...
            if int(percent_done*100) % 5 == 0:
                print("   --- {0:0.0f}% DONE ---".format(percent_done * 100))

            # check if we have enough history        to calc year high / low
            if len(sub_df) <= _business_days_in_a_year:
                print('ticker {} does not have enough data to calc year high'.format(ticker))
            else:
                # count the amount of year ranges available in the dataframe - eq. len(df) 253 means 2 ranges of 252
                new_size = len(sub_df) - _business_days_in_a_year + 1
                new_df = _create_feature_data_frame(new_size)
                curr_date = sub_df.iloc[_business_days_in_a_year]['date']
                for i in range(new_size):
                    start_loc = i
                    end_loc = _business_days_in_a_year + i
                    year_df = sub_df.iloc[start_loc:end_loc]

                    year_adj_close = np.asarray(year_df['adj. close'][-_business_days_in_a_year:])
                    year_low = min(year_adj_close)
                    year_high = max(year_adj_close)
                    curr_row = year_df.iloc[-1]
                    prev_row = year_df.iloc[-2]
                    two_days_ago_row = year_df.iloc[-3]
                    last_date = curr_date
                    curr_date = curr_row['date']
                    curr_close = curr_row['adj. close']

                    # Date Check
                    actual_curr_date = datetime.datetime.strptime(curr_date, '%Y-%m-%d')
                    actual_last_date = datetime.datetime.strptime(last_date, '%Y-%m-%d')
                    if (actual_curr_date - actual_last_date).days > 10:
                        raise Exception(f"{ticker} has gaps in their time series - {curr_date}")

                    # Fundamental Measures here - remember to unitize
                    roe = curr_row['roe']
                    eps = curr_row['eps basic'] / curr_close
                    net_margin = curr_row['net margin']
                    bv = curr_row['book value'] * curr_row['shares']
                    z_score = ((1.2*curr_row['cash'] + 1.4*curr_row['earnings'] + 3.3*curr_row['revenue'])/bv)/5.0
                    # for PE Ratio i really should be using the total of the past 4 quarters earnings, but can't get that
                    # reliably, as some rows may have been deleted due to missing data
                    pe_ratio = (curr_close / (float(curr_row['earnings']) / float(curr_row['shares']))) / 150.0
                    # for  PB ratio, the value can get really big, like over 30, so divide by 20 to capture val
                    pb_ratio = (curr_close / curr_row['book value']) / 20.0
                    rpsop = (curr_row['revenue'] / curr_row['shares']) / curr_close

                    return_1d = curr_close / prev_row['adj. close'] - 1
                    year_return = curr_close / year_df.iloc[0]['adj. close'] - 1
                    curr_return_open = curr_close / curr_row['adj. open'] - 1
                    curr_return_high = curr_close / curr_row['adj. high'] - 1
                    curr_return_low = curr_close / curr_row['adj. low'] - 1
                    day_returns = [curr_close/x - 1 for x in year_df.iloc[-31:-1]['adj. close']]
                    month_returns = [curr_close/x - 1 for x in year_df.iloc[-253:-1:23]['adj. close']]
                    return_4d = curr_close / year_df.iloc[-5]['adj. close'] - 1  # DEBUG: REMOVE THIS LATER
                    if not day_returns[-4] == return_4d:
                        print("RETURN CALC ERROR")
                        raise ValueError("RETURN CALC ERROR")
                    curr_year_high_pct = (curr_close - year_low) / (year_high - year_low)
                    curr_year_low_pct = (year_high - curr_close) / (year_high - year_low)

                    # now calc the multi day stats
                    prev_row = year_df.iloc[-10]  # prev 9 day
                    return_9_day = curr_close / prev_row['adj. close'] - 1
                    ma_9_day = np.asarray(year_df['adj. close'][-10:]).mean() / curr_close - 1
                    past_volumes = np.asarray(year_df['adj. volume'][-10:])
                    volume_high = past_volumes.max()
                    volume_low = past_volumes.min()
                    volume_percent = (curr_row['adj. volume'] - volume_low) / (volume_high - volume_low)
                    volume_average = year_df['adj. volume'].values.mean()
                    volume_stddev = year_df['adj. volume'].values.std(ddof=1)
                    # want to capture up to 3 std deviations off of the average
                    volume_deviation = ((curr_row['adj. volume'] - volume_average) / volume_stddev)/3

                    prev_row = year_df.iloc[-16]
                    return_15_day = curr_close / prev_row['adj. close'] - 1
                    ma_15_day = np.asarray(year_df['adj. close'][-16:]).mean() / curr_close - 1

                    prev_row = year_df.iloc[-31]
                    return_30_day = curr_close / prev_row['adj. close'] - 1
                    array_30 = np.asarray(year_df['adj. close'][-31:])
                    ma_30_day = array_30.mean() / curr_close - 1
                    stddev_30 = array_30.std(ddof=1) / curr_close

                    prev_row = year_df.iloc[-61]
                    return_60_day = curr_close / prev_row['adj. close'] - 1
                    array_60 = np.asarray(year_df['adj. close'][-61:])
                    ma_60_day = array_60.mean() / curr_close - 1
                    stddev_60 = array_60.std(ddof=1) / curr_close

                    stddev_year = np.asarray(year_df['adj. close']).std(ddof=1) / curr_close
                    macd_12 = np.asarray(year_df[ema_12_column_name][-9:])
                    macd_26 = np.asarray(year_df[ema_26_column_name][-9:])
                    macd_diff = macd_12 - macd_26
                    macd = (macd_diff[-1] - macd_diff.mean()) / curr_close

                    new_values = [ticker, curr_date, return_1d, day_returns[-2], day_returns[-3], day_returns[-4],
                                 day_returns[-5], day_returns[-6], day_returns[-7], day_returns[-8], day_returns[-9],
                                 day_returns[-10], day_returns[-11], day_returns[-12], day_returns[-13], day_returns[-14],
                                 day_returns[-15], day_returns[-16], day_returns[-17], day_returns[-18], day_returns[-19],
                                 day_returns[-20], day_returns[-21], day_returns[-22], day_returns[-23], day_returns[-24],
                                 day_returns[-25], day_returns[-26], day_returns[-27], day_returns[-28], day_returns[-29],
                                 day_returns[-30], month_returns[-2], month_returns[-3], month_returns[-4],
                                  month_returns[-5], month_returns[-6], month_returns[-7], month_returns[-8],
                                  month_returns[-9], month_returns[-10], month_returns[-11], year_return,
                                  volume_percent, volume_deviation, curr_year_high_pct, stddev_30, stddev_60, stddev_year,
                                  net_margin, z_score, *sectors_features]

                    new_df.loc[i] = new_values

                # Now normalize-ish the data by clipping it to acceptable ranges for ML
                for col in get_feature_columns():
                    new_df[col] = np.clip(new_df[col], -1., 1.)

                # drop any rows with nan
                old_row_count = new_df.shape
                new_df.dropna(how='any')
                new_row_count = new_df.shape
                if new_row_count < old_row_count:
                    print("Features removed - from {} to {}".format(old_row_count, new_row_count))

                with open(_feature_path + _get_calc_filename(ticker), 'wt', encoding='utf-8') as f:
                    f.write(new_df.to_csv())
        except Exception as ex:
            print("error with {} - {}".format(ticker, ex))


def _get_calc_filename(ticker, extension=".csv"):
    # ticker may have odd symbols that cannot be in a filename like a forward slash
    ticker = ticker.replace('/', '+')
    return '{}{}'.format(ticker, extension)


def _create_feature_data_frame(df_size):
    df_features = pd.DataFrame(index=range(df_size),
                               columns=(_get_feature_dataframe_columns()))
    return df_features


def _create_training_data_frame(df_size):
    df_label = pd.DataFrame(index=range(df_size), columns=(_get_label_dataframe_columns()))
    return df_label


def get_descriptive_columns():
    return ['ticker', 'date']


def get_label_columns():
    return ['future_return_label']


def get_feature_columns():
    sectors_df = du.get_sectors()
    sector_list, sector_dict = get_sector_features(sectors_df)
    return ['return_1d', 'return_2d', 'return_3d', 'return_4d', 'return_5d', 'return_6d',
            'return_7d', 'return_8d', 'return_9d', 'return_10d', 'return_11d', 'return_12d', 'return_13d',
            'return_14d', 'return_15d', 'return_16d', 'return_17d', 'return_18d', 'return_19d', 'return_20d',
            'return_21d', 'return_22d', 'return_23d', 'return_24d', 'return_25d', 'return_26d', 'return_27d',
            'return_28d', 'return_29d', 'return_30d', 'return_2m', 'return_3m', 'return_4m',
            'return_5m', 'return_6m', 'return_7m', 'return_8m',
            'return_9m', 'return_10m', 'return_11m', 'year_return',
            'volume_percent', 'volume_deviation', 'year_high_percent', 'stddev_30_day', 'stddev_60_day', 'stddev_year',
            'net_margin', 'z_score', *sector_list]


def _get_feature_dataframe_columns():
    return get_descriptive_columns() + get_feature_columns()


def _get_label_dataframe_columns():
    return get_descriptive_columns() + get_label_columns() + ['future_return']


def calc_all():
    calc_feature_data()
    calc_label_data()
    df = get_all_feature_data()
    print('--------------------------------------------')
    print('FEATURE DATA {} rows.'.format(len(df)))
    print('--------------------------------------------')
    print(df.describe())
    print(df.tail())
    df = get_all_label_data()
    print('--------------------------------------------')
    print('LABEL DATA {} rows.'.format(len(df)))
    print('--------------------------------------------')
    print(df.describe())
    df = get_all_ml_data()
    print('--------------------------------------------')
    print('COMBINED DATA {} rows.'.format(len(df)))
    print('--------------------------------------------')
    print(df.describe())
    print(df.tail())


if __name__ == '__main__':
    calc_all()
