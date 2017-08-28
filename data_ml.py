import os
import pandas as pd
import numpy as np
import data_universe as du
from utils import timing


_label_dir = "\\data\\labels\\"
_feature_dir = "\\data\\features\\"
_combined_filename = "special.csv"
_cwd = os.getcwd()
_feature_path = _cwd + _feature_dir
_label_path = _cwd + _label_dir
_business_days_in_a_year = 252  # according to NYSE
_forecast_days = 15  # numbers of days in the future to train on
_forecast_buy_threshold = 3  # train for positive results above this percent return
_forecast_sell_threshold = -3  # train for positive results below this percent return
_forecast_slope = 0.2  # the steep climb from 0 to 1 as x approaches the threshold precentage


# ensure paths are there...
if not os.path.exists(_feature_path):
    os.makedirs(_feature_path)

if not os.path.exists(_label_path):
    os.makedirs(_label_path)


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

    main_df = du.get_all_prices()

    ticker_set = {t for t in main_df['ticker']}
    ticker_count = len(ticker_set)
    tickers = iter(ticker_set)
    counter = 0

    # for each ticker, sort and process calculated data for ml
    while True:
        counter += 1
        ticker = next(tickers)
        if ticker is None:
            break
        sub_df = main_df[main_df.ticker == ticker]
        sub_df = sub_df.sort_values(by='date')
        print('ticker: {} - rows: {}'.format(ticker, len(sub_df)))
        start_date = sub_df.head(1)['date'].iloc[0]
        end_date = sub_df.tail(1)['date'].iloc[0]
        print('   date_range: {} - {}'.format(start_date, end_date))
        percent_done = counter / ticker_count
        yield ticker, sub_df, percent_done


def _get_aggregated_data(a_path):
    ttl_data = pd.DataFrame()
    file_list = [a_path + a_file for a_file in os.listdir(a_path)]
    latest_file = ""
    if len(file_list) > 0:
        latest_file = max(file_list, key=os.path.getmtime)
    if latest_file.find(_combined_filename) > -1:
        print('Reading cached file: {}'.format(_combined_filename))
        with open(a_path + _combined_filename, 'rt') as f:
            all_data = pd.read_csv(f)
            # convert datetime column
            all_data['date'].apply(pd.to_datetime)  # TODO: THIS returns the modified column.. test the fix
            return all_data
    print('latest file found: {}'.format(latest_file))
    print('Reading raw price files...')
    for file_found in file_list:
        if (file_found != a_path + _combined_filename) & file_found.endswith('.csv'):
            with open(file_found, 'rt') as f:
                current_data = pd.read_csv(f)
                if current_data['date'].dtype == np.int64:
                    print("file: {} - type: {} ".format(file_found, current_data['date'].dtype))
                ttl_data = pd.concat([current_data, ttl_data])

    # convert datetime column
    ttl_data['date'].apply(pd.to_datetime)  # TODO: THIS returns the modified column.. test the fix

    # process munged data
    ttl_data.reset_index(drop=True, inplace=True)

    with open(a_path + _combined_filename, 'wt', encoding='utf-8') as f:
        f.write(ttl_data.to_csv())
    return ttl_data


@timing
def get_all_feature_data():
    """ Returns a dataframe with all calculated data for ml to consume """
    return _get_aggregated_data(_feature_path)


@timing
def get_all_label_data():
    """ Returns a dataframe with all label data for ml to consume """
    return _get_aggregated_data(_label_path)


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

                # Try shaping the label more smoothly
                buy_label = step(future_return, _forecast_buy_threshold, True)
                sell_label = step(future_return, _forecast_sell_threshold, False)

                label_row_values = [ticker, curr_date, buy_label, sell_label, future_return]
                new_df.loc[i] = label_row_values

                if i > (new_data_size - 2):
                    print('date: {} adj close: {:4.3f} - future return: {:3.2f}'.format(
                        curr_date, curr_close, future_return))
                    print('   buy_label: {:1.3f} sell_label: {:1.3f} '.format(buy_label, sell_label))
            # SAVE
            with open(_label_path + _get_calc_filename(ticker), 'wt', encoding='utf-8') as f:
                f.write(new_df.to_csv())


@timing
def calc_feature_data():
    """ Generates ml data by calculating specific values off of daily prices """
    # for each ticker, sort and process calculated data for ml
    for ticker, sub_df, percent_done in ticker_data():

        # add EMA to the ticker data
        ema_12_column_name = "ema_12"
        ema_26_column_name = "ema_26"
        ema_12 = pd.ewma(sub_df['adj. close'], span=12)
        ema_26 = pd.ewma(sub_df['adj. close'], span=26)
        sub_df[ema_12_column_name] = ema_12
        sub_df[ema_26_column_name] = ema_26

        # Let people know how long this might take...
        if int(percent_done*100) % 5 == 0:
            print("   --- {0:0.0f}% DONE ---".format(percent_done * 100))

        # check if we have enough history to calc year high / low
        if len(sub_df) <= _business_days_in_a_year:
            print('ticker {} does not have enough data to calc year high'.format(ticker))
        else:
            # count the amount of year ranges available in the dataframe - eq. len(df) 253 means 2 ranges of 252
            new_size = len(sub_df) - _business_days_in_a_year + 1
            new_df = _create_feature_data_frame(new_size)
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
                curr_date = curr_row['date']
                curr_close = curr_row['adj. close']

                curr_return = curr_close / prev_row['adj. close'] - 1
                year_return = curr_close / year_df.iloc[0]['adj. close'] - 1
                curr_return_open = curr_close / curr_row['adj. open'] - 1
                curr_return_high = curr_close / curr_row['adj. high'] - 1
                curr_return_low = curr_close / curr_row['adj. low'] - 1
                two_days_ago_return = curr_close / two_days_ago_row['adj. close'] - 1
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

                # print out end of series data for debugging
                if i > (new_size-3):
                    print('date: {} adj close: {:4.3f} - year high: {:3.2f} '
                          'macd: {:1.3f}'.format(curr_date,
                                                curr_close, curr_year_high_pct, macd))
                    print('   volume %: {:1.3f} volume stddev: {:1.3f} '.format(volume_percent, volume_deviation))
                    print('   9 day rtn: {:1.3f} stddev 60: {:1.3f} '
                          'stddev yr: {:1.3f} MA 60 day: {:1.3f}'.format(return_9_day, stddev_60,
                                                                          stddev_year, ma_60_day))

                new_values = [ticker, curr_date, curr_return, year_return, volume_percent, volume_deviation,
                              return_60_day, ma_30_day, ma_60_day, macd,
                              curr_year_high_pct, stddev_30, stddev_60, stddev_year]

                new_df.loc[i] = new_values

            # Now normalize-ish the data by clipping it to acceptable ranges for ML
            for col in get_feature_columns():
                new_df[col] = np.clip(new_df[col], -1., 1.)

            with open(_feature_path + _get_calc_filename(ticker), 'wt', encoding='utf-8') as f:
                f.write(new_df.to_csv())


def _get_calc_filename(ticker, extension=".csv"):
    # ticker may have odd symbols that cannot be in a filename like a forward slash
    ticker = ticker.replace('/', '+')
    return '{}{}'.format(ticker, extension)


def _create_training_data_frame(df_size):
    df_label = pd.DataFrame(index=range(df_size), columns=('ticker', 'date',
                                                           'buy_label', 'sell_label', 'future_return'))
    return df_label


def get_descriptive_columns():
    return ['ticker', 'date']


def get_label_columns():
    return ['buy_label', 'sell_label']


def get_feature_columns():
    return ['curr_return', 'year_return', 'volume_percent', 'volume_deviation',
            'return_60_day', 'ma_30_day', 'ma_60_day', 'macd',
            'year_high_percent', 'stddev_30_day', 'stddev_60_day', 'stddev_year']


def _get_feature_dataframe_columns():
    return get_descriptive_columns() + get_feature_columns()


def _create_feature_data_frame(df_size):
    # TODO: use column listings above instead of duplicating strings
    df_features = pd.DataFrame(index=range(df_size),
                               columns=(_get_feature_dataframe_columns()))
    return df_features


def calc_all():
    calc_feature_data()
    calc_label_data()
    df = get_all_feature_data()
    print('FEATURE DATA {} rows.'.format(len(df)))
    print(df.describe())
    df = get_all_label_data()
    print('LABEL DATA {} rows.'.format(len(df)))
    print(df.describe())
    df = get_all_ml_data()
    print('COMBINED DATA {} rows.'.format(len(df)))
    print(df.describe())
    print(df.tail())


if __name__ == '__main__':
    calc_all()
