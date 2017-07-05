import os
import datetime
import calendar
import pandas as pd
import numpy as np
import data_universe as du

_label_dir = "\\data\\label\\"
_calced_dir = "\\data\\calced\\"
_cwd = os.getcwd()
_calced_path = _cwd + _calced_dir
_label_path = _cwd + _label_dir
_business_days_in_a_year = 252  # according to NYSE
_forecast_days = 10  # numbers of days in the future to train on
_forecast_threshold = 2  # train for positive results above/below this percent return
_forecast_slope = 0.3  # the steep climb from 0 to 1 as x approaches the threshold precentage


def adjusted_double_sigmoid(x, target_value, slope):
    # return 2*(1/(1 + np.exp((-4/adjust)*x))) - 1
    return (1 / (1 + np.exp((-4 / slope) * (x - target_value)))) + \
           (1 / (1 + np.exp((-4 / slope) * (x + target_value)))) - 1


def ticker_data():
    """ Iterator to get the next ticker and corresponding dataframe """
    if not os.path.exists(_calced_path):
        os.makedirs(_calced_path)

    if not os.path.exists(_label_path):
        os.makedirs(_label_path)

    df = du.get_all_prices()
    tickers = iter({t for t in df['ticker']})
    # for each ticker, sort and process calculated data for ml
    while True:
        ticker = next(tickers)
        if ticker is None:
            break
        sub_df = df[df.ticker == ticker]
        sub_df = sub_df.sort_values(by='date')
        print('ticker: {} - rows: {}'.format(ticker, len(sub_df)))
        start_date = sub_df.head(1)['date'].iloc[0]
        end_date = sub_df.tail(1)['date'].iloc[0]
        print('   date_range: {} - {}'.format(start_date, end_date))
        yield ticker, sub_df


def calc_training_data():
    # for each ticker, sort and process label data for ml training
    for ticker, sub_df in ticker_data():
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
                label = adjusted_double_sigmoid(future_return, _forecast_threshold, _forecast_slope)
                label_row_values = [ticker, curr_date, label, future_return]
                new_df.loc[i] = label_row_values
            with open(_label_path + _get_calc_filename(ticker, curr_date), 'wt') as f:
                f.write(new_df.to_json())
            with open(_label_path + _get_calc_filename(ticker, curr_date, extension='.csv'), 'wt') as f:
                f.write(new_df.to_csv())


def calc_ml_data():
    # for each ticker, sort and process calculated data for ml
    for ticker, sub_df in ticker_data():

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
                curr_date = curr_row['date']
                curr_close = curr_row['adj. close']

                curr_return = curr_close / prev_row['adj. close'] - 1
                curr_return_open = curr_close / curr_row['adj. open'] - 1
                curr_return_high = curr_close / curr_row['adj. high'] - 1
                curr_return_low = curr_close / curr_row['adj. low'] - 1
                curr_year_high_pct = (curr_close - year_low) / (year_high - year_low)
                curr_year_low_pct = (year_high - curr_close) / (year_high - year_low)

                # now calc the multi day stats
                prev_row = year_df.iloc[-10]  # prev 9 day
                return_9_day = curr_close / prev_row['adj. close'] - 1
                ma_9_day = np.asarray(year_df['adj. close'][-10:]).mean() / curr_close - 1

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


                # print out end of series data for debugging
                if i > (new_size-5):
                    print('date: {} adj close: {:4.3f} - year high percent: {:3.2f} '
                          'low: {:3.2f}'.format(curr_date,
                                                curr_close, curr_year_high_pct, curr_year_low_pct))
                    print('   high: {} low: {} '.format(year_high, year_low))
                    print('   9 day rtn: {} 15 day rtn: {} '
                          '30 day rtn: {} 60 day rtn: {}'.format(return_9_day, return_15_day,
                                                                 return_30_day, return_60_day))

                new_values = [ticker, curr_date, curr_return, curr_return_open, curr_return_high, curr_return_low,
                              return_9_day, return_15_day, return_30_day, return_60_day,
                              ma_9_day, ma_15_day, ma_30_day, ma_60_day,
                              curr_year_high_pct, curr_year_low_pct, stddev_30, stddev_60]
                new_df.loc[i] = new_values

            with open(_calced_path + _get_calc_filename(ticker, curr_date), 'wt') as f:
                f.write(new_df.to_json())
            with open(_calced_path + _get_calc_filename(ticker, curr_date, extension='.csv'), 'wt') as f:
                f.write(new_df.to_csv())


def _get_calc_filename(ticker, last_date, extension=".json"):
    # ticker may have odd symbols that cannot be in a filename like a forward slash
    if type(last_date) == str:
        last_date = datetime.datetime.strptime(last_date, '%Y-%m-%d')
    ticker = ticker.replace('/', '+')
    return (ticker + ".{:04d}.{:02d}" + extension).format(last_date.year, last_date.month)


def _create_training_data_frame(df_size):
    df = pd.DataFrame(index=range(df_size), columns=('ticker', 'date', 'label', 'return'))
    return df


def _create_feature_data_frame(df_size):
    df = pd.DataFrame(index=range(df_size),
                      columns=('ticker', 'date', 'return_daily', 'return_open', 'return_high', 'return_low',
                               'return_9_day', 'return_15_day', 'return_30_day', 'return_60_day',
                               'ma_9_day', 'ma_15_day', 'ma_30_day', 'ma_60_day',
                               'year_high_percent', 'year_low_percent', 'stddev_30_day', 'stddev_60_day'))
    return df


if __name__ == '__main__':
    # calc_ml_data()
    calc_training_data()