import os
import datetime
import calendar
import pandas as pd
import numpy as np
import data_universe as du


_calced_dir = "\\data\\calced\\"
_cwd = os.getcwd()
_calced_path = _cwd + _calced_dir
year_days = 252 # according to NYSE


def calc_ml_data():
    if not os.path.exists(_calced_path):
        os.makedirs(_calced_path)
    df = du.get_all_prices()
    tickers = {t for t in df['ticker']}

    # for each ticker, sort and process calculated data for ml
    for ticker in tickers:
        sub_df = df[df.ticker == ticker]
        sub_df = sub_df.sort_values(by='date')
        print('ticker: {} - rows: {}'.format(ticker, len(sub_df)))

        # print(sub_df.head())
        # check if we have enough history to calc year high / low
        start_date = sub_df.head(1)['date'].iloc[0]
        end_date = sub_df.tail(1)['date'].iloc[0]
        print('   date_range: {} - {}'.format(start_date, end_date))
        if len(sub_df) <= year_days:
            print('ticker {} does not have enough data to calc year high'.format(ticker))
        else:
            # count the amount of year ranges available in the dataframe - eq. len(df) 253 means 2 ranges of 252
            new_size = len(sub_df) - year_days + 1
            new_df = _create_data_frame(new_size)
            for i in range(new_size):
                start_loc = i
                end_loc = year_days + i
                year_df = sub_df.iloc[start_loc:end_loc]
                # print('i = {} startloc: {} endloc: {}'.format(i, start_loc, end_loc))
                start_date = year_df.head(1)['date'].iloc[0]
                end_date = year_df.tail(1)['date'].iloc[0]
                # print('   year date_range: {} - {}'.format(start_date, end_date))
                year_data = np.asarray(year_df['adj. close'][-year_days:])
                year_low = min(year_data)
                year_high = max(year_data)
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
                stddev_30 = array_30.std(ddof=1)

                prev_row = year_df.iloc[-61]
                return_60_day = curr_close / prev_row['adj. close'] - 1
                array_60 = np.asarray(year_df['adj. close'][-61:])
                ma_60_day = array_60.mean() / curr_close - 1
                stddev_60 = array_60.std(ddof=1)


                # print out end of series data for debugging
                if i > (new_size-5):
                    print('date: {} adj close: {:4.3f} - year high percent: {:3.2f} '
                          'low: {:3.2f}'.format(curr_date,
                                                curr_close, curr_year_high_pct, curr_year_low_pct))
                    print('   high: {} low: {} '.format(year_high, year_low))
                    print('   9 day rtn: {} 15 day rtn: {} '
                          '30 day rtn: {} 60 day rtn: {}'.format(return_9_day, return_15_day,
                                                                 return_30_day, return_60_day))
                # print(latest_row)
                new_vals = [ticker, curr_date, curr_return, curr_return_open, curr_return_high, curr_return_low,
                            return_9_day, return_15_day, return_30_day, return_60_day,
                            ma_9_day, ma_15_day, ma_30_day, ma_60_day,
                            curr_year_high_pct, curr_year_low_pct, stddev_30, stddev_60]
                new_df.loc[i] = new_vals

            with open(_calced_path + _get_calc_filename(ticker, curr_date), 'wt') as f:
                f.write(new_df.to_json())


def _get_calc_filename(ticker, lastdate, extension=".json"):
    # ticker may have odd symbols that cannot be in a filename like a forward slash
    ticker = ticker.replace('/', '+')
    return (ticker + ".{:04d}.{:02d}" + extension).format(lastdate.year, lastdate.month)


def _create_data_frame(df_size):
    df = pd.DataFrame(index=range(df_size),
                      columns=('ticker', 'date', 'return_daily', 'return_open', 'return_high', 'return_low',
                               'return_9_day', 'return_15_day', 'return_30_day', 'return_60_day',
                               'ma_9_day', 'ma_15_day', 'ma_30_day', 'ma_60_day',
                               'year_high_percent', 'year_low_percent', 'stddev_30_day', 'stddev_60_day'))
    return df


if __name__ == '__main__':
    calc_ml_data()
