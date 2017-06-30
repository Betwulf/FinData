import os
import datetime
import calendar
import pandas as pd
import numpy as np
import data_universe as du


def calc_ml_data():
    df = du.get_all_prices()
    tickers = {t for t in df['ticker']}

    # for each ticker, sort and process calculated data for ml
    for ticker in tickers:
        sub_df = df[df.ticker == ticker]
        sub_df = sub_df.sort_values(by='date')
        print(sub_df.head())




if __name__ == '__main__':
    calc_ml_data()
