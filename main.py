import numpy as np
import pandas as pd
import quandl
import sys
import os
import datetime
import calendar
import pandas as pd
import quandl
from utils import timing
import data_universe as du
import data_ml as dm
import run_ml as rml

""" This project uses Quandl - make sure you have an API key to Quandl in order to access their data """

api_key = ""
if len(sys.argv) > 1:
    api_key = sys.argv[1]
else:
    print("Please paste in your quandl api key:")
    api_key = sys.stdin.readline().replace('\n', '')
quandl.ApiConfig.api_key = api_key
du.update_all_price_caches()
fin_data = du.get_all_prices()
print(fin_data.describe())
print('Total number of rows: {}'.format(len(fin_data)))
print('Got data for the following tickers:')
print({t for t in fin_data['ticker']})

dm.calc_ml_data()
dm.calc_training_data()
df = dm.get_all_feature_data()
print('FEATURE DATA {} rows.'.format(len(df)))
print(df.describe())
df = dm.get_all_label_data()
print('LABEL DATA {} rows.'.format(len(df)))
print(df.describe())
df = dm.get_all_ml_data()
print('COMBINED DATA {} rows.'.format(len(df)))
print(df.describe())


rml.RNN()
