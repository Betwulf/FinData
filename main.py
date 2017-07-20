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
import data_ml as dml
import training_data as td

""" This project uses Quandl - make sure you have an API key to Quandl in order to access their data """
#
# api_key = ""
# if len(sys.argv) > 1:
#     api_key = sys.argv[1]
# else:
#     print("Please paste in your quandl api key:")
#     api_key = sys.stdin.readline().replace('\n', '')
# quandl.ApiConfig.api_key = api_key
# du.update_all_price_caches()
# fin_data = du.get_all_prices()
# print(fin_data.describe())
# print('Total number of rows: {}'.format(len(fin_data)))
# print('Got data for the following tickers:')
# print({t for t in fin_data['ticker']})
#
# dm.calc_ml_data()
# dm.calc_training_data()
# df = dm.get_all_feature_data()
# print('FEATURE DATA {} rows.'.format(len(df)))
# print(df.describe())
# df = dm.get_all_label_data()
# print('LABEL DATA {} rows.'.format(len(df)))
# print(df.describe())
# df = dm.get_all_ml_data()
# print('COMBINED DATA {} rows.'.format(len(df)))
# print(df.describe())
#
#
# rml.RNN()

label_count = 2
feature_count = 10
feature_series_count = 30  # The number of inputs back-to-back to feed into the RNN
hidden_neurons = 512
last_hidden_neurons = 32
test_data_date = datetime.datetime(2016, 6, 30)


data_df = dml.get_all_ml_data()
training_df = data_df[data_df.date < test_data_date]
test_df = data_df[data_df.date >= test_data_date]

training_data_class = td.TrainingData(training_df, feature_series_count, feature_count, label_count)
with open('./file.json', 'wt') as f:
    new_df = pd.DataFrame()
    for _ in range(400):
        feature_data, label_data, descriptive_df = training_data_class.get_next_training_data()
        np_feature = np.array(feature_data)
        np_feature = np.reshape(np_feature, [300])
        np_label = np.array(label_data)
        np_label = np.reshape(np_label, [2])
        np_feature = np.append(np_feature, np_label, axis=0)
        # temp_data = np.append(np_feature, descriptive_df.as_matrix()[0], axis=0)
        temp_data = descriptive_df.as_matrix()[0]
        temp_data = np.reshape(temp_data, [1, 3])
        print(pd.to_datetime(temp_data[0][0]))
        temp_data[0][0] = pd.to_datetime(temp_data[0][0]).strftime('%x')
        temp_df = pd.DataFrame(temp_data)
        new_df = new_df.append(temp_df)
    new_df.reset_index(drop=True, inplace=True)
    f.write(new_df.to_json(orient='index'))
