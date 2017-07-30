import os
import numpy as np
import data_ml as dml
from utils import timing
import datetime
import training_data as td
import run_ml as rml
import pandas as pd


def _get_trades_columns():
    return ['date', 'ticker', 'buy', 'sell', 'quantity', 'price', 'fee', 'total_cost']


@timing
def simulate(start_cash, start_date, end_date, buy_threshold, sell_threshold, trx_cost, prediction_file):
    the_curr_time = datetime.datetime.now().strftime('%X')
    print("Starting Simulation... time: {}".format(the_curr_time))

    print_string = "   params - "
    print_string += " from: " + start_date
    print_string += " to: " + start_date
    print_string += " , buy thresh= {:1.4f}".format(buy_threshold)
    print_string += " , sell thresh= {:1.4f".format(sell_threshold)
    print_string += " , fees= {:2.3f".format(trx_cost)

    print(print_string)

    # get the data frame, this may be a lot of data....
    data_df = dml.get_all_ml_data()
    # test_df = data_df[data_df.date >= start_date and data_df.date <= end_date]
    test_df = data_df[start_date <= data_df.date <= end_date]
    del data_df

    curr_cash = start_cash
    trades_df = pd.DataFrame()

    training_data_class = td.TrainingData(test_df, rml.feature_series_count, rml.feature_count, rml.label_count)
    feature_data, label_data, descriptive_df = training_data_class.get_next_training_data()
    curr_date = descriptive_df['date']


if __name__ == '__main__':
    a_start_date = rml.test_data_date
    simulate(100000, a_start_date, datetime.date.today(), 0.6, 0.5, 0.0, rml.prediction_file)



