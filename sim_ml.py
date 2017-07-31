import os
import numpy as np
import data_ml as dml
from utils import timing
import datetime
import training_data as td
import run_ml as rml
import pandas as pd
import data_universe as du


def add_a_day(a_date):
    return a_date + datetime.timedelta(days=1)


@timing
def simulate(start_cash, start_date, end_date, buy_threshold, sell_threshold, difference_threshold, trx_cost, prediction_file):
    the_curr_time = datetime.datetime.now().strftime('%X')
    print("Starting Simulation... time: {}".format(the_curr_time))

    print_string = "   params - "
    print_string += " from: " + start_date
    print_string += " to: " + start_date
    print_string += " , buy thresh= {:1.4f}".format(buy_threshold)
    print_string += " , sell thresh= {:1.4f".format(sell_threshold)
    print_string += " , diff thresh= {:1.4f".format(difference_threshold)
    print_string += " , fees= {:2.3f".format(trx_cost)

    print(print_string)

    # get the data frame, this may be a lot of data....
    prices_df = du.get_all_prices()
    with open(rml.prediction_file, 'rt', encoding='utf-8') as f:
        predictions_df = pd.read_csv(f)

    # convert datetime column
    prices_df['date'].apply(pd.to_datetime)
    predictions_df['date'].apply(pd.to_datetime)

    # predictions can create transactions based off of the prices for the next day...
    # so adjust the days of the predictions to be applied to the next days' prices
    predictions_df['date'].apply(add_a_day)

    prediction_price_df = pd.merge(prices_df, predictions_df, how='inner', on=['date', 'ticker'])
    prediction_price_df.sort_values('date', inplace=True)
    prediction_price_df.reset_index(drop=True, inplace=True)

    transactions_df = pd.DataFrame(columns=_get_transaction_columns())
    curr_transactions_df = pd.DataFrame(columns=_get_transaction_columns())
    positions_df = pd.DataFrame(columns=_get_position_columns())
    old_positions_df = pd.DataFrame(columns=_get_position_columns())
    new_positions_df = pd.DataFrame(columns=_get_position_columns())
    curr_cash = start_cash
    old_date = None

    for index, row in prediction_price_df.iterrows():
        buy_prediction = row['buy_prediction']
        sell_prediction = row['sell_prediction']
        curr_ticker = row['ticker']
        curr_date = row['date']
        if old_date is None:
            old_date = curr_date
        print(" {} - {} ({}, {})".format(row['date'], row['ticker'], buy_prediction, sell_prediction))

        # check to see if the date is rolling forward
        if curr_date > old_date:
            old_date = curr_date
            # set quantities for transactions, save them to the main set, and clear temp set
            # create positions for yesterday, save them to main set, set old_positions

        is_buy = False
        is_sell = False
        if sell_prediction > buy_prediction:
            buy_prediction = 0.0
        if (buy_prediction > (sell_prediction + difference_threshold)) and buy_prediction > buy_threshold:
            is_buy = True
            sell_prediction = 0.0
        if sell_prediction > sell_threshold:
            is_sell = True

        owned_df = old_positions_df['ticker == {}'.format(curr_ticker)]
        if len(owned_df) > 0:
            # If we already own it, go through this logic
            if is_sell:
                # create a transaction
                new_transaction = [curr_ticker, curr_date, row['price'], 0.0, trx_cost, 0.0, False, True]
                curr_transactions_df.loc[-1] = new_transaction
        elif is_buy:
            # don't own any and signal is a buy...
            new_transaction = [curr_ticker, curr_date, row['price'], 0.0, trx_cost, 0.0, True, False]
            curr_transactions_df.loc[-1] = new_transaction


def _get_transaction_columns():
    return ['ticker', 'date', 'price', 'quantity', 'fee', 'total_cost', 'buy', 'sell']


def _get_position_columns():
    return ['ticker', 'date', 'quantity', 'price', 'value', 'buy_date']


if __name__ == '__main__':
    a_start_date = rml.test_data_date
    simulate(100000, a_start_date, datetime.date.today(), 0.6, 0.5, 0.4, 0.0, rml.prediction_file)



