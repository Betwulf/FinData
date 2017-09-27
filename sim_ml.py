import os
import numpy as np
import data_ml as dml
from utils import timing
import datetime
import run_ml as rml
import pandas as pd
import data_universe as du


_sim_dir = "\\sim\\"
_positions_filename = "positions.csv"
_transactions_filename = "transactions.csv"
_cwd = os.getcwd()
_sim_path = _cwd + _sim_dir
_sim_positions_file = _sim_path + _positions_filename
_sim_transactions_file = _sim_path + _transactions_filename


# ensure paths are there...
if not os.path.exists(_sim_path):
    os.makedirs(_sim_path)


def add_a_day(a_date):
    return a_date + datetime.timedelta(days=1)


def simulate_all(start_cash, start_date, end_date, buy_threshold, sell_threshold, difference_threshold,
                 max_position_percent, trx_cost, prediction_file):

    the_curr_time = datetime.datetime.now().strftime('%X')
    print("Starting All Simulations... time: {}".format(the_curr_time))

    print_string = "   params - "
    print_string += " DATE: " + start_date.strftime('%x')
    print_string += " - " + end_date.strftime('%x')
    print_string += " : fees= {:2.2f}".format(trx_cost)

    print(print_string)

    print_string = "          - "
    print_string += "buy thresh= {:1.2f}".format(buy_threshold)
    print_string += " , sell thresh= {:1.2f}".format(sell_threshold)
    print_string += " , diff thresh= {:1.2f}".format(difference_threshold)

    print(print_string)

    print_string = "          - "
    print_string += "max pos % = {:1.2f}".format(max_position_percent)
    print_string += " , pred file = {}".format(prediction_file)

    print(print_string)

    # get the data frame, this may be a lot of data....
    prices_df = du.get_all_prices()
    # convert datetime column
    prices_df['date'] = prices_df['date'].apply(pd.to_datetime)

    # get the data frame, this may be a lot of data....
    with open(prediction_file, 'rt', encoding='utf-8') as f:
        predictions_df = pd.read_csv(f)

    model_file_list = set(predictions_df['model_file'].values)

    for model_file in model_file_list:
        model_predictions_df = predictions_df[predictions_df.model_file == model_file].copy()
        # convert datetime column
        model_predictions_df['date'] = model_predictions_df['date'].apply(pd.to_datetime)

        # predictions can create transactions based off of the prices for the next day...
        # so adjust the days of the predictions to be applied to the next days' prices
        model_predictions_df['date'] = model_predictions_df['date'].apply(add_a_day)

        prediction_price_df = pd.merge(prices_df, model_predictions_df, how='inner', on=['date', 'ticker'])
        prediction_price_df = prediction_price_df[(start_date <= prediction_price_df.date) &
                                                  (prediction_price_df.date <= end_date)].copy()
        prediction_price_df.sort_values('date', inplace=True)
        prediction_price_df.reset_index(drop=True, inplace=True)

        simulate(model_file, start_cash, buy_threshold, sell_threshold, difference_threshold,
                 max_position_percent, trx_cost, prediction_price_df)


@timing
def simulate(model_file, start_cash, buy_threshold, sell_threshold, difference_threshold,
             max_position_percent, trx_cost, prediction_price_df):
    print("Begin Simulation...")
    transactions_df = pd.DataFrame(columns=_get_transaction_columns())
    positions_df = pd.DataFrame(columns=_get_position_columns())
    old_positions_df = pd.DataFrame(columns=_get_position_columns())
    curr_cash = start_cash
    old_date = None
    curr_buys = []
    curr_sells = []
    curr_rebalances = []

    for index, row in prediction_price_df.iterrows():
        buy_prediction = row['buy_prediction']
        sell_prediction = row['sell_prediction']
        curr_ticker = row['ticker']
        curr_date = row['date']

        # initial setup for old_date
        if old_date is None:
            old_date = curr_date

        # check to see if the date is rolling forward
        if curr_date > old_date:
            # set quantities for transactions, save them to the main set, and clear temp set
            new_positions_df = pd.DataFrame(columns=_get_position_columns())
            # new position count minus 1 for the cash position
            new_position_count = float(len(old_positions_df) + len(curr_buys) - len(curr_sells) - 1)
            curr_total_value = max(old_positions_df['value'].sum(), curr_cash)
            target_position_value = min(max_position_percent * curr_total_value,
                                        curr_total_value/(max(new_position_count, 1.0)))
            print("target position value : {} ".format(target_position_value))
            # rebal tickers that are staying
            for a_ticker, a_price, a_buy_bool in curr_rebalances:
                a_row_df = old_positions_df.loc[old_positions_df['ticker'] == a_ticker]
                old_quantity = a_row_df['quantity'].iloc[0]
                old_age = a_row_df['age'].iloc[0]
                rebal_quantity = (target_position_value - old_quantity * a_price) / a_price
                ttl_trx_cost = rebal_quantity*a_price - trx_cost
                new_transaction = [model_file, a_ticker, curr_date, a_price, rebal_quantity,
                                   trx_cost, ttl_trx_cost, False, False, True]
                transactions_df.loc[transactions_df.shape[0]] = new_transaction
                curr_cash += -rebal_quantity*a_price - trx_cost
                new_quantity = old_quantity + rebal_quantity
                new_age = 0 if a_buy_bool else old_age + 1
                new_position = [model_file, a_ticker, curr_date, a_price, new_quantity, a_price*new_quantity, new_age]
                new_positions_df.loc[new_positions_df.shape[0]] = new_position

            # buy new tickers
            for a_ticker, a_price in curr_buys:
                buy_quantity = target_position_value/a_price
                ttl_trx_cost = target_position_value - trx_cost
                new_transaction = [model_file, a_ticker, curr_date, a_price, buy_quantity,
                                   trx_cost, ttl_trx_cost, True, False, False]
                transactions_df.loc[transactions_df.shape[0]] = new_transaction
                curr_cash -= target_position_value + trx_cost
                new_position = [model_file, a_ticker, curr_date, a_price, buy_quantity, a_price * buy_quantity, 0]
                new_positions_df.loc[new_positions_df.shape[0]] = new_position

            # sell dying tickers
            for a_ticker, a_price in curr_sells:
                a_row_df = old_positions_df.loc[old_positions_df['ticker'] == a_ticker]
                if len(a_row_df) == 0:
                    print("what?")
                old_quantity = a_row_df['quantity'][0]
                ttl_trx_cost = - a_price*old_quantity - trx_cost
                new_transaction = [model_file, a_ticker, curr_date, a_price, old_quantity,
                                   trx_cost, ttl_trx_cost, False, True, False]
                transactions_df.loc[transactions_df.shape[0]] = new_transaction
                curr_cash += a_price*old_quantity - trx_cost

            # Add cash as a position
            new_position = [model_file, '$', curr_date, curr_cash, 1, curr_cash, 0]
            new_positions_df.loc[new_positions_df.shape[0]] = new_position

            # clean up
            curr_buys = []
            curr_sells = []
            curr_rebalances = []
            positions_df = pd.concat([positions_df, new_positions_df])
            old_positions_df = new_positions_df
            old_date = curr_date

        is_buy = False
        is_sell = False
        # if sell_prediction > buy_prediction:
        #     buy_prediction = 0.0
        if (buy_prediction > (sell_prediction + difference_threshold)) and buy_prediction > buy_threshold:
            is_buy = True
            sell_prediction = 0.0
        if sell_prediction > sell_threshold:
            is_sell = True

        owned_df = old_positions_df.loc[old_positions_df['ticker'] == curr_ticker]
        if len(owned_df) > 0:
            # If we already own it, go through this logic
            if is_sell:
                curr_sells.append((curr_ticker, row['adj. close']))
            else:
                curr_rebalances.append((curr_ticker, row['adj. close'], is_buy))
        elif is_buy:
            # don't own any and signal is a buy...
            curr_buys.append((curr_ticker, row['adj. close']))

    # The main loop is over, save to disk the results
    print(" --- Sim Complete --- ")
    print("curr_buys")
    print(curr_buys)
    positions_df.reset_index(drop=True, inplace=True)
    transactions_df.reset_index(drop=True, inplace=True)
    transactions_df['buy'] = transactions_df['buy'].astype(int)
    transactions_df['sell'] = transactions_df['sell'].astype(int)
    transactions_df['rebalance'] = transactions_df['rebalance'].astype(int)
    print("positions")
    print(positions_df.describe())
    print("positions tail")
    print(positions_df.tail())
    print("transactions")
    print(transactions_df.describe())
    with open(_sim_positions_file, 'wt', encoding='utf-8') as f:
        positions_df.to_csv(f)
    with open(_sim_transactions_file, 'wt', encoding='utf-8') as f:
        transactions_df.to_csv(f)


def _get_transaction_columns():
    return ['model_file', 'ticker', 'date', 'price', 'quantity', 'fee', 'total_cost', 'buy', 'sell', 'rebalance']


def _get_position_columns():
    return ['model_file', 'ticker', 'date', 'price', 'quantity', 'value', 'age']


if __name__ == '__main__':
    a_start_date = rml.test_data_date
    simulate_all(100000.0, a_start_date, datetime.date.today(), 0.6, 0.6, -1.4, 1.0, 0.0, rml.prediction_file)
