import os
import numpy as np
import data_ml as dml
from utils import timing
import datetime
import run_ml as rml
import pandas as pd
import data_universe as du


_sim_dir = "/sim/"
_positions_filename = "positions.csv"
_transactions_filename = "transactions.csv"
_returns_filename = "returns.csv"
_cwd = os.getcwd()
_sim_path = _cwd + _sim_dir
_sim_positions_file = _sim_path + _positions_filename
_sim_transactions_file = _sim_path + _transactions_filename
_sim_returns_file = _sim_path + _returns_filename


# ensure paths are there...
if not os.path.exists(_sim_path):
    os.makedirs(_sim_path)


# Buy if over threshold, Sell if under threshold - expects a list
def _threshold_function(prediction, buy_threshold, sell_threshold):
    buys = [x > buy_threshold for x in prediction]
    sells = [x < sell_threshold for x in prediction]
    return buys, sells


def _add_a_day(a_date):
    return a_date + datetime.timedelta(days=1)


def _simulate_parameters_print(start_cash, start_date, end_date, buy_threshold, sell_threshold, difference_threshold,
                               sell_age, max_position_percent, trx_fee, prediction_file, one_signal=True):
    the_curr_time = datetime.datetime.now().strftime('%X')
    print("Starting All Simulations... time: {}".format(the_curr_time))

    print_string = "   params -"
    print_string += " DATE: " + start_date.strftime('%x')
    print_string += " - " + end_date.strftime('%x')
    print_string += " : fees= {:2.2f}".format(trx_fee)
    print_string += " : sell age= {}".format(sell_age)
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


def simulate_all(start_cash, start_date, end_date, buy_threshold, sell_threshold, difference_threshold, sell_age,
                 max_position_percent, trx_fee, prediction_file, one_signal=True):

    _simulate_parameters_print(start_cash, start_date, end_date, buy_threshold, sell_threshold, difference_threshold,
                               sell_age, max_position_percent, trx_fee, prediction_file, one_signal)

    # get the data frame, this may be a lot of data....
    prices_df = du.get_all_prices()
    # convert datetime column
    prices_df['date'] = prices_df['date'].apply(pd.to_datetime)

    # get the data frame, this may be a lot of data....
    with open(prediction_file, 'rt', encoding='utf-8') as f:
        predictions_df = pd.read_csv(f)

    model_file_list = set(predictions_df['model_file'].values)

    for model_file in model_file_list:
        print("Starting model file: {}".format(model_file))
        model_predictions_df = predictions_df[predictions_df.model_file == model_file].copy()
        # convert datetime column
        model_predictions_df['date'] = model_predictions_df['date'].apply(pd.to_datetime)

        # predictions can create transactions based off of the prices for the next day...
        # so adjust the days of the predictions to be applied to the next days' prices
        model_predictions_df['date'] = model_predictions_df['date'].apply(_add_a_day)

        print("merging prices and predictions....")
        prediction_price_df = pd.merge(prices_df, model_predictions_df, how='inner', on=['date', 'ticker'])
        prediction_price_df = prediction_price_df[(start_date <= prediction_price_df.date) &
                                                  (prediction_price_df.date <= end_date)].copy()
        print("ordering data by date....")
        prediction_price_df.sort_values('date', inplace=True)
        prediction_price_df.reset_index(drop=True, inplace=True)

        _simulate_new(model_file, start_cash, buy_threshold, sell_threshold, difference_threshold, sell_age,
                                max_position_percent, trx_fee, prediction_price_df, _threshold_function)
        if one_signal:
            simulate_one_signal(model_file, start_cash, buy_threshold, sell_threshold, difference_threshold, sell_age,
                                max_position_percent, trx_fee, prediction_price_df)
        else:
            simulate_two_signals(model_file, start_cash, buy_threshold, sell_threshold, difference_threshold, sell_age,
                                 max_position_percent, trx_fee, prediction_price_df)


# Build dictionaries for prices and predictions that are easier to use in set operations
def build_dictionaries(prediction_price_df):
    prices = {}
    predictions = {}
    for index, row in prediction_price_df.iterrows():
        curr_prediction = row['prediction']
        curr_ticker = row['ticker']
        curr_date = row['date']
        curr_price = row['adj. close']
        if curr_date not in prices.keys():
            prices[curr_date] = {}
            predictions[curr_date] = {}
        prices[curr_date][curr_ticker] = curr_price
        predictions[curr_date][curr_ticker] = curr_prediction
    return prices, predictions


def _simulate_new(model_file, start_cash, buy_threshold, sell_threshold, difference_threshold, sell_age,
                  max_position_percent, trx_fee, prediction_price_df, threshold_function):

    transactions_df = pd.DataFrame(columns=_get_transaction_columns())
    positions_df = pd.DataFrame(columns=_get_position_columns())
    old_position_list = []
    old_positions = []
    curr_positions = {}
    old_date = None
    curr_cash = start_cash
    prices, predictions = build_dictionaries(prediction_price_df)

    # main loop - daily
    for curr_date in sorted(predictions):
        day_prices = sorted(list(prices[curr_date].items()))
        day_predictions = sorted(list(predictions[curr_date].items()))
        buys, sells = threshold_function(list(zip(*day_predictions))[1], buy_threshold, sell_threshold)
        print("date: {} \t ticker: {}".format(curr_date.strftime('%x'), day_predictions[0][0]))
        print("prediction: {}".format(day_predictions[0][1]))
        print("buy: {}".format(buys[0]))

        # calculate trade size
        new_position_list = []
        new_buys = []
        current_total_value = 0
        if len(old_position_list) > 0:
            current_total_value = list(zip(*old_positions))[5]  # get the total_values
            buys_and_positions = [a or b for a, b in zip(old_position_list, buys)]
            new_position_list = [a and not b for a, b in zip(buys_and_positions, sells)]
            new_buys = [a and not b for a, b in zip(buys, old_position_list)]
        else:
            # first run only
            current_total_value = curr_cash
            new_position_list = [a and not b for a, b in zip(buys, sells)]
            new_buys = buys
        new_position_count = sum(new_position_list)
        target_position_value = current_total_value - new_position_count

        # iterate through new buys
        for idx, abuy in enumerate(new_buys):
            aticker = day_prices[idx][0]
            aprice = day_prices[idx][1]
            aticker = day_predictions[idx][0]  # debug - ensure we have the same ticker - order is important
            prediction = day_predictions[idx][1]
            quantity = target_position_value/aprice
            new_pos = _new_position(model_file, aticker, curr_date, aprice, quantity, target_position_value, 0)
            new_trx = _new_transaction(model_file, aticker, curr_date, aprice, quantity, trx_fee, target_position_value,
                                       prediction, True, False, False)
            positions_df.loc[positions_df.shape[0]] = new_pos  # save for file later
            transactions_df.loc[transactions_df.shape[0]] = new_trx  # save for file later
            curr_positions.append(new_pos)




@timing
def simulate_two_signals(model_file, start_cash, buy_threshold, sell_threshold, difference_threshold, sell_age,
                         max_position_percent, trx_fee, prediction_price_df):
    print("Begin Simulation...")
    transactions_df = pd.DataFrame(columns=_get_transaction_columns_two_signal())
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
        curr_price = row['adj. close']

        # initial setup for old_date
        if old_date is None:
            old_date = curr_date
            # Add cash as a position
            new_position = [model_file, '$', curr_date, curr_cash, 1, curr_cash, 0]
            old_positions_df.loc[old_positions_df.shape[0]] = new_position

        # check to see if the date is rolling forward
        if curr_date > old_date:
            # set quantities for transactions, save them to the main set, and clear temp set
            new_positions_df = pd.DataFrame(columns=_get_position_columns())
            # new position count minus 1 for the cash position
            new_position_count = float(len(old_positions_df) + len(curr_buys) - len(curr_sells) - 1)
            curr_total_value = max(old_positions_df['value'].sum(), curr_cash)  # should not need max anymore
            target_position_value = min(max_position_percent * curr_total_value,
                                        curr_total_value/(max(new_position_count, 1.0)))
            print("date: {} \t value: {:7.2f}".format(old_date.strftime('%x'), curr_total_value))
            # rebalance tickers that are staying, only if there is a trade
            if len(curr_buys) == 0:
                # need to create positions for the day
                for a_ticker, a_date, a_price, a_buy_bool in curr_rebalances:
                    a_row_df = old_positions_df.loc[old_positions_df['ticker'] == a_ticker]
                    old_quantity = a_row_df['quantity'].iloc[0]
                    old_age = a_row_df['age'].iloc[0]
                    new_value = old_quantity * a_price
                    new_age = 0 if a_buy_bool else old_age + 1
                    if new_age < sell_age:
                        new_position = [model_file, a_ticker, a_date, a_price, old_quantity, new_value, new_age]
                        new_positions_df.loc[new_positions_df.shape[0]] = new_position
                    else:
                        # need to sell old signal off
                        curr_sells.append((a_ticker, a_date, a_price, -666, -666))
            else:
                # need to rebal to make cash for new buys
                for a_ticker, a_date, a_price, a_buy_bool in curr_rebalances:
                    a_row_df = old_positions_df.loc[old_positions_df['ticker'] == a_ticker]
                    old_quantity = a_row_df['quantity'].iloc[0]
                    old_age = a_row_df['age'].iloc[0]
                    rebal_quantity = (target_position_value - old_quantity * a_price) / a_price
                    ttl_trx_cost = -rebal_quantity*a_price - trx_fee
                    new_transaction = [model_file, a_ticker, a_date, a_price, rebal_quantity,
                                       trx_fee, ttl_trx_cost, buy_prediction, sell_prediction, False, False, True]
                    transactions_df.loc[transactions_df.shape[0]] = new_transaction
                    curr_cash += -rebal_quantity*a_price - trx_fee
                    new_quantity = old_quantity + rebal_quantity
                    new_age = 0 if a_buy_bool else old_age + 1
                    new_position = [model_file, a_ticker, a_date, a_price, new_quantity, a_price*new_quantity, new_age]
                    new_positions_df.loc[new_positions_df.shape[0]] = new_position

            # buy new tickers
            for a_ticker, a_date, a_price, a_buy_prediction, a_sell_prediction in curr_buys:
                buy_quantity = target_position_value/a_price
                ttl_trx_cost = -target_position_value - trx_fee
                new_transaction = [model_file, a_ticker, a_date, a_price, buy_quantity,
                                   trx_fee, ttl_trx_cost, a_buy_prediction, a_sell_prediction, True, False, False]
                transactions_df.loc[transactions_df.shape[0]] = new_transaction
                curr_cash -= target_position_value + trx_fee
                new_position = [model_file, a_ticker, a_date, a_price, buy_quantity, a_price * buy_quantity, 0]
                new_positions_df.loc[new_positions_df.shape[0]] = new_position

            # sell dying tickers
            for a_ticker, a_date, a_price, a_buy_prediction, a_sell_prediction in curr_sells:
                a_row_df = old_positions_df.loc[old_positions_df['ticker'] == a_ticker]
                if len(a_row_df) == 0:
                    print("can't sell what you don't have.... CRASH")
                old_quantity = a_row_df['quantity'].iloc[0]
                ttl_trx_cost = a_price*old_quantity - trx_fee
                new_transaction = [model_file, a_ticker, a_date, a_price, -old_quantity,
                                   trx_fee, ttl_trx_cost, a_buy_prediction, a_sell_prediction, False, True, False]
                transactions_df.loc[transactions_df.shape[0]] = new_transaction
                curr_cash += a_price*old_quantity - trx_fee

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
                curr_sells.append((curr_ticker, curr_date, curr_price, buy_prediction, sell_prediction))
            else:
                curr_rebalances.append((curr_ticker, curr_date, curr_price, is_buy))
        elif is_buy:
            # don't own any and signal is a buy...
            curr_buys.append((curr_ticker, curr_date, curr_price, buy_prediction, sell_prediction))

    # The main loop is over, save to disk the results
    print(" --- Sim Complete --- ")
    print("curr_buys")
    print(curr_buys)
    positions_df.reset_index(drop=True, inplace=True)
    transactions_df.reset_index(drop=True, inplace=True)
    returns_df = calculate_returns(transactions_df)
    transactions_df['buy'] = transactions_df['buy'].astype(int)
    transactions_df['sell'] = transactions_df['sell'].astype(int)
    transactions_df['rebalance'] = transactions_df['rebalance'].astype(int)
    print("positions")
    print(positions_df.describe())
    print("positions tail")
    print(positions_df.tail())
    print("transactions")
    print(transactions_df.describe())
    print("returns")
    print(returns_df.describe())
    with open(_sim_positions_file, 'wt', encoding='utf-8') as f:
        positions_df.to_csv(f)
    with open(_sim_transactions_file, 'wt', encoding='utf-8') as f:
        transactions_df.to_csv(f)
    with open(_sim_returns_file, 'wt', encoding='utf-8') as f:
        returns_df.to_csv(f)


@timing
def simulate_one_signal(model_file, start_cash, buy_threshold, sell_threshold, difference_threshold, sell_age,
                        max_position_percent, trx_fee, prediction_price_df):
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
        prediction = row['prediction']
        curr_ticker = row['ticker']
        curr_date = row['date']
        curr_price = row['adj. close']

        # initial setup for old_date
        if old_date is None:
            old_date = curr_date
            # Add cash as a position
            new_position = [model_file, '$', curr_date, curr_cash, 1, curr_cash, 0]
            old_positions_df.loc[old_positions_df.shape[0]] = new_position

        # check to see if the date is rolling forward
        if curr_date > old_date:
            # set quantities for transactions, save them to the main set, and clear temp set
            new_positions_df = pd.DataFrame(columns=_get_position_columns())
            # new position count minus 1 for the cash position
            new_position_count = float(len(old_positions_df) + len(curr_buys) - len(curr_sells) - 1)
            curr_total_value = max(old_positions_df['value'].sum(), curr_cash)  # should not need max anymore
            target_position_value = min(max_position_percent * curr_total_value,
                                        curr_total_value/(max(new_position_count, 1.0)))
            print("date: {} \t value: {:7.2f}".format(old_date.strftime('%x'), curr_total_value))
            # rebalance tickers that are staying, only if there is a trade
            if len(curr_buys) == 0:
                # need to create positions for the day
                for a_ticker, a_date, a_price, a_buy_bool in curr_rebalances:
                    a_row_df = old_positions_df.loc[old_positions_df['ticker'] == a_ticker]
                    old_quantity = a_row_df['quantity'].iloc[0]
                    old_age = a_row_df['age'].iloc[0]
                    new_value = old_quantity * a_price
                    new_age = 0 if a_buy_bool else old_age + 1
                    if new_age < sell_age:
                        new_position = [model_file, a_ticker, a_date, a_price, old_quantity, new_value, new_age]
                        new_positions_df.loc[new_positions_df.shape[0]] = new_position
                    else:
                        # need to sell old signal off
                        curr_sells.append((a_ticker, a_date, a_price, -666))
            else:
                # need to rebal to make cash for new buys
                for a_ticker, a_date, a_price, a_buy_bool in curr_rebalances:
                    a_row_df = old_positions_df.loc[old_positions_df['ticker'] == a_ticker]
                    old_quantity = a_row_df['quantity'].iloc[0]
                    old_age = a_row_df['age'].iloc[0]
                    rebal_quantity = (target_position_value - old_quantity * a_price) / a_price
                    ttl_trx_cost = -rebal_quantity*a_price - trx_fee
                    new_transaction = [model_file, a_ticker, a_date, a_price, rebal_quantity,
                                       trx_fee, ttl_trx_cost, prediction, False, False, True]
                    transactions_df.loc[transactions_df.shape[0]] = new_transaction
                    curr_cash += -rebal_quantity*a_price - trx_fee
                    new_quantity = old_quantity + rebal_quantity
                    new_age = 0 if a_buy_bool else old_age + 1
                    new_position = [model_file, a_ticker, a_date, a_price, new_quantity, a_price*new_quantity, new_age]
                    new_positions_df.loc[new_positions_df.shape[0]] = new_position

            # buy new tickers
            for a_ticker, a_date, a_price, a_prediction in curr_buys:
                buy_quantity = target_position_value/a_price
                ttl_trx_cost = -target_position_value - trx_fee
                new_transaction = [model_file, a_ticker, a_date, a_price, buy_quantity,
                                   trx_fee, ttl_trx_cost, a_prediction, True, False, False]
                transactions_df.loc[transactions_df.shape[0]] = new_transaction
                curr_cash -= target_position_value + trx_fee
                new_position = [model_file, a_ticker, a_date, a_price, buy_quantity, a_price * buy_quantity, 0]
                new_positions_df.loc[new_positions_df.shape[0]] = new_position

            # sell dying tickers
            for a_ticker, a_date, a_price, a_prediction in curr_sells:
                a_row_df = old_positions_df.loc[old_positions_df['ticker'] == a_ticker]
                if len(a_row_df) == 0:
                    print("can't sell what you don't have.... CRASH")
                old_quantity = a_row_df['quantity'].iloc[0]
                ttl_trx_cost = a_price*old_quantity - trx_fee
                new_transaction = [model_file, a_ticker, a_date, a_price, -old_quantity,
                                   trx_fee, ttl_trx_cost, a_prediction, False, True, False]
                transactions_df.loc[transactions_df.shape[0]] = new_transaction
                curr_cash += a_price*old_quantity - trx_fee

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
        if prediction > buy_threshold:
            is_buy = True
        if prediction < sell_threshold:
            is_sell = True

        owned_df = old_positions_df.loc[old_positions_df['ticker'] == curr_ticker]
        if len(owned_df) > 0:
            # If we already own it, go through this logic
            if is_sell:
                curr_sells.append((curr_ticker, curr_date, curr_price, prediction))
            else:
                curr_rebalances.append((curr_ticker, curr_date, curr_price, is_buy))
        elif is_buy:
            # don't own any and signal is a buy...
            curr_buys.append((curr_ticker, curr_date, curr_price, prediction))

    # The main loop is over, save to disk the results
    print(" --- Sim Complete --- ")
    print("curr_buys")
    print(curr_buys)
    positions_df.reset_index(drop=True, inplace=True)
    transactions_df.reset_index(drop=True, inplace=True)
    returns_df = calculate_returns(transactions_df)
    transactions_df['buy'] = transactions_df['buy'].astype(int)
    transactions_df['sell'] = transactions_df['sell'].astype(int)
    transactions_df['rebalance'] = transactions_df['rebalance'].astype(int)
    print("positions")
    print(positions_df.describe())
    print("positions tail")
    print(positions_df.tail())
    print("transactions")
    print(transactions_df.describe())
    print("returns")
    print(returns_df.describe())
    with open(_sim_positions_file, 'wt', encoding='utf-8') as f:
        positions_df.to_csv(f)
    with open(_sim_transactions_file, 'wt', encoding='utf-8') as f:
        transactions_df.to_csv(f)
    with open(_sim_returns_file, 'wt', encoding='utf-8') as f:
        returns_df.to_csv(f)


def calculate_returns(transactions_df):
    returns_df = pd.DataFrame(columns=['model_file', 'buy_date', 'sell_date', 'ticker',
                                       'prediction', 'real_return', 'pure_return'])
    model_file_list = set(transactions_df['model_file'].values)

    for model_file in model_file_list:
        ticker_buy_prediction = {}
        ticker_buys = {}
        ticker_sells = {}
        ticker_buy_date = {}
        ticker_run_cost = {}
        ticker_buy_price = {}  # contains the price at the initial buy for pure return, not including rebalances
        print("Starting model file: {}".format(model_file))
        model_transactions_df = transactions_df[transactions_df.model_file == model_file].copy()
        for index, row in model_transactions_df.iterrows():
            curr_date = row['date']
            curr_ticker = row['ticker']
            curr_cost = row['total_cost']
            curr_price = row['price']
            curr_buy_prediction = row['prediction']
            is_buy = row['buy']
            is_sell = row['sell']
            is_rebalance = row['rebalance']
            if is_buy:
                ticker_buy_price[curr_ticker] = curr_price
                ticker_run_cost[curr_ticker] = curr_cost
                ticker_buy_date[curr_ticker] = curr_date
                ticker_buy_prediction[curr_ticker] = curr_buy_prediction
                ticker_buys[curr_ticker] = curr_cost
                ticker_sells[curr_ticker] = 0.0
            if is_sell:
                ticker_sells[curr_ticker] = ticker_sells[curr_ticker] + curr_cost
                pure_return = curr_price / ticker_buy_price[curr_ticker]
                real_return = ticker_sells[curr_ticker]/(-ticker_buys[curr_ticker])
                new_return_row = [model_file, ticker_buy_date[curr_ticker], curr_date, curr_ticker,
                                  ticker_buy_prediction[curr_ticker], real_return, pure_return]
                returns_df.loc[returns_df.shape[0]] = new_return_row
                ticker_run_cost = {key: val for key, val in ticker_run_cost.items() if key != curr_ticker}
                ticker_buy_price = {key: val for key, val in ticker_buy_price.items() if key != curr_ticker}
                ticker_buy_date = {key: val for key, val in ticker_buy_date.items() if key != curr_ticker}
                ticker_buy_prediction = {key: val for key, val in ticker_buy_prediction.items() if key != curr_ticker}
                ticker_buys = {key: val for key, val in ticker_buys.items() if key != curr_ticker}
                ticker_sells = {key: val for key, val in ticker_sells.items() if key != curr_ticker}
            if is_rebalance:
                if curr_cost < 0:
                    ticker_buys[curr_ticker] = ticker_buys[curr_ticker] + curr_cost
                else:
                    ticker_sells[curr_ticker] = ticker_sells[curr_ticker] + curr_cost
            # convert
    return returns_df


def _get_transaction_columns_two_signal():
    return ['model_file', 'ticker', 'date', 'price', 'quantity', 'fee', 'total_cost',
            'buy_prediction', 'sell_prediction', 'buy', 'sell', 'rebalance']


def _new_transaction(model_file, aticker, adate, aprice, quantity, fee, total_cost, prediction, buy, sell, rebalance):
    return [model_file, aticker, adate, aprice, quantity, fee, total_cost, prediction, buy, sell, rebalance]


def _get_transaction_columns():
    return ['model_file', 'ticker', 'date', 'price', 'quantity', 'fee', 'total_cost',
            'prediction', 'buy', 'sell', 'rebalance']


def _new_position(model_file, aticker, adate, aprice, quantity, total_value, age):
    return [model_file, aticker, adate, aprice, quantity, total_value, age]


def _get_position_columns():
    return ['model_file', 'ticker', 'date', 'price', 'quantity', 'value', 'age']


if __name__ == '__main__':
    a_start_date = rml.test_data_date
    an_end_date = datetime.datetime(2017, 7, 11)
    simulate_all(100000.0, a_start_date, an_end_date, 0.03, 0.02, -1.4, 15, 0.05, 0.0, rml.prediction_file)
