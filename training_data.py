import numpy as np
import pandas as pd
import data_ml as dml


class TrainingData:

    def __init__(self, data_df, feature_series_count, feature_count, label_count):
        # setup training data selection
        print("Getting Training Data...")
        self.feature_series_count = feature_series_count
        self.feature_count = feature_count
        self.label_count = label_count
        self.data_df = data_df
        self.tickers = list({t for t in data_df['ticker']})
        self.tickers.sort()
        self.valid_tickers = []
        self.ticker_count = len(self.tickers)
        self.curr_ticker = 0

        self.tickers_df_list = []
        for ticker_num in range(self.ticker_count):
            ticker_df = data_df[data_df.ticker == self.tickers[ticker_num]].copy()
            ticker_df.sort_values('date', inplace=True)
            ticker_df.reset_index(drop=True, inplace=True)
            if len(ticker_df) < feature_series_count:
                print("not enough data - {} - {}".format(ticker_df['date'][0], ticker_df['ticker'][0]))
                self.ticker_count = self.ticker_count - 1
            else:
                self.tickers_df_list.append(ticker_df)
                self.valid_tickers.append(self.tickers[ticker_num])

        self.ticker_df_curr_row = np.zeros(self.ticker_count, dtype=np.int)

    def get_next_training_data_random(self, until_exhausted=False):
        # Get random ticker
        rnd_ticker_num = np.random.randint(0, len(self.valid_tickers))
        ticker_df = self.data_df[self.data_df.ticker == self.valid_tickers[rnd_ticker_num]]

        # get random series for this ticker
        series_count = len(ticker_df) - self.feature_series_count
        if (series_count < 1):
            print("WTF {}".format(self.valid_tickers[rnd_ticker_num]))
        rnd_series_num = np.random.randint(0, series_count)
        train_df = ticker_df.iloc[rnd_series_num:rnd_series_num + self.feature_series_count]

        # get data for ml
        feature_matrix = train_df.as_matrix(columns=dml.get_feature_columns())
        feature_shaped = np.reshape(feature_matrix, [self.feature_series_count, self.feature_count])
        label_values = np.array(train_df[dml.get_label_columns()].values[-1])
        label_array = np.reshape(label_values, [1, self.label_count])
        descriptive_df = train_df.drop(dml.get_feature_columns(), axis=1)
        return feature_shaped, label_array, descriptive_df

    def get_batch(self, batch_size, until_exhausted=False):
        batch_feature_shaped = np.ndarray((batch_size, self.feature_series_count, self.feature_count))
        batch_label_array = np.ndarray((batch_size, 1, self.label_count))
        batch_descriptive = []

        for i in range(batch_size):
            feature_data, label_data, descriptive_df = self.get_next_training_data(until_exhausted)
            batch_feature_shaped[i] = feature_data
            batch_label_array[i] = label_data
            batch_descriptive.append(descriptive_df)

        return batch_feature_shaped, batch_label_array, batch_descriptive

    def get_next_training_data(self, call_random=False, until_exhausted=False):
        if call_random:
            return self.get_next_training_data_random()
        # if we have used up all data possible, then reset and reuse training data
        if self.ticker_df_curr_row.sum() == -self.ticker_count:
            print("Used up all the training labels...")
            self.ticker_df_curr_row = np.zeros(self.ticker_count, dtype=np.int)
            self.curr_ticker = 0
            if until_exhausted:
                return None, None, None

        # if this ticker has used up all of its data, then try the next ticker
        while self.ticker_df_curr_row[self.curr_ticker] == -1:
            self.curr_ticker = (self.curr_ticker + 1) % self.ticker_count

        # Get the current ticker's training data
        ticker_df = self.tickers_df_list[self.curr_ticker]
        curr_row = self.ticker_df_curr_row[self.curr_ticker]
        train_df = ticker_df.iloc[curr_row:curr_row + self.feature_series_count]

        # get data for ml
        feature_matrix = train_df.as_matrix(columns=dml.get_feature_columns())
        feature_shaped = np.reshape(feature_matrix, [self.feature_series_count, self.feature_count])
        label_values = np.array(train_df[dml.get_label_columns()].values[-1])
        label_array = np.reshape(label_values, [1, self.label_count])
        descriptive_df = train_df.drop(dml.get_feature_columns(), axis=1)
        descriptive_df = descriptive_df.drop(dml.get_label_columns(), axis=1)

        # all data is gathered, setup counters for next time.
        self.ticker_df_curr_row[self.curr_ticker] = int(self.ticker_df_curr_row[self.curr_ticker] + 1)
        if len(ticker_df) < self.ticker_df_curr_row[self.curr_ticker] + self.feature_series_count:
            self.ticker_df_curr_row[self.curr_ticker] = -1  # we have exhausted this tickers data
            # print("Exhausted data for {}".format(descriptive_df.iloc[0]))
        self.curr_ticker = (self.curr_ticker + 1) % self.ticker_count

        return feature_shaped, label_array, descriptive_df


class TrainingDataTicker:

    def __init__(self, data_df, feature_series_count, feature_count, label_count, ticker):
        # setup training data selection
        print("Getting Training Data...")
        self.feature_series_count = feature_series_count
        self.feature_count = feature_count
        self.label_count = label_count
        self.data_df = data_df

        self.ticker = ticker
        self.ticker_df = data_df[data_df.ticker == ticker].copy()
        self.ticker_df.sort_values('date', inplace=True)
        self.ticker_df.reset_index(drop=True, inplace=True)
        self.ticker_df_curr_row = 0
        if len(self.ticker_df) == 0:
            error_string = "NO data - {}".format(ticker)
            print(error_string)
            raise ValueError(error_string)
        if len(self.ticker_df) < feature_series_count:
            error_string = "not enough data - {} - {}".format(self.ticker_df['date'][0], self.ticker_df['ticker'][0])
            print(error_string)
            raise ValueError(error_string)

    def get_next_training_data(self, until_exhausted=False):

        # Get the current ticker's training data
        curr_row = self.ticker_df_curr_row
        train_df = self.ticker_df.iloc[curr_row:curr_row + self.feature_series_count]

        # get data for ml
        feature_matrix = train_df.as_matrix(columns=dml.get_feature_columns())
        feature_shaped = np.reshape(feature_matrix, [self.feature_series_count, self.feature_count])
        label_values = np.array(train_df[dml.get_label_columns()].values[-1])
        label_array = np.reshape(label_values, [1, self.label_count])
        descriptive_df = train_df.drop(dml.get_feature_columns(), axis=1)
        descriptive_df = descriptive_df.drop(dml.get_label_columns(), axis=1)

        # all data is gathered, setup counters for next time.
        self.ticker_df_curr_row = int(self.ticker_df_curr_row + 1)
        if len(self.ticker_df) < self.ticker_df_curr_row + self.feature_series_count:
            self.ticker_df_curr_row = 0  # we have exhausted this tickers data
            if until_exhausted:
                return None, None, None
            # print("Exhausted data for {}".format(self.ticker))
        return feature_shaped, label_array, descriptive_df

