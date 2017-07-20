import numpy as np
import pandas as pd
import data_ml as dml


class TrainingData:

    def __init__(self, data_df):
        # setup training data selection
        print("Getting Training Data...")
        self.tickers = list({t for t in data_df['ticker']})
        self.ticker_count = len(self.tickers)
        self.curr_ticker = 0

        self.tickers_df_list = []
        for ticker_num in range(self.ticker_count):
            ticker_df = data_df[data_df.ticker == self.tickers[ticker_num]]
            self.tickers_df_list.append(ticker_df)

        self.ticker_df_curr_row = np.zeros(self.ticker_count, dtype=np.int)

    def get_next_training_data(self, feature_series_count, feature_count, label_count):
        # if we have used up all data possible, then reset and reuse training data
        if self.ticker_df_curr_row.sum() == -self.ticker_count:
            self.ticker_df_curr_row = np.zeros(self.ticker_count)
            self.curr_ticker = 0

        # if this ticker has used up all of its data, then try the next ticker
        while self.ticker_df_curr_row[self.curr_ticker] == -1:
            self.curr_ticker = (self.curr_ticker + 1) % self.ticker_count

        # Get the current ticker's training data
        ticker_df = self.tickers_df_list[self.curr_ticker]
        curr_row = self.ticker_df_curr_row[self.curr_ticker]
        train_df = ticker_df.iloc[curr_row:curr_row + feature_series_count]

        # get data for ml
        feature_matrix = train_df.as_matrix(columns=dml.get_feature_columns())
        feature_shaped = np.reshape(feature_matrix, [feature_series_count, feature_count])
        label_values = np.array(train_df[dml.get_label_columns()].values[-1])
        label_array = np.reshape(label_values, [1, label_count])
        descriptive_df = train_df.drop(dml.get_feature_columns(), axis=1)
        descriptive_df = descriptive_df.drop(dml.get_label_columns(), axis=1)

        # all data is gathered, setup counters for next time.
        self.ticker_df_curr_row[self.curr_ticker] = self.ticker_df_curr_row[self.curr_ticker] + 1
        if len(ticker_df) < self.ticker_df_curr_row[self.curr_ticker] + feature_series_count:
            self.ticker_df_curr_row[self.curr_ticker] = -1  # we have exhausted this tickers data
        self.curr_ticker = (self.curr_ticker + 1) % self.ticker_count

        return feature_shaped, label_array, descriptive_df
