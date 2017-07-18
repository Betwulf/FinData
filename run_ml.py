import os
import pandas as pd
import numpy as np
import data_universe as du
import data_ml as dml
from utils import timing
import tensorflow as tf
from tensorflow.contrib import rnn
import collections
import time
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# LSTM Parameters
learning_rate = 0.001
epochs = 400000
display_step = 2000
feature_count = 10
feature_series_count = 30  # The number of inputs back-to-back to feed into the RNN
hidden_neurons = 512
last_hidden_neurons = 32


# Target log path
logs_path = './logs'
writer = tf.summary.FileWriter(logs_path)

# get the dataframe, this may be a lot of data....
data_df = dml.get_all_ml_data()


def get_next_data():
    # Get random ticker
    tickers = list({t for t in data_df['ticker']})
    rnd_ticker_num = np.random.randint(0, len(tickers))
    ticker_df = data_df[data_df.ticker == tickers[rnd_ticker_num]]

    # get random series for this ticker
    series_count = len(ticker_df) - feature_series_count
    rnd_series_num = np.random.randint(0, series_count)
    train_df = ticker_df.iloc[rnd_series_num:rnd_series_num + feature_series_count]

    # get data for ml
    feature_matrix = train_df.as_matrix(columns=dml.get_feature_columns())
    feature_shaped = np.reshape(feature_matrix, [feature_series_count, feature_count])
    label_value = np.array(train_df[dml.get_label_column()[0]].values[-1])
    label_array = np.reshape(label_value, [1, 1])
    descriptive_df = train_df.drop(dml.get_feature_columns(), axis=1)
    return feature_shaped, label_array, descriptive_df


@timing
def RNN():

    # tf graph input
    x = tf.placeholder("float", [feature_series_count, feature_count])
    y = tf.placeholder("float", [1])

    # RNN output node weights and biases
    weights = {
        'out': tf.Variable(tf.random_normal([last_hidden_neurons, 1]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([1]))
    }
    # reshape to [1, n_input]
    x2 = tf.reshape(x, [feature_series_count, feature_count])

    # Generate a n_input-element sequence of inputs
    x3 = tf.split(x2, num_or_size_splits=feature_series_count, axis=0)

    # 2-layer LSTM, each layer has n_hidden units.
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(hidden_neurons),
                                 rnn.BasicLSTMCell(hidden_neurons),
                                 rnn.BasicLSTMCell(last_hidden_neurons)])

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x3, dtype=tf.float32)

    # there are feature_series_count outputs but
    # we only want the last output
    prediction = (tf.matmul(outputs[-1], weights['out']) + biases['out'])
    # prediction_adjust = tf.round(prediction)

    # Loss and optimizer
    # temp_cost = tf.reduce_mean(y - prediction)
    # cost = -tf.reduce_sum(y * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
    cost = tf.reduce_mean(tf.square(y - prediction[0]))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as session:
        print("Starting tensorflow...")
        session.run(init)
        step = 0
        acc_total = 0
        cost_total = 0

        writer.add_graph(session.graph)

        while step < epochs:
            # Generate a minibatch. Add some randomness on selection process.
            feature_data, label_data, descriptive_df = get_next_data()
            _, cost_out, prediction_out = session.run([optimizer, cost, prediction],
                                                          feed_dict={x: feature_data, y: label_data[0]})

            cost_total += cost_out
            acc_total += np.abs(label_data[0][0] - prediction_out[0][0])
            if ((step+1) % display_step == 0) or step < 10:
                the_curr_time = datetime.datetime.now().strftime('%X')
                print_string = "Time: {}".format(the_curr_time)
                print_string += " Iter= " + str(step+1)
                print_string += " , Average Loss= {:1.4f}".format(cost_total/display_step)
                print_string += " , Average Accuracy= {:3.2f}%".format(100*acc_total/display_step)

                print(print_string)
                acc_total = 0
                cost_total = 0
                ticker = descriptive_df['ticker'].iloc[-1]
                data_date = descriptive_df['date'].iloc[-1]
                print("Prediction for: {} - {}".format(ticker, data_date.strftime('%x')))
                print("Actual {:1.4f} vs {:1.4f} (cost {:1.4f} )".format(label_data[0][0], prediction_out[0][0], cost_out))
                print("")
                # print("outputs_out: {}".format(outputs_out))
            step += 1
        print("Optimization Finished!")
        print("Run on command line.")
        print("\ttensorboard --logdir=%s" % logs_path)
        print("Point your web browser to: http://localhost:6006/")

        prompt = "Hit enter to finish."
        sentence = input(prompt)


if __name__ == '__main__':
    RNN()

