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


# LSTM Parameters

learning_rate = 0.001
epochs = 50000
display_step = 500
n_features = 16
n_input_series = 10  # The number of inputs back-to-back to feed into the RNN
hidden_neurons = 512


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
    series_count = len(ticker_df) - n_input_series
    rnd_series_num = np.random.randint(0, series_count)
    train_df = ticker_df.iloc[rnd_series_num:rnd_series_num+n_input_series]

    # get data for ml
    feature_matrix = train_df.as_matrix(columns=dml.get_feature_columns())
    feature_shaped = np.reshape(feature_matrix, [-1, 10, 16])
    label_value = np.array(train_df[dml.get_label_column()[0]].values[-1])
    label_array = np.reshape(label_value, [1, 1])
    descriptive_df = train_df.drop(dml.get_feature_columns(), axis=1)
    return feature_shaped, label_array, descriptive_df


@timing
def RNN():

    # tf graph input
    x = tf.placeholder("float", [1, n_input_series, n_features])
    y = tf.placeholder("float", [None, 1])

    # RNN output node weights and biases
    weights = {
        'out': tf.Variable(tf.random_normal([hidden_neurons, 1]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([1]))
    }
    # reshape to [1, n_input]
    x2 = tf.reshape(x, [n_input_series, n_features])

    # Generate a n_input-element sequence of inputs
    x3 = tf.split(x2, num_or_size_splits=n_input_series, axis=0)

    # 2-layer LSTM, each layer has n_hidden units.
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(hidden_neurons), rnn.BasicLSTMCell(hidden_neurons)])

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x3, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    prediction = tf.matmul(outputs[-1], weights['out']) + biases['out']

    # Loss and optimizer
    # temp_cost = tf.reduce_mean(y - prediction)
    # cost = -tf.reduce_sum(y * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    # Model evaluation
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as session:
        session.run(init)
        step = 0
        acc_total = 0
        cost_total = 0

        writer.add_graph(session.graph)

        while step < epochs:
            # Generate a minibatch. Add some randomness on selection process.
            feature_data, label_data, descriptive_df = get_next_data()
            outputs_out = session.run(outputs, feed_dict={x: feature_data, y: label_data})
            _, accuracy_out, cost_out, prediction_out = session.run([optimizer, accuracy, cost, prediction],
                                                          feed_dict={x: feature_data, y: label_data})
            cost_total += cost_out
            acc_total += accuracy_out
            if (step+1) % display_step == 0:
                print("Iter= " + str(step+1) + ", Average Loss= " +
                      "{:.6f}".format(cost_total/display_step) + ", Average Accuracy= " +
                      "{:.2f}%".format(100*acc_total/display_step))
                acc_total = 0
                cost_total = 0
                ticker = descriptive_df['ticker'].iloc[-1]
                data_date = descriptive_df['date'].iloc[-1]
                print("Prediction for: {} - {}".format(ticker, data_date))
                print("Actual [%s] vs [%s]" % (label_data, prediction_out))
                print("cost: {}".format(cost_out))
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

