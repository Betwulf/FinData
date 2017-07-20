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
import training_data as td

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# LSTM Parameters
learning_rate = 0.001
epochs = 400000
display_step = 2000
label_count = 2
feature_count = 10
feature_series_count = 30  # The number of inputs back-to-back to feed into the RNN
hidden_neurons = 512
last_hidden_neurons = 32
test_data_date = datetime.datetime(2016, 6, 30)
model_file = './model/model.ckpt'

# Target log path
logs_path = './logs'
writer = tf.summary.FileWriter(logs_path)




@timing
def RNN(training_data_class):

    # tf graph input
    x = tf.placeholder("float", [feature_series_count, feature_count])
    y = tf.placeholder("float", [label_count])

    # RNN output node weights and biases
    weights = {
        'out': tf.Variable(tf.random_normal([last_hidden_neurons, label_count]))
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
    # cost = tf.reduce_mean(tf.square(y - prediction[0]))
    sub_cost = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
    cost = tf.reduce_mean(sub_cost)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as session:
        print("Starting tensorflow...")
        session.run(init)
        step = 0
        acc_total = 0.0
        cost_total = 0.0

        writer.add_graph(session.graph)

        while step < epochs:
            # get data
            feature_data, label_data, descriptive_df = \
                training_data_class.get_next_training_data(feature_series_count, feature_count, label_count)

            # Run the Optimizer
            _, sub_cost_out, cost_out, prediction_out = session.run([optimizer, sub_cost, cost, prediction],
                                                          feed_dict={x: feature_data, y: label_data[0]})

            cost_total += cost_out
            average_difference = np.mean(np.abs(label_data[0] - prediction_out[0]))
            acc_total += 1 - min([average_difference, 1])
            if ((step+1) % display_step == 0) or step == 0:
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
                print("   Buy - Actual {:1.4f} vs {:1.4f} ".format(label_data[0][0], prediction_out[0][0]))
                print("   Sell - Actual {:1.4f} vs {:1.4f} ".format(label_data[0][1], prediction_out[0][1]))
                print("sub cost: {:1.4f} - cost: {:1.4f}".format(sub_cost_out[0], cost_out))
                print("")
                # print("outputs_out: {}".format(outputs_out))
            step += 1
        print("Optimization Finished!")
        print("Run on command line.")
        print("\ttensorboard --logdir=%s" % logs_path)
        print("Point your web browser to: http://localhost:6006/")

        prompt = "Hit enter to finish."
        sentence = input(prompt)

        saver = tf.train.Saver()
        # Save the variables to disk.
        save_path = saver.save(session, model_file)
        print("Model saved in file: %s" % save_path)


if __name__ == '__main__':

    # get the dataframe, this may be a lot of data....
    data_df = dml.get_all_ml_data()
    training_df = data_df[data_df.date < test_data_date]
    test_df = data_df[data_df.date >= test_data_date]
    del data_df

    training_data_class = td.TrainingData(training_df)
    RNN(training_data_class)

