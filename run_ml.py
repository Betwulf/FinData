import os
import numpy as np
import pandas as pd
import data_ml as dml
from utils import timing
import tensorflow as tf
from tensorflow.contrib import rnn
import datetime
import training_data as td

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# LSTM Input / Output Parameters
feature_count = len(dml.get_feature_columns())
label_count = len(dml.get_label_columns())

# TODO: Turn these into parameters for training
learning_rate = 0.001
epochs = 500000
display_step = 4000
save_step = 100000
test_data_date = datetime.datetime(2016, 6, 30)

# Parameters for LSTM Shape
feature_series_count = 30  # The number of inputs back-to-back to feed into the RNN
hidden_neurons = 512
last_hidden_neurons = 32

# File parameters
_prediction_dir = "\\prediction\\"
_model_dir = "\\model\\"
_combined_filename = "special.json"
_cwd = os.getcwd()
_model_path = _cwd + _model_dir
_model_file = _model_path + 'findata'
_prediction_path = _cwd + _prediction_dir
prediction_file = _prediction_path + _combined_filename

# Target log path
logs_path = './logs'
writer = tf.summary.FileWriter(logs_path)

# ensure paths are there...
if not os.path.exists(_model_path):
    os.makedirs(_model_path)
if not os.path.exists(_prediction_path):
    os.makedirs(_prediction_path)


@timing
def build_rnn():
    # tf graph input
    x = tf.placeholder("float", [feature_series_count, feature_count], name="x")
    y = tf.placeholder("float", [label_count], name="y")

    tf.add_to_collection('vars', x)
    tf.add_to_collection('vars', y)

    # RNN output node weights and biases
    weights = {
        'out': tf.Variable(tf.random_normal([last_hidden_neurons, label_count]), name="out_weights")
    }
    biases = {
        'out': tf.Variable(tf.random_normal([1]), name="out_bias")
    }
    # reshape to [1, n_input]
    x2 = tf.reshape(x, [feature_series_count, feature_count], name="x2")

    # Generate a n_input-element sequence of inputs
    x3 = tf.split(x2, num_or_size_splits=feature_series_count, axis=0, name="x3")

    # 2-layer LSTM, each layer has n_hidden units.
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(hidden_neurons),
                                 rnn.BasicLSTMCell(hidden_neurons),
                                 rnn.BasicLSTMCell(last_hidden_neurons)])

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x3, dtype=tf.float32)

    # there are feature_series_count outputs but
    # we only want the last output
    prediction = tf.add(tf.matmul(outputs[-1], weights['out']), biases['out'], name="prediction")
    # prediction_adjust = tf.round(prediction)

    # Loss and optimizer
    # cost = tf.nn.softmax_cross_entropy_with_logits(logits=prediction[0], labels=y)
    cost = tf.reduce_mean(tf.square(y - prediction[0]), name="cost")
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name="rms_optimizer").minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()
    return init, x, y, prediction, cost, optimizer


@timing
def restore_rnn():
    file_list = [_model_path + a_file for a_file in os.listdir(_model_path) if ".meta" in a_file]
    latest_file = max(file_list, key=os.path.getmtime)
    print("Found file to restore: {}".format(latest_file))

    session = tf.Session()
    saver = tf.train.import_meta_graph(latest_file)
    saver.restore(session, tf.train.latest_checkpoint(_model_path))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    prediction = graph.get_tensor_by_name("prediction:0")
    cost = graph.get_tensor_by_name("cost:0")

    return session, x, y, prediction, cost


@timing
def train_rnn(training_data_cls):
    print("Start training model...")
    init, x, y, prediction, cost, optimizer = build_rnn()
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as session:
        print("Starting tensorflow...")
        session.run(init)
        step = 0
        acc_total = 0.0
        cost_total = 0.0
        cost_df = pd.DataFrame(columns=('iteration', 'cost'))

        writer.add_graph(session.graph)

        while step < epochs:
            # get data
            feature_data, label_data, descriptive_df = training_data_cls.get_next_training_data()

            # Run the Optimizer
            _, cost_out, prediction_out = session.run([optimizer, cost, prediction],
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

                cost_df.loc[cost_df.shape[0]] = [step+1, cost_total/display_step]
                acc_total = 0.0
                cost_total = 0.0
                ticker = descriptive_df['ticker'].iloc[-1]
                data_date = descriptive_df['date'].iloc[-1]
                print("Prediction for: {} - {} (cost: {:1.4f} )".format(ticker, data_date.strftime('%x'), cost_out))
                print("   Buy - Actual {:1.4f} vs {:1.4f} ".format(label_data[0][0], prediction_out[0][0]))
                print("   Sell - Actual {:1.4f} vs {:1.4f} ".format(label_data[0][1], prediction_out[0][1]))
                print("")
                # print("outputs_out: {}".format(outputs_out))
            step += 1

            # Save the variables to disk.
            if (step + 1) % save_step == 0:
                save_path = saver.save(session, _model_file, global_step=step+1)
                print("Model saved in file: %s" % save_path)

        print("Optimization Finished!")
        print("Run on command line.")
        print("\ttensorboard --logdir=%s" % logs_path)
        print("Point your web browser to: http://localhost:6006/")

        # Save the variables to disk.
        save_path = saver.save(session, _model_file, global_step=epochs)
        cost_df.to_csv(_model_path + "cost.csv")
        print("Model saved in file: %s" % save_path)


@timing
def test_rnn(testing_data_cls, test_epochs, test_display_step, buy_threshold, sell_threshold):
    print("Start testing model...")
    session, x, y, prediction, cost = restore_rnn()

    step = 0
    acc_total = 0.0
    cost_total = 0.0
    buy_accuracy_total = 0.0
    sell_accuracy_total = 0.0

    predictions_df = pd.DataFrame(columns=('date', 'ticker',
                                           'buy_prediction', 'buy_signal', 'sell_prediction', 'sell_signal'))

    while step < test_epochs:
        # get data
        feature_data, label_data, descriptive_df = testing_data_cls.get_next_training_data()

        # Run the Optimizer
        cost_out, prediction_out = session.run([cost, prediction],
                                               feed_dict={x: feature_data, y: label_data[0]})

        cost_total += cost_out
        buy_accuracy = abs(((0, 1)[label_data[0][0] > buy_threshold]) -
                           ((0, 1)[prediction_out[0][0] > buy_threshold]))
        sell_accuracy = abs(((0, 1)[label_data[0][1] > sell_threshold]) -
                            ((0, 1)[prediction_out[0][1] > sell_threshold]))
        buy_accuracy_total += (1 - buy_accuracy)
        sell_accuracy_total += (1 - sell_accuracy)
        average_difference = np.mean(np.abs(label_data[0] - prediction_out[0]))
        acc_total += 1 - min([average_difference, 1])

        # save test predictions
        ticker = descriptive_df['ticker'].iloc[-1]
        data_date = descriptive_df['date'].iloc[-1]
        prediction_row = [data_date, ticker, prediction_out[0][0], label_data[0][0], prediction_out[0][1], label_data[0][1]]
        predictions_df.loc[predictions_df.shape[0]] = prediction_row

        if (step + 1) % test_display_step == 0:
            the_curr_time = datetime.datetime.now().strftime('%X')
            print_string = "Time: {}".format(the_curr_time)
            print_string += " Iter= " + str(step + 1)
            print_string += " , Average Loss= {:1.4f}".format(cost_total / test_display_step)
            print_string += " , Average Accuracy= {:3.2f}%".format(100 * acc_total / test_display_step)
            print(print_string)

            print("   Buy  Accuracy: {:2.3f}%".format(100 * buy_accuracy_total / test_display_step))
            print("   Sell Accuracy: {:2.3f}%".format(100 * sell_accuracy_total / test_display_step))
            acc_total = 0.0
            cost_total = 0.0
            buy_accuracy_total = 0.0
            sell_accuracy_total = 0.0
            print("Prediction for: {} - {} (cost: {:1.4f} )".format(ticker, data_date.strftime('%x'), cost_out))
            print("   Buy  - Actual {:1.4f} vs {:1.4f} ".format(label_data[0][0], prediction_out[0][0]))
            print("   Sell - Actual {:1.4f} vs {:1.4f} ".format(label_data[0][1], prediction_out[0][1]))
            print("")
        step += 1
    predictions_df.to_csv(prediction_file)
    print("Testing Finished!")


if __name__ == '__main__':

    # GET DATA
    data_df = dml.get_all_ml_data()
    training_df = data_df[data_df.date < test_data_date].copy()
    test_df = data_df[data_df.date >= test_data_date].copy()
    oh_eight_start_date = datetime.datetime(2008, 1, 1)
    oh_eight_end_date = datetime.datetime(2008, 12, 31)
    test_oh_eight_df = data_df[(oh_eight_start_date <= data_df.date) & (data_df.date <= oh_eight_end_date)].copy()
    del data_df

    # TRAIN
    training_data_class = td.TrainingData(training_df, feature_series_count, feature_count, label_count)
    train_rnn(training_data_class)

    # TEST
    testing_data_class = td.TrainingData(test_df, feature_series_count, feature_count, label_count)
    test_rnn(testing_data_class, 6000, 6000, 0.6, 0.6)
    testing_oh_eight_data_class = td.TrainingData(test_oh_eight_df, feature_series_count, feature_count, label_count)
    test_rnn(testing_oh_eight_data_class, 6000, 6000, 0.6, 0.6)

    # prompt = "Hit enter to finish."
    # sentence = input(prompt)


