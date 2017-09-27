import os
import sys
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
epochs = 250000  # 1600000
display_step = 10000  # 10000
save_step = 50000  # 100000
test_data_date = datetime.datetime(2016, 6, 30)

# Parameters for LSTM Shape
feature_series_count = 30  # The number of inputs back-to-back to feed into the RNN, aka Batch size, sequence length
hidden_neurons = 128
last_hidden_neurons = 32

# File parameters
_prediction_dir = "/prediction/"
_model_dir = "/model/"
_prediction_filename = "predictions.csv"
_cwd = os.getcwd()
_model_path = _cwd + _model_dir
_prediction_path = _cwd + _prediction_dir
prediction_file = _prediction_path + _prediction_filename

# Target log path
logs_path = './logs'
writer = tf.summary.FileWriter(logs_path)

# ensure paths are there...
if not os.path.exists(_model_path):
    os.makedirs(_model_path)
if not os.path.exists(_prediction_path):
    os.makedirs(_prediction_path)


def get_prediction_filename(a_model_file):
    return '{}{}'.format(a_model_file, '.csv')


def get_state_variables(batch_size, cell):
    # For each layer, get the initial state and make a variable out of it
    # to enable updating its value.
    state_variables = []
    for state_c, state_h in cell.zero_state(batch_size, tf.float32):
        state_variables.append(rnn.LSTMStateTuple(
            tf.Variable(state_c, trainable=True),
            tf.Variable(state_h, trainable=True)))
    # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
    return tuple(state_variables)


@timing
def build_rnn():
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        # global_step - Created here, not sure if i increment, or if minimize does
        # TODO: pass back global_step to increment? and use in saver
        global_step = tf.Variable(0, name='global_step', trainable=False)

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
            'out': tf.Variable(tf.random_normal([label_count]), name="out_bias")
        }
        # reshape to [1, n_input]
        x2 = tf.reshape(x, [feature_series_count, feature_count], name="x2")

        # Generate a n_input-element sequence of inputs
        x3 = tf.split(x2, num_or_size_splits=feature_series_count, axis=0, name="x3")

        # 2-layer LSTM, each layer has n_hidden units.
        rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(hidden_neurons, activation=tf.nn.relu),
                                     rnn.BasicLSTMCell(hidden_neurons, activation=tf.nn.relu),
                                     rnn.BasicLSTMCell(last_hidden_neurons, activation=tf.nn.relu)])

        # trying to save state or rnn, this should work
        # states = get_state_variables(feature_count, rnn_cell)

        # generate prediction
        # outputs, states = tf.nn.dynamic_rnn(rnn_cell, x3, sequence_length=[feature_series_count], dtype=tf.float32)
        outputs, states = rnn.static_rnn(rnn_cell, x3, dtype=tf.float32)

        # there are feature_series_count outputs but
        # we only want the last output
        prediction = tf.add(tf.matmul(outputs[-1], weights['out']), biases['out'], name="prediction")
        # prediction_adjust = tf.round(prediction)

        # Loss and optimizer
        # cost = tf.nn.softmax_cross_entropy_with_logits(logits=prediction[0], labels=y)
        cost = tf.reduce_mean(tf.square(y - prediction[0]), name="cost")
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name="rms_optimizer").minimize(cost)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001,
                                           name="adam_optimizer").minimize(cost, global_step=global_step)

        # Initializing the variables
        init = tf.global_variables_initializer()
        return init, x, y, prediction, cost, optimizer, g


def _get_rnn_model_files():
    return [_model_path + a_file for a_file in os.listdir(_model_path) if ".meta" in a_file]


def _name_model_file_from_path(path, steps):
    return path + 'findata.{}'.format(steps)


@timing
def restore_rnn(meta_file):

    print("RNN model to restore: {}".format(meta_file))
    tf_data_path = meta_file[:-5]  # take off the '.meta' ending
    g = tf.Graph()
    with g.as_default():
        session = tf.Session()
        saver = tf.train.import_meta_graph(meta_file)
        # latest_dataset = tf.train.latest_checkpoint(checkpoint_directory)
        # checkpoint_state = tf.train.get_checkpoint_state(meta_file)
        # From the checkpoint state you could get all the tf datasets in the dir, but dont need it now
        saver.restore(session, tf_data_path)
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        prediction = graph.get_tensor_by_name("prediction:0")
        cost = graph.get_tensor_by_name("cost:0")
        return session, x, y, prediction, cost, g


@timing
def train_rnn(training_data_cls, train_model_path):
    the_curr_time = datetime.datetime.now().strftime('%X')
    print_string = "Time: {}".format(the_curr_time)
    print("Start training model... {}".format(print_string))
    init, x, y, prediction, cost, optimizer, tf_graph = build_rnn()
    with tf_graph.as_default():
        saver = tf.train.Saver(max_to_keep=20)

        # Launch the graph
        with tf.Session() as session:
            print("Starting tensorflow...")
            session.run(init)
            step = 0
            acc_total = 0.0
            cost_total = 0.0
            cost_df = pd.DataFrame(columns=('iteration', 'cost', 'accuracy'))

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
                if (step+1) % display_step == 0:
                    the_curr_time = datetime.datetime.now().strftime('%X')
                    print_string = "Time: {}".format(the_curr_time)
                    print_string += " Iter= " + str(step+1)
                    print_string += " , Average Loss= {:1.4f}".format(cost_total/display_step)
                    print_string += " , Average Accuracy= {:3.2f}%".format(100*acc_total/display_step)
                    print(print_string)

                    cost_df.loc[cost_df.shape[0]] = [step+1, cost_total/display_step, 100*acc_total/display_step]
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
                    save_path = saver.save(session, _name_model_file_from_path(train_model_path, step + 1))
                    print("Model saved in file: %s" % save_path)

            print("      ***** Optimization Finished! ***** ")

            # Save the variables to disk.
            save_path = saver.save(session, _name_model_file_from_path(train_model_path, epochs))
            cost_df.to_csv(train_model_path + "cost.csv")
            print("Model saved in file: %s" % save_path)
            return save_path


@timing
def test_rnn(testing_data_cls, test_epochs, test_display_step, buy_threshold, sell_threshold, specific_files=None):
    the_curr_time = datetime.datetime.now().strftime('%X')
    print_string = "Time: {}".format(the_curr_time)
    print("Start testing model...{}".format(print_string))

    predictions_df = pd.DataFrame(columns=('model_file', 'date', 'ticker',
                                           'buy_prediction', 'buy_signal', 'sell_prediction', 'sell_signal'))

    file_list = _get_rnn_model_files()
    if specific_files is None:
        file_list = [max(file_list, key=os.path.getmtime)]
    else:
        file_list = specific_files

    for each_file in file_list:
        session, x, y, prediction, cost, tf_graph = restore_rnn(each_file)
        with tf_graph.as_default():
            step = 0
            curr_display_steps = 0
            acc_total = 0.0
            cost_total = 0.0
            buy_accuracy_total = 0.0
            sell_accuracy_total = 0.0

            while step < test_epochs:
                # get data
                feature_data, label_data, descriptive_df = testing_data_cls.get_next_training_data(until_exhausted=True)

                if feature_data is None:
                    print(" --- Data Exhausted --- ")
                    the_curr_time = datetime.datetime.now().strftime('%X')
                    print_string = "Time: {}".format(the_curr_time)
                    print_string += " Iter= " + str(step + 1)
                    print_string += " , Average Loss= {:1.4f}".format(cost_total / curr_display_steps)
                    print_string += " , Average Accuracy= {:3.2f}%".format(100 * acc_total / curr_display_steps)
                    print(print_string)

                    print("   Buy  Accuracy: {:2.3f}%".format(100 * buy_accuracy_total / curr_display_steps))
                    print("   Sell Accuracy: {:2.3f}%".format(100 * sell_accuracy_total / curr_display_steps))
                    break

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
                prediction_row = [each_file, data_date, ticker,
                                  prediction_out[0][0], label_data[0][0], prediction_out[0][1], label_data[0][1]]
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
                    curr_display_steps = -1
                    print("Prediction for: {} - {} (cost: {:1.4f} )".format(ticker, data_date.strftime('%x'), cost_out))
                    print("   Buy  - Actual {:1.4f} vs {:1.4f} ".format(label_data[0][0], prediction_out[0][0]))
                    print("   Sell - Actual {:1.4f} vs {:1.4f} ".format(label_data[0][1], prediction_out[0][1]))
                    print("")
                step += 1
                curr_display_steps += 1
            print("{} Testing Finished!".format(each_file))
            print("Saving Predictions...")
            predictions_df.to_csv(get_prediction_filename(each_file))
            session.close()
    print("ALL TESTS FINISHED.")


def train_and_test_by_ticker(test_epochs, test_display_step, buy_threshold, sell_threshold):
    # GET DATA
    data_df = dml.get_all_ml_data()
    tickers = list({t for t in data_df['ticker']})
    training_df = data_df[data_df.date < test_data_date].copy()
    test_df = data_df[data_df.date >= test_data_date].copy()
    del data_df

    for ticker in tickers:
        print(" ----- Begin Training for {} ----- ".format(ticker))
        # TRAIN
        training_data_class = td.TrainingDataTicker(training_df, feature_series_count,
                                                    feature_count, label_count, ticker)
        ticker_path = _model_path + ticker + '/'
        if not os.path.exists(ticker_path):
            os.makedirs(ticker_path)
        saved_model_file = train_rnn(training_data_class, ticker_path)
        saved_model_file = saved_model_file + '.meta'

        # TEST
        testing_data_class = td.TrainingDataTicker(test_df, feature_series_count,
                                                   feature_count, label_count, ticker)
        test_rnn(testing_data_class, test_epochs, test_display_step, buy_threshold, sell_threshold, [saved_model_file])


def get_data_train_and_test_rnn(test_epochs, test_display_step, buy_threshold, sell_threshold):
    # GET DATA
    data_df = dml.get_all_ml_data()
    training_df = data_df[data_df.date < test_data_date].copy()
    test_df = data_df[data_df.date >= test_data_date].copy()
    del data_df

    # TRAIN
    training_data_class = td.TrainingData(training_df, feature_series_count, feature_count, label_count)
    # TODO: switch rnn to use batch data, testing below
    # fff, lll, ddd = training_data_class.get_batch(3)
    train_rnn(training_data_class, _model_path)

    # TEST
    testing_data_class = td.TrainingData(test_df, feature_series_count, feature_count, label_count)
    test_rnn(testing_data_class, test_epochs, test_display_step, buy_threshold, sell_threshold)


def get_data_and_test_rnn(test_epochs, test_display_step, buy_threshold, sell_threshold, specific_file=None):
    # GET DATA
    data_df = dml.get_all_ml_data()
    test_df = data_df[data_df.date >= test_data_date].copy()
    del data_df

    # TEST
    testing_data_class = td.TrainingData(test_df, feature_series_count, feature_count, label_count)
    test_rnn(testing_data_class, test_epochs, test_display_step, buy_threshold, sell_threshold, specific_file)


if __name__ == '__main__':
    if len(sys.argv) > 2:
        get_data_and_test_rnn(4000, 4000, 0.8, 0.7, [sys.argv[2]])
    elif len(sys.argv) > 1:
        if sys.argv[1] == "test":
            get_data_and_test_rnn(4000, 4000, 0.8, 0.7)
    else:
        train_and_test_by_ticker(4000, 4000, 0.8, 0.7)
