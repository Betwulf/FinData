import os
import sys
import glob
import math
import numpy as np
import pandas as pd
import data_ml as dml
from utils import timing
from utils import get_file_friendly_datetime_string
import tensorflow as tf
from tensorflow.contrib import rnn
import datetime
import training_data as td

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# LSTM Input / Output Parameters
feature_count = len(dml.get_feature_columns())
feature_series_count = 1
label_count = len(dml.get_label_columns())

# TODO: Turn these into parameters for training
nn_learning_rate = 0.0001
epochs = 900000  # 1600000
display_step = 20000  # 10000
save_step = 100000  # 100000
test_data_date = datetime.datetime(2017, 6, 30)

# Parameters for LSTM Shape
layers_count = 10
hidden_neurons = 2048

# File parameters
_prediction_dir = "/prediction/"
_model_dir = "/model/"
_prediction_filename = "predictions.csv"
_cwd = "C:/Temp/"
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


@timing
def build_rnn():
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        # tf graph input
        x = tf.placeholder("float", [1, feature_count], name="x")
        y = tf.placeholder("float", [label_count], name="y")
        # reshape to [1, n_input]
        x2 = tf.reshape(x, [1, feature_count], name="x2")

        tf.add_to_collection('vars', x)
        tf.add_to_collection('vars', y)

        layers = []
        last_layer = x2
        for i in range(layers_count):
            new_layer = tf.layers.dense(inputs=last_layer, units=hidden_neurons, activation=tf.nn.relu,
                                        name="layer_{}".format(i))
            layers.append(new_layer)
            last_layer = new_layer

        # Last Layer
        prediction = tf.layers.dense(inputs=last_layer, units=label_count, activation=tf.nn.relu, name="prediction")
        cost = tf.reduce_mean(tf.square(y - prediction), name="cost")

        # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name="rms_optimizer").minimize(cost)
        optimizer = tf.train.AdamOptimizer(learning_rate=nn_learning_rate,
                                           name="adam_optimizer").minimize(cost)  # , global_step=global_step)

        # Initializing the variables
        init = tf.global_variables_initializer()
        return init, x, y, prediction, cost, optimizer, g


def _get_rnn_model_files():
    # [_model_path + a_file for a_file in os.listdir(_model_path) if ".meta" in a_file]
    return [f for f in glob.glob(_model_path + "**/*.meta", recursive=True)]


def _name_model_file_from_path(path, steps):
    return path + 'findata.{}'.format(steps)


def _name_test_meta_file_from_path(path):
    return path + 'findata.test.{}.csv'.format(get_file_friendly_datetime_string())


def _name_run_cost_file_from_path(path):
    return path + 'findata.run_cost.{}.csv'.format(get_file_friendly_datetime_string())


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
        cost = graph.get_tensor_by_name("cost:0")
        prediction = graph.get_tensor_by_name("prediction/Relu:0")
        return session, x, y, prediction, cost, g


@timing
def train_rnn(training_data_cls, train_model_path, use_random_data):
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
            close_enuf_total = 0.0
            less_close_total = 0.0
            cost_total = 0.0
            cost_df = pd.DataFrame(columns=('time', 'iteration', 'cost', 'accuracy2p', 'accuracy3p'))

            writer.add_graph(session.graph)

            while step < epochs:
                # get data
                feature_data, label_data, descriptive_df = \
                    training_data_cls.get_next_training_data(call_random=use_random_data)
                # Adjust feature data for NOT an RNN
                # feature_data = feature_data[0]
                # Run the Optimizer
                _, cost_out, prediction_out = session.run([optimizer, cost, prediction],
                                                          feed_dict={x: feature_data, y: label_data[0]})

                cost_total += cost_out
                if math.isnan(cost_total) or abs(prediction_out[0][0]) > 1000.0:
                    ticker = descriptive_df['ticker'].iloc[-1]
                    data_date = descriptive_df['date'].iloc[-1]
                    print("*** WEIRD *** Prediction for: {} - {} (cost: {:1.4f} )".format(ticker,
                                                                                          data_date.strftime('%x'),
                                                                                          cost_out))
                    print("   Prediction - Actual: {:1.4f} vs {:1.4f} ".format(label_data[0][0], prediction_out[0][0]))
                    if math.isnan(cost_total):
                        break

                # average_difference = np.mean(np.abs(label_data[0] - prediction_out[0]))
                close_enough = (prediction_out[0][0] - 0.02) < label_data[0][0] < (prediction_out[0][0] + 0.02)
                less_close = (prediction_out[0][0] - 0.03) < label_data[0][0] < (prediction_out[0][0] + 0.03)
                less_close_total += less_close
                close_enuf_total += close_enough
                if (step + 1) % display_step == 0:
                    the_curr_time = datetime.datetime.now().strftime('%X')
                    print_string = "Time: {}".format(the_curr_time)
                    print_string += " Iter= " + str(step + 1)
                    print_string += " , Loss= {:1.4f}".format(cost_total / display_step)
                    print(print_string)

                    print_string = "   2% Accuracy: {:3.2f}%".format(100 * close_enuf_total / display_step)
                    print(print_string)
                    print_string = "   3% Accuracy: {:3.2f}%".format(100 * less_close_total / display_step)
                    print(print_string)

                    cost_df.loc[cost_df.shape[0]] = [get_file_friendly_datetime_string(), step + 1,
                                                     cost_total / display_step, 100 * close_enuf_total / display_step,
                                                     100 * less_close_total / display_step]
                    close_enuf_total = 0.0
                    less_close_total = 0.0
                    cost_total = 0.0
                    # ticker = descriptive_df['ticker'].iloc[-1]
                    # data_date = descriptive_df['date'].iloc[-1]
                    # print("Prediction for: {} - {} (cost: {:1.4f} )".format(ticker, data_date.strftime('%x'), cost_out))
                    # print("    - Actual: {:1.4f} vs Prediction: {:1.4f} ".format(label_data[0][0], prediction_out[0][0]))
                    # print("   Sell - Actual: {:1.4f} vs {:1.4f} ".format(label_data[0][1], prediction_out[0][1]))
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
            # Save the meta data of the run to disk...
            with open(_name_run_cost_file_from_path(train_model_path), 'wt') as f:
                f.write(cost_df.to_csv())

            # Can't make cost files in model path anymore, will throw off prediction aggregation
            # cost_df.to_csv(train_model_path + "cost.csv")
            print("Model saved in file: %s" % save_path)
            return save_path


def predict_rnn(after_date, specific_files=None):
    the_curr_time = datetime.datetime.now().strftime('%X')
    print_string = "Time: {}".format(the_curr_time)
    print("START PREDICTING MODEL...{}".format(print_string))

    data_df = dml.get_all_predictable_data()
    test_df = data_df[data_df.date >= after_date].copy()
    del data_df
    predict_data_cls = td.TrainingData(test_df, feature_series_count, feature_count, label_count)

    predictions_df = pd.DataFrame(columns=('model_file', 'date', 'ticker', 'prediction'))

    file_list = _get_rnn_model_files()
    if specific_files is None:
        print("Got these files:")
        print(file_list)
    else:
        file_list = specific_files

    for each_file in file_list:
        session, x, y, prediction, cost, tf_graph = restore_rnn(each_file)
        with tf_graph.as_default():
            feature_data = 'first run'
            while feature_data is not None:
                feature_data, label_data, descriptive_df = predict_data_cls.get_next_training_data(until_exhausted=True)
                if feature_data is None:
                    print(" --- Data Exhausted --- ")
                    the_curr_time = datetime.datetime.now().strftime('%X')
                    print_string = "Time: {}".format(the_curr_time)
                    print(print_string)
                    break

                prediction_out = session.run([prediction], feed_dict={x: feature_data})[0]
                ticker = descriptive_df['ticker'].iloc[-1]
                data_date = descriptive_df['date'].iloc[-1]
                predictions_df.loc[predictions_df.shape[0]] = [each_file, data_date,
                                                               ticker, prediction_out[0][0]]
    with open(_prediction_path + get_file_friendly_datetime_string() + "." + _prediction_filename, 'wt') as f:
        f.write(predictions_df.to_csv())


@timing
def test_rnn(testing_data_cls, test_epochs, test_display_step, buy_threshold, sell_threshold, specific_files=None):
    the_curr_time = datetime.datetime.now().strftime('%X')
    print_string = "Time: {}".format(the_curr_time)
    print("START TESTING MODEL...{}".format(print_string))

    file_list = _get_rnn_model_files()
    if specific_files is None:
        print("Got these files:")
        print(file_list)
    else:
        file_list = specific_files

    test_cost_df = pd.DataFrame(columns=('time', 'file', 'iteration', 'cost', 'accuracy2p', 'accuracy3p'))
    for each_file in file_list:
        predictions_df = pd.DataFrame(columns=('model_file', 'date', 'ticker',
                                               'prediction', 'label'))
        session, x, y, prediction, cost, tf_graph = restore_rnn(each_file)
        with tf_graph.as_default():
            step = 0
            curr_display_steps = 0
            close_enuf_total = 0.0
            less_close_total = 0.0
            close_positive_total = 0.0
            cost_total = 0.0
            buy_accuracy_total = 0.0

            # sell_accuracy_total = 0.0

            while step < test_epochs:
                # get data
                feature_data, label_data, descriptive_df = testing_data_cls.get_next_training_data(until_exhausted=True)

                if feature_data is None:
                    print(" --- Data Exhausted --- ")
                    the_curr_time = datetime.datetime.now().strftime('%X')
                    print_string = "Time: {}".format(the_curr_time)
                    print_string += " Iter= " + str(step + 1)
                    print_string += " , Average Loss= {:1.4f}".format(cost_total / curr_display_steps)
                    print(print_string)
                    print("   test step: {} - curr step: {}".format(test_display_step, curr_display_steps))
                    print("       2% Accuracy: {:3.2f}%".format(100 * close_enuf_total / test_display_step))
                    print("       3% Accuracy: {:3.2f}%".format(100 * less_close_total / test_display_step))
                    print("       2% Pos Accu: {:3.2f}%".format(100 * close_positive_total / test_display_step))

                    test_cost_df.loc[test_cost_df.shape[0]] = [get_file_friendly_datetime_string(), each_file,
                                                               step + 1, cost_total / curr_display_steps,
                                                               100 * close_enuf_total / curr_display_steps,
                                                               100 * less_close_total / curr_display_steps]
                    break

                # Run the Optimizer
                cost_out, prediction_out = session.run([cost, prediction],
                                                       feed_dict={x: feature_data, y: label_data[0]})

                cost_total += cost_out
                close_enough = (prediction_out[0][0] - 0.02) < label_data[0][0] < (prediction_out[0][0] + 0.02)
                less_close = (prediction_out[0][0] - 0.03) < label_data[0][0] < (prediction_out[0][0] + 0.03)
                close_positive = (close_enough or (prediction_out[0][0] - 0.52) * (label_data[0][0] - 0.50)) > 0
                less_close_total += less_close
                close_enuf_total += close_enough
                close_positive_total += close_positive

                # save test predictions
                ticker = descriptive_df['ticker'].iloc[-1]
                data_date = descriptive_df['date'].iloc[-1]
                prediction_row = [each_file, data_date, ticker,
                                  prediction_out[0][0], label_data[0][0]]
                predictions_df.loc[predictions_df.shape[0]] = prediction_row

                if (step + 1) % test_display_step == 0:
                    the_curr_time = datetime.datetime.now().strftime('%X')
                    print_string = "Time: {}".format(the_curr_time)
                    print_string += " Iter= " + str(step + 1)
                    print_string += " , Average Loss= {:1.4f}".format(cost_total / test_display_step)
                    print(print_string)

                    print("   2% Accuracy: {:3.2f}%".format(100 * close_enuf_total / test_display_step))
                    print("   3% Accuracy: {:3.2f}%".format(100 * less_close_total / test_display_step))
                    print("   2% Pos Accu: {:3.2f}%".format(100 * close_positive_total / test_display_step))

                    test_cost_df.loc[test_cost_df.shape[0]] = [get_file_friendly_datetime_string(), each_file,
                                                               step + 1, cost_total / test_display_step,
                                                               100 * close_enuf_total / test_display_step,
                                                               100 * less_close_total / test_display_step]
                    close_enuf_total = 0.0
                    less_close_total = 0.0
                    close_positive_total = 0.0
                    cost_total = 0.0
                    buy_accuracy_total = 0.0
                    # sell_accuracy_total = 0.0
                    curr_display_steps = -1
                    print("")
                step += 1
                curr_display_steps += 1
            print("{} Testing Finished!".format(each_file))
            print("Saving Predictions...")
            predictions_df.to_csv(get_prediction_filename(each_file))
            predictions_df.drop(predictions_df.index, inplace=True, axis=0)
            del predictions_df
            session.close()
    # Save the meta data of the run to disk...
    with open(_name_test_meta_file_from_path(_model_path), 'wt') as f:
        f.write(test_cost_df.to_csv())
    print("ALL TESTS FINISHED.")


def train_and_test_by_ticker(test_epochs, test_display_step, buy_threshold, sell_threshold, use_random_data):
    # GET DATA
    data_df = dml.get_all_ml_data()
    tickers = list({t for t in data_df['ticker']})
    training_df = data_df[data_df.date < test_data_date].copy()
    test_df = data_df[data_df.date >= test_data_date].copy()
    del data_df
    prediction_files = []
    print("BULK TRAINING for {} tickers.".format(len(tickers)))

    for ticker in tickers:
        try:
            print(" ----- Begin Training for {} ----- ".format(ticker))
            # DATA
            training_data_class = td.TrainingDataTicker(training_df, feature_series_count,
                                                        feature_count, label_count, ticker)
            testing_data_class = td.TrainingDataTicker(test_df, feature_series_count,
                                                       feature_count, label_count, ticker)
            # TRAIN
            ticker_path = _model_path + ticker + '/'
            if os.path.exists(ticker_path):
                print("Found ticker directory, skipping {}".format(ticker))
                # add existing file to predictions list
                file_list = [ticker_path + a_file for a_file in os.listdir(ticker_path)]
                if len(file_list) > 0:
                    latest_file = max(file_list, key=os.path.getmtime)
                    if latest_file.endswith('.csv'):
                        print("adding prediction file: {}".format(latest_file))
                        prediction_files.append(latest_file)
                    else:
                        print("couldn't find csv as latest file in {}".format(ticker_path))
            else:
                os.makedirs(ticker_path)
                saved_model_file = train_rnn(training_data_class, ticker_path, use_random_data)
                saved_model_file = saved_model_file + '.meta'
                prediction_files.append(saved_model_file + '.csv')

                # TEST
                test_rnn(testing_data_class, test_epochs, test_display_step,
                         buy_threshold, sell_threshold, [saved_model_file])
        except ValueError as ve:
            print(ve)
    # GENERATE PREDICTION AGGREGATE FILE
    merge_predictions(prediction_files)


def merge_predictions(prediction_files):
    file_out = open(prediction_file, "wt", encoding='utf-8')
    try:
        # first file:
        for line in open(prediction_files[0], "rt", encoding='utf-8'):
            file_out.write(line)
        # now the rest:
        for num in range(1, len(prediction_files) - 1):
            f = open(prediction_files[num], "rt", encoding='utf-8')
            f.readline()  # skip the header
            for line in f:
                file_out.write(line)
            f.close()  # not really needed
    except ValueError as ve:
        print(ve)
    finally:
        file_out.close()


def get_data_train_and_test_rnn(test_epochs, test_display_step, buy_threshold, sell_threshold, use_random_data):
    # GET DATA
    data_df = dml.get_all_ml_data()
    training_df = data_df[data_df.date < test_data_date].copy()
    test_df = data_df[data_df.date >= test_data_date].copy()
    del data_df
    # TRAIN
    training_data_class = td.TrainingData(training_df, feature_series_count, feature_count, label_count)
    # TODO: switch rnn to use batch data, testing below
    # fff, lll, ddd = training_data_class.get_batch(3)
    train_rnn(training_data_class, _model_path, use_random_data)
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


def get_data_and_test_rnn_by_ticker(test_epochs, test_display_step, buy_threshold, sell_threshold, specific_file):
    # Get ticker
    parse_filename = specific_file.replace('\\', "/")
    print(parse_filename)
    ticker = "WIKI/" + parse_filename.split("/")[-2]
    print("Using ticker - {}".format(ticker))
    # GET DATA
    data_df = dml.get_all_ml_data()
    test_df = data_df[data_df.date >= test_data_date].copy()
    del data_df
    # TEST
    testing_data_class = td.TrainingDataTicker(test_df, feature_series_count, feature_count, label_count, ticker)
    test_rnn(testing_data_class, test_epochs, test_display_step, buy_threshold, sell_threshold, [specific_file])


def get_data_and_test_all_tickers(test_epochs, test_display_step, buy_threshold, sell_threshold):
    # TODO: instead of a file search, use checkpoints to get latest meta in each directory
    meta_files = [f for f in glob.glob(_model_path + "**/*.meta", recursive=True)]
    print(meta_files)
    for file in meta_files:
        get_data_and_test_rnn_by_ticker(test_epochs, test_display_step, buy_threshold, sell_threshold, file)
    # merge predictions
    prediction_files = [f for f in glob.glob(_model_path + "**/*.csv", recursive=True)]
    merge_predictions(prediction_files)


if __name__ == '__main__':
    if len(sys.argv) > 2:
        if sys.argv[1] == "testticker":
            get_data_and_test_rnn_by_ticker(200000, 200000, 0.6, 0.6, [str(sys.argv[2])])
        elif sys.argv[1] == "test":
            get_data_and_test_rnn(200000, 200000, 0.03, 0.02, [str(sys.argv[2])])
        elif sys.argv[1] == "predict":
            predict_rnn(datetime.datetime.today() - datetime.timedelta(days=27 + feature_series_count),
                        [str(sys.argv[2])])
    elif len(sys.argv) > 1:
        if sys.argv[1] == "testticker":
            get_data_and_test_all_tickers(200000, 200000, 0.6, 0.6)
        elif sys.argv[1] == "test":
            get_data_and_test_rnn(200000, 200000, 0.03, 0.02)
        elif sys.argv[1] == "predict":
            predict_rnn(datetime.datetime.today() - datetime.timedelta(days=27 + feature_series_count))
    else:
        # get_data_and_test_rnn(20000, 20000, 0.03, 0.02)
        # predict_rnn(datetime.datetime.today() - datetime.timedelta(days=27 + feature_series_count))
        # train_and_test_by_ticker(4000, 4000, 0.6, 0.6)
        get_data_train_and_test_rnn(200000, 200000, 0.03, 0.02, use_random_data=False)
