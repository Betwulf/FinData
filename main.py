import numpy as np
import pandas as pd
import quandl
import sys
import os
import datetime
import calendar
import pandas as pd
import quandl
from utils import timing
import data_universe as du
import data_ml as dm
import run_ml as rml
import data_ml as dml
import sim_ml as sml
import training_data as td

""" This project uses Quandl - make sure you have an API key to Quandl in order to access their data """

prompt = " --- Fin Data -- \n"
prompt += "\tType  calc  [ENTER] to calc features and labels\n"
prompt += "\tType  train [ENTER] to train and test the RNN\n"
prompt += "\tType  test  [ENTER] to test the RNN\n"
prompt += "\tType  all   [ENTER] to calc features and labels, then train and test the RNN\n"
prompt += "\tType  sim   [ENTER] to simulate trading\n"
sentence = input(prompt)

main_buy_threshold = 0.9
main_sell_threshold = 0.8

if "calc" in sentence:
    dml.calc_all()
if "train" in sentence:
    rml.get_data_train_and_test_rnn(200000, 200000, main_buy_threshold, main_sell_threshold)
if "test" in sentence:
    rml.get_data_and_test_rnn(200000, 200000, main_buy_threshold, main_sell_threshold)
if "all" in sentence:
    dml.calc_all()
    rml.get_data_train_and_test_rnn(200000, 200000, main_buy_threshold, main_sell_threshold)
if "sim" in sentence:
    a_start_date = rml.test_data_date
    sml.simulate_all(100000.0, a_start_date, datetime.date.today(), main_buy_threshold, main_sell_threshold,
                     -1.4, 0.05, 0.0, rml.prediction_file)

