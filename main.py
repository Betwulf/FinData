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
import training_data as td

""" This project uses Quandl - make sure you have an API key to Quandl in order to access their data """

prompt = "Type 1 [ENTER] to calc features and labels\n"
prompt += "Type 2 [ENTER] to train and test the RNN\n"
prompt += "Type 3 [ENTER] to calc features and labels, then train and test the RNN\n"

sentence = input(prompt)

if "1" in sentence:
    dml.calc_all()
if "2" in sentence:
    rml.get_data_train_and_test_rnn(200000, 200000, 0.6, 0.6)
if "3" in sentence:
    dml.calc_all()
    rml.get_data_train_and_test_rnn(200000, 200000, 0.6, 0.6)

