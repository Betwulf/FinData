
import requests
import pandas as pd

iex_base_url = "https://api.iextrading.com/1.0/"
ticker = "TSLA"
iex_data = requests.get(iex_base_url+"deep/trades?symbols="+ticker, "").text
trades_df = pd.read_json(iex_data[8:-1])



