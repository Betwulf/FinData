import pandas as pd
import io
import requests

url = "http://www.stockpup.com/data/A_quarterly_financial_data.csv"
s = requests.get(url).content
c = pd.read_csv(io.StringIO(s.decode('utf-8')))
print(c)

