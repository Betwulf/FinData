import numpy as np
import pandas as pd
import quandl

snp_list = []
with open('S&P500.json', 'rb') as f:
    json = pd.read_json(f)
    snp_list = ['WIKI/' + x['ticker'] for x in json['constituents']]

snp_list.sort()
with open('universe.txt', 'wt', encoding='utf-8') as f:
    for x in snp_list:
        f.write(x + '\n')

data = quandl.get('WIKI/GOOG')
print(data.tail())
