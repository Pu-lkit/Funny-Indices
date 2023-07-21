import pandas as pd 
import numpy as np



dfi = pd.read_csv('./data_challenge_index_prices.csv')
dfs = pd.read_csv('data_challenge_stock_prices.csv')


rs = dfs.pct_change().dropna()*10000
ri = dfi.pct_change().dropna()*10000



indexList = [0,2,4,5,6,14]
al = pd.read_csv('allocation_matrix.csv')
# print(al)
al = al.drop(0)
al = al.drop(columns=['Unnamed: 0'])
al = al.rename(columns={'0':'0', '1':'2', '2':'4', '3':'5', '4':'6' , '5':'14'})

# print(al)


ri2 = ri.drop(columns=['1', '3', '7','8', '9', '10', '11', '12', '13'])

# print(ri2)

c = al*ri2 
c = c.sum(axis='columns')
c_mean = c.mean(axis='index')
print(c_mean)

c_std = c.std(axis='index') 
print(c_std)

c_sharpe = c_mean/c_std
print(c_sharpe)


























