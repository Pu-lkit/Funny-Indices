import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


dfi = pd.read_csv('./data_challenge_index_prices.csv')
dfs = pd.read_csv('data_challenge_stock_prices.csv')


ri = dfs.pct_change().dropna()*10000
Ri = dfi.pct_change().dropna()*10000



# CALCULATING RETURNS FOR EACH STOCK, STORING IT IN "ret_df"
ri_trans = ri.transpose()
x = ri_trans

# CLUSTERING STOCKS USING KMEANS ALGORITHM
n_clusters = 4
kmeans = KMeans(n_clusters = n_clusters, n_init = 10, random_state = 42)
kmeans.fit(ri_trans)
x['sector'] = kmeans.labels_

# MAKING A LIST OF SECTORS TO SEPARATE EACH CLUSTER
sectors = []
for i in range(n_clusters):
    sector = x[x['sector']==i].drop('sector', axis=1).transpose()
    sectors.append(sector)

#list_of_sector_numbers = [1l, 0, 1r, 2, 3l, 0r, 2l, 1, 2, 3, 0, 1, 3, 2, 3r]
list_of_sector_numbers = [1, 0, 1, 2, 3, 0, 2, 1, 2, 3, 0, 1, 3, 2, 3]

predictors = [None]*15



def cosaa(lr, X, y):
    y_pred = lr.predict(X)
    y_test_np = y.values

    dot_product = np.dot(y_pred, y_test_np)

    # calculate the magnitudes of the two vectors
    magnitude1 = np.linalg.norm(y_pred)
    magnitude2 = np.linalg.norm(y_test_np)

    # calculate the cosine of the angle between the two vectors
    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    print(cosine_similarity)



X_train, X_test, y_train, y_test = train_test_split(sectors[1], Ri['0'], test_size=0.3, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
predictors[0] = lr
cosaa(lr, X_test, y_test)

X_train, X_test, y_train, y_test = train_test_split(sectors[3], Ri['4'], test_size=0.3, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
predictors[4] = lr
cosaa(lr, X_test, y_test)

X_train, X_test, y_train, y_test = train_test_split(sectors[2], Ri['6'], test_size=0.3, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
predictors[6] = lr
cosaa(lr, X_test, y_test)

X_train, X_test, y_train, y_test = train_test_split(sectors[1], Ri['2'], test_size=0.3, random_state=42)
rf = RandomForestRegressor(n_estimators=10, random_state=0)
rf.fit(X_train, y_train)
predictors[2] = rf
cosaa(rf, X_test, y_test)

X_train, X_test, y_train, y_test = train_test_split(sectors[0], Ri['5'], test_size=0.3, random_state=42)
rf = RandomForestRegressor(n_estimators=5, random_state=0)
rf.fit(X_train, y_train)
predictors[5] = rf
cosaa(rf, X_test, y_test)

X_train, X_test, y_train, y_test = train_test_split(sectors[3], Ri['14'], test_size=0.3, random_state=42)
rf = RandomForestRegressor(n_estimators=5, random_state=0)
rf.fit(X_train, y_train)
predictors[14] = rf
cosaa(rf, X_test, y_test)

# tuple (Ri index, sector number)
list_sector_k = [(0, 1), (2, 1), (4, 3), (5, 0), (6, 2), (14, 3)]
mu = predictors[0].predict(sectors[1])
mu = np.column_stack((mu, predictors[2].predict(sectors[1])))
mu = np.column_stack((mu, predictors[4].predict(sectors[3])))
mu = np.column_stack((mu, predictors[5].predict(sectors[0])))
mu = np.column_stack((mu, predictors[6].predict(sectors[2])))
mu = np.column_stack((mu, predictors[14].predict(sectors[3])))

mu = mu.transpose()
print(mu)

covariance_matrix = np.cov(mu)
mu = pd.DataFrame(mu)
covariance_matrix = pd.DataFrame(covariance_matrix)
mu.to_csv('mu.csv')
covariance_matrix.to_csv('covariance_matrix')















































