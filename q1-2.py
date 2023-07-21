import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
from mpl_toolkits.mplot3d import Axes3D


df = pd.read_csv('data_challenge_stock_prices.csv')
ROW = len(df.index) # 2,00,000
COL = len(df.columns) # 100


# CALCULATING RETURNS FOR EACH STOCK, STORING IT IN "ret_df"
ret_df = df.pct_change().dropna()*100
ret_dftrans = ret_df.transpose()
ret_dftrans.columns = ret_dftrans.columns.astype(str)


# REDUCING DIMENSIONS FROM 2,00,000 to 3 USING PCA
pca = PCA(n_components=3)
data2 = pca.fit_transform(ret_dftrans)
data2 = pd.DataFrame(data2)


# CLUSTERING STOCKS USING KMEANS ALGORITHM
n_clusters = 4
kmeans = KMeans(n_clusters = n_clusters, random_state = 42)
kmeans.fit(ret_dftrans)
data2['sector'] = kmeans.labels_


# MAKING A LIST OF SECTORS TO SEPARATE EACH CLUSTER
sectors = []
for i in range(n_clusters):
    sector = data2[data2['sector']==i].drop('sector', axis=1).transpose()
    sectors.append(sector)


# PLOTTING ALL CLUSTERS IN A 3D PLOT
colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray', 'yellow', 'cyan']
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
for i in range(n_clusters):
    ax.scatter(sectors[i].loc[0], sectors[i].loc[1], sectors[i].loc[2], color=colors[i], alpha=0.75)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('3D Scatter Plot')
plt.show()










