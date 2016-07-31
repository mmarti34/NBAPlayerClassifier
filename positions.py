# Michael Martinez

# This program creates a matrix of advanced player statistics
# to cluster players into new positions. 


import pandas as pd
url = '2016NBAstats.txt'
nba = pd.read_csv(url, sep=' ')
nba

nba.fillna(0)
nba = nba[(nba['G'] > 10)]

X = nba.drop('G', axis = 0)
X = nba.drop('Player', axis=1)


from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=1)
km.fit(X)


km.labels_

nba['cluster'] = km.labels_
nba.sort('cluster')

km.cluster_centers_

nba.groupby('cluster').mean()

centers = nba.groupby('cluster').mean()


import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 13


import numpy as np
colors3 = np.array(['red', 'green', 'blue'])
colors5 = np.array(['red', 'green', 'blue', 'yellow', 'orange'])
colors7 = np.array(['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan'])


#------------------------------------------------------
# 3 Clusters


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


km = KMeans(n_clusters=3, random_state=1)
km.fit(X_scaled)


nba['cluster'] = km.labels_
nba.sort('cluster')

# Cluster by k means
nba.groupby('cluster').mean()

# Plot scatter
pd.scatter_matrix(X, c=colors3[nba.cluster], lw=0.2,  figsize=(10,10), s=100)


# Save data to files
plt.savefig(r"Results/3pos.png")
(nba.cluster).to_csv('data/3Pos.csv', sep="\t", index=False)

# ------------------------------------------------------
# 5 Clusters

km = KMeans(n_clusters=5, random_state=1)
km.fit(X_scaled)


nba['cluster'] = km.labels_
nba.sort('cluster')


nba.groupby('cluster').mean()



pd.scatter_matrix(X, c=colors5[nba.cluster], lw=0.2, figsize=(10,10), s=100)
plt.savefig(r"Results/5pos.png")
(nba.cluster).to_csv('data/5Pos.csv', sep="\t", index=False)



# ------------------------------------------------------
# 7 Clusters

km = KMeans(n_clusters=7, random_state=1)
km.fit(X_scaled)


nba['cluster'] = km.labels_
nba.sort('cluster')


nba.groupby('cluster').mean()


pd.scatter_matrix(X, c=colors7[nba.cluster], lw=0.2, figsize=(10,10), s=100)
plt.savefig(r"Results/7pos.png")
(nba.cluster).to_csv('data/7Pos.csv', sep="\t", index=False)


#Evaluation
from sklearn import metrics
metrics.silhouette_score(X_scaled, km.labels_)

k_range = range(2, 20)
scores = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=1)
    km.fit(X_scaled)
    scores.append(metrics.silhouette_score(X_scaled, km.labels_))

print(scores)


#PCA 

#from sklearn import decomposition
#from sklearn.preprocessing import OneHotEncoder

#df = data.copy()

#nba = nba.copy()

# Initialize a PCA object with two components.
#pca = decomposition.PCA(n_components=10)

# Fit and transform the (continuous) data with the PCA object.
#X_pca = pca.fit_transform(nba.select_dtypes(include=['float64']))

# Transform the class from categorical to numeric.

# Plot the data according to the first two principal components.
#plt.scatter(X_pca[:,0], X_pca[:,1], c=nba['TS'])
#plt.xlabel('First Principal Component')
#plt.ylabel('Second Principal Component')
