###IDENTIFYING CUSTOMER PERSONAS & SEGMENTS VIA CLUSTERING

#Package Imports
import pandas
import scipy 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

#### PART A: LOADING & PREPROCESSING DATA ####

#Importing data from Uncle Steve's Github
url_jewelry = "https://raw.githubusercontent.com/stepthom/sandbox/master/data/jewelry_customers.csv"
jewelry_df = pandas.read_csv(url_jewelry)

#Check data
jewelry_df.info()
jewelry_df.head()

#Converting df to np array
X = jewelry_df.to_numpy()
X.shape

#Scaling data
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Plotting scaled data
plt.figure(figsize = (20,5))
plt.scatter(X[:,0],X[:,1],c='red')
plt.scatter(X[:,0],X[:,2],c='green')
plt.scatter(X[:,0],X[:,3],c='blue')

#### PART B: CLUSTERING THE DATA ####

#Running algo for various values of K
K = range(1,10)
fits = [KMeans(n_clusters=k, init='k-means++', max_iter=500, n_init=30, verbose=False, random_state=666).fit(X) for k in K]
centroids = [fit.cluster_centers_ for fit in fits]
inertias = [fit.inertia_ for fit in fits]

#Evaluating clustering visually
clusters_df = pandas.DataFrame( { "num_clusters":K, "cluster_errors": inertias } )
plt.figure(figsize=(20,5))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o",c='red' )
plt.show()

#Running the algo with k=5 [heuristic curve elbow value]
k = 5
fit = KMeans(n_clusters=k, init='k-means++', max_iter=500, n_init=30, verbose=False, random_state=1234).fit(X)
centroids = fit.cluster_centers_
inertia = fit.inertia_ 
labels = fit.labels_

##Visualizing the clustering
predictions = fit.fit_predict(X)

#Plotting the clusters
plt.figure(figsize=(20,5))
plt.scatter(
    X[predictions == 0, 0],X[predictions == 0, 1],s=125,c='blue',marker='s',edgecolor='black',
    label='cluster 1')
plt.scatter(
    X[predictions == 1, 0],X[predictions == 1, 1],s=125,c='red',marker='o',edgecolor='black',
    label='cluster 2')
plt.scatter(
    X[predictions == 2, 0],X[predictions == 2, 1],s=125,c='green',marker='v',edgecolor='black',
    label='cluster 3')
plt.scatter(
    X[predictions == 3, 0],X[predictions == 3, 1],s=125,c='pink',marker='h',edgecolor='black',
    label='cluster 4')
plt.scatter(
    X[predictions == 4, 0],X[predictions == 4, 1],s=125,c='purple',marker='D',edgecolor='black',
    label='cluster 5')

#Plotting the centroids
plt.scatter(
    centroids[:, 0],centroids[:, 1],s=500,marker='*',c='yellow',edgecolor='black',label='centroids')
plt.legend()
plt.show()

#Checking silhouette score
print("Silhouette Score = {:.2f}".format(silhouette_score(X, labels)))

#### PART C PRINTING SUMMARY STATS FOR EACH CLUSTER ####

#Inverse transforming all features back to their original values
cluster_1 = scaler.inverse_transform(X[predictions == 0])
cluster_2 = scaler.inverse_transform(X[predictions == 1])
cluster_3 = scaler.inverse_transform(X[predictions == 2])
cluster_4 = scaler.inverse_transform(X[predictions == 3])
cluster_5 = scaler.inverse_transform(X[predictions == 4])

#Converting arrays to DFs
cluster_1_df = pandas.DataFrame.from_records(cluster_1,columns = ["Age",'Income','SpendingScore','Savings'])
cluster_2_df = pandas.DataFrame.from_records(cluster_2,columns = ["Age",'Income','SpendingScore','Savings'])
cluster_3_df = pandas.DataFrame.from_records(cluster_3,columns = ["Age",'Income','SpendingScore','Savings'])
cluster_4_df = pandas.DataFrame.from_records(cluster_4,columns = ["Age",'Income','SpendingScore','Savings'])
cluster_5_df = pandas.DataFrame.from_records(cluster_5,columns = ["Age",'Income','SpendingScore','Savings'])

#Printing summary statisics for each cluster (by dataframe)
print("Cluster 1 Summary Stats:",(cluster_1_df.describe()))
print("Cluster 2 Summary Stats:",(cluster_2_df.describe()))
print("Cluster 3 Summary Stats:",(cluster_3_df.describe()))
print("Cluster 4 Summary Stats:",(cluster_4_df.describe()))
print("Cluster 5 Summary Stats:",(cluster_5_df.describe()))
