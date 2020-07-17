###PREDICTING WHICH BRAND A CUSTOMER WILL PURCHASE

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

#Importing data from Uncle Steve's Github
url_oj = "https://raw.githubusercontent.com/stepthom/sandbox/master/data/OJ.csv"
oj_df = pandas.read_csv(url_oj)
oj_df.shape
oj_df.head()

#Check for nulls
oj_df.isnull().sum()

#Number of purchases of each brand
fig1 = plt.figure(figsize = (20,5))
sns.countplot(oj_df['Purchase'])
print(oj_df.groupby('Purchase').size())

#Price_diff visualization
fig2 = plt.figure(figsize = (20,5))
fig2.add_subplot(1,2,1)
sns.distplot(oj_df['PriceDiff'])
fig2.add_subplot(1,2,2)
sns.boxplot(oj_df['PriceDiff'])

#Loyalty Visualization
fig3 = plt.figure(figsize = (20,5))
fig3.add_subplot(1,2,1)
sns.distplot(oj_df['LoyalCH'])
fig3.add_subplot(1,2,2)
sns.boxplot(oj_df['LoyalCH'])

#Subsetting important features
X = oj_df[["LoyalCH","PriceDiff"]]
Y = oj_df["Purchase"]

#Preprocessing data (scaling variables to a common scale for the algorithm)
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(
X,Y, test_size = 0.25, random_state = 666)

#Model 1, Similarity-Based (KNN)

#Hyperparams
n_neighbors = list(range(1,50))
p = [1,2,3]
hyperparams = dict(n_neighbors = n_neighbors, p=p)

#Tuning algo
knn_tuner = KNeighborsClassifier()
knn_tuner_clf = GridSearchCV(knn_tuner,hyperparams,cv=10)
model_knn = knn_tuner_clf.fit(X_train,Y_train)
print(model_knn.best_estimator_)

#KNN algo using tuned hyperparams
knn_clf = KNeighborsClassifier(p=1,n_neighbors=47)
knn_clf = knn_clf.fit(X_train,Y_train)
Y_pred_knn = knn_clf.predict(X_test)

#KNN tuned model performance
print(confusion_matrix(Y_test,Y_pred_knn))
print("KNN Model Accuracy = {:.2f}".format(accuracy_score(Y_test, Y_pred_knn)))

#Model 2, Probability-Based (Gaussian Naive Bayes)

#Algo
bayes = GaussianNB()
model_bayes = bayes.fit(X_train,Y_train)
Y_pred_bayes = model_bayes.predict(X_test)

#Model params (tuned by the preprocessing of the data itself)
model_bayes.theta_ #Mean of each feature
model_bayes.sigma_ #Variance of each feature

#Hyperparams for this algo are the prior probs of the classes (priors)
#the portion of the largest variance of all features (var_smoothing)

#Naive Bayes model performance
print(confusion_matrix(Y_test,Y_pred_bayes))
print("GNB Model Accuracy = {:.2f}".format(accuracy_score(Y_test, Y_pred_bayes)))

#Model 3, Error-Based (SVM)

#Hyperparams
C = [0.1,1,10,100]
gamma = [1,0.1,0.01,0.001]
kernel = ['rbf','poly','sigmoid']
hyperparams = dict(C=C,gamma = gamma, kernel = kernel)

#Tuning algo
svm_tuner = SVC()
svm_tuner = GridSearchCV(svm_tuner,hyperparams,cv=3)
model_svm = svm_tuner.fit(X_train,Y_train)
print(model_svm.best_estimator_)

#Algo using tuned hyperparams
svm = SVC(kernel="rbf",gamma = 0.01 ,degree = 3, C=100)
svm = svm.fit(X_train, Y_train)
Y_pred_svm = svm.predict(X_test)

#SVM model performance
print(confusion_matrix(Y_test,Y_pred_svm))
print("SVM Model Accuracy = {:.2f}".format(accuracy_score(Y_test, Y_pred_svm)))
