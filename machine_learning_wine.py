#Package Imports
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns
sns.set_style('dark')

#Pull dataset from UCI Machine Learning Repository
#Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

url = "https://raw.githubusercontent.com/BrandonToushan/Datasets/master/wine.data"
names  = ['class','alcohol','malic acid','ash','alcalinity of ash','magnesium','total phenols','flavanoids','nonflavanoid phenols','proanthocyanins','color intensity','hue','OD280/OD315 of diluted wines','proline']
dataset = pandas.read_csv(url ,names = names)

#Summarizing Datasets

#shape gives (instances, attributes)
print(dataset.shape)

#head allows you see x number of rows of your data
print(dataset.head(20))

#decriptions gives the count, mean, min and max values and some percentiles
print(dataset.describe())

#number of instances (rows) that belong to each class
print(dataset.groupby('class').size())

#scatter plot matrix
scatter_matrix(dataset, figsize = (25,25),grid = True, alpha = 0.9)
plt.show()

#Creating a validation dataset

#trim the data into arrays for training
array = dataset.values
X = array[:,13:]
Y = array[:,13]

#split the loaded data set into two, 75% to train models 25% to validate
validation_size = 0.25

#test options and evaluation metric
seed = 7 #the seed is a random number and the specific choice does not matter
scoring = 'accuracy'

#training data in the X_train and Y_train
#validation data in X_validation and Y_validation
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#use 10-fold cross validation to estimate accuracy

#split dataset into 10 parts, train on 9 and test on 1

#build models

#Trying 6 different algorithms on the data
models = []
#models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB() ))
models.append(('SVM', SVC(gamma='auto')))

#Evaluating each model 
results = [] 
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed,shuffle = True)
    cv_results = model_selection.cross_val_score(model,X_train,Y_train,cv =kfold, scoring = scoring)
    results.append(cv_results) #adding the results to lists
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) #organizing results
    print(msg) 

#Plotting the mode eval results to compare spread and mean accuracy
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#Making predictions based on validation dataset
NB = GaussianNB()
NB.fit(X_train, Y_train)
predictions = NB.predict(X_validation)

print(accuracy_score(Y_validation, predictions))
#prints the accuracy score as a decimal 0.90 = 90%

print(confusion_matrix(Y_validation, predictions))
#provides a visual representation of errors made

print(classification_report(Y_validation, predictions))
#provides a breakdown of each class by several parameters

