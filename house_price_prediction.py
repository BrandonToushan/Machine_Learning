#Brandon Toushan 10178340
#July 16, 2020
#GMMA 867 Predictive Modelling

#Package Imports
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import skew 
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import sklearn.linear_model as linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet

#Loading in datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

#Training data
train_df.head()

#Testing Data
test_df.head()

#Looking at Target Feature (Sale Price) & Correlations

#Distribution Plot
sns.distplot(train_df['SalePrice'])
plt.show()

#Kurtosis & Skew
print("Kurtosis: %f" % train_df['SalePrice'].kurt())
print("Skewness: %f" % train_df['SalePrice'].skew())

#Creating correlation matrix
corr_mat = train_df.corr()
fig, ax = plt.subplots()
sns.heatmap(corr_mat)

#Quantifying the Correlation Matrix
i = 10
columns = corr_mat.nlargest(i,'SalePrice')['SalePrice'].index
corrmat = np.corrcoef(train_df[columns].values.T)
heatmap = sns.heatmap(corrmat,
            cbar=True,
            annot=True, 
            xticklabels = columns.values,
            yticklabels = columns.values)
plt.show()


#Dealing with Missing Values

#Finding all missing values in training data
missing_train = train_df.isnull().sum().reset_index()
missing_train.columns = ['column_name', 'missing_count']
missing_train = missing_train.loc[missing_train['missing_count']>0]

#Visualizng as %
index = np.arange(missing_train.shape[0])
fig, ax= plt.subplots(figsize = (30,20))
rects = ax.barh(index,missing_train.missing_count.values/1460,color = 'purple')
ax.set_yticks(index)
ax.set_yticklabels(missing_train.column_name.values)
plt.show()

#Repeating for test data
missing_test = test_df.isnull().sum().reset_index()
missing_test.columns = ['column_name', 'missing_count']
missing_test = missing_test.loc[missing_test['missing_count']>0]
print(missing_test.count())

#Visualizng as %
index = np.arange(missing_test.shape[0])
fig, ax= plt.subplots(figsize = (30,20))
rects = ax.barh(index,missing_test.missing_count.values/1459,color = 'orange')
ax.set_yticks(index)
ax.set_yticklabels(missing_test.column_name.values)
plt.show()

#Outlier Analysis for Most Correlated Features

#Overall Qual 
plt.plot()
plt.scatter(train_df['OverallQual'],train_df['SalePrice'])

#GrLivArea
plt.plot()
plt.scatter(train_df['GrLivArea'],train_df['SalePrice'],color='red')

#Deleting Outliers
train_df = train_df.drop(train_df[train_df.GrLivArea >4500].index)
train_df.reset_index(drop = True, inplace = True)

#Garage Cars
plt.plot()
plt.scatter(train_df['GarageCars'],train_df['SalePrice'],color='purple')

#Deleting Outliers
train_df = train_df.drop(train_df[train_df.GarageCars >3.5].index)
train_df.reset_index(drop = True, inplace = True)              

#Basement SF
plt.plot()
plt.scatter(train_df['TotalBsmtSF'],train_df['SalePrice'],color='green')

#Deleting Outliers
train_df = train_df.drop(train_df[train_df.TotalBsmtSF >6000].index)
train_df.reset_index(drop = True, inplace = True)

#1st SF
plt.plot()
plt.scatter(train_df['1stFlrSF'],train_df['SalePrice'],color='orange')

#Garage Cars
plt.plot()
plt.scatter(train_df['TotRmsAbvGrd'],train_df['SalePrice'],color='pink')

#Deleting Outliers
train_df = train_df.drop(train_df[train_df.TotRmsAbvGrd >13].index)
train_df = train_df.drop(train_df[(train_df['SalePrice'] > 600000) & (train_df['TotRmsAbvGrd'] == 10)].index)
train_df.reset_index(drop = True, inplace = True)

#Year Built
plt.plot()
plt.scatter(train_df['YearBuilt'],train_df['SalePrice'],color='black')

#Deleting Outliers
train_df = train_df.drop(train_df[(train_df['SalePrice'] > 250000) & (train_df['YearBuilt'] < 1900)].index)
train_df = train_df.drop(train_df[(train_df['SalePrice'] > 600000) & (train_df['YearBuilt'] < 2000)].index)
train_df.reset_index(drop = True, inplace = True)

#Transforming Target Variable

#Creating normality via log transform
train_df['SalePrice'] = np.log1p(train_df['SalePrice'])

#Visualizing results
sns.distplot(train_df['SalePrice'])
fig = plt.figure()
res = stats.probplot(train_df['SalePrice'], plot=plt)

#Prepping Datasets for FE

#Dropping ID col
train_df.drop(columns=['Id'],axis=1, inplace=True)
test_df.drop(columns=['Id'],axis=1, inplace=True)

#Defining target variable
y = train_df['SalePrice'].reset_index(drop=True)

#Merging train_df & test_df for convenience
data = pd.concat((train_df, test_df)).reset_index(drop = True)

#Dropping the target variable. 
data.drop(['SalePrice'], axis = 1, inplace = True)

#Dealing with missing data

#Imputing missing values for LotFrontage with Median Val by Neighborhood
data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform( lambda i: i.fillna(i.mean()))

##Remaning NaNs have two cases: those with a purpose, and null values without a purpose

#NaN values without a purpose (fill with 'None')
NaN_cols_none = ["Alley", "PoolQC", "MiscFeature","Fence","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond",
                    'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','MasVnrType']
#Fill NaNs
for i in NaN_cols_none:
    data[i] = data[i].fillna('None')
    
#NaN values with a purpose (fill with 0)
NaN_cols_zero = ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt',
                    'GarageArea','GarageCars','MasVnrArea']
#Fill NaNs
for i in NaN_cols_zero:
    data[i] = data[i].fillna(0)

#Converting subclass and zoning vars from numeric to categorical
data['MSSubClass'] = data['MSSubClass'].astype(str)
data['MSZoning'] = data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

#Converting important dates from numeric to categorical 
data['YrSold'] = data['YrSold'].astype(str)
data['MoSold'] = data['MoSold'].astype(str)

#Filling NaN like vals for the remaining vars
data['Functional'] = data['Functional'].fillna('Typ') 
data['Utilities'] = data['Utilities'].fillna('AllPub') 
data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0]) 
data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])
data['KitchenQual'] = data['KitchenQual'].fillna("TA") 
data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])
data['Electrical'] = data['Electrical'].fillna("SBrkr")

#Checking for Skew
numeric_features = data.dtypes[data.dtypes != "object"].index
skew_features = data[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)
print(skew_features)

#Checking for Normality
sns.distplot(data['1stFlrSF']);

#Define a function to adjust the skew of every feature

def fixing_skewness(df):
    
    #Select all numeric features
    numeric_features = df.dtypes[df.dtypes != "object"].index

    #Check skew
    skewed_features = df[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)
    high_skew = skewed_features[abs(skewed_features) > 0.5]
    high_skew_features = high_skew.index

    for feature in high_skew_features:
        df[feature] = boxcox1p(df[feature], boxcox_normmax(df[feature] + 1))

fixing_skewness(data)

sns.distplot(data['1stFlrSF']);

#Creating feature for total square footage (finished basement)
data['Total_Sqft_Finished'] = (data['BsmtFinSF1'] + data['BsmtFinSF2'] + data['1stFlrSF'] + data['2ndFlrSF'])

#Creating feature for total square footage (finished + Unfinished)
data['Total_Sqft'] = (data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF'])

#Creating feature for total porch square footage
data['Total_Porch_sqft'] = (data['OpenPorchSF'] + data['3SsnPorch'] + data['EnclosedPorch'] + data['ScreenPorch'] + data['WoodDeckSF'])

#Creating feature for year built and remodelled
data['YrBltAndRemod'] = data['YearBuilt'] + data['YearRemodAdd']

#Creating feature for total # of bathrooms
data['NumBathrooms'] = (data['FullBath'] + (0.5 * data['HalfBath']) + data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath'])) 

#Creating feature for houses with pools
data['HasPool'] = data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

#Creating a feature for houses with multiple floors
data['Has2ndfloor'] = data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

#Creating a feature for houses with garages
data['HasGarage'] = data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

#Creating a feature for houses with basements
data['HasBsmt'] = data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

#Creating a feature for houses with fireplaces
data['HasFireplace'] = data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

#Dropping Excess Features (Utilities, Street, PoolQC)
data = data.drop(['Utilities', 'Street', 'PoolQC',], axis=1)

#Creating dummy variables and splitting the data

#One-Hot Enconding via SkLearn
features = pd.get_dummies(data).reset_index(drop=True)

#Storing training and testing data
X = features.iloc[:len(y), :]
X_sub = features.iloc[len(y):, :]

#Performing train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

#Fitting a basic linear model

#Training model with training set
basic_reg = LinearRegression(normalize = True, n_jobs = -1)
basic_reg.fit(X_train,y_train)

#Predicting results
pred = basic_reg.predict(X_test)

#Evaluating Predictions
print('MSE (Basic Reg): %.2f' % mean_squared_error(y_test, pred))

#Performing CV

lin_reg = LinearRegression()
cv = KFold(shuffle=True, random_state=7, n_splits=10)
lin_reg.fit(X_train,y_train)
scores = cross_val_score(lin_reg, X,y,cv = cv, scoring = 'neg_mean_absolute_error')

print ('%.8f'%scores.mean())

#Improving Linear Model with Regularization Techniques

#Ridge Method

#Creating list of different alphas to try with model
alpha_ridge = [-3,-2,-1,1e-15, 1e-10, 1e-8,1e-5,1e-4, 1e-3,1e-2,0.5,1,1.5, 2,3,4, 5, 10, 20, 30, 40]
temp_rss = {}
temp_mse = {}

for i in alpha_ridge: 
    ridge = Ridge(alpha= i, normalize=True) #Assigning model
    ridge.fit(X_train, y_train) #Fitting model
    y_pred = ridge.predict(X_test) #Predicting on X_test

    mse = mean_squared_error(y_test, y_pred) #Evaluating model performance
    rss = sum((y_pred-y_test)**2)
    temp_mse[i] = mse
    temp_rss[i] = rss

#Lasso Method
temp_rss = {}
temp_mse = {}

for i in alpha_ridge:
    lasso_reg = Lasso(alpha= i, normalize=True) #Assigning model
    lasso_reg.fit(X_train, y_train) #Fitting model
    y_pred = lasso_reg.predict(X_test) #Predicting on X_test

    mse = mean_squared_error(y_test, y_pred) #Evaluating Model
    rss = sum((y_pred-y_test)**2)
    temp_mse[i] = mse
    temp_rss[i] = rss

#ElasticNet Method
temp_mse = {}

for i in alpha_ridge:
    lasso_reg = ElasticNet(alpha= i, normalize=True) #Assigning each model
    lasso_reg.fit(X_train, y_train) #Fitting the model
    y_pred = lasso_reg.predict(X_test) #Predicting on X_test

    mse = mean_squared_error(y_test, y_pred) #Evaluating Model
    rss = sum((y_pred-y_test)**2)
    temp_mse[i] = mse
    temp_rss[i] = rss

#kfolds CV
kfolds = KFold(n_splits=10, shuffle=True, random_state=7)

#Defining a function to calculate MSE for all folds
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

#Defining a function to calculated -MSE for all folds
def cross_validation_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))
    return (rmse)

#Assembling cross-validation steps togeteher via sklearn pipelines

#elasticnet
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, cv=kfolds))      

#lasso
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, random_state=7, cv=kfolds))
                          
#ridge
ridge = make_pipeline(RobustScaler(), RidgeCV(cv=kfolds))

#Performing cv
cv_rmse(ridge)
cv_rmse(lasso)
cv_rmse(elasticnet)

#Evaluating different models

#Defining the models

#ElasticNet
elastic_model = elasticnet.fit(X, y)

#Lasso
lasso_model = lasso.fit(X, y)

#Ridge
ridge_model = ridge.fit(X, y)

#Evaluating results predicting on X_test
print('RMSLE score on train data (ElasticNet):')
print(rmsle(y, elastic_model.predict(X)))

print('RMSLE score on train data (Lasso):')
print(rmsle(y, lasso_model.predict(X)))

print('RMSLE score on train data (Ridge):')
print(rmsle(y, ridge_model.predict(X)))

#Printing out resulting final model (ridge)
print(ridge['ridgecv'])
print(ridge['ridgecv'].intercept_)
print(ridge['ridgecv'].coef_)

#Creating submission .csv
submission = pd.read_csv("sample_submission.csv")
submission.iloc[:,1] = np.floor(np.expm1(ridge_model.predict(X_sub))) #expm1 is the exact inverse of log1p
submission.to_csv("submission_ridge.csv", index=False)
