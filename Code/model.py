# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 11:35:15 2021

@author: Cibin
"""

# Following Libraries are being used
import pandas as pd
import numpy as np


from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


data = pd.read_csv("C:/Users/Cibin/OneDrive/Masters/final_project/Datasets/playdata.csv")

pd.set_option('display.max_columns', 500)

data.head()





data.describe()
data.dtypes
#we copy the dataframe to a new varibale so the original is kept intact.
# corr = data.corr() 
# plt.figure(figsize=(9, 8))

# sns.heatmap(corr[(corr >= 0.1) | (corr <= -0.1)], 
#             cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
#             annot=True, annot_kws={"size": 8}, square=True)

model1 = data.copy()

#preprocessing


"""We start modeling the data we have.

First, we have to encode the categorical values into numeric to be able to use in the model.
for that we use label and one hot encoding technique to convert the categoricals into numbers. """

def fun_preprocess(model):

    #here we convert categories into numbers using one hot encoding.
    cat_list = model['Category'].unique().tolist() 
    cat_list = ['cat_' + word for word in cat_list]
    model = pd.concat([model, pd.get_dummies(model['Category'], prefix='cat')], axis=1)

    #Here  we use label encoder for encoding Content rating 
    le = preprocessing.LabelEncoder()
    model['Content Rating'] = le.fit_transform(model['Content Rating'])


    #same label encoder converts the boolean into 1 and 0.
    model['Free'] = model['Free'].astype('category')
    model['Free'] = model['Free'].cat.codes

    model[['Installs','Size','Rating Count', 'Price']].head(10)
    """The next step to the model is standardising the values which are gonna be used for modeling. """
    #Standerdising

    #columns that needs standardising
    cols = ['Installs','Size','Rating Count', 'Price']

    scaler = StandardScaler()
    scaler.fit(model[cols])
    model[cols] = scaler.transform(model[cols])
    
    return model,cat_list

def fun_traintestplit(model,features_,cat_list):

    #initialsing Model
    features = features_
    features.extend(cat_list)
    X = model[features]
    y = model['Rating']

    #train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print("Training Set Dimensions:", X_train.shape)
    print("Validation Set Dimensions:", X_test.shape)

    X_train.head()
    X_train.dtypes
    
    return X_train, X_test, y_train, y_test,X,y



#---------------------------------------------------------------------------------------

def model_randomforest(X_train, X_test, y_train, y_test,estimators,X,y):

    randomf = RandomForestRegressor(n_estimators=estimators)
    randomf.fit(X_train, y_train)

    #predicted values

    y_pred = randomf.predict(X_test)

    #calculating the errors
    errors = abs(y_pred - y_test)


    # Measure mean squared error for training and validation sets
    print('Mean squared Error for Training Set:', mean_squared_error(y_train, randomf.predict(X_train)))
    print('Mean squared Error for Test Set:', mean_squared_error(y_test, randomf.predict(X_test)))
    print('Mean Absolute Error for Test Sets:', round(np.mean(errors), 2), 'scores.')

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')

    # Important features for random forest regressor
    for name, importance in zip(X.columns, randomf.feature_importances_):
        print('feature:', name, "=", importance)
    
    importances = randomf.feature_importances_
    indices = np.argsort(importances)
    features = X.columns
    plt.figure(figsize=(6, 12))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='g', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()


    plt.figure(figsize=(5, 7))
    ax = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
    sns.distplot(y_pred, hist=False, color="b", label="Fitted Values" , ax=ax)
    plt.title('Actual vs Fitted Values for Rating')
    plt.show()
    plt.close()
    
    #prediction

    #row = [[-0.045302,-0.278646,1-0.046049,1,-0.043101 ,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

    #yhat = randomf.predict(row)

    #print(yhat)

 
#---------------------------------------------------------------------------------
"""Initial Model"""
#---------------------------------------------------------------------------------
#preprocessing
model2 = fun_preprocess(model1) 

#traintest split
features = ['Rating Count', 'Size', 'Installs', 'Free', 'Price', 'Content Rating']
X_train1, X_test1, y_train1, y_test1,X1,y1 = fun_traintestplit(model2,features)
esti = 10
model_randomforest(X_train1, X_test1, y_train1, y_test1,esti,X1,y1)
#-------------------------------------------------------------------------------------------------------
"""Best possible model."""

model2 = fun_preprocess(model1) 

#traintest split
features = ['Rating Count', 'Size', 'Installs']
X_train1, X_test1, y_train1, y_test1,X1,y1 = fun_traintestplit(model2,features)
esti = 100
model_randomforest(X_train1, X_test1, y_train1, y_test1,esti,X1,y1)
#-----------------------------------------------------------------------------------------------------

"""Model which has ratings between 1 and 5 """
new_model = data.loc[data['Rating'].between(1,5)]

model2 = fun_preprocess(new_model) 


features = ['Rating Count', 'Size', 'Installs', 'Free', 'Price', 'Content Rating']
X_train1, X_test1, y_train1, y_test1,X1,y1 = fun_traintestplit(model2,features)
esti = 200
model_randomforest(X_train1, X_test1, y_train1, y_test1,esti,X1,y1)

#------------------------------------------------------------------------------------------------------------------------------
"""Optimized model with ratings between 1to 5 and installs greater than 100000 """
new_model = data.loc[data['Rating'].between(1,5) & (data['Installs'] > 100000)]

model2 = fun_preprocess(new_model) 


features = ['Rating Count', 'Size', 'Installs', 'Free', 'Price', 'Content Rating']
X_train1, X_test1, y_train1, y_test1,X1,y1 = fun_traintestplit(model2,features)
esti = 300
model_randomforest(X_train1, X_test1, y_train1, y_test1,esti,X1,y1)
#-----------------------------------------------------------------------------------------------------------------
"""Model for developers """
new_model = data.loc[data['Rating'].between(1,5)]

model2,cat_list1 = fun_preprocess(new_model) 


features = ['Content Rating','Size','Free','Price']
X_train1, X_test1, y_train1, y_test1,X1,y1 = fun_traintestplit(model2,features,cat_list1)
esti = 20
model_randomforest(X_train1, X_test1, y_train1, y_test1,esti,X1,y1)
#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------
