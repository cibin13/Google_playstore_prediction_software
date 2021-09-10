# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 17:38:45 2021

@author: Cibin
"""
import pandas as pd  #importing the required packages
import numpy as np
import sys
from sklearn import linear_model

pd.set_option('display.max_columns', 500) #set display for all columns

np.set_printoptions(threshold=sys.maxsize) #print all values 

data = pd.read_csv("C:/Users/Cibin/OneDrive/Masters/final_project/Datasets/Google-Playstore.csv") #importing the data to a dataframe

data.head() #first 5 values.

data.tail() #last 10 values

data.shape #shows number of rows and columns

data.info() #name and types.

data.describe().apply(lambda s: s.apply('{0:.5f}'.format)) #statistics of the variables

#we will start by removing the columns that we dont need.

''' Columns like minimum android, developer website,developer email , last updated,  privacy policy, currency are not revelvant in our data,
model and for the analysis.'''


data = data.drop(columns=['Currency', 'Minimum Android', 'Developer Website','Developer Email','Last Updated','Privacy Policy','Scraped Time'  ])

data.info()

#checking for null values.
data.isnull().sum()

#The amount of null values are fine for now as they are a small percentage of the total value.
data["App Name"].isnull().sum()
data["App Id"].isnull().sum()

data[data['App Name'].isnull()] # we have 2 apps with no name sbut with cross checking with App id we found the app names from it.

data.loc[45686 ,'App Name'] = 'Promedia'
data.loc[2117918  ,'App Name'] = 'Nuli' 

""" We replace the missing values by finding out the real names name sfrom the app id"""

data['Category'].unique() #the categories seem to have all eligible values and just 3 nulls.

data['Rating'].unique() #values are in float and have nans.
data["Rating"].isnull().sum() 

data['Size'].unique() 
"""Here we have the size which contains characters like 'M','K','G' which denotes Megabytes,kilobyte,Gigabyte respectively. There is also an 'Varies with device' entry which is of no use to us so we make it a nan.i will convert the sizes to kilobytes so the M will be changed to 000."""
                        
data['Size'] = data.Size.str.replace("M","000") #replacing M with 000
data['Size'] = data.Size.str.replace("k","") #replacing k with ""
data['Size'] = data.Size.str.replace("G","000000") #replacing G with 000000
data['Size'] = data.Size.str.replace(",","") #replacing , with ""
data['Size'] = data.Size.replace("Varies with device",np.nan) #replacing varies with device with with np.nan
data['Size'] =data['Size'].astype(float)


"""We have a lot of null values for our main label that we are going to to 
predict for our system so we need an way to fill these in. Since there are a lot of missing values
there is no point in using mean or median as the values will be redudndant and no suitable for modeling.
The method i have chosen is to use linear regression to fill in the missing values so that is gives an estimated value count rather than a mean or median value. """

#first step is to use the required features for predictions the features that we will use are 
# 1. Category  3. Content Rating 3. Rating 4. Rating Count 5. Maximum Installs 6. Price 7. Size 

# the columns Category and Content Rating are objects so they must be converted to numbers so for that we need to use encoding techniques.



data_model = data.copy()

data_model = data_model[['Category','Rating','Rating Count','Size','Maximum Installs','Content Rating','Free']]

#encoding category and content rating 
data_model.info()
col_list = ['Category','Content Rating','Free','Size']
for c in col_list:
    if c != 'Size':
        data_model[c] = data_model[c].astype('category')
        data_model[c] = data_model[c].cat.codes
#converting the data we use to floats
# train_data['Price'] = train_data['Price'].astype(float)
# train_data['Installs'] = train_data['Installs'].astype(float)
# train_data['Reviews'] = train_data['Reviews'].astype(float)
data_model = data_model.reset_index()
data_model.isnull().sum()
X = data_model[data_model['Size'].notnull()] #allocating the non null value to X
y = data_model.loc[data_model['Size'].notnull(), 'Size'] 
X = X.drop(columns=['Rating','Size','Rating Count']) #droping the columns with nulls.

X.isnull().sum()

# Fit model
model = linear_model.LinearRegression() 
model.fit(X, y)

# Get all rows with null values
X_miss = data_model[data_model['Size'].isnull()]
X_miss = X_miss.drop(columns = ['Rating','Size','Rating Count'])
X_miss.isnull().sum()
# Fill the predicted values
data_model.loc[data_model['Size'].isnull(), 'Size'] = model.predict(X_miss)

#-----------------------------------------------------------------------------------------

X = data_model[data_model['Rating'].notnull()]
y = data_model.loc[data_model['Rating'].notnull(), 'Rating']
X = X.drop(columns=['Rating','Rating Count'])

# Fit model
model = linear_model.LinearRegression()
model.fit(X, y)

# Get all rows with null values
X_miss = data_model[data_model['Rating'].isnull()]
X_miss = X_miss.drop(columns = ['Rating','Rating Count'])

X_miss.isnull().sum()

# Fill the predicted values
data_model.loc[data_model['Rating'].isnull(), 'Rating'] = model.predict(X_miss)


data_model["Rating"].isnull().sum() 
#------------------------------------------------------------------------------------
X = data_model[data_model['Rating Count'].notnull()]
y = data_model.loc[data_model['Rating Count'].notnull(), 'Rating']
X = X.drop(columns=['Rating Count'])

# Fit model
model = linear_model.LinearRegression()
model.fit(X, y)

# Get all rows with null values
X_miss = data_model[data_model['Rating Count'].isnull()]
X_miss = X_miss.drop(columns = ['Rating Count'])

X_miss.isnull().sum()

# Fill the predicted values
data_model.loc[data_model['Rating Count'].isnull(), 'Rating Count'] = model.predict(X_miss)
#-------------------------------------------------------------------------------------------

data_model["Rating Count"].isnull().sum()





#--------------------------------------------------------------------------------------
data_model.isnull().sum()

data_model.describe().apply(lambda s: s.apply('{0:.5f}'.format))

data.info()

data["Rating"] = data_model["Rating"] #finally, copying the predicted values to the original dataframe.

data["Size"] = data_model["Size"]

data["Rating Count"] = data_model["Rating Count"]

data.isnull().sum()

"""The only missing values now are thge released dates and the developer id which we cannot fill.
rest all the values are filled in."""

data[["Installs","Minimum Installs","Maximum Installs"]].head() 
""""Here we can see something interesting. The value of installls is based on the minimum number of installs 
but we can see here is that we can get the correct number of installs with maximum number of installs. So its better to keep 
maximum number of installs and remove both installs and minimum installs. """

data = data.drop(columns=['Installs', 'Minimum Installs']) #dropped the installs and minimum install columns.

data.info()

data["Price"].unique() #unique values for price.


#converting the realeased column from an object to a datetime object.
data["Released"]=pd.to_datetime(data['Released'], format ='%b %d, %Y', errors='coerce')

data["Released"].head(8)

data["Maximum Installs"].unique() # everything looks normal



data["Content Rating"].unique() #values looks good.

""" Basic Cleaning is completed, Now we convert the file to csv for basic visulaizations"""

data.duplicated(subset=('App Name')).value_counts() #there are many duplicate entries with the same app name. 
data[data.duplicated(['App Name'], keep= False)].sort_values('App Name')

data.duplicated(subset=('App Id')).value_counts() #while looking at the app id there are no duplicates

""" So here we can see duplicates with app names but infact these apps have unique app id so they must be from different developer so they are different apps.  """

data = data.rename(columns={"Maximum Installs": "Installs"})



data.head()

data.to_csv("C:/Users/Cibin/OneDrive/Masters/final_project/Datasets/playdata.csv", index = False)
