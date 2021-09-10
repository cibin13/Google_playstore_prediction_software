# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 19:27:26 2021

@author: Cibin
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams.update({'font.size': 15}) #increase font size for all graphs

data = pd.read_csv("C:/Users/Cibin/OneDrive/Masters/final_project/Datasets/playdata.csv")



data.head()




#start of by plaotting the count of apps in category
plt.figure(figsize=(10, 5))
a = sns.countplot(x="Category",data=data, palette = "Set1")
a.set_xticklabels(a.get_xticklabels(), rotation=90, ha="right")
a 
plt.title('Count of app in each category')

#---------------------------------------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
a = sns.countplot(x="Content Rating",data=data, palette = "Set2")
a.set_xticklabels(a.get_xticklabels(), rotation=90, ha="right")
plt.title('Count of Content Rating')

#---------------------------------------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(10,10))
labels = ["Free","Paid"]
sizes = data['Free'].value_counts(sort = True)
colors = ["Grey","Blue"]
explode = (0.2,0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=0,textprops={'fontsize': 25})
plt.title('Percent of Free Vs Paid Apps',size = 20)
plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------

rt = data.loc[data['Rating'].between(1,5) & (data['Installs'] > 1000)] # we subset the values of rating to only values between 1-5 because
rt.count()                                           #the min value for rating is 1 and there are a lot of values that are 0 which means the
                                            #those apps didnt get rated.
rt.head()
plt.show()

plt.figure(figsize=(15,9))
plt.xlabel("Rating")
plt.ylabel("Frequency")
graph = sns.kdeplot(rt.Rating, color="Blue", shade = True)
plt.title('Distribution of Rating',size = 20);

#---------------------------------------------------------------------------------------------------------------------------------------------------
"""The most number of installs in all the categories"""
highest_Install = data.groupby('Category')[['Installs']].sum().sort_values(by='Installs', ascending=False)

x = []
y = []

for i in range(len(highest_Install)):
    x.append(highest_Install.Installs[i])
    y.append(highest_Install.index[i])

plt.figure(figsize=(18,13))

plt.xlabel("Installs")
plt.ylabel("Category")
graph = sns.barplot(x = x, y = y, alpha =0.9, palette= "viridis")
graph.set_title("Installs", fontsize = 25);

#-------------------------------------------------------------------------------------------------------------------------------------------------

"""Here we find out the most installed apps in any category that we want to see """

#since we need to  see the apps in all the categories we must use a function to get the top 10 apps from the category we want
data["Category"].value_counts()

def top10(str):
    
    top10 = data[data['Category'] == str]
    top10apps = top10.sort_values(by='Installs', ascending=False).head(10)
    plt.figure(figsize=(15,12))
    plt.title('Top 10 Installed Apps in {a}'.format(a = str),size = 20);    
    graph = sns.barplot(x = top10apps["App Name"], y = top10apps.Installs)
    graph.set_xticklabels(graph.get_xticklabels(), rotation= 45, horizontalalignment='right')

# here we can change the category to get the top 10 in any category
top10('Communication')

#-------------------------------------------------------------------------------------------------------------------------------------------------------
#finding out the top paid apps on installs
top10PaidApps = data[data['Free'] == False].sort_values(by='Price', ascending=False)

top10PaidApps = top10PaidApps.loc[(top10PaidApps['Installs'] > 1000)].head(10)

plt.figure(figsize=(15,12));
plt.title('Top 10 most expensive apps against installs',size = 20);    
graph = sns.barplot(x = top10PaidApps["App Name"], y = top10PaidApps.Installs)
graph.set_xticklabels(graph.get_xticklabels(), rotation= 45, horizontalalignment='right')
#-----------------------------------------------------------------------------------------------------------------
top10FreeApps = data[data['Free'] == True].sort_values(by='Installs', ascending=False).head(11)


plt.figure(figsize=(15,12));
plt.title('Top 10 Free Installed Apps',size = 20);    
graph = sns.barplot(x = top10FreeApps["App Name"], y = top10FreeApps.Installs)
graph.set_xticklabels(graph.get_xticklabels(), rotation= 45, horizontalalignment='right')

#-----------------------------------------------------------------------------------------------------------------
"""Apps with most number of ratings given """

apps_most_rated = data.sort_values(by='Rating Count', ascending=False).head(20)

plt.figure(figsize=(15,12));
plt.title('Top 20 Most Rated  Apps',size = 20);    
graph = sns.barplot(x = apps_most_rated["App Name"], y = apps_most_rated["Rating Count"])
graph.set_xticklabels(graph.get_xticklabels(), rotation= 45, horizontalalignment='right');

#---------------------------------------------------------------------------------------------------------------

""" To find out the apps with the highest revenue made.  """

Paid_Apps = data[data['Free'] == False]

earning = Paid_Apps[['App Name', 'Installs', 'Price']]

earning['Earnings'] = earning['Installs'] * earning['Price'];

earning_sorted = earning.sort_values(by='Earnings', ascending=False).head(20)

earning_sorted_Price = earning_sorted.sort_values(by='Price', ascending=False)



plt.figure(figsize=(15,9))
plt.bar(earning_sorted_Price["App Name"], earning_sorted_Price.Earnings, width=1.1, label=earning_sorted_Price.Earnings)
plt.xlabel("Apps")
plt.ylabel("Earnings")
plt.tick_params(rotation=90)
plt.title("Top Earning Apps");

#-------------------------------------------------------------------------------------------------------------------------------

""" The percentage of apps which have Ad support """

plt.figure(figsize=(10,20))
labels = ["No","Yes"]
sizes = data['Ad Supported'].value_counts(sort = True)
colors = ["Grey","Blue"]
explode = (0.2,0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=0,textprops={'fontsize': 25})
plt.title('Percentage of Ads Supported apps',size = 20)
plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------

plt.figure(figsize=(10,20))
labels = ["No","Yes"]
sizes = data['In App Purchases'].value_counts(sort = True)
colors = ["Grey","Blue"]
explode = (0.2,0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=0,textprops={'fontsize': 25})
plt.title('Percenage of apps with in app purchases',size = 20)
plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------

plt.figure(figsize=(10,20))
labels = ["No","Yes"]
sizes = data['Editors Choice'].value_counts(sort = True)
colors = ["Grey","Blue"]
explode = (0.2,0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%.2f%%', shadow=True, startangle=0,textprops={'fontsize': 25})
plt.title('Percenage of apps which are featured in Editors Choice ',size = 20)
plt.show()

#-----------------------------------------------------------------------------------------------
"""Popularity of developers """
devloper_id = data.sort_values(by='Installs', ascending=False).head(40)

plt.figure(figsize=(15,12));
plt.title('Top 10 Most famous Developers by the total number of installs their apps have',size = 20);    
graph = sns.barplot(x = devloper_id["Developer Id"], y = devloper_id["Installs"])
graph.set_xticklabels(graph.get_xticklabels(), rotation= 45, horizontalalignment='right');
#-------------------------------------------------------------------------------------

data.head()

"""Top 10 Categories with most number of Rating Count """
cat = data.sort_values(by='Rating Count', ascending=False).head(20)

plt.figure(figsize=(15,12));
plt.title('Top 10 Categories with most number of Rating Count',size = 20);    
graph = sns.barplot(x = cat["Category"], y = cat["Rating Count"])
graph.set_xticklabels(graph.get_xticklabels(), rotation= 45, horizontalalignment='right');

#------------------------------------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------------------------------------------------

