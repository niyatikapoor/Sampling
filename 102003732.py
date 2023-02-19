# -*- coding: utf-8 -*-
"""102003732.py
Name- Niyati Kapoor
Roll No. 102003732
COE23
"""

import pandas as pd
import numpy as np
import math
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

heading=['Simple Random','Systematic','Cluster','Stratified','Convenience']
ans=pd.DataFrame(columns=heading, index=['Logistic Regression','SVM','KNN','XGboost','Gradient Boosting'])
# print(ans)

# Reading the dataset
data=pd.read_csv('Creditcard_data.csv')
# print(data.head())
# print(len(data.axes[0]))

x=data.loc[ : , data.columns != 'Class']
y=data['Class']

# Dataset is imbalanced
# print((y==0).sum()) # 0 for not fraud.
# print((y==1).sum())

# Balancing the dataset usig oversampling
ros = RandomOverSampler(random_state=42)
x_ros, y_ros = ros.fit_resample(x, y)

# Dataset is now balanced
# print((y_ros==0).sum()) # 0 for not fraud.
# print((y_ros==1).sum())

# New Balanced Dataset
df=pd.DataFrame(x_ros)
df['Class']=y_ros

#------------------------------------- Simple Random Sampling--------------------------------------------------------
z=1.96
p=0.5
E=0.05
sample_size = math.ceil((z*z*p*(1-p))/(E*E))
# print(sample_size)

samples=[]
s1 = df.sample(n=sample_size, random_state=0)
samples.append(s1)
# -----------------------------------------------Systematic Sampling----------------------------------------------------
n = len(df)
k = int(math.sqrt(n))
s2 = df.iloc[::k]
samples.append(s2)
# ----------------------------------------------Cluster Sampling--------------------------------------------------------
z=1.96
p=0.5
E=0.05
C=1.5
sample_size = round(((z**2)*p*(1-p))/((E/C)**2))
num_select_clusters=2
df_new=df
# print(sample_size)
N = len(df)
K = int(N/sample_size)
data = None
for k in range(K):
    sample_k = df_new.sample(sample_size)
    sample_k["cluster"] = np.repeat(k,len(sample_k))
    df_new = df_new.drop(index = sample_k.index)
    data = pd.concat([data,sample_k],axis = 0)

random_chosen_clusters = np.random.randint(0,K,size = num_select_clusters)
s3 = data[data.cluster.isin(random_chosen_clusters)]
s3.drop(['cluster'], axis=1, inplace=True)
samples.append(s3)
# print(len(df))
# --------------------------------------------Stratified sampling------------------------------------------------------------------------------

s4=df.groupby('Class', group_keys=False).apply(lambda x: x.sample(190))
samples.append(s4)
# print("Stratified")

# ---------------------------------------------------Convenience sampling-------------------------------------------------------------------------
s5=df.head(400)
samples.append(s5)
# print(s5)


# Applying Models
for i in range(5):
    j=0
    x_s=samples[i].drop('Class',axis=1)
    y_s=samples[i]['Class']

    # Splitting into train and test
    xtrain, xtest, y_train, y_test = train_test_split(x_s ,y_s , random_state=104, test_size=0.25, shuffle=True)

    # Applying Logistic Regression
    classifier = LogisticRegression(random_state = 0,max_iter=2000)
    classifier.fit(xtrain, y_train)
    y_pred = classifier.predict(xtest)
    acc = accuracy_score(y_test, y_pred)
    ans.iloc[j,i]=acc*100
    # print("Logistic")

    # Applying SVM
    clf = SVC(kernel='rbf')
    clf.fit(xtrain, y_train) 
    y_pred=clf.predict(xtest)
    acc = accuracy_score(y_test, y_pred)
    ans.iloc[j+1,i]=acc*100
    # print("SVM")

    # Applying KNN
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(xtrain, y_train)
    y_pred=knn.predict(xtest)
    acc = accuracy_score(y_test, y_pred)
    ans.iloc[j+2,i]=acc*100
    # print("KNN")
    #Applying XGBoost Classifier
    model = XGBClassifier()
    model.fit(xtrain, y_train)
    y_pred=model.predict(xtest)
    acc=accuracy_score(y_test,y_pred)
    ans.iloc[j+3,i]=acc*100
    # Applying Gradient Boosting Classifier
    gbc=GradientBoostingClassifier()
    gbc.fit(xtrain,y_train)
    y_pred=gbc.predict(xtest)
    acc = accuracy_score(y_test, y_pred)
    ans.iloc[j+4,i]=acc*100
print(ans)

