# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 00:15:18 2017

@author: NafiS
"""
#removing sklearn wornings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#imporing packages
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
import numpy as np
import pandas as pd

#reading data
data = pd.read_csv('mll.csv')

#print(data.describe().transpose())

#print(data.info())

#removing 'Player' attribute
data.drop('AGE',axis=1, inplace=True)

#Target in numpy array
y = data['GTIME'].values

#correlation metrix
#print(data.corr(method='kendall'))


#dropping attribute depending on correlation metrix
data.drop('ECMO_TCR',axis=1, inplace=True)
data.drop('GTIME',axis=1, inplace=True)
data.drop('HEMO_PA_MN_TRR',axis=1, inplace=True)
data.drop('PRAMR_CL2',axis=1, inplace=True)
data.drop('ABO_MAT',axis=1, inplace=True)

#print(data.corr(method='kendall'))

#removing 'Transfer' attribute or target attribute
data.drop('INIT_AGE',axis=1, inplace=True)



#features in numpy array
X = data.values
try:
    float(X)
except ValueError:
    pass

#creating objects of the classifier
rfc = RandomForestClassifier(n_estimators=50,max_depth=1,random_state=10)

dtc = DecisionTreeClassifier(random_state=50)

svc = svm.SVC(random_state=50)

knn = KNeighborsClassifier(n_neighbors=5)

gbc = GradientBoostingClassifier(n_estimators=3, learning_rate=1.0, max_depth=1, random_state=50)


#declaring lists
accuracy_rfc = []
accuracy_dtc = []
accuracy_svc = []
accuracy_knn = []
accuracy_gbc = []

precision_rfc = []
precision_dtc = []
precision_svc = []
precision_knn = []
precision_gbc = []

recall_rfc = []
recall_dtc = []
recall_svc = []
recall_knn = []
recall_gbc = []

F1_rfc = []
F1_dtc = []
F1_svc = []
F1_knn = []
F1_gbc = []

#Scaling features
"""scaler = preprocessing.StandardScaler().fit(X.astype(float))
X = scaler.transform(X.astype(float))

try:
    float(X)
except ValueError:
    pass
"""
#K Fold Cross validation
kf = KFold(n_splits=5,random_state = 0,shuffle = True)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #Model Traning
    rfc.fit(X_train, y_train)
    dtc.fit(X_train, y_train)
    svc.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    gbc.fit(X_train, y_train)
    
    #predection
    y_pred_rfc = rfc.predict(X_test)
    y_pred_dtc = dtc.predict(X_test)
    y_pred_svc = svc.predict(X_test)
    y_pred_knn = knn.predict(X_test)
    
    
    y_pred_gbc = gbc.predict(X_test) 
    
    
    
    accuracy_rfc.append( metrics.accuracy_score(y_test, y_pred_rfc))
    accuracy_dtc.append( metrics.accuracy_score(y_test, y_pred_dtc))
    accuracy_svc.append( metrics.accuracy_score(y_test, y_pred_svc))
    accuracy_knn.append( metrics.accuracy_score(y_test, y_pred_knn))
    accuracy_gbc.append( metrics.accuracy_score(y_test, y_pred_gbc))
    
    precision_rfc.append(metrics.precision_score(y_test, y_pred_rfc,average='macro'))
    precision_dtc.append(metrics.precision_score(y_test, y_pred_dtc,average='macro'))
    precision_svc.append(metrics.precision_score(y_test, y_pred_svc,average='macro'))
    precision_knn.append(metrics.precision_score(y_test, y_pred_knn,average='macro'))
    precision_gbc.append(metrics.precision_score(y_test, y_pred_gbc,average='macro'))
    
    recall_rfc.append(metrics.recall_score(y_test, y_pred_rfc,average='macro'))
    recall_dtc.append(metrics.recall_score(y_test, y_pred_dtc,average='macro'))
    recall_svc.append(metrics.recall_score(y_test, y_pred_svc,average='macro'))
    recall_knn.append(metrics.recall_score(y_test, y_pred_knn,average='macro'))
    recall_gbc.append(metrics.recall_score(y_test, y_pred_gbc,average='macro'))
    
    
    F1_rfc.append(metrics.f1_score(y_test, y_pred_rfc,average='macro'))
    F1_dtc.append(metrics.f1_score(y_test, y_pred_dtc,average='macro'))
    F1_svc.append(metrics.f1_score(y_test, y_pred_svc,average='macro'))
    F1_knn.append(metrics.f1_score(y_test, y_pred_knn,average='macro'))
    F1_gbc.append(metrics.f1_score(y_test, y_pred_gbc,average='macro'))
    

#Performance metric
print ("  FOR Random Forest Classifier:")

print ("    Avg Accuracy : ",np.mean(accuracy_rfc)*1000)
print ("    Avg Precision : ",np.mean(precision_rfc)*1000)
print ("    Avg Recall : ",np.mean(recall_rfc)*1000)
print ("    Avg F1 : ",np.mean(F1_rfc)*1000)
print ("         ")
print ("         ")


print ("  FOR Decision Tree Classifier:")

print ("    Avg Accuracy : ",np.mean(accuracy_dtc)*1000)
print ("    Avg Precision : ",np.mean(precision_dtc)*1000)
print ("    Avg Recall : ",np.mean(recall_dtc)*1000)
print ("    Avg F1 : ",np.mean(F1_dtc)*1000)
print ("         ")
print ("         ")

