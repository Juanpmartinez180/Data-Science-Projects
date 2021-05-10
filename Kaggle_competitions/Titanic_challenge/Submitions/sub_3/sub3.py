#-- TITANIC KAGGLE CHALLENGE ----------- SUBMITION 1

"""Third attempt to the titanic challenge.
The idea is use another classification model.. 
Random Forest is a good option.. 
Firts i need to run the model and see the accuracy, the improve it
using kfold and grid search.
As a last step once the model is in their best shape, y pretend to use the whole
training set to train the model.. So maybe the model can reach an even better 
accuracy due the use of a 20%more of data to traing.
Then, re-run the model to see if there is any improvement... LET'S DO IT!
"""

# --- DATA PREPROCESSING ---
#---------------------------
#----Library import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#----Dataset import
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, [2,4,5,6,7,9,11]].values
y = dataset.iloc[:, 1].values

to_predict = pd.read_csv('test.csv')
X_pred = to_predict.iloc[:, [1,3,4,5,6,8,10]].values

#----Nans values cleaning
A = X[:,2].reshape(-1,1)
B = X_pred[:,2].reshape(-1,1)
C = X_pred[:,5].reshape(-1,1)
D = X[:,6].reshape(-1,1)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent', verbose=0) 

imputer_1 = imputer.fit(A)
X[:,2] = (imputer_1.transform(A)).reshape(-1,)

imputer_2 = imputer.fit(B)
X_pred[:,2] = (imputer_2.transform(B)).reshape(-1,)

imputer_3 = imputer.fit(C)
X_pred[:,5] = (imputer_3.transform(C)).reshape(-1,)

imputer_4 = imputer.fit(D)
X[:,6] = (imputer_4.transform(D)).reshape(-1,)

#----Categorical data codification
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#TRAINING SET
labelencoder_X1 = LabelEncoder()
labelencoder_X2 = LabelEncoder()
labelencoder_X5 = LabelEncoder()
X[:, 0] = labelencoder_X1.fit_transform(X[:,0])
X[:, 1] = labelencoder_X2.fit_transform(X[:,1])
X[:, 6] = labelencoder_X5.fit_transform(X[:,6])
ct = ColumnTransformer( [('one_hot_encoder', OneHotEncoder(categories='auto'), [0,6])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

X = np.delete(X, [0,3], 1)

"""
i need to delete the column 0 and 3 to prevent data redundacy

"""

#TO PREDICT SET
labelencoder_X3 = LabelEncoder()
labelencoder_X4 = LabelEncoder()
labelencoder_X6 = LabelEncoder()
X_pred[:, 0] = labelencoder_X3.fit_transform(X_pred[:,0])
X_pred[:, 1] = labelencoder_X4.fit_transform(X_pred[:,1])
X_pred[:, 6] = labelencoder_X6.fit_transform(X_pred[:,6])

ct = ColumnTransformer( [('one_hot_encoder', OneHotEncoder(categories='auto'), [0,6])], remainder='passthrough')
X_pred = np.array(ct.fit_transform(X_pred), dtype = np.float)

X_pred = np.delete(X_pred, [0,3], 1)

#---Variable scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
X_pred = sc_X. transform(X_pred)

#---Dataset split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.01, random_state = 0)


#---------------- MODEL SELECTION ----------------------
#---Model selection and dataset fitting
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators = 950,
#                                     criterion = 'entropy',
#                                     min_samples_leaf = 2,
#                                     min_samples_split = 22,
#                                     n_jobs = -1,
#                                     random_state = 0  )
# classifier.fit(X_train, y_train)


#XGboost model training
from xgboost import XGBClassifier

classifier = XGBClassifier()
classifier.fit(X_train, y_train)

#---Grid search to enhance the model and hyperparameters
# from sklearn.model_selection import GridSearchCV

# parameters =    [{'n_estimators' :[800,850,900,950,1000],
#                   'criterion': ['gini'],
#                   'min_samples_leaf' : [1,2,3],
#                   'min_samples_split' : [18,20,22,30]}
#                   ]

# grid_search = GridSearchCV(estimator = classifier, 
#                             param_grid = parameters,
#                             scoring = 'accuracy',
#                             cv = 10,
#                             n_jobs = -1)
# grid_search = grid_search.fit(X_train, y_train)

# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_

"""GRID SEARCH ITERATIONS
#1 - criterior = gini, min_samples_leaf = 1, min_samples_plit = 20, n_estimators = 900 - best_acc = 0.83
#2 - criterior = gini, min_samples_leaf = 2, min_samples_plit = 22, n_estimators = 950 - best_acc = 0.8342
"""

#---Result predictions and file saving 

y_pred = classifier.predict(X_test)
y_pred_test = classifier.predict(X_pred)

passenger_ID = to_predict.iloc[:,0].values
submit_test = np.vstack((passenger_ID, y_pred_test)).T

np.savetxt('submission3.csv', submit_test, delimiter = ',', fmt = '%s',header = 'PassengerID, Survived' )

#-------------------- FINAL MODEL METRICS --------------
#---Confussion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

TP = cm[1,1]    #True positive
TN = cm[0,0]    #True negative
FP = cm[1,0]    #False positive
FN = cm[0,1]    #False negative

accuracy = (TP+TN) / (np.sum(cm))
pressition = (TP)/(TP+FP)
recall = (TP)/(TP+FN)
F1_score = (2*pressition*recall) / (pressition+recall)

"""ITERATIONS RESULTS
SVM FIRST TRY - accuracy = 0.832402
Grid search enhance - accuracy = 0.8379

"""

