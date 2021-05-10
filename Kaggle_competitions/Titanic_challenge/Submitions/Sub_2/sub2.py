#-- TITANIC KAGGLE CHALLENGE ----------- SUBMITION 1

"""Second attempt to the titanic challenge.
The objetive is using the same SVM regressor, improve the dataset analysis
to find more correlations and in consecuence a better accuracy
I want to add the lastest column of the dataset (Embarked), then codify due is
a categorical variable.
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

#---Dataset split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#---Variable scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
X_pred = sc_X.transform(X_pred)

#---------------- MODEL SELECTION ----------------------
#---Model selection and dataset fitting
from sklearn.svm import SVC
classifier = SVC(C = 125, kernel = 'rbf',gamma = 0.007 ,random_state = 0)
classifier.fit(X_train, y_train)

#---Grid search to enhance the model and hyperparameters
# from sklearn.model_selection import GridSearchCV
# parameters =    [{'C' :[133,130,125, 120, 123], 'kernel': ['rbf'], 'gamma':[0.003,0.005,0.007,0.008,0.009,0.01,0.015]}
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
#1 - kernel = rbf, C = 135, gamma = 0,012 - accuracy = -
#2 - kernel = rbf, C = 125, gamma = 0,007 - accuracy = 0.82586
"""

#---Result predictions and file saving 

y_pred = classifier.predict(X_test)
y_pred_test = classifier.predict(X_pred)

passenger_ID = to_predict.iloc[:,0].values
submit_test = np.vstack((passenger_ID, y_pred_test)).T

np.savetxt('submission2.csv', submit_test, delimiter = ',', fmt = '%s',header = 'PassengerID, Survived' )

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
Grid search enhance - accuracy = 0.84357

"""

