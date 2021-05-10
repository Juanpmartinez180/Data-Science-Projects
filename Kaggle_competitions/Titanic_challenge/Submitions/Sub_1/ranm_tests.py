# -*- coding: utf-8 -*-

 #--- DATA PREPROCESSING ---
#---------------------------
#----Library import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#----Dataset import
to_predict = pd.read_csv('test.csv')
X_pred = to_predict.iloc[:, [1,3,4,5,6,8]].values

#----Nans cleaning
B = X_pred[:,2].reshape(-1,1)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent', verbose=0) 

imputer_2 = imputer.fit(B)

X_pred[:,2] = (imputer_2.transform(B)).reshape(-1,)


#----Categorical data codification
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#TO PREDICT SET
labelencoder_X3 = LabelEncoder()
labelencoder_X4 = LabelEncoder()
X_pred[:, 0] = labelencoder_X3.fit_transform(X_pred[:,0])
X_pred[:, 1] = labelencoder_X4.fit_transform(X_pred[:,1])

ct = ColumnTransformer( [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
                       remainder='passthrough'                         # Leave the rest of the columns untouched
)

X_pred = np.array(ct.fit_transform(X_pred))
X_pred = X_pred[:, 1:]