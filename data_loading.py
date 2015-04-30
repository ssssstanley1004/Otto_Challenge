"""
Functions to load dataset with preprocessing

"""

import pandas as pd
import numpy as np
import os
from sklearn import preprocessing


def load(dir):

    # @Return: X_train, X_test, y, sample_submission format in pd.DataFrame 

    # import data
    os.chdir(dir)
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    sample = pd.read_csv('sampleSubmission.csv')

    X = train.values.copy()
    X_id, X, labels = X[:,0], X[:, 1:-1].astype(np.float32), X[:,-1]

    scaler = StandardScaler()
    X = np.log10(X+1)

    #Add ID back for ensembling
    #df=pd.DataFrame(X)
    #df['id']=pd.Series(X_id)
    #X = df.values


    # encode labels 
    lbl_enc = preprocessing.LabelEncoder()
    labels = lbl_enc.fit_transform(labels)

    X_test = test.values.copy()
    X_test, ids_test = X_test[:, 1:].astype(np.float32), X_test[:, 0].astype(str)
    X_test = np.log10(X_test+1)

    y=labels

    return X_train, X_test, y, sample
    
    
