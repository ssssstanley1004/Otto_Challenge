"""
Random Forest Classifier with calibration

"""

import pandas as pd
import numpy as np
from sklearn import ensemble
import util



class RandomForestClfCal(object):
	
	def __init__(self,
                 n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features="auto",
                 max_leaf_nodes=None,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 min_density=None,
                 compute_importances=None,
				 calibration=True):
				 
		
		self.calibration = calibration	
	
		self.clf = ensemble.RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
													min_samples_leaf=min_samples_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, oob_score=oob_score,
													n_jobs=n_jobs, random_state=random_state, verbose=verbose, min_density=min_density, compute_importances=compute_importances)
	
	def fit(self, X, y):
		self.clf.fit(X,y)
		return self
		
		
	def predict_proba(self, X):
		proba = self.clf.predict_proba(X)
		
		if self.calibration:
			proba = util.get_calibrated_pred(proba)
			
		return proba
