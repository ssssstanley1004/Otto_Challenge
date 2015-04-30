"""
Utility functions 

"""

import numpy as np
import pandas as pd

def calculate_logloss(source, target):
	
	"""
	Log loss function
	"""
	
    # @Params: source = predicted probabilities. Target = target value. Both in pd.DataFrame
    # @Return: logloss
    sumloss = 0.0
    zeros = 0.0
    n = len(target)
    for i in range(n):
        prob = source.iloc[i][target.iloc[i]]
        if prob != 0:
            sumloss += math.log(prob)
        else:
            zeros += 1
    return -sumloss/(n-zeros)

def add_feature_sum(data, feature_name):
    
    """
    Add a new feature of sum of values of all other features in place
    """
    data[feature_name] = data.sum(axis=1)

def add_nonzero_counts(data, feature_name):

    """
    Add a new feature of counting # of non-zero features
    """
    non_zero_count = [len(np.unique(data.iloc[i].values))-1 for i in range(len(data))]
    data[feature_name] = pd.Series(non_zero_count)

def calibrate_prob(prob, r):

    """
    Calibration function for Random Forest Classifier
    """
    max = prob.max()
    isMax = [max==x for x in prob]
    calibrated_prob = [prob[i] + r*(1-prob[i]) if isMax[i] 
                           else prob[i]*(1-r) for i in range(len(prob))]
    return calibrated_prob

def get_calibrated_pred(pred, r=0.33):
    calibrated_pred = []
    for prob in pred:
        calibrated_pred.append(calibrate_prob(prob,r))
    return calibrated_pred
    
    
def check_bad_pred(pred_proba, target):
	
	"""
	Check back classifications (Predicted class <> target class) in training
	"""
	
    bad_preds = pd.DataFrame(columns=['max_proba_class','max_proba','target_class','target_proba'])
    for i in range(len(pred_proba)):
        pred = pred_proba.iloc[i]
        max_class = pred.idxmax(axis=1)
        if max_class <> target.iloc[i]:
            bad_pred=pd.DataFrame({'max_proba_class':max_class,'max_proba':pred[max_class],'target_class':target.iloc[i],'target_proba':pred[target.iloc[i]]},
                                  index=[i])
            bad_preds=bad_preds.append(bad_pred)
    return bad_preds
