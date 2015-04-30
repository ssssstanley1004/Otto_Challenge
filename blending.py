"""
Blending functions

Blending with multiple classifiers to get predicted probabilities for each class out of each classifier. Apply Logistic Regression on predicted probabilities to calculate
blended probability predictions. In blending, 10-fold Cross-Validation is used. Logistic Regression is trained using predicted probabilities and labels and applied on blended
test set to get final predictions. Out of the pool of classifiers, a forward stepwise selection is applied to get the best combination of classifiers. 

"""



def train_classifier(X, y, X_test, classifier_list, n_fold, seed):
    
    """
    Blend_train is a collection of predicted probabilities out of 10-fold cross validation on training set (X). 
    Blend_test is a collection of predicted probabilities out of 10-fold cross validation on test set (X_test).
    
    train_classifier process can be regarded as a transformation which is applied on both training and test set. 
    Final prediction or final blending is based on Logistic Regression
    """
	
	#Params: X, y, X_test in np.narray
    
    print "Start training with preselected classifier on both training and test dataset"
    
    np.random.seed(seed)
    n_classifier = len(classifier_list)
    stratified_KFold = list(cross_validation.StratifiedKFold(y, n_fold))
    n_class = len(np.unique(y))

    blend_train = np.zeros((len(X), n_classifier, n_class))  # Dimensions: (# of observations, # of classifier, # of target class)
    blend_test = np.zeros((len(X_test), n_classifier, n_class))

    for i, clf in enumerate(classifier_list):
        print "current classifier ", i, clf
        blend_test_subfold = np.zeros((len(X_test), n_fold, n_class))
        for j, (tr, cv) in enumerate(stratified_KFold):
            print j, "fold"
            X_tr, X_cv = X[tr], X[cv]
            y_tr, y_cv = y[tr], y[cv]
            
            clf.fit(X_tr, y_tr)
            blend_train[cv, i] = clf.predict_proba(X_cv)
            blend_test_subfold[:, j] = clf.predict_proba(X_test)
        blend_test[:,i]=blend_test_subfold.mean(axis = 1)
    return blend_train, blend_test

def predict_with_blending_lgr(blend_train, blend_test,y):
    
    """
    Blending predictions with Logistic Regression
    
    """
    
    print "Start blending..."
    n_class = len(np.unique(y))
    class_index = range(n_class)
    y_submission = np.zeros((len(blend_test), n_class))
    
    for index in class_index:
        print "Applying logistics regression on class %d " % (index)
        y_class = y==index
        X_class = blend_train[:,:,index]
        
        lgr = linear_model.LogisticRegression()
        lgr.fit(X_class, y_class)
        y_submission[:,index] = lgr.predict_proba(blend_test[:,:,index])[:,1]
		y_submission = normalize_prediction(y_submission)
		
    return y_submission
    
    
def forward_stepwise_selection(X,y,classifier_list, n_fold, seed):

    """
    Apply forward stepwise selection to get the optimal subset of classifier combination
    
    """

    np.random.seed(seed)
    n_classifier = len(classifier_list)
    n_class = len(y.unique())
    
    # Split training set into _tr and _cv and use _cv as test to get logloss
    
    stratified_KFold = list(cross_validation.StratifiedKFold(y, 2))[0]
    tr, cv = stratified_KFold[0], stratified_KFold[1]
    X_tr, X_cv, y_tr, y_cv = X.iloc[tr], X.iloc[cv], y.iloc[tr], y.iloc[cv]S
    blend_tr, blend_cv = train_classifier(X_tr, y_tr, X_cv, classifier_list, n_fold, seed)
    
    clf_index_list = range(n_classifier)
    opt_clf_list = []
    min_logloss = float("inf")
    local_opt_found=True
    
    Print "Start forward stepwise selection" 
    while clf_index_list and local_opt_found:
        
        local_opt_found=False
        opt_sub_clf=[]
        for i in clf_index_list:
            
            opt_clf_list.append(i)
            i_blend_tr = blend_tr[:,opt_clf_list,:]
            i_blend_cv = blend_cv[:,opt_clf_list,:]
            i_prediction = predict_with_blending_lgr(i_blend_tr,i_blend_cv, y_tr)
            prediction_norm = pd.DataFrame(normalize_prediction(i_prediction))
            i_logloss = calculate_logloss(prediction_norm, y_cv) 
            opt_clf_list.pop(-1)
            
            if i_logloss < min_logloss:
                opt_sub_clf = i
                local_opt_found = True
                min_logloss = i_logloss
            print i, i_logloss, opt_sub_clf, min_logloss, local_opt_found, opt_clf_list
        
        if local_opt_found:
            opt_clf_list.append(opt_sub_clf)
            clf_index_list.remove(opt_sub_clf)
            print opt_clf_list, clf_index_list
    
    return classifier_list[opt_clf_list.sorted()]
    

def get_all_subset(list, interval):
    return [x for x in (list[a:a+b] for a in range(0,len(list)) for b in range(interval, len(list)-a+1))]

def normalize_prediction(prediction):
    n = prediction.shape[0]
    pred_norm = np.zeros(prediction.shape)
    for i in range(n):
        pred_norm[i] = prediction[i]/prediction[i].sum(0)
    return pred_norm
    


