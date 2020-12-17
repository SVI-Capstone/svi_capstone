from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

######################## Logistic Regression ##########################

def logistic_regression(X_train, y_train, X_bow, X_tfidf):
    '''
    This function takes in X_train (features using for model) and y_train (target) and performs logistic
    regression giving us accuracy of the model and the classification report
    '''
    # Calling out funtion
    lm = LogisticRegression()

    # Array of the predicitons
    lm_bow = lm.fit(X_bow, y_train)
    X_train['predicted'] = lm_bow.predict(X_bow)

    # X_bow
    print('X_bow Accuracy: {:.0%}\n'.format(accuracy_score(y_train, X_train.predicted)))
    print('-----------------------')
    print(f'X_bow Confusion Matrix: \n\n {pd.crosstab(y_train.language, X_train.predicted)}\n' )
    print('-----------------------')
    print("X_bow Logistic Regression Classification Report:\n", classification_report(y_train, X_train.predicted))
    
    lm_tfidf = lm.fit(X_tfidf, y_train)
    X_train['pred_tfidf'] = lm_tfidf.predict(X_tfidf)

    # TF-IDF
    print('-----------------------')
    print('TF-IDF Accuracy: {:.0%}\n'.format(accuracy_score(y_train, X_train.pred_tfidf)))
    print('-----------------------')
    print(f'TF-IDF Confusion Matrix: \n\n {pd.crosstab(y_train.language, X_train.pred_tfidf)}\n' )
    print('-----------------------')
    print("TF-IDF Logistic Regression Classification Report:\n", classification_report(y_train, X_train.pred_tfidf))
    return lm_bow, lm_tfidf
    ######################## Decesion Tree ##########################

def decesion_tree(X_train, y_train, X_bow, X_tfidf, k):
    '''
    This function requires X_train, y_train and k (max_depth). A confusion matrix, models accuracy and 
    classification report are outputed
    '''
    # Creating the decision tree object
    clf = DecisionTreeClassifier(max_depth=k, random_state=123)

    # Fitting the data to the trained data using xbow
    clf.fit(X_bow, y_train)

    # Array of the predicitons
    X_train['predicted'] = clf.predict(X_bow)

    # Confusion matrix
    print('X_bow Accuracy: {:.0%}\n'.format(accuracy_score(y_train, X_train.predicted)))
    print('-----------------------')
    print(f'X_bow Confusion Matrix: \n\n {pd.crosstab(y_train.language, X_train.predicted)}\n' )
    print('-----------------------')
    print("X_bow Decesion Tree Classification Report:\n", classification_report(y_train, X_train.predicted))
                                        
    # tfidf
    clf_tfidf = clf.fit(X_tfidf, y_train)
    X_train['pred_tfidf'] = clf_tfidf.predict(X_tfidf)
                                    
    # Confusion matrix
    print('-----------------------')
    print('TF-IDF Accuracy: {:.0%}\n'.format(accuracy_score(y_train, X_train.pred_tfidf)))
    print('-----------------------')
    print(f'TF-IDF Confusion Matrix: \n\n {pd.crosstab(y_train.language, X_train.pred_tfidf)}\n' )
    print('-----------------------')
    print("TF-IDF Decesion Tree Classification Report:\n", classification_report(y_train, X_train.pred_tfidf))

    ######################## Random Forest ##########################

# def random_forest(X_train, y_train, X_bow, X_tfidf, k):
#     '''
#     This function takes in X_train (features using for model) and y_train (target) and performs random
#     forest giving us accuracy of the model and the classification report
#     '''
#     # Random forest object
#     rf = RandomForestClassifier(n_estimators=500, max_depth=k, random_state=123)

#     # Fitting the data to the trained data
#     rf_bow = rf.fit(X_bow, y_train)

#     # Array of the predicitons
#     X_train['predicted'] = rf_bow.predict(X_bow)

#     # Confusion matrix
#     print('X_bow Accuracy: {:.0%}\n'.format(accuracy_score(y_train.language, X_train.predicted)))
#     print('-----------------------')
#     print(f'X_bow Confusion Matrix: \n\n {pd.crosstab(y_train.language, X_train.predicted)}\n' )
#     print('-----------------------')
#     print("X_bow Random Forest Classification Report:\n", classification_report(y_train.language, X_train.predicted))

#     rf_tfidf = rf.fit(X_tfidf, y_train)
#     X_train['pred_tfidf'] = rf_tfidf.predict(X_tfidf)

#    # Confusion matrix
#     print('-----------------------')
#     print('TF-IDF Accuracy: {:.0%}\n'.format(accuracy_score(y_train.language, X_train.pred_tfidf)))
#     print('-----------------------')
#     print(f'TF-IDF Confusion Matrix: \n\n {pd.crosstab(y_train.language, X_train.pred_tfidf)}\n' )
#     print('-----------------------')
#     print("TF-IDF Random Forest Classification Report:\n", classification_report(y_train.language, X_train.pred_tfidf))
#     return rf_bow, rf_tfidf
    ######################## KNN ##########################

def knn(X_train, y_train, X_bow, X_tfidf, k):
    '''
    This function takes in X_train (features using for model) and y_train (target) and performs 
     K nearest neighbors giving us accuracy of the model and the classification report
    '''
    # KNN object
    knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')

    # Fit the model
    knn.fit(X_bow, y_train)

    # Make predictions
    X_train['predicted'] = knn.predict(X_bow)
    
    # X_bow
    print('X_bow Accuracy: {:.0%}\n'.format(accuracy_score(y_train.language, X_train.predicted)))
    print('-----------------------')
    print(f'X_bow Confusion Matrix: \n\n {pd.crosstab(y_train.language, X_train.predicted)}\n' )
    print('-----------------------')
    print("X_bow KNN Classification Report:\n", classification_report(y_train.language, X_train.predicted))
    
    knn_tfidf = knn.fit(X_tfidf, y_train)
    X_train['pred_tfidf'] = knn_tfidf.predict(X_tfidf)

    # TF-IDF
    print('-----------------------')
    print('TF-IDF Accuracy: {:.0%}\n'.format(accuracy_score(y_train.language, X_train.pred_tfidf)))
    print('-----------------------')
    print(f'TF-IDF Confusion Matrix: \n\n {pd.crosstab(y_train.language, X_train.pred_tfidf)}\n' )
    print('-----------------------')
    print("TF-IDF KNN Classification Report:\n", classification_report(y_train.language, X_train.pred_tfidf))

    ######################## Complement Naive Bayes ##########################

def complement_naive_bayes(X_train, y_train, X_tfidf):
    '''
    This function takes in X_train (features using for model) and y_train (target) and performs 
    complement Naive Bayes giving us accuracy of the model and the classification report
    '''    
    # Call function and fit
    cnb = ComplementNB().fit(X_tfidf, y_train)
    
    cnb_tfidf = cnb.fit(X_tfidf, y_train)
    X_train['pred_tfidf'] = cnb_tfidf.predict(X_tfidf)

    # TF-IDF
    print('TF-IDF Accuracy: {:.0%}\n'.format(accuracy_score(y_train, X_train.pred_tfidf)))
    print('-----------------------')
    print(f'TF-IDF Confusion Matrix: \n\n {pd.crosstab(y_train.language, X_train.pred_tfidf)}\n' )
    print('-----------------------')
    print("TF-IDF Complement Niave Bayes Classification Report:\n", classification_report(y_train, X_train.pred_tfidf))
    return cnb_tfidf
    ######################## Multinomial Naive Bayes ##########################

def multinomial_naive_bayes(X_train, y_train, X_tfidf):
    '''
    This function takes in X_train (features using for model) and y_train (target) and performs 
    multinomial naive bayes giving us accuracy of the model and the classification report
    '''    
    # Call function and fit
    mnb = MultinomialNB().fit(X_tfidf, y_train)
    
    mnb_tfidf = mnb.fit(X_tfidf, y_train)
    X_train['pred_tfidf'] = mnb_tfidf.predict(X_tfidf)

    # TF-IDF
    print('TF-IDF Accuracy: {:.0%}\n'.format(accuracy_score(y_train, X_train.pred_tfidf)))
    print('-----------------------')
    print(f'TF-IDF Confusion Matrix: \n\n {pd.crosstab(y_train.language, X_train.pred_tfidf)}\n' )
    print('-----------------------')
    print("TF-IDF Multinomial Niave Bayes Classification Report:\n", classification_report(y_train, X_train.pred_tfidf))

    ######################## Validate Logistic Regression ##########################

def validate_logistic_regression(X_validate, y_validate, V_bow, V_tfidf, lm_bow, lm_tfidf):
    '''
    This function takes in X_train (features using for model) and y_train (target) and performs logistic
    regression giving us accuracy of the model and the classification report
    '''
    # Calling out funtion
    #lm = LogisticRegression().fit(V_bow, y_validate)

    # Array of the predicitons
    X_validate['predicted'] = lm_bow.predict(V_bow)

     # X_bow
    print('X_bow Accuracy: {:.0%}\n'.format(accuracy_score(y_validate, X_validate.predicted)))
    print('-----------------------')
    print(f'X_bow Confusion Matrix: \n\n {pd.crosstab(y_validate.language, X_validate.predicted)}\n' )
    print('-----------------------')
    print("X_bow Logistic Regression Classification Report:\n", classification_report(y_validate, X_validate.predicted))
    
    #lm_tfidf = lm.fit(V_tfidf, y_validate)
    X_validate['pred_tfidf'] = lm_tfidf.predict(V_tfidf)

    # TF-IDF
    print('-----------------------')
    print('TF-IDF Accuracy: {:.0%}\n'.format(accuracy_score(y_validate, X_validate.pred_tfidf)))
    print('-----------------------')
    print(f'TF-IDF Confusion Matrix: \n\n {pd.crosstab(y_validate.language, X_validate.pred_tfidf)}\n' )
    print('-----------------------')
    print("TF-IDF Logistic Regression Classification Report:\n", classification_report(y_validate, X_validate.pred_tfidf))

    ######################## Validate Random Forest ##########################

def validate_random_forest(X_validate, y_validate, V_bow, V_tfidf, k, rf_bow, rf_tfidf):
    '''
    This function takes in X_train (features using for model) and y_train (target) and performs random
    forest giving us accuracy of the model and the classification report
    '''    
    # Array of the predicitons
    X_validate['predicted'] = rf_bow.predict(V_bow)

    # Confusion matrix
    print('X_bow Accuracy: {:.0%}\n'.format(accuracy_score(y_validate.language, X_validate.predicted)))
    print('-----------------------')
    print(f'X_bow Confusion Matrix: \n\n {pd.crosstab(y_validate.language, X_validate.predicted)}\n' )
    print('-----------------------')
    print("X_bow Random Forest Classification Report:\n", classification_report(y_validate.language, X_validate.predicted))

    #rf_tfidf = rf.fit(V_tfidf, y_validate)
    X_validate['pred_tfidf'] = rf_tfidf.predict(V_tfidf)

    # Confusion matrix
    print('-----------------------')
    print('TF-IDF Accuracy: {:.0%}\n'.format(accuracy_score(y_validate.language, X_validate.pred_tfidf)))
    print('-----------------------')
    print(f'TF-IDF Confusion Matrix: \n\n {pd.crosstab(y_validate.language, X_validate.pred_tfidf)}\n' )
    print('-----------------------')
    print("TF-IDF Random Forest Classification Report:\n", classification_report(y_validate.language, X_validate.pred_tfidf))

######################## Validate Complement Naive Bayes ##########################

def validate_complement_naive_bayes(X_validate, y_validate, V_tfidf, cnb_tfidf):
    '''
    This function takes in X_train (features using for model) and y_train (target) and performs 
    complement Naive Bayes giving us accuracy of the model and the classification report
    '''       
    # Array of the predicitons
    X_validate['pred_tfidf'] = cnb_tfidf.predict(V_tfidf)

    # TF-IDF
    print('TF-IDF Accuracy: {:.0%}\n'.format(accuracy_score(y_validate, X_validate.pred_tfidf)))
    print('-----------------------')
    print(f'TF-IDF Confusion Matrix: \n\n {pd.crosstab(y_validate.language, X_validate.pred_tfidf)}\n' )
    print('-----------------------')
    print("TF-IDF Complement Niave Bayes Classification Report:\n", classification_report(y_validate, X_validate.pred_tfidf))

    ######################## Test Random Forest ##########################

def test_random_forest(X_test, y_test, T_tfidf, k, rf_tfidf):
    '''
    This function takes in X_train (features using for model) and y_train (target) and performs random
    forest giving us accuracy of the model and the classification report
    '''      
    # Array of the predicitons
    X_test['pred_tfidf'] = rf_tfidf.predict(T_tfidf)

   # Confusion matrix
    print('TF-IDF Accuracy: {:.0%}\n'.format(accuracy_score(y_test.language, X_test.pred_tfidf)))
    print('-----------------------')
    print(f'TF-IDF Confusion Matrix: \n\n {pd.crosstab(y_test.language, X_test.pred_tfidf)}\n' )
    print('-----------------------')
    print("TF-IDF Random Forest Classification Report:\n", classification_report(y_test.language, X_test.pred_tfidf))



def test_logistic_regression(X_test, y_test, T_bow, T_tfidf, lm_bow, lm_tfidf):
    '''
    This function takes in X_train (features using for model) and y_train (target) and performs logistic
    regression giving us accuracy of the model and the classification report
    '''   
    # Array of the predicitons
    X_test['predicted'] = lm_bow.predict(T_bow)

     # X_bow
    print('X_bow Accuracy: {:.0%}\n'.format(accuracy_score(y_test, X_test.predicted)))
    print('-----------------------')
    print(f'X_bow Confusion Matrix: \n\n {pd.crosstab(y_test.language, X_test.predicted)}\n' )
    print('-----------------------')
    print("X_bow Logistic Regression Classification Report:\n", classification_report(y_test, X_test.predicted))
    
    #lm_tfidf = lm.fit(T_tfidf, y_test)
    X_test['pred_tfidf'] = lm_tfidf.predict(T_tfidf)

    # TF-IDF
    print('-----------------------')
    print('TF-IDF Accuracy: {:.0%}\n'.format(accuracy_score(y_test, X_test.pred_tfidf)))
    print('-----------------------')
    print(f'TF-IDF Confusion Matrix: \n\n {pd.crosstab(y_test.language, X_test.pred_tfidf)}\n' )
    print('-----------------------')
    print("TF-IDF Logistic Regression Classification Report:\n", classification_report(y_test, X_test.pred_tfidf))


def random_forest_class(X, y, max_depth = 4, n_estimators = 100, ):
    # Setting the features:
    X_train_rf = X
    y_train_rf = y

    # Using Random Forest Model
    rf = RandomForestClassifier(bootstrap=True, 
                                class_weight=None, 
                                criterion='gini',
                                min_samples_leaf=3,
                                n_estimators = n_estimators,
                                max_depth = max_depth, 
                                random_state=123)

    # Fitting the model using the train data:
    rf.fit(X_train_rf, y_train_rf)

    # Making prediction:
    y_pred_rf = rf.predict(X_train_rf)
    
    # Estimating the case count bin using the training dataset:
    y_pred_proba_rf = rf.predict_proba(X_train_rf)
    
    # Cross Validation:
    rfc_cv_score = cross_val_score(rf, X_train_rf, y_train_rf, cv=10, scoring='roc_auc_ovr')
    
    print('Accuracy of Random Forest Model on training set: {:.2f}'
     .format(rf.score(X_train_rf, y_train_rf)))
    
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_train_rf, y_pred_rf))
    print('\n')
    print("=== Classification Report ===")
    print(classification_report(y_train_rf, y_pred_rf))
    print('\n')
    print("=== All AUC Scores ===")
    print(rfc_cv_score)
    print('\n')
    print("=== Mean AUC Score ===")
    print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())

    return rf.score(X_train_rf, y_train_rf)