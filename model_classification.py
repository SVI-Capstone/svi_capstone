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

# Functions

def random_forest_class(X, y, max_depth = 4, n_estimators = 100):
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
    rfc_cv_score = cross_val_score(rf, X_train_rf, y_train_rf, cv=10, scoring='accuracy')
    print(f'RF CV scores mean: {round(rfc_cv_score.mean(), 2)}')
    
    # print('Accuracy of Random Forest Model on training set: {:.2f}'
    #  .format(rf.score(X_train_rf, y_train_rf)))
    
#     print("=== Confusion Matrix ===")
#     print(confusion_matrix(y_train_rf, y_pred_rf))
#     print('\n')
#     print("=== Classification Report ===")
#     print(classification_report(y_train_rf, y_pred_rf))
#     print('\n')
#     print("=== All AUC Scores ===")
#     print(rfc_cv_score)
#     print('\n')
#     print("=== Mean AUC Score ===")
#     print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())
    return rfc_cv_score.mean()


# KNN Function:

def knn_classification(X_train_knn, y_train_knn, n_neighbors=5, cv = 5):
    
    # Making the model:
    knn = KNeighborsClassifier(n_neighbors = n_neighbors, weights='uniform')

    # Fitting the model:
    knn.fit(X_train_knn, y_train_knn)

    # Getting the score:
    knn.score(X_train_knn, y_train_knn)

    # Cross Validation:
    knn_cv_scores = cross_val_score(knn, X_train_knn, y_train_knn, cv = cv, scoring='accuracy')

    # Printing the CV scores:
    # print each cv score (accuracy) and average them
    # print(knn_cv_scores)
    print(f'KNN CV scores mean: {round(knn_cv_scores.mean(), 2)}')
    
    # predict y values
    y_pred_knn = knn.predict(X_train_knn)
    y_pred_proba_knn = knn.predict_proba(X_train_knn)

    # print('Accuracy of KNN classifier on training set: {:.2f}'
    #      .format(knn.score(X_train_knn, y_train_knn)))
    # print("\n---------------------------------")
    # print(confusion_matrix(y_train_knn, y_pred_knn))
    # print("\n---------------------------------")
    # print(classification_report(y_train_knn, y_pred_knn))
    
    # print("\n\nArray of cross validation scores:")
    
    return knn_cv_scores.mean()

# RF test function:

def rf_test_class(X, y, X_test, y_test, max_depth = 4, n_estimators = 100):
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
    y_pred_rf = rf.predict(X_test)
    
    # Estimating the case count bin using the training dataset:
#     y_pred_proba_rf = rf.predict_proba(X_test)

    
    rf_score = rf.score(X_test, y_test)
    
    print('Accuracy of Random Forest Model on test set: {:.2f}'
     .format(rf.score(X_test, y_test)))

    return rf_score