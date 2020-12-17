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
    print(f'RF CV scores mean: {round(rfc_cv_score.mean(), 2)}')
    
    print('Accuracy of Random Forest Model on training set: {:.2f}'
     .format(rf.score(X_train_rf, y_train_rf)))
    
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
    print("\n\nArray of cross validation scores:")
    return rfc_cv_score

    return rfc_cv_score.mean()