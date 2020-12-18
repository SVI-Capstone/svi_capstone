import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from scipy import stats

import warnings
warnings.filterwarnings("ignore")

from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression, SelectKBest, RFE 
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures


def feature_ranking(X_train_scaled, y_train):
    lm = LinearRegression()
    rfe = RFE(lm, 1)
    rfe.fit(X_train_scaled, y_train)
    ranks = rfe.ranking_
    names = X_train_scaled.columns.tolist()
    rankdf = pd.DataFrame({'features': names, 'rank': ranks}).set_index('rank').sort_values('rank')
    return rankdf


def linear_reg_train(x_scaleddf, target):
    '''
    runs linear regression algorithm
    '''
    lm = LinearRegression()
    lm.fit(x_scaleddf, target)
    y_hat = lm.predict(x_scaleddf)

    LM_RMSE = sqrt(mean_squared_error(target, y_hat))
    return LM_RMSE

def get_baseline_mean(y_train):
    '''
    Using mean gets baseline for y dataframe
    '''
    # determine Baseline to beat
    rows_needed = y_train.shape[0]
    # create array of predictions of same size as y_train.logerror based on the mean
    y_hat = np.full(rows_needed, np.mean(y_train))
    # calculate the MSE for these predictions, this is our baseline to beat
    baseline = sqrt(mean_squared_error(y_train, y_hat))
    print("Baseline RMSE:", baseline)
    return baseline, y_hat

def get_baseline_median(y_train):
    '''
    Using median gets baseline for y dataframe
    '''
    # determine Baseline to beat
    rows_needed = y_train.shape[0]
    # create array of predictions of same size as y_train.logerror based on the median
    y_hat = np.full(rows_needed, np.median(y_train))
    # calculate the MSE for these predictions, this is our baseline to beat
    baseline = sqrt(mean_squared_error(y_train, y_hat))
    print("Baseline RMSE:", baseline)
    return baseline, y_hat

def lasso_lars(x_scaleddf, target):
    '''
    runs Lasso Lars algorithm
    ''' 
    # Make a model
    lars = LassoLars(alpha=1)
    # Fit a model
    lars.fit(x_scaleddf, target)
    # Make Predictions
    lars_pred = lars.predict(x_scaleddf)
    # Computer root mean squared error
    lars_rmse = sqrt(mean_squared_error(target, lars_pred))
    return lars_rmse

def polynomial2(X_trainsdf, target):
    '''
    runs polynomial algorithm
    ''' 
    # Make a model
    pf = PolynomialFeatures(degree=2)
    # Fit and Transform model to get a new set of features...which are the original features squared
    X_train_squared = pf.fit_transform(X_trainsdf)
    
    # Feed new features in to linear model. 
    lm_squared = LinearRegression(normalize=True)
    lm_squared.fit(X_train_squared, target)
    # Make predictions
    lm_squared_pred = lm_squared.predict(X_train_squared)
    # Compute root mean squared error
    pf2_rmse = sqrt(mean_squared_error(target, lm_squared_pred))
    return pf2_rmse


def poly_val_test(X_train_scaled, X_validate_scaled, y_train, y_validate):
    '''
    runs polynomial algorithm for validate and test dataframes
    needs to be fixed/evaluated, should be combined with above and above needs to return transformed
    for later use
    ''' 
    # Make a model
    pf = PolynomialFeatures(degree=2)
    X_train_squared = pf.fit_transform(X_train_scaled)
    X_validate_squared = pf.transform(X_validate_scaled)
    #X_test_squared = pf.transform(X_test_scaled)
    # Feed new features in to linear model. 
    lm_squared = LinearRegression(normalize=True)
    lm_squared.fit(X_train_squared, y_train)
    
    # Make Predictions
    lm_pred_train = lm_squared.predict(X_train_squared)
    lm_pred_val = lm_squared.predict(X_validate_squared)

    # Compute root mean squared error
    lm_rmse_train = sqrt(mean_squared_error(y_train, lm_pred_train))
    lm_rmse_val = sqrt(mean_squared_error(y_validate, lm_pred_val))
    print('RMSE train=', lm_rmse_train)
    print('RMSE validate=', lm_rmse_val) 
    return lm_rmse_train, lm_rmse_val 


def linear_reg_vt(X_train_scaled, X_validate_scaled, y_train, y_validate):
    '''
    runs linear regression algorithm on validate and test
    but fits model on train
    '''
    lm = LinearRegression()
    lm.fit(X_train_scaled, y_train)

    y_hat = lm.predict(X_validate_scaled)

    LM_RMSE = sqrt(mean_squared_error(y_validate, y_hat))
    return LM_RMSE, y_hat    

def tweedie05(X_train_scaled, y_train):
    '''
    runs tweedie algorithm
    ''' 
    # Make Model
    tw = TweedieRegressor(power=0, alpha=.5) # 0 = normal distribution
    # Fit Model
    tw.fit(X_train_scaled, y_train)
    # Make Predictions
    tw_pred = tw.predict(X_train_scaled)
    # Compute root mean squared error
    tw_rmse = sqrt(mean_squared_error(y_train, tw_pred))
    return tw_rmse

def tweedie_vt(X_train_scaled, X_validate_scaled, y_train, y_validate):
    '''
    runs tweedie algorithm on validate and test
    but fits model on train
    '''
    # Make Model
    tw = TweedieRegressor(power=0, alpha=0.001) # 0 = normal distribution
    # Fit Model
    tw.fit(X_train_scaled, y_train)
    # Make Predictions
    tw_pred = tw.predict(X_validate_scaled)
    # Compute root mean squared error
    tw_rmse = sqrt(mean_squared_error(y_validate, tw_pred))
    return tw_rmse