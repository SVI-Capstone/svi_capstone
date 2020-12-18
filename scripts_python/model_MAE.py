import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from scipy import stats

import warnings
warnings.filterwarnings("ignore")

from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
from sklearn.feature_selection import f_regression, SelectKBest, RFE 
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

from numpy import mean
from numpy import std, absolute
from sklearn.datasets import make_blobs
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

def feature_ranking(X_train_scaled, y_train):
    lm = LinearRegression()
    rfe = RFE(lm, 1)
    rfe.fit(X_train_scaled, y_train)
    ranks = rfe.ranking_
    names = X_train_scaled.columns.tolist()
    rankdf = pd.DataFrame({'features': names, 'rank': ranks}).set_index('rank').sort_values('rank')
    return rankdf

def cvLinearReg(X_train, y_train):
    # create loocv procedure
    cvLR = LeaveOneOut()
    # create model
    modelLR = LinearRegression()
    # evaluate model
    scoresLR = cross_val_score(modelLR, X_train, y_train, scoring='neg_mean_absolute_error', cv=cvLR, n_jobs=-1)
    # force positive
    scoresLR = absolute(scoresLR)
    # report performance
    print('MAE: %.3f (%.3f)' % (mean(scoresLR), std(scoresLR)))
    meanMAE = mean(scoresLR)
    stddevMAE = std(scoresLR)
    return meanMAE

def cvLassoLars(X_train, y_train, x):
    # LassoLars
    # create loocv procedure
    cvLL = LeaveOneOut()
    # create model
    modelLL = LassoLars(alpha=x)
    # evaluate model
    scoresLL = cross_val_score(modelLL, X_train, y_train, scoring='neg_mean_absolute_error', cv=cvLL, n_jobs=-1)
    # force positive
    scoresLL = absolute(scoresLL)
    # report performance
    print('MAE: %.3f (%.3f)' % (mean(scoresLL), std(scoresLL)))
    meanMAE = mean(scoresLL)
    stddevMAE = std(scoresLL)
    return meanMAE

def cvTweedie(X_train, y_train, pwr, alf):
    # Tweedie Regressor
    # create loocv procedure
    cvTW = LeaveOneOut()
    # create model
    modelTW = TweedieRegressor(power=pwr, alpha=alf) # 0 = normal distribution
    # evaluate model
    scoresTW = cross_val_score(modelTW, X_train, y_train, scoring='neg_mean_absolute_error', cv=cvTW, n_jobs=-1)
    # force positive
    scoresTW = absolute(scoresTW)
    # report performance
    print('MAE: %.3f (%.3f)' % (mean(scoresTW), std(scoresTW)))
    meanMAE = mean(scoresTW)
    stddevMAE = std(scoresTW)
    return meanMAE

def cvRandomForest(X_train, y_train, x):
    # Random Forest Regressor
    # create loocv procedure
    cvRF = LeaveOneOut()
    # create model
    modelRF = RandomForestRegressor(n_estimators=x, random_state = 123)
    # evaluate model
    scoresRF = cross_val_score(modelRF, X_train, y_train, scoring='neg_mean_absolute_error', cv=cvRF, n_jobs=-1)
    # force positive
    scoresRF = absolute(scoresRF)
    # report performance
    print('MAE: %.3f (%.3f)' % (mean(scoresRF), std(scoresRF)))
    meanMAE = mean(scoresRF)
    stddevMAE = std(scoresRF)
    return meanMAE

def cvSVR(X_train, y_train, x):
    # Support Vector Regressor
    # create loocv procedure
    cvSVR = LeaveOneOut()
    # create model
    modelSVR = SVR(kernel = x)
    # evaluate model
    scoresSVR = cross_val_score(modelSVR, X_train, y_train, scoring='neg_mean_absolute_error', cv=cvSVR, n_jobs=-1)
    # force positive
    scoresSVR = absolute(scoresSVR)
    # report performance
    print('MAE: %.3f (%.3f)' % (mean(scoresSVR), std(scoresSVR)))
    meanMAE = mean(scoresSVR)
    stddevMAE = std(scoresSVR)
    return meanMAE





def get_baseline_mean(y_train):
    '''
    Using mean gets baseline for y dataframe
    '''
    # determine Baseline to beat
    rows_needed = y_train.shape[0]
    # create array of predictions of same size as y_train.logerror based on the mean
    y_hat = np.full(rows_needed, np.mean(y_train))
    # calculate the MSE for these predictions, this is our baseline to beat
    baseline = mean_absolute_error(y_train, y_hat)
    print("Baseline MAE:", baseline)
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
    baseline = mean_absolute_error(y_train, y_hat)
    print("Baseline MAE:", baseline)
    return baseline, y_hat

def linear_reg_train(x_scaleddf, target):
    '''
    runs linear regression algorithm
    '''
    lm = LinearRegression()
    lm.fit(x_scaleddf, target)
    y_hat = lm.predict(x_scaleddf)

    LM_MAE = mean_absolute_error(target, y_hat)
    return LM_MAE

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
    lars_MAE = mean_absolute_error(target, lars_pred)
    return lars_MAE

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
    pf2_MAE = mean_absolute_error(target, lm_squared_pred)
    return pf2_MAE

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
    tw_MAE = mean_absolute_error(y_train, tw_pred)
    return tw_MAE

def randomforest_test(x_scaleddf, target, X_test, y_test, est):
    '''
    runs random forest regressor
    '''
    # make model
    regressor = RandomForestRegressor(n_estimators = est, random_state = 123)
    # fit the model
    regressor.fit(x_scaleddf, target)
    # make predictions
    y_pred = regressor.predict(X_test)
    # calculate MAE
    randMAE = mean_absolute_error(y_test, y_pred)
    return randMAE, regressor

def lasso_lars_test(x_scaleddf, target, X_test, y_test):
    '''
    runs Lasso Lars algorithm
    ''' 
    # Make a model
    lars = LassoLars(alpha=1)
    # Fit a model
    lars.fit(x_scaleddf, target)
    # Make Predictions
    lars_pred = lars.predict(X_test)
    # calculate MAE
    lars_MAE = mean_absolute_error(y_test, lars_pred)
    return lars_MAE, lars

def linear_test(x_scaleddf, target, X_test, y_test):
    '''
    runs Lasso Lars algorithm
    ''' 
    # Make a model
    lm = LinearRegression()
    # Fit model on train dataset
    lm.fit(x_scaleddf, target)
    # Make Predictions on test dataset
    y_hat = lm.predict(X_test)
    # calculate MAE
    LM_MAE = mean_absolute_error(y_test, y_hat)
    return LM_MAE, lm


def SVR_test(x_scaleddf, target, X_test, y_test, kern):
    '''
    runs Support Vector Regressor algorithm
    ''' 
    # Make a model
    regressor = SVR(kernel = kern)
    # Fit model on train dataset
    regressor.fit(x_scaleddf, target)
    # Make Predictions on test dataset
    y_hat = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))
    # calculate MAE
    svr_MAE = mean_absolute_error(y_test, y_hat)
    return svr_MAE, regressor

def tweedie_test(X_train, y_train, X_test, y_test, pwr, alf):
    '''
    runs tweedie algorithm
    ''' 
    # Make Model
    tw = TweedieRegressor(power=pwr, alpha=alf) # 0 = normal distribution
    # Fit Model
    tw.fit(X_train, y_train)
    # Make Predictions
    tw_pred = tw.predict(X_test)
    # Compute root mean squared error
    tw_MAE = mean_absolute_error(y_test, tw_pred)
    return tw_MAE, tw

