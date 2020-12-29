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
    return lars_MAE, lars, lars_pred

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
    return LM_MAE, lm, y_hat


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
    return tw_MAE, tw, tw_pred


def create_visualdf(y_test, y_train, y_test_predLL, y_test_predLR, y_test_predTW, y_test_predrTW):
    '''
    creates dataframe for making visualizations
    '''
    visualdf = pd.DataFrame()
    visualdf['actual'] = y_test.tract_cases_per_100k
    visualdf['baseline'] = y_train.tract_cases_per_100k.mean()
    visualdf['TWpred'] = y_test_predTW
    visualdf['LRpred'] = y_test_predLR
    visualdf['LLpred'] = y_test_predLL
    visualdf['SVI_only'] = y_test_predrTW
    return visualdf


def plot_actual_vs_predicted(visualdf):
    '''
    Produces subplots of actual VS predicted for the top models
    '''
    plt.figure(figsize=(16,8))
    #plt.suptitle('Plotting Actual Cases per 100K vs Predicted Cases per 100K')
    plt.plot(visualdf.actual, visualdf.baseline, alpha=.5, color="gray", label='_nolegend_')
    #plt.annotate("Baseline: Predict Using Mean", (15, 8))
    plt.plot(visualdf.actual, visualdf.actual, alpha=.5, color="blue", label='_nolegend_')
    #plt.annotate("The Ideal Line: Predicted = Actual", (.5, 1), rotation=15.5)
    #plt.subplot(1,3,1,)
    plt.scatter(visualdf.actual, visualdf.SVI_only, 
               alpha=.5, color="blue", s=50, label="Model: TW SVI only")
    #plt.subplot(1,3,2)
    plt.scatter(visualdf.actual, visualdf.TWpred, 
                alpha=.5, color="green", s=100, label="Model: TW Top 4 Features")
    #plt.subplot(1,3,3)
    # plt.scatter(visualdf.actual, visualdf.LLpred, 
    #           alpha=.5, color="orange", s=75, label="Model: LassoLars")
    plt.legend()
    plt.xlabel("Actual Cases per 100K")
    plt.ylabel("Predicted Cases per 100K")
   
    # plt.annotate("The polynomial model appears to overreact to noise", (2.0, -10))
    # plt.annotate("The OLS model (LinearRegression)\n appears to be most consistent", (15.5, 3))
    plt.show()

def plotting_errors(visualdf):
    '''
    Plots the errors of the top models with zero error line
    '''
    # plotting Errors in Predictions
    plt.figure(figsize=(16,8))
    plt.axhline(label="No Error")
    plt.scatter(visualdf.actual, visualdf.SVI_only-visualdf.actual, 
                alpha=.5, color="blue", s=50, label="Model: TW SVI only")
    plt.scatter(visualdf.actual, visualdf.TWpred-visualdf.actual, 
                alpha=.5, color="green", s=100, label="Model: TW Top 4 Features")
    # plt.scatter(visualdf.actual, visualdf.LLpred-visualdf.actual, 
    #             alpha=.5, color="orange", s=75, label="Model: LassoLars")
    plt.legend()
    plt.xlabel("Actual Cases per 100K")
    plt.ylabel("Residual/Error: Predicted Cases per 100K - Actual Cases per 100K")
    plt.title("Do the size of errors change as the actual value changes?")
    # plt.annotate("The polynomial model appears to overreact to noise", (2.0, -10))
    # plt.annotate("The OLS model (LinearRegression)\n appears to be most consistent", (15.5, 3))
    plt.show()


def plotting_histograms(visualdf):
    '''
    Plots Histograms of top models, currently only shows actual vs best performing
    '''
    plt.figure(figsize=(16,8))
    plt.hist(visualdf.actual, color='blue', alpha=.5, label="Actual Cases per 100K", histtype=u'step', linewidth=5)
    plt.hist(visualdf.LRpred, color='orange', alpha=.5, label="Model: TW SVI only")
    plt.hist(visualdf.TWpred, color='green', alpha=.5, label="Model: TW Top 4 Features")
    #plt.hist(visualdf.LLpred, color='red', alpha=.5, label="Model: LassoLars")
    plt.xlabel("Actual Cases per 100K")
    plt.ylabel("predictions")
    plt.title("SVI alone is not enough")
    plt.legend()
    plt.show()