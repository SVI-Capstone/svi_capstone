import pandas as pd
import numpy as np
import scipy as sp 
import os

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

import sklearn
import acquire
import prepare

#################### Wrangle ##################

####### Split dataframe ########
def split(df, target_var):
    '''
    This splits the dataframe for train, validate, and test, and creates X and y dataframes for each
    '''
    # split df into train (80%) and test (20%)
    # cross validation with be used instead of a validate dataset
    train, test = train_test_split(df, test_size=.20, random_state = 123, stratify=df.rank_svi)
    
    # for explore create copy of train without x/y split
    train_exp = train.copy()

    # create X_train by dropping the target variable 
    X_train = train.drop(columns=[target_var])
    # create y_train by keeping only the target variable.
    y_train = train[[target_var]]

    # create X_test by dropping the target variable 
    X_test = test.drop(columns=[target_var])
    # create y_test by keeping only the target variable.
    y_test = test[[target_var]]
    
    return train_exp, X_train, y_train, X_test, y_test


######## Scale #########

def add_scaled_columns(X_train, X_test, scaler, columns_to_scale):
    """This function takes the inputs from scale_zillow and scales the data"""
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    scaler.fit(X_train[columns_to_scale])

    X_train_scaled = pd.concat([
        X_train,
        pd.DataFrame(scaler.transform(X_train[columns_to_scale]), columns=new_column_names, index=X_train.index),
    ], axis=1)
    X_test_scaled = pd.concat([
        X_test,
        pd.DataFrame(scaler.transform(X_test[columns_to_scale]), columns=new_column_names, index=X_test.index),
    ], axis=1)
    
    return X_train_scaled, X_test_scaled

def scale_data(X_train, X_test):
    """This function provides the inputs and runs the add_scaled_columns function"""
    X_train_scaled, X_test_scaled = add_scaled_columns(
    X_train,
    X_test,
    scaler = sklearn.preprocessing.MinMaxScaler(),
    columns_to_scale=['f_soci_total', 'f_comp_total', 'f_status_total', 'f_trans_total', 'all_flags_total', 'rank_svi'])
    return X_train_scaled, X_test_scaled


def wrangle_data():
    '''This function makes all necessary changes to the dataframe for exploration and modeling'''
    # acquire data
    df = acquire.run()
    # prepare data
    df = prepare.run(df)

    # split dataset
    target_var = 'tract_cases_per_100k'
    train_exp, X_train, y_train, X_test, y_test = split(df, target_var)
    print(X_train.shape, X_test.shape)

    # drop rows not needed for modeling
    X_train = X_train.drop(columns=['tract','zip','bin_svi'])
    X_test = X_test.drop(columns=['tract','zip','bin_svi'])
    
    # df is now ready to scale
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # drop rows now scaled from scaled dataframes
    X_train_scaled = X_train_scaled.drop(columns=['f_soci_total', 'f_comp_total', 'f_status_total', 'f_trans_total', 'all_flags_total', 'rank_svi'])
    X_test_scaled = X_test_scaled.drop(columns=['f_soci_total', 'f_comp_total', 'f_status_total', 'f_trans_total', 'all_flags_total', 'rank_svi'])
    
    return df, train_exp, X_train_scaled, y_train, X_test_scaled, y_test