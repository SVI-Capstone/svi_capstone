import pandas as pd
import numpy as np
import scipy as sp 
import os

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

import sklearn
from scripts_python import acquire
from scripts_python import prepare
from scripts_python import acquire_dallas
from scripts_python import acquire_all_counties
from scripts_python import prepare_counties

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
    columns_to_scale=['f_soci_total', 'f_comp_total', 'f_status_total', 'f_trans_total', 'all_flags_total', 'rank_svi', 'spl_theme1', 'ep_pov', 'e_pov'])
    return X_train_scaled, X_test_scaled

def scale_data_centroid(X_train, X_test):
    """This function provides the inputs and runs the add_scaled_columns function"""
    X_train_scaled, X_test_scaled = add_scaled_columns(
    X_train,
    X_test,
    scaler = sklearn.preprocessing.MinMaxScaler(),
    columns_to_scale=['centroid_spl_theme1', 'centroid_ep_pov', 'centroid_e_pov'])
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

def wrangle_dallas_data():
    '''This function makes all necessary changes to the dataframe for exploration and modeling'''
    # acquire data
    df = acquire_dallas.run()
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
    X_train_scaled = X_train_scaled.drop(columns=['f_soci_total', 'f_comp_total', 'f_status_total', 'f_trans_total', 'all_flags_total', 'rank_svi', 'spl_theme1', 'ep_pov', 'e_pov'])
    X_test_scaled = X_test_scaled.drop(columns=['f_soci_total', 'f_comp_total', 'f_status_total', 'f_trans_total', 'all_flags_total', 'rank_svi', 'spl_theme1', 'ep_pov', 'e_pov'])
    
    return df, train_exp, X_train_scaled, y_train, X_test_scaled, y_test

def wrangle_countylevelonly_data():
    '''This function makes all necessary changes to the dataframe for exploration and modeling'''
    # acquire data
    df = acquire_all_counties.get_countylevelonly_data()
    # prepare data
    df = prepare_counties.prepare_countylevelonly_data(df)

    # split dataset
    target_var = 'tract_cases_per_100k'
    train_exp, X_train, y_train, X_test, y_test = split(df, target_var)
    print(X_train.shape, X_test.shape)

    # drop rows not needed for modeling
    X_train = X_train.drop(columns=['county','e_totpop', 'cases', 'state_pop', 'pop_percentage', 'calculated_cases', 'bin_svi'])
    X_test = X_test.drop(columns=['county','e_totpop', 'cases', 'state_pop', 'pop_percentage', 'calculated_cases', 'bin_svi'])
    
    # df is now ready to scale
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # drop rows now scaled from scaled dataframes
    X_train_scaled = X_train_scaled.drop(columns=['f_soci_total', 'f_comp_total', 'f_status_total', 'f_trans_total', 'all_flags_total', 'rank_svi'])
    X_test_scaled = X_test_scaled.drop(columns=['f_soci_total', 'f_comp_total', 'f_status_total', 'f_trans_total', 'all_flags_total', 'rank_svi'])
    
    return df, train_exp, X_train_scaled, y_train, X_test_scaled, y_test
    
# Wrangle for data for classification models.

def wrangle_data_class():
    '''This function makes all necessary changes to the dataframe for exploration and modeling'''
    # acquire data
    df = acquire.run()
    # prepare data
    df = prepare.run(df)
    
    df['bin_cases'] = pd.cut(df.tract_cases_per_100k, bins = [780, 2280, 2860, 3850, 8900], labels = ['low', 'low_mod', 'mod_high', 'high'])
    df['rank_cases'] = pd.cut(df.tract_cases_per_100k, bins = [780, 2280, 2860, 3850, 8900], labels = [4, 3, 2, 1])
    
    # split dataset
    target_var = 'rank_cases'
    train_exp, X_train, y_train, X_test, y_test = split(df, target_var)
    print(X_train.shape, X_test.shape)

   # drop rows not needed for modeling
    X_train = X_train.drop(columns=['tract','zip','bin_svi', 'bin_cases'])
    X_test = X_test.drop(columns=['tract','zip','bin_svi', 'bin_cases'])
    
    # df is now ready to scale
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # drop rows now scaled from scaled dataframes
    X_train_scaled = X_train_scaled.drop(columns=['f_soci_total', 'f_comp_total', 'f_status_total', 'f_trans_total', 'all_flags_total', 'rank_svi', 'spl_theme1', 'ep_pov', 'e_pov'])
    X_test_scaled = X_test_scaled.drop(columns=['f_soci_total', 'f_comp_total', 'f_status_total', 'f_trans_total', 'all_flags_total', 'rank_svi', 'spl_theme1', 'ep_pov', 'e_pov'])
    
    return df, train_exp, X_train_scaled, y_train, X_test_scaled, y_test

print("wrangle_class.py functions successfully loaded")
