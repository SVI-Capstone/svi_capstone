# Importing libraries

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import os

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
#import acquire.py
#import prepare.py

import sklearn
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

############################################################# Prepare Function #############################################################

def prepare_data(df):
    """
    This function identfies colums in the acqure df we wish to keep, renames those columns and drops 4 null values
    """

    # Creating a new column which assigns the % of cases by zip code to the correct % of addresses each census tract "owns"
    df['tract_cases_per_100k'] = df.casesp100000 * df.address_ratio



    # Columns we want to keep (can be changed as needed):
    columns_to_keep = [
                    # 'st_abbr',
                    # 'county',
                    'tract',
                    # 'area_sqmi',
                    'rpl_themes',
                    'f_pov',
                    'f_unemp',
                    'f_pci',
                    'f_nohsdp',
                    'f_theme1',
                    'f_age65',
                    'f_age17',
                    'f_disabl',
                    'f_sngpnt',
                    'f_theme2',
                    'f_minrty',
                    'f_limeng',
                    'f_theme3',
                    'f_munit',
                    'f_mobile',
                    'f_crowd',
                    'f_noveh',
                    'f_groupq',
                    'f_theme4',
                    'f_total',
                    'zip',
                    # 'population',
                    # 'positive',
                    'tract_cases_per_100k']
    # Using list comprehension to create a dataframe. Because there are more columns we want to remove than we want to keep, I simply iterated thru the list made above and in essence dropped all columns we didn't want to keep. Easier than using pd.drop.
    df = df[[c for c in df.columns if c in columns_to_keep]]
    
    # Renaming the columns:
    df.rename(columns = {'rpl_themes': 'raw_svi', 
                     "f_theme1": "f_soci_total", 
                     "f_theme2": "f_comp_total", 
                     "f_theme3": "f_status_total", 
                     "f_theme4": "f_trans_total", 
                     "f_total": "all_flags_total",
                     "f_pov": "f_pov_soci",
                     "f_unemp": "f_unemp_soci", 
                     "f_pci": "f_pci_soci", 
                     "f_nohsdp": "f_nohsdp_soci", 
                     "f_age65": "f_age65_comp", 
                     "f_age17": "f_age17_comp", 
                     "f_disabl": "f_disabl_comp", 
                     "f_sngpnt": "f_sngpnt_comp", 
                     "f_minrty": "f_minrty_status", 
                     "f_limeng": "f_limeng_status", 
                     "f_munit": "f_munit_trans", 
                     "f_mobile": "f_mobile_trans", 
                     "f_crowd": "f_crowd_trans", 
                     "f_noveh": "f_noveh_trans",
                     "casesp100000": "case_p_hunth", 
                     "f_groupq": "f_groupq_trans"}, inplace = True)
    
    # Dropping rows:
    # There are 4 rows which are military bases according to the tract information. These rows all return -999 for all the flags and svi score, thus they are not useful for our analysis. Since 4 rows only 1% of our total rows, we opted to simply drop those 4 rows.

    df = df[df.raw_svi > 0]

    # Agg all columns by zipcode
    # groupdf = df.groupby(['zip'])['raw_svi', 'f_soci_total',  'f_comp_total', 'f_status_total', 'f_trans_total', 'all_flags_total', 'case_p_hunth'].\
    #                     agg({'raw_svi': ['min', 'max', 'mean'], 'f_soci_total' : ['sum'], 'f_comp_total': ['sum'], \
    #                          'f_status_total': ['sum'], 'f_trans_total': ['sum'], 'all_flags_total': ['sum'], 'case_p_hunth': ['first']})


    # Unstacking the columns index
    # groupdf.columns = [' '.join(col).strip() for col in groupdf.columns.values]
    
    # Replacing the spaces in the column names with "_"
    # groupdf.columns = groupdf.columns.str.replace(" ", "_")
    
    # Categorizing (or binning) the raw_svi_mean scores
    df['bin_svi'] = pd.cut(df.raw_svi, bins = [0, .27, .5, .75, 1], labels = ['Low', 'Low Moderate', 'Moderate High', 'High'])
    df['rank_svi'] = pd.cut(df.raw_svi, bins = [0, .27, .5, .75, 1], labels = [4, 3, 2, 1])
    
    return df
    
# To use:
# df = pd.read_csv('full_san_antonio.csv', index_col = 0)
# df = prepare_data(df)

def run(df):
    print("Prepare: preparing data files...")
    df = prepare_data(df)
    print("Prepare: Completed!")
    return df

############################################################# Split Function #############################################################

def split(df, target_var):
    '''
    This splits the dataframe for train, validate, and test, and creates X and y dataframes for each
    '''
    # split df into train_validate (80%) and test (20%)
    train_validate, test = train_test_split(df, test_size=.20, random_state = 123, stratify=df.bin_svi)
    # split train_validate into train(70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.25, random_state = 123, stratify=train_validate.bin_svi)
    
    # create X_train by dropping the target variable 
    X_train = train.drop(columns=[target_var])
    # create y_train by keeping only the target variable.
    y_train = train[[target_var]]

    # create X_validate by dropping the target variable 
    X_validate = validate.drop(columns=[target_var])
    # create y_validate by keeping only the target variable.
    y_validate = validate[[target_var]]

    # create X_test by dropping the target variable 
    X_test = test.drop(columns=[target_var])
    # create y_test by keeping only the target variable.
    y_test = test[[target_var]]
    
    # for explore create copy of train without x/y split
    train_exp = train.copy()
    return train_exp, X_train, y_train, X_validate, y_validate, X_test, y_test

############################################################# Scale Function #############################################################

def scale_data(X_train, X_validate, X_test):
    """
    This function is used to scale the numeric data using a MinMaxScaler
    """
    scaler = MinMaxScaler(copy=True).fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, 
                                columns=X_train.columns.values).\
                                set_index([X_train.index.values])

    X_validate_scaled = pd.DataFrame(X_validate_scaled, 
                                    columns=X_validate.columns.values).\
                                set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled, 
                                    columns=X_test.columns.values).\
                                set_index([X_test.index.values])

    return X_train_scaled, X_validate_scaled, X_test_scaled

def prepare_countylevelonly_data(df):
    """
    This function identfies colums in the acquire df we wish to keep, renames those columns,
    and calculates cases per 100k based on population ratio
    """
    # Creating a new column of the total state population = sum of e_totpop
    df['state_pop'] = sum(df.e_totpop)
    # create new column that is the county percentage of the state population
    df['pop_percentage'] = df.e_totpop/df.state_pop
    # create column of cases based on percentage of population
    df['calculated_cases'] = df.pop_percentage * sum(df.cases)
    # create cases per 100k column
    df['tract_cases_per_100k'] = df.cases/df.e_totpop*100000
    # Columns we want to keep (can be changed as needed):
    columns_to_keep = [
                    # 'st_abbr',
                    'county',
                    'e_totpop',
                    'state_pop',
                    'pop_percentage',
                    'calculated_cases',
                    'cases',
                    'tract_cases_per_100k',
                    # 'tract',
                    # 'area_sqmi',
                    'rpl_themes',
                    'f_pov',
                    'f_unemp',
                    'f_pci',
                    'f_nohsdp',
                    'f_theme1',
                    'f_age65',
                    'f_age17',
                    'f_disabl',
                    'f_sngpnt',
                    'f_theme2',
                    'f_minrty',
                    'f_limeng',
                    'f_theme3',
                    'f_munit',
                    'f_mobile',
                    'f_crowd',
                    'f_noveh',
                    'f_groupq',
                    'f_theme4',
                    'f_total',
                    # 'zip',
                    # 'population',
                    # 'positive'
                    ]
    # Using list comprehension to create a dataframe. Because there are more columns we want to remove than we want to keep, I simply iterated thru the list made above and in essence dropped all columns we didn't want to keep. Easier than using pd.drop.
    df = df[[c for c in df.columns if c in columns_to_keep]]
    
    # Renaming the columns:
    df.rename(columns = {'rpl_themes': 'raw_svi', 
                     "f_theme1": "f_soci_total", 
                     "f_theme2": "f_comp_total", 
                     "f_theme3": "f_status_total", 
                     "f_theme4": "f_trans_total", 
                     "f_total": "all_flags_total",
                     "f_pov": "f_pov_soci",
                     "f_unemp": "f_unemp_soci", 
                     "f_pci": "f_pci_soci", 
                     "f_nohsdp": "f_nohsdp_soci", 
                     "f_age65": "f_age65_comp", 
                     "f_age17": "f_age17_comp", 
                     "f_disabl": "f_disabl_comp", 
                     "f_sngpnt": "f_sngpnt_comp", 
                     "f_minrty": "f_minrty_status", 
                     "f_limeng": "f_limeng_status", 
                     "f_munit": "f_munit_trans", 
                     "f_mobile": "f_mobile_trans", 
                     "f_crowd": "f_crowd_trans", 
                     "f_noveh": "f_noveh_trans", 
                     "f_groupq": "f_groupq_trans"}, inplace = True)

    df['bin_svi'] = pd.cut(df.raw_svi, bins = [0, .27, .5, .75, 1], labels = ['Low', 'Low Moderate', 'Moderate High', 'High'])
    df['rank_svi'] = pd.cut(df.raw_svi, bins = [0, .27, .5, .75, 1], labels = [4, 3, 2, 1])
    
    return df
    
# To use:
# df = pd.read_csv('full_san_antonio.csv', index_col = 0)
# df = prepare_data(df)

def run(df):
    print("Prepare: preparing data files...")
    df = prepare_data(df)
    print("Prepare: Completed!")
    return df

