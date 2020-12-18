#### Environment Imports #####
import pandas as pd 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#### Functions #####

def get_san_antonio_data():
    '''
    This function reads the Bexar case data, normalizes headers, and returns dataframe
    '''
    bexardf = pd.read_csv('data_csv_files/Bexar_County_COVID-19_Data_by_Zip_Code.csv', index_col=0)
    bexardf.columns = bexardf.columns.str.lower()
    bexardf = bexardf.rename(columns={'zip_code': 'zip', 'populationtotals_totpop_cy': 'population'})
    return bexardf

def get_sa_svi_data():
    '''
    This function reads the svi data, normalizes headers, and returns dataframe
    '''
    svidf = pd.read_csv('data_csv_files/san_antonio_2018_tract.csv')
    svidf.columns = svidf.columns.str.lower()
    svidf = svidf.rename(columns={'fips': 'tract'})
    return svidf

def get_dallas_data():
    '''
    This function reads the Dallas case data, normalizes headers, and returns dataframe
    '''
    dallasdf = pd.read_csv('data_csv_files/dallas_zip_covid_8_8_2020.csv')
    dallasdf.columns = dallasdf.columns.str.lower()
    dallasdf = dallasdf.rename(columns={'zipcode': 'zip'})
    return dallasdf

def get_dallas_svi_data():
    '''
    This function reads the Dallas svi data, normalizes headers, and returns dataframe
    '''
    svidf = pd.read_csv('data_csv_files/dallas_2018_tract.csv')
    svidf.columns = svidf.columns.str.lower()
    svidf = svidf.rename(columns={'fips': 'tract'})
    return svidf

def get_HUD(citydf):
    '''
    This function gets the HUD Track to Zip crosswalk data, filters for the city zip codes,
    sorts and orders by % of addresses in Zip code. Then return a dataframe with 1 zip code per tract for the city.
    '''
    #create a list of zip codes for the city
    city_zip_list = citydf.zip.tolist()
    
    # # get list of all tracts from svi
    # tract_list = svidf.tract.tolist()

    # # import track to zip dataframe
    # tracts = pd.read_csv('TRACT_ZIP_032019.csv')
    # import track to zip dataframe
    zips = pd.read_csv('data_csv_files/TRACT_ZIP_032019.csv')
    #filter the zips df to only those in the city zip list
    zips = zips[zips.zip.isin(city_zip_list)]
    
    # # filter for tracts in list
    # tracts = tracts[tracts.tract.isin(tract_list)]
    # aggregate the data frame to get the zip code with the max ratio by tract
    zipsdf = zips.groupby(['tract'])['tot_ratio', 'zip'].agg({'tot_ratio':['max'], 'zip':['first']})
    zipsdf.columns = [' '.join(col).strip() for col in zipsdf.columns.values]
    zipsdf = zipsdf.reset_index()
    # rename columns
    zipsdf = zipsdf.rename(columns={'zip first':'zip', 'tot_ratio max':'address_ratio'})
    return zipsdf


def compile_san_antonio_data():
    '''
    This function gets the data from the 3 .csv files and compiles them together
    '''
    # get SVI data
    svidf = get_sa_svi_data()
    # get the Bexar data
    bexar = get_san_antonio_data()
    # create the merge dataframe
    merge_bexar = bexar[['zip', 'population', 'positive', 'casesp100000']]
    # get the HUD data
    merge_zip = get_HUD(bexar)  
    # create new df merging svi and 2nd merge zip file on tract
    svi_zip = pd.merge(svidf, merge_zip, on='tract', how='left')
    svi_zip_cases = pd.merge(svi_zip, merge_bexar, on='zip', how='left')
    return svi_zip_cases


def compile_dallas_data():
    '''
    This function gets the data from the 3 .csv files and compiles them together
    '''
    # get SVI data
    svidf = get_dallas_svi_data()
    # get the Dallas data
    dallas = get_dallas_data()
    # create the merge dataframe
    merge_dallas = dallas[['zip', 'population', 'cases_per_100k']]
    # get the HUD data
    merge_zip = get_HUD(dallas)  
    # create new df merging svi and 2nd merge zip file on tract
    svi_zip = pd.merge(svidf, merge_zip, on='tract', how='left')
    svi_zip_cases = pd.merge(svi_zip, merge_dallas, on='zip', how='left')
    # drop nulls - these are tracts not in the Dallas cases file
    svi_zip_cases.dropna(inplace=True)
    # rename cases per 100k for prepare function
    svi_zip_cases = svi_zip_cases.rename(columns={'cases_per_100k': 'casesp100000'})
    return svi_zip_cases

def compile_sa_data():
    '''
    This function gets the data from the 3 .csv files and compiles them together
    '''
    # get SVI data
    svidf = get_san_antonio_data()
    # get the sa data
    bexar = get_sa_svi_data()
    # create the merge dataframe
    merge_sa = bexar[['zip', 'population', 'positive', 'casesp100000']]
    # get the HUD data
    merge_zip = get_HUD(svidf)  
    # create new df merging svi and 2nd merge zip file on tract
    svi_zip = pd.merge(svidf, merge_zip, on='tract', how='left')
    svi_zip_cases = pd.merge(svi_zip, merge_sa, on='zip', how='left')
    return svi_zip_cases

def run():
    print("Acquire: compiling raw data files...")
    df = compile_dallas_data()
    print("Acquire: Completed!")
    return df