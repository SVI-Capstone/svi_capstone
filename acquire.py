#### Environment Imports #####
import pandas as pd 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#### Functions #####

def get_san_antonio_data():
    '''
    This function reads the Bexar case data, normalizes headers, and returns dataframe
    '''
    bexardf = pd.read_csv('Bexar_County_COVID-19_Data_by_Zip_Code.csv', index_col=0)
    bexardf.columns = bexardf.columns.str.lower()
    bexardf = bexardf.rename(columns={'zip_code': 'zip', 'populationtotals_totpop_cy': 'population'})
    return bexardf

def get_svi_data():
    '''
    This function reads the svi data, normalizes headers, and returns dataframe
    '''
    svidf = pd.read_csv('san_antonio_2018_tract.csv')
    svidf.columns = svidf.columns.str.lower()
    svidf = svidf.rename(columns={'fips': 'tract'})
    return svidf

def get_HUD(citydf):
    '''
    This function gets the HUD Track to Zip crosswalk data, filters for the city zip codes,
    sorts and orders by % of addresses in Zip code. Then return a dataframe with 1 zip code per tract for the city.
    '''
    # create a list of zip codes for the city
    city_zip_list = citydf.zip.tolist()
    # import track to zip dataframe
    zips = pd.read_csv('TRACT_ZIP_122018_78s_only.csv')
    # filter the zips df to only those in the city zip list
    zips = zips[zips.zip.isin(city_zip_list)]
    # aggregate the data frame to get the zip code with the max ratio by tract
    zipsdf = zips.groupby(['tract'])['tot_ratio', 'zip'].agg({'tot_ratio':['max'], 'zip':['first']})
    zipsdf.columns = [' '.join(col).strip() for col in zipsdf.columns.values]
    zipsdf = zipsdf.reset_index()
    # create dataframe of only items to merge with svi data
    merge_zip = zipsdf[['tract', 'zip first']]
    merge_zip = merge_zip.rename(columns={'zip first':'zip'})
    return merge_zip


def compile_san_antonio_data():
    '''
    This function gets the data from the 3 .csv files and compiles them together
    '''
    # get SVI data
    svidf = get_svi_data()
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


def run():
    print("Acquire: compiling raw data files...")
    df = compile_san_antonio_data()
    print("Acquire: Completed!")
    return df