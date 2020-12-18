#### Environment Imports #####
import pandas as pd 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#### Functions #####

def get_county_data():
    '''
    This function  gathers input from the user for state and county and date
    reads the SVI2018_US.csv and returns dataframe of selected counties census tracks.
    It also reads the county data by date and gets the COVID cases for that county by date.
    Also gets uszips.csv info for all zip codes and associated population for the requested county and state
    Gets county population from SVI2018_US_COUNTY.csv for requested state and county
    '''
    # get user input for state abbreviation and county name and date
    date_req = input('Enter the requested date in YYYY-MM-DD format: ')
    state_req = input('Enter the requested state full name: ')
    county_req = input('Enter the requested full name of county: ') 
    # state must be in uppercase, county must be in titlecase
    state_req = state_req.upper()
    county_req = county_req.title()
    # should probably add a date format check here
    
    # get svi .csv
    svidf = pd.read_csv('SVI2018_us.csv')
    # based on CDC info should drop tracts with 0 population
    svidf = svidf[svidf.E_TOTPOP != 0]
    # filter for selected state and county
    svidf = svidf[svidf.STATE == state_req]
    svidf = svidf[svidf.COUNTY == county_req]
    print(svidf.shape)
    
    # read the all counties COVID data .csv
    casesdf = pd.read_csv('COVID20201208_county', index_col = 0)
    # need state in title case for this dataset
    state_req2 = state_req.title()
    # filter for only the selected date, state, and county
    casesdf = casesdf[casesdf['date'] == date_req]
    casesdf = casesdf[casesdf.state == state_req2]
    casesdf = casesdf[casesdf.county == county_req]
    print(casesdf.shape)
    
    # read in full uszips.csv
    zipsdf = pd.read_csv('uszips.csv')
    # filter for selected state and county
    zipsdf = zipsdf[zipsdf.state_name == state_req2]
    zipsdf = zipsdf[zipsdf.county_name == county_req]
    # get just the county_name, state_name, zip code, and population
    zipsdf = zipsdf[['state_name', 'county_name', 'zip', 'population']]
    print(zipsdf.shape)

    # get county population for requested state and county
    county_pop = pd.read_csv('SVI2018_US_COUNTY.csv')
    # filter for requested state and county
    cpopdf = county_pop[county_pop.STATE == state_req]
    cpopdf = cpopdf[cpopdf.COUNTY == county_req]
    # get only county, state, and population
    cpopdf = cpopdf[['STATE', 'COUNTY', 'E_TOTPOP']]
    print(cpopdf.shape)
    return svidf, casesdf, zipsdf, cpopdf


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
    zips = pd.read_csv('TRACT_ZIP_032019.csv')
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


def get_countylevelonly_data():
    '''
    This function  gathers input from the user for state and date
    reads the SVI2018_US_COUNTY.csv and returns dataframe of selected state.
    It also reads the county data by date and gets the COVID cases for that county by date.
    '''
    # get user input for state abbreviation and county name and date
    date_req = input('Enter the requested date in YYYY-MM-DD format: ')
    state_req = input('Enter the requested state full name: ')
    # state must be in uppercase, county must be in titlecase
    state_req = state_req.upper()
    # should probably add a date format check here
    
    # get svi .csv
    svidf = pd.read_csv('SVI2018_US_COUNTY.csv')
    svidf.columns = svidf.columns.str.lower()
    # based on CDC info should drop counties with 0 population
    # not needed no null population counties
    # filter for selected state and county
    svidf = svidf[svidf.state == state_req]
    print(svidf.shape)
    
    # read the all counties COVID data .csv
    casesdf = pd.read_csv('COVID20201208_county', index_col = 0)
    # need state in title case for this dataset
    state_req2 = state_req.title()
    # filter for only the selected date, state, and county
    casesdf = casesdf[casesdf['date'] == date_req]
    casesdf = casesdf[casesdf.state == state_req2]
    print(casesdf.shape)
    # merge dataframes on county
    cmerged = pd.merge(svidf, casesdf, on='county')
    print(cmerged.shape)
    return cmerged