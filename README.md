# Evaluating the CDC's Social Vulnerability Index (SVI) as a tool to predict COVID infections in San Antonio Texas

## About the Project
The CDC's social vulnerability index (SVI) is a scale that predicts the vulnerability of a population in the event of an emergency or natural disaster. COVID is the first global pandemic since the development of this measure. We will evaluate the association between SVI score and COVID case count in San Antonio, Texas. Features from this measure will be incorporated into a predictive model that can be used to guide recovery resource prioritization. 

### Goals

*Goal # 1* - To evaluate the association between SVI score and COVID case count in San Antonio, TX  

*Goal # 2* - To build a model based SVI score component features that can predict COVID cases by census tract within San Antonio, TX    

### Background
The SVI (Social Vulnerability Index) was developed to help city governments and first responders predict areas that are particularly vulnerable in emergency situations so that resources can be prioritized to help areas at high risk (Citation CDC Website). The CDC’s Social Vulnerability Index (CDC SVI) uses 15 U.S. census variables to classify census tracts with a composite score between 0 and 1 (lower scores = less vulnerability, higher score = greater vulnerability. This socre is calculated by first ranking every census tract, in every country, in every state, in the United States. Those ranked tracks are then broken up to 4 themes (socioeconomic status, household composition and disability, minority status and language, household type and transportation) and reclassified.  This overall score is then tallied by summing the themed percentiles and ranked on a score between 0 and 1.  

While SVI was designed to help city goverments repsond to emergency situations, the efficacy of the systems has never been tested on in response to a global pandemic. COVID-19 is the disease caused by a new coronavirus called SARS-CoV-2. WHO first learned of this new virus on 31 December 2019, following a report of a cluster of cases of ‘viral pneumonia’ in Wuhan, People’s Republic of China. (Citation WHO). As of 9 December 2020, more than 68.4 million cases have been confirmed, with more than 1.56 million deaths attributed to COVID-19. 

### Deliverables
1. Model to predict COVID 19 symptomatic infection by census tract in Bexar county
2. Clean and reproducable notebook documenting worflow and findings
3. 5-10 min presentation

### Acknowledgments
Thank you to the Codeup faculty and staff that have helped us every step of the way.  Also thank you to our friends and family who have supported us on this 22 week journey.  

## Data Dictionary
  ---                                ---
| **Terms**                        | **Definition**        |
| ---                              | ---                   |
| Social Vulnerability Index (SVI) | A composite of 15 U.S. census variables to help local officials identify communities that may need support before, during, or after disasters. |
| Census Tract | Small, relatively permanent statistical subdivisions of a county or equivalent entity that are updated by local participants prior to each decennial census as part of the Census Bureau's Participant Statistical Areas Program |
| Federal Information Processing Standards (FIPS) | A set of standards that describe document processing, encryption algorithms and other information technology standards for use within non-military government agencies and by government contractors and vendors who work with the agencies. |
| COVID - 19 | Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus |
| Regression Model | A predictive modelling technique investigating the relationship between a dependent (*target*) and independent variable(s) (*predictor*)                      
| Target Variable | Dependent variable: The feature the model predicts ***(COVID Infection Count)*** |
| Predictive Variable | Independent variable: The features used to create the prediction |
| f_pov_soci | Flag - the percentage of person in povery is in the 90th percentile nationally (1= yes, 0 = no)                                                   | f_unemp_soci | Flag - the percentage of civilian unemployed is in the 90th percentile nationally (1= yes, 0 = no) |
| f_pci_soci | Flag - per capita income is in the 90th percentile nationally (1= yes, 0 = no)|
| f_nohsdp_soci | Flag - the percentage of persons with no high school diploma is in the 90th percentile nationally (1= yes, 0 = no)|
| f_soci_total | Sum of flages for Socioeconomic Status theme|
| f_age65_comp | Flag - the percentage of persons aged 65 and older is in the 90th percentile nationally (1= yes, 0 = no)|
| f_age17_comp | Flag - the percenage of persons aged 17 and younger is in the 90th percentile nationally (1= yes, 0 = no)|
| f_disabl_comp | Flag - the percentage of persons with a disability is in the 90th percentile nationally (1= yes, 0 = no)|
| f_sngpnt_comp | Flag - the percentage of single parent households is in the 90th percentile nationally (1= yes, 0 = no)|
| f_comp_total | Sum of flags for Household Compensation theme |
| f_minrty_status | Flag - the percentage of minority is in the 90th percentile nationally (1= yes, 0 = no)|
| f_limeng_status  | Flag - the perentage those with limited English is in the 90th percentile nationally (1= yes, 0 = no)|
| f_status_total | Sum of flags for Minority Status/Language theme |
| f_munit_trans | Flag = the percentage of households in mulit-unit housing in the 90th percentile nationally (1= yes, 0 = no)|
| f_mobile_trans | Flag - the percentage of mobile homes is in the 90th percentile nationally (1= yes, 0 = no)|
| f_crowd_trans | Flag - the percentage of crowded households is in the 90th percentile nationally (1= yes, 0 = no)|
| f_noveh_trans | Flag - the percentage of households with no vehicles is in the 90th percentile nationally (1= yes, 0 = no)|
| f_groupq_trans | Flag - the percentage of persons in institutionalized group quarters is in the 90th percentile nationally (1= yes, 0 = no)|
| f_trans_total | Sum of flags for Housing Type/Transportation theme |
| all_flags_total| Sum of flags for the four themes |
| tract_cases_per_100k | Derrived density of cases per Cencus Tract |
| bin_svi | raw_svi percentages broken up in to cateogries based on CDC precident *low* < 0.27                 |
|         |                                                                       *low_med* > 0.27 and < 0.50  |
|         |                                                                       *med_high* > 0.50 and < 0.75 |
|         |                                                                       *high* < 0.75                |
|
  ---                     ---  


## Initial Thoughts & Hypotheses

1. Does the SVI score correlate to the number of COVID cases (or cases/10K) observed per cencus tract?
   -San Anotnio
   -Dallas

2. Is SVI better at predicting COVID cases in cencus tracts with overall high/med/low SVI scores?

3. Are the individual components of SVI better at predicting COVID cases then the aggregate score?

4. Is this pattern different from other cities in TX (Comporable size and SVI demigraphics) ?

## Project Steps
### Acquire

We acquired SVI data from the CDC's website amd downloaded COVID data for San Antonio and Dallas from the cities respective COVID data web portals. In order to merge the data we developed programatic solutions to translate federal FIPS codes in to discernable local Zip codes.  The HUD crosswalk provided a guide to transform the data, however, HUD info is complicated as there are many census tracts that may be in one zip code, or they may overlap into multiple other Zip code areas. In order to progratically solve this probelm we found the Zip code that accounted for the highest percentage of addresses within the tract and assigned that as the sole Zip code for the tract. This allowed us to merge the tables by matching to tract then Zip code linking all of the data together in a signle dataframe for prepare. The ratio of addresses for the census tract was then used to calculate a cases per 100K measure for each tract.


### Wrangle

*Prepare:*
In order to prepare the df for exploration 29 features were selected and remaned for clarity. Four null rows associated with miliary bases were removed from the data frame according to the tract information. Finally we categorized (or binning) the raw_svi_mean scores in to a bin_svi column and a rank_svi column. The bin_svi coulmn returns a label (low, low-mod, mod-high, high) in relation to the raw_svi score, while the rank_svi column is a numeric represention of SVI (1 represetning a high score, 4 represetning a low score).

*Split/Scale*
Prepaed data was split using...
scaled using ....
return .....
shape...


### Explore

### Model

### Conclusions

## How to Reproduce
### Steps

### Tools & Requirements

## Sources

[CDC's Social Vulnerability Index (SVI)](https://www.atsdr.cdc.gov/placeandhealth/svi/index.html)   
- Centers for Disease Control and Prevention/ Agency for Toxic Substances and Disease Registry/ Geospatial Research, Analysis, and Services Program. CDC Social Vulnerability Index 2018 Database Texas. Accessed on 12-8-2020.

[City of San Antonio: Bexar County Covid Data](https://cosacovid-cosagis.hub.arcgis.com/datasets/bexar-county-covid-19-data-by-zip-code/data?geometry=-100.416%2C29.018%2C-96.502%2C29.855&showData=true)   
- Daily COVID counts by ZIP  

[Dallas County COVID Data](https://www.dallascounty.org/covid-19/)   
- Daily COVID counts by ZIP   

[vLookup in Python ](https://www.geeksforgeeks.org/how-to-do-a-vlookup-in-python-using-pandas/)   
- Converts FIPS to ZIP

[HUD](https://www.huduser.gov/portal/datasets/usps_crosswalk.html)
- Schema to correlate ZIP to Census tracts, FIPS codes, and CBSA's

## Creators

Corey Solitaire
Ryvyn Young
Luke Becker
