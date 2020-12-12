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
| bin_svi | raw_svi percentages broken up in to categories based on CDC precident  *low* < 0.27, *low_med* > 0.27 and < 0.50, *med_high* > 0.50 and < 0.75, *high* < 0.75 |                
| rank_svi | raw_svi percentages broken up in to categories based on bin_svi  *low* = 4, *low_med* = 3, *med_high* = 2, *high* = 1 |
  ---                     ---  


## Initial Thoughts & Hypotheses

1. Is there a correlation between the CDC's Range Category SVI Score and COVID-19 Infection Cases per 100k Individuals?

2. Is there a correlation between raw_svi and cases per 100k?

3. Is SVI better at predicting COVID cases in cencus tracts with overall high/med/low SVI scores?

4. Are the individual components of SVI better at predicting COVID cases then the aggregate score?

5. Is this pattern different from other cities in TX (Comporable size and SVI demigraphics)?

## Project Steps
### Acquire
We acquired SVI data from the CDC's website and downloaded COVID data for San Antonio and Dallas from the cities respective COVID data web portals. In order to merge the data we developed programatic solutions to translate federal FIPS codes in to discernable local Zip codes.  The HUD crosswalk provided a guide to transform the data, however, HUD info is complicated as there are many census tracts that may be in one zip code, or they may overlap into multiple other Zip code areas. In order to progratically solve this probelm we found the Zip code that accounted for the highest percentage of addresses within the tract and assigned that as the sole Zip code for the tract. This allowed us to merge the tables by matching to tract then Zip code linking all of the data together in a signle dataframe for prepare. The ratio of addresses for the census tract was then used to calculate a cases per 100K measure for each tract.


### Wrangle
In order to prepare the df for exploration 29 features were selected and renamed for clarity. Four observations associated with military bases were removed from the data frame. Binning the raw SVI score created a bin_svi column and a rank_svi column. The bin_svi column returns a label (low, low-mod, mod-high, high) in relation to the raw_svi score, while the rank_svi column is a numeric representation of SVI (1 representing a high score, 4 representing a low score). Prepared data was split in to train and test for later modeling with cross validation.  Numeric columns with values greater than 4 were scaled using sklearn's MinMaxScaler.  Six data frames were returned at then end of wrangle including *train_explore* for exploration and individual scaled data frames for modeling  *X_train_scaled, y_train, X_test_scaled, y_test*.

### Explore
Exploration focused on answering questions regarding the relationship between the CDC's range category SVI score and cases of COVID-19 per 100k.  After identifying that average number of cases per 100k individuals appeared to be distinct, statistical testing was preformed to validate this observation.  Variability in the data set required us to preform a parametric ANOVA test (Kruskal).  The test was performed at a confidence interval of 99%, and returned a p-value less than alpha requiring us to reject the null hypothesis and accept the alternate hypothesis that the average number of COVID-19 cases per 100k *is significantly different* across all CDC SVI range categories.  This conclusion then prompted us to verify our initial assumption and test for a statistically significant correlation between the raw_svi score and cases per 100k.  This verification was performed using the pearson's correlation coefficient test.  The test was preformed at a confidence interval of 99% and returned a r-value (correlation coefficient) of .55 and a p-value less than alpha requiring us to reject the null hypothesis and accept the alternate hypothesis that there *is a statistically significant difference* between raw_svi and cases per 100K. 

After we were able to infer with 99% confidence that there is a significant difference between CDC SVI range categories we explored the distribution of cases and SVI scores.  When viewed with hue = svi_cat distinct boundaries were observed separating range categories.  Dispersed clustering within categories was observed with the greatest variation occurring in the 'low' SVI vulnerability category.  Several observations within this category were located outside the IQR and identified as outliers. It was decided to leave this data alone, but to further investigate the census tracts and zip codes associated in the next iteration.  Also examine was the relationships between the distribution of cases and number of specific SVI flags.  This plot displayed the same dispersed distribution of cases however the wide distribution of flags under the 'high' vulnerability category was unexpected, as the values ranged from 0-9 flags. This suggests that there is a wide range of flagged vulnerabilities even within the census tracts identified as highly vulnerable.  In the next iteration of research we would love to focus on answering questions such as the mean number of flags in San Antonio, the identification of census tracts with greater then 5 flags, and the geographic (census tract and zip code) distribution of those tracts.  

### Model
Baseline for modeling was determined by plotting the histogram distribution of COVID-19 cases per 100k.  The skew observed in the distribution led us to use the median for this value instead of the mean **(verify which to use)**. Used cross validation due to limited size of dataset. Size of dataset limited by San Antonio number of census tracts. Three of the 4 models used all of the features in the dataset, one model used only the top4 features identified by RFE. Linear Regression, LassoLars, and 2 degree polynomial features used all features and a 2nd version of 2 degree polynomial was run with just the top4 features. Of these the LassoLars had the least MAE (mean absolute error) and was run on out of sample data (test). This model had nearly identical MAE when run on out of sample data, only a 0.7 difference in MAE. Overall this is a 25% improvement from mean baseline MAE.

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
