# Project Title

## About the Project
The CDC's social vulnerability index (SVI) is a scale that predicts the vulnerability of a population in the event of an emergency or natural disaster. COVID is the first global pandemic since the development of this measure. We will evaluate the association between SVI score and COVID case count in San Antonio, Texas. Feature engineering will evaluate the predictive value of categorical SVI score, SVI flags, and change in SVI. The goal is to determine which features best predict COVID cases by zip code with in San Antonio.

### Goals

*Goal # 1* - To evaluate the association between SVI score and COVID case count in San Antonio, TX  

*Goal # 2* - To build a model based SVI score component features that can predict COVID cases by zip code with in San Antonio, TX    

### Background
The SVI (Social Vulnerability Index) was developed to help city governments and first responders predict areas that are particularly vulnerable in emergency situations so that resources can be prioritized to help areas at high risk (Citation CDC Website). The CDC’s Social Vulnerability Index (CDC SVI) uses 15 U.S. census variables to classify census tracts with a composite score between 0 and 1 (lower scores = less vulnerability, higher score = greater vulnerability. This socre is calculated by first ranking every census tract, in every country, in every state, in the United States. Those ranked tracks are then broken up to 4 themes (socioeconomic status, household composition and disability, minority status and language, household type and transportation) and reclassified.  This overall score is then tallied by summing the themed percentiles and ranked on a score between 0 and 1.  

While SVI was designed to help city goverments repsond to emergency situations, the efficacy of the systems has never been tested on in response to a global pandemic. COVID-19 is the disease caused by a new coronavirus called SARS-CoV-2. WHO first learned of this new virus on 31 December 2019, following a report of a cluster of cases of ‘viral pneumonia’ in Wuhan, People’s Republic of China. (Citation WHO). As of 9 December 2020, more than 68.4 million cases have been confirmed, with more than 1.56 million deaths attributed to COVID-19. 

### Deliverables
1. Model to predict COVID 19 symptomatic infection by zipcode in Bexar county
2. Clean and reproducable notebook documenting worflow and findings
3. 5-10 min presentation

### Acknowledgments
Thank you to the Codeup faculty and staff that have helped us every step of the way.  Also thank you to our friends and family who have supported us on this 22 week journey.  

## Data Dictionary
  ---                                ---
| **Terms**                         | **Definition**        |
| ---                               | ---                   |
| Social Vulnerability Index (SVI)  | A composite of 15 U.S. census variables to help local officials identify communities that may need support before, during, or after disasters. |
| Federal Information Processing Standards (FIPS) | A set of standards that describe document processing, encryption algorithms and other information technology standards for use within non-military government agencies and by government contractors and vendors who work with the agencies. |
| COVID - 19 | Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus |
| Regression Model | A predictive modelling technique investigating the relationship between a dependent (*target*) and independent variable(s) (*predictor*)                      
| Target Variable                   | Dependent variable: The feature the model predicts ***(COVID Infection Count)*** |
| Predictive Variable               | Independent variable: The features used to create the prediction |
|                   |                       |
|                   |                       |
  ---                     ---  


## Initial Thoughts & Hypotheses

- Does the SVI score correlate to the number of COVID cases (or cases/10K) observed per zipcode?
   -San Anotnio
   -Dallas

- Is SVI better at predicting COVID cases in zipcodes with overall high/med/low SVI scores?

- Are the individual components of SVI better at predicting COVID cases then the aggregate score?

- Is this pattern different from other cities in TX (Comporable size and SVI demigraphics) ?

## Project Steps

### Acquire

SVI data was acquired from the CDC's website. COVID data for San Antonio and Dallas was downloaded from the cities respective COVID data web portals. In order to merge the data programatic solutions were developed to translate federal FIPS codes in to discernable local Zip codes.  HUD crosswalk provided a guide to transform the data, however, HUD info is complicated as there are many census tracts that may be in one zip code, or they may overlap into multiple other Zip code areas. In order to progratically solve this probelm we found the Zip code that accounted for the highest percentage of addresses within the tract and assigned that as the sole Zip code for the tract. This allowed us to merge the tables by matching to tract then Zip code linking all of the data together in a signle dataframe for prepare.


### Prepare
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
