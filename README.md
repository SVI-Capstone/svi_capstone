# Project Title
## About the Project

The purpose of this project is to discover if a city’s SVI score is an accurate predictor of Covid infection rates. The purpose of the SVI is to predict which communities within a city (or county) will need the most help in the event of a disaster, thus we are trying to determine if the SVI is indeed acting currently as a good predictor for a pandemic type disaster, like Covid is. Once we discover if SVI is a good predictor or not of Covid infection rates, then we will be able to break down the SVI index further and search for sub-components which may act as a better predictor for covid transmission than the overall SVI index. We could then use those SVI flags to predict where Covid infections would be the highest based on the SVI features we use, and test to see if that model we create is more accurate than simply using the base SVI index by itself to predict Covid infection. (Draft) 

### Goals


### Background
The SVI (Social Vulnerability Index) was developed to help city governments and first responders predict areas that are particularly vulnerable in emergency situations so that resources can be prioritized to help areas at high risk (citation). The CDC’s Social Vulnerability Index (CDC SVI) uses 15 U.S. census variables to classify census tracts with a composite score between 0 and 1 (lower scores = less vulnerability, higher score = greater vulnerability.  This socre is calculated by first ranking every census tract, in every country, in every state, in the United States.  Those ranked tracks are then broken up to 4 themes (  socioeconomic status, household composition and disability,  minority status and language, household type and transportation) and reclassified.  This overall score is then tallied by summing the themed percentiles and ranked on a score between 0 and 1.  

SVI 


### Deliverables
1. Model to predict COVID 19 symptomatic infection by zipcode in Bexar county
2. Clean and reproducable notebook documenting worflow and findings
3. 5-10 min presentation

### Acknowledgments

## Data Dictionary
  ---                                ---
| **Terms**                         | **Definition**        |
| ---                               | ---                   |
| Social Vulnerability Index (SVI)  | A predictive modelling technique investigating the relationship between a dependent (target) and independent variable (s) (predictor) |
| FIPS                              |                       |
| COVID - 19                        |                       |
| Regression Model                  |                       |
| Target Variable                   | Dependent variable: The feature the model predicts ***(COVID Infection Count)***                     |
| Predictive Variabvle              | Independent variable: The features used to create the prediction                                     |
|                   |                       |
|                   |                       |
  ---                     ---  


## Initial Thoughts & Hypotheses

- Does the SVI score correlate to the number of COVID cases (or cases/10K) per zipcode?
   -San Anotnio
   -Dallas

- Is SVI better at predicting COVID cases in cities with overall high/med/low SVI scores?

### Thoughts
### Hypotheses
## Project Steps
### Acquire
### Prepare
### Explore
### Model
### Conclusions
## How to Reproduce
### Steps
### Tools & Requirements

## Sources

[CDC's Social Vulnerability Index (SVI)](https://www.atsdr.cdc.gov/placeandhealth/svi/index.html)   
Centers for Disease Control and Prevention/ Agency for Toxic Substances and Disease Registry/ Geospatial Research, Analysis, and Services Program. CDC Social Vulnerability Index 2018 Database Texas. Accessed on 12-8-2020.

[City of San Antonio: Bexar County Covid Data](https://cosacovid-cosagis.hub.arcgis.com/datasets/bexar-county-covid-19-data-by-zip-code/data?geometry=-100.416%2C29.018%2C-96.502%2C29.855&showData=true)   
Daily COVID counts by zipcode   

[Dallas County COVID Data](https://www.dallascounty.org/covid-19/)   
Daily COVID counts by zipcode   

[vLookup in Python ](https://www.geeksforgeeks.org/how-to-do-a-vlookup-in-python-using-pandas/)   
- Converts FIPS to zipcode

## Creators

Corey Solitaire
Ryvyn Young
Luke Becker
