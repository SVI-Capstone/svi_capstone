# Evaluating the CDC's Social Vulnerability Index (SVI) as a tool to predict COVID infections in San Antonio and Dallas Texas

## About the Project
In 2011 The CDC created the social vulnerability index (SVI).  The SVI is a scale that predicts the vulnerability of a population in the event of an emergency or natural disaster. COVID is the first global pandemic since the development of this measure. We evaluated the association between SVI score and COVID case count between San Antonio and Dallas, Texas.  Using modeling, we were able to identify localized community-specific and highly vulnerable subgroups.  It was observed that the most vulnerable subgroups in San Antonio included persons over 25 years of age with no high school diploma, minority status (non-white), institutional group homes, and those who are generally unemployed.  In Dallas, these groups focused on minority populations, those over 25 years of age with no high school diploma, and individuals that are identified as having limited English proficiency (LEP).  While SVI shared a strong correlation (.55) with COVID case count per 100k in San Antonio, this correlation was not observed in Dallas (.19).  This finding suggests that resources would be well allocated in San Antonio using the SVI index, but in Dallas, that correlation does not hold. We hope that our work helps to better inform our local government about specific sub-populations were aid allocation should be prioritized.
  
### Goals

*Goal # 1* - Evaluate the association between SVI score and COVID case count in San Antonio and Dallas, Texas.

*Goal # 2* - Evaluate the correlation between raw SVI score and case count per 100k.

*Goal # 3* - Compare and contrast the patterns observed between SVI score and COVID case count. 

*Goal # 4* - Predict local communities most at risk for COVID infection using SVI score.

*Goal # 5* - To identify subgroups inside identified communities that need particular attention or focused support.

### Background
The SVI (Social Vulnerability Index) was developed to help city governments and first responders predict areas that are particularly vulnerable in emergencies to prioritize resources to help regions at high risk (CDC's Social Vulnerability Index, 2020). The CDC's Social Vulnerability Index (CDC SVI) uses 15 U.S. census variables to classify census tracts with a composite score between 0 and 1 (lower scores = less vulnerability, higher score = greater vulnerability. This score is calculated by first ranking every census tract, in every country, in every state, in the United States. Those ranked tracks are then broken up into four themes (socioeconomic status, household composition, disability, minority status and language, household type, and transportation) and reclassified.  This overall score is then tallied by summing the themed percentiles and ranked between 0 and 1.  

While SVI was designed to help city governments respond to emergencies, the systems' efficacy has never been tested in response to a global pandemic. COVID-19 is a disease caused by a new coronavirus called SARS-CoV-2. WHO first learned of this new virus on 31 December 2019, following a report of a cluster of cases of 'viral pneumonia' in Wuhan, People's Republic of China. (World Health Organization, 2020). As of 9 December 2020, more than 68.4 million cases have been confirmed, with more than 1.56 million deaths attributed to COVID-19. 

Separated by 274 miles San Antonio and Dallas are two cities that share both comparable population sizes and SVI scores. Cities will be evaluated separately and then together for comparison.  

### Deliverables
1. Model to predict COVID 19 symptomatic infection by census tract in San Antonio and Dallas, TX.
2. Subgroup per community identified for focused support
2. Clean and reproducable notebook documenting worflow and findings
3. 5-10 min presentation

### Acknowledgments
Thank you to the Codeup faculty and staff that have helped us every step of the way.  Also thank you to our friends and family who have supported us on this 22 week journey.  

## Data Dictionary

[Click on link to see full dictionary](https://github.com/SVI-Capstone/svi_capstone/blob/main/data_dictionary.md)
  ---                                ---
| **Terms**                        | **Definition**        |
| ---                              | ---                   |
| Social Vulnerability Index (SVI) | A composite of 15 U.S. census variables to help local officials identify communities that may need support before, during, or after disasters. |
| Census Tract | Small, relatively permanent statistical subdivisions of a county or equivalent entity that are updated by local participants before each decennial census as part of the Census Bureau's Participant Statistical Areas Program |
| Federal Information Processing Standards (FIPS) | A set of standards that describe document processing, encryption algorithms, and other information technology standards for use within non-military government agencies and by government contractors and vendors who work with the agencies. |
| COVID - 19 | Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus |
| Regression | A predictive modeling technique investigating the relationship between a dependent (*target*) and independent variable(s) (*predictor*)                      
| LassoLars Model | Algorithm that performs both feature selection (LASSO) and noise reduction within the same model.|
| Algorithm | A process or set of rules to be followed in calculations or other problem-solving operations, especially by a computer. |
| Clustering| Clustering is an unsupervised process of grouping similar observations or objects together. In this process similarities are based on comparing a vector of information for each observation or object, often using various mathematical distance functions.|
| Target Variable | Dependent variable: The feature the model predicts ***(COVID Infection Count)*** |
| Predictive Variable | Independent variable: The features used to create the prediction |
| SVI flag | For a theme, the flag value is the number of flags for variables comprising the theme. We calculated the overall flag value for each tract as the number of all variable flags (Tracts in the top 10%, i.e., at the 90th percentile of values).  |
| e_pov | Persons below poverty estimate |
| ep_pov | Percent persons below poverty estimate |
| spl_theme1 | Sum of series for socioeconomic theme |
| f_pov_soci | Flag - the percentage of person in poverty is in the 90th percentile nationally (1= yes, 0 = no) |                                                   | f_unemp_soci | Flag - the percentage of civilian unemployed is in the 90th percentile nationally (1= yes, 0 = no) |
| f_pci_soci | Flag - per capita income is in the bottom 90th percentile nationally (1= yes, 0 = no)|
| f_nohsdp_soci | Flag - the percentage of persons with no high school diploma is in the 90th percentile nationally (1= yes, 0 = no)|
| f_soci_total | Sum of flags for Socioeconomic Status theme|
| f_age65_comp | Flag - the percentage of persons aged 65 and older is in the 90th percentile nationally (1= yes, 0 = no)|
| f_age17_comp | Flag - the percentage of persons aged 17 and younger are in the 90th percentile nationally (1= yes, 0 = no)|
| f_disabl_comp | Flag - the percentage of persons with a disability is in the 90th percentile nationally (1= yes, 0 = no)|
| f_sngpnt_comp | Flag - the percentage of single-parent households is in the 90th percentile nationally (1= yes, 0 = no)|
| f_comp_total | Sum of flags for Household Compensation theme |
| f_minrty_status | Flag - the percentage of minority is in the 90th percentile nationally (1= yes, 0 = no)|
| f_limeng_status  | Flag - the percentage of those with limited English is in the 90th percentile nationally (1= yes, 0 = no)|
| f_status_total | Sum of flags for Minority Status/Language theme |
| f_munit_trans | Flag = the percentage of households in multi-unit housing in the 90th percentile nationally (1= yes, 0 = no)|
| f_mobile_trans | Flag - the percentage of mobile homes is in the 90th percentile nationally (1= yes, 0 = no)|
| f_crowd_trans | Flag - the percentage of crowded households is in the 90th percentile nationally (1= yes, 0 = no)|
| f_noveh_trans | Flag - the percentage of households with no vehicles is in the 90th percentile nationally (1= yes, 0 = no)|
| f_groupq_trans | Flag - the percentage of persons in institutionalized group quarters is in the 90th percentile nationally (1= yes, 0 = no)|
| f_trans_total | Sum of flags for Housing Type/Transportation theme |
| all_flags_total| Sum of flags for the four themes |
| tract_cases_per_100k | Derived density of cases per Census Tract |
| bin_svi | raw_svi percentages broken up in to categories based on CDC precedent  *low* < 0.27, *low_med* > 0.27 and < 0.50, *med_high* > 0.50 and < 0.75, *high* < 0.75 |                
| rank_svi | raw_svi percentages broken up in to categories based on bin_svi  *low* = 4, *low_med* = 3, *med_high* = 2, *high* = 1 |
| Mean Absolute Error (MAE) | MAE measures the average magnitude of the errors in a set of predictions without considering their direction. It's the average over the test sample of the absolute differences between prediction and actual observation where all individual differences have equal weight. |
| Centroid | One of the cluster centers in a K-Means clustering |
| rising | Raw svi score 2018>2016>2014 (getting more vulnerable), bool |
| falling | Raw svi score 2018<2016<2014 (getting less vulnerable), bool |
| delta | Change in raw svi score from 2014 to 2018, float |
| avg3yr | Added raw svi for 2018, 2016, 2014 and divided by 3, float |
| r_soci_rise | Raw score for socioeconomic subgroup rising (getting more vulnerable) year over year, bool |
| r_comp_rise | Raw score for household compensation and disability subgroup rising (getting more vulnerable) year over year, bool |
| r_status_rise | Raw score for minority and language subgroup rising (getting more vulnerable) year over year, bool |
| r_trans_rise | Raw score for transportation and housing type subgroup rising (getting more vulnerable) year over year, bool |
| r_soci_fall | Raw score for socioeconomic subgroup falling (getting less vulnerable) year over year, bool |
| r_comp_fall | Raw score for household compensation and disability subgroup falling (getting less vulnerable) year over year, bool |
| r_status_fall | Raw score for minority and language subgroup falling (getting less vulnerable) year over year, bool |
| r_trans_fall | Raw score for transportation and housing type subgroup falling (getting less vulnerable) year over year, bool |
  ---                                ---

## Initial Thoughts & Hypotheses

1. Is the average number of COVID cases per 100k is the same across CDC SVI Range Categories?

2. Is there a correlation between raw_svi and the number of cases per 100k?

3. Is SVI a useful feature for predicting the number of cases per 100k?

4. Are the individual components of SVI better at predicting COVID cases, then the rank score?

5. Are the features identified in modeling consistent across communities (similar size and SVI score)? 

## Project Steps
### Acquire
We acquired SVI data from the CDC's website and downloaded COVID data for San Antonio and Dallas from the cities' respective COVID data web portals. We developed programmatic solutions to translate federal FIPS codes into discernable local Zip Codes to merge the data.  The HUD crosswalk provided a guide to transforming the data; however, HUD info is complicated as many census tracts may be in one Zip Code or overlap into multiple other Zip Code areas. To programmatically solve this problem, we found the Zip Code that accounted for the highest percentage of addresses within the tract and assigned that as the tract's sole Zip Code. This allowed us to merge the tables by matching them to a tract. Then, Zip Code links all of the data together in a single data frame for preparation. The ratio of addresses for the census tract was then used to calculate cases per 100K measure for each tract.

### Wrangle
Twenty-nine features were selected and renamed for clarity to prepare the data for exploration. Four observations associated with military bases were removed from the data frame. Binning, the raw SVI score created a bin_svi column and a rank_svi column. The bin_svi column returns a label (low, low-mod, mod-high, high) in relation to the raw_svi score, while the rank_svi column is a numeric representation of SVI (1 representing a high score, four representing a low score). Prepared data was split into train and test for later modeling with cross-validation.  Numeric columns with values that are greater than four were scaled using sklearn's MinMaxScaler.  Six data frames were returned at the end of wrangle, including *train_explore* for exploration and individual scaled data frames for modeling  *X_train_scaled, y_train, X_test_scaled, y_test*.

### Explore
Exploration focused on answering questions regarding the relationship between the CDC's range category SVI score and cases of COVID-19 per 100k. After identifying that an average number of instances per 100k individuals appeared to be distinct, statistical testing was performed to validate this observation. Variability in the data set required us to perform a parametric ANOVA test (Kruskal). The test was performed at a confidence interval of 99% and returned a p-value less than alpha, requiring us to reject the null hypothesis and accept the alternate hypothesis that the average number of COVID-19 cases per 100k is significantly different across all CDC SVI range categories. This conclusion then prompted us to verify our initial assumption and test for a statistically significant correlation between the raw_svi score and cases per 100k. This verification was performed using Pearson's correlation coefficient test. The test was performed at a confidence interval of 99% and returned an r-value (correlation coefficient) of .55 for San Antonio and .29 for Dallas. This test also returned a p-value less than alpha, allowing us to reject the null hypothesis and accept the alternate hypothesis that there * a statistically significant difference* between raw_svi and cases per 100K.

As we were able to infer with 99% confidence that there is a significant difference between CDC SVI range categories, we explored the distribution of cases and SVI scores. When viewed with hue = svi_cat distinct boundaries were observed separating range categories. Dispersed clustering within categories was observed, with the most significant variation occurring in the 'low' SVI vulnerability category. Several observations within this category were located outside the IQR and identified as outliers. It was decided to leave this data alone but to further investigate the census tracts and zip codes associated in the next iteration. Also, examine was the relationships between the distribution of cases and the number of specific SVI flags. In San Antonio, a wide distribution of flags under the 'high' vulnerability category was observed. This came as an unexpected observation and suggested the need to identify subgroups inside identified communities that need more focused assistance or support. Similar trends were observed in the Dallas dataset. The most significant difference is that flags' specific distribution under the 'high' vulnerability category change.  This unique distribution of flags per city suggests that while cities may use SVI to identify vulnerable communities, they may need alternate tools to key in on particularly vulnerable subgroups. 

After it was identified that unique flags might be used to better identify groupings within the data, an exploratory clustering analysis was performed. We hoped to use clustering to generate a new feature that could then be fed into a regression model.  Raw SVI data were examined, and three columns were identified as closely related to the prediction of COVID counts per 100k.  These features were e_pov (the estimate of persons below poverty), ep_pov (the percentage of persons below poverty), and spl_theme1 (the sum of features associated with socioeconomic themes).  These features were combined into a new feature identified as poverty_cluster.  This feature, along with centroids, were added to the data frame for modeling.   

### Model
Two rounds of modeling were performed during this investigation.  The first being a regression model that assessed SVI's impact on the prediction of COVID cases per 100k, and the second being a classification model that assessed SVI's impact on case rank (defined as low, low_mod, mod_high, high). 

*Regression Model*     
   
The mean value for COVID cases per 100k was identified as the baseline for modeling. We used cross-validation instead of a three-way split into train, validate, and test datasets due to the dataset's limited size. The size of the dataset is limited by the number of census tracts in each city. Linear Regression and LassoLars algorithms were used to evaluate multiple combinations of feature selection. Of these, the LassoLars had the least MAE (mean absolute error) when using all of the possible features and was run on the out of sample (test) data. The MAE of a model is the mean of the individual prediction errors' absolute values over all instances in the test set. We chose to assess model performance in terms of MAE due to its ease of interpretation. San Antonio's top-performing model was a TweedieRegressor, using Top 4 features as identified by RFE. The model demonstrated a 21% improvement over baseline. Dallas' top-performing model was also a TweedieRegressor, using Top 4 features as determined by RFE. The model showed a 2% improvement over baseline.  

*Classification Model*      
   
For the classification model, we ran a series of classification models using the Random Forest and KNN algorithms, which sought to use SVI components as features in the models to predict the severity of COVID cases based on our constructed rankings of point counts: Low, Low-Moderate, Moderate-High, and High bins. The mean value of the most common ranked bin was identified as the baseline for modeling. We found that our classification models' most useful features were also similar to the features we found as applicable in regression modeling. Using the Random Forest algorithm and using the top 4 RFE features, our best model yielded an accuracy result of 55%, which is an improvement over the baseline of 7% (or an increase in accuracy of 14.5%).

### Conclusions
**1. Is the average number of COVID cases per 100k is the same across CDC SVI Range Categories?**
- Based on the Kruskal test, we are 99% confident that there is a significant difference between the average number of cases across the CDC SVI range categories in San Antonio and Dallas. This suggests that SVI is useful in predicting vulnerable communities during this pandemic and that SVI values should be examined as modeling features.    

**2. Is there a correlation between raw_svi and the number of cases per 100k?**
- Based on a Pearson R correlation test, we are 99% confident that there is a correlation between raw_svi and the number of cases per 100k in San Antonio and Dallas.  This correlation does not suggest causation yet describes a linear relationship that exists between the two features.  This relationship is characterized by a strong correlation in San Antonio (0.54) and a weaker yet still significant correlation in Dallas (0.19).   

**3. Is SVI a useful feature for predicting the number of cases per 100k?**   
   
- *Yes, in San Antonio*. Using a TweedieRegressor with SVI as a feature, our model can predict the number of cases per 100k better than baseline (21%).    
   
- *Yes, in Dallas*.  Using a TweedieRegressor with SVI as a feature, our model can predict the number of cases per 100k better than baseline (2%).   
   
This observation suggests that while there are statistically significant correlations between SVI score and cases per 100k in San Antonio and Dallas, the SVI score's predictive power is more significant in San Antonio.  Further investigation is necessary to explain this disparity.     

**4. Are the individual components of SVI better at predicting COVID cases than the rank score?**    
- *Yes, in San Antonio*, individual SVI components are better than the raw or binned score at predicting cases per 100k.  Three of the top features were derived from looking at how the SVI score has changed over time.  In San Antonio, the top four features identified as necessary in predicting count per 100k included total socioeconomic themes, the change in SVI for minority and language subgroups, the change in SVI status between 2014 – 2018, and the average SVI score between 2014 -2018.     
   
- *Yes, in Dallas*, individual SVI components are better than the raw or binned score at predicting cases per 100k.  In Dallas, the top four features identified as necessary in predicting count per 100k included the centroids of persons below poverty, percent of persons below poverty, and the scaled count of total socioeconomic themes.  The only significant factor that was not derived from clustering was derived from the SVI change over time, the difference in SVI for minority and language subgroup.     
    
These observations suggest that changes in SVI have a considerable influence on how our models perform.  Local governments should consider the shift in SVI score over time when thinking about the distribution of aid, as communities that have seen improvement in SVI scores between 2014-2018 are less likely to need the number of resources that communities with stagnant scores will require.  
    
**5. Are the features identified in modeling consistent across communities (similar size and SVI score)?**
- In San Antonio and Dallas, COVID cases per 100k are greatest in communities where most residents are of minority status and lack educational opportunities (individuals >25 and no diploma).  What is different about these two cities is the predictive ability SVI has on COVID case count.  In San Antonio, resources would be well allocated using the SVI index, but in Dallas, that correlation does not hold. For two cities with the same approximate SVI score and population, this is an interesting observation that will require further research to understand better. Regardless we hope that our work helps to better inform our local government about specific sub-populations where aid allocation should be prioritized.

## How to Reproduce
### Steps
1. Obtain SVI data set from CDC website
2. Obtain HUD crosswalk data set
3. Identify COVID case count by county
4. Run functions in notebook on county data set.  

### Tools & Requirements
1. Web-based interactive development environment
2. County data of your choosing

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

[World Health Organization: COVID-19](https://www.who.int/emergencies/diseases/novel-coronavirus-2019)
- WHO website and resource on COVID - 19

## Creators

[Corey Solitaire](https://github.com/CSolitaire)  
[Ryvyn Young](https://github.com/RyvynYoung)   
[Luke Becker](https://github.com/lukewbecker)   
