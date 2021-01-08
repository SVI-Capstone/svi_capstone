# Evaluating the CDC's Social Vulnerability Index (SVI) as a tool to predict COVID infections in San Antonio and Dallas Texas

## About the Project
In 2011 The CDC created the Social Vulnerability Index (SVI). The SVI is a scale that predicts a population's vulnerability in the event of an emergency or natural disaster. COVID is the first global pandemic since the development of this measure. We evaluated the effectiveness of utilizing SVI as a tool to predict COVID case count per 100,000 individuals in San Antonio and Dallas, Texas. Using exploratory data analysis, we were able to identify localized community-specific and highly vulnerable subgroups and from them engineer features that were better than SVI at predicting case count. We found that the best indicator of case count in both cities was total socioeconomic need. Over-reliance on SVI during pandemic type natural disasters may lead to vulnerable populations missing out on critical resources.  With this in mind, we recommend targeting resources based on socioeconomic need, exploring additional cities to determine how predictive SVI is more broadly, and investigating underlying city differences that make SVI more or less predictive. 
  
### Goals

*Goal # 1* - Evaluate the correlation between SVI and case count per 100k in San Antonio and Dallas, Texas.

*Goal # 2* - Compare and contrast the patterns observed between SVI and COVID case count per city. 

*Goal # 3* - Predict local communities most at risk for COVID infection.

*Goal # 4* - Identify key features necessary to identify communities in need of attention and or focused support.

### Background
The SVI (Social Vulnerability Index) was developed to help city governments and first responders predict areas that are particularly vulnerable in emergencies to prioritize resources to help regions at high risk (CDC's Social Vulnerability Index, 2020). The CDC's Social Vulnerability Index (SVI) uses 15 U.S. census variables to classify census tracts with a composite score between 0 and 1 (lower scores = less vulnerability, higher score = greater vulnerability). This score is calculated by ranking every census tract, in every country, in every state, in the United States. Those ranked tracks are then broken up into four themes (socioeconomic status, household composition, disability, minority status and language, household type, and transportation) and reclassified.  This overall score is then tallied by summing the themed percentiles and ranked between 0 and 1.  

While SVI was designed to help city governments respond to emergencies, the systems' efficacy has never been tested in response to a global pandemic. COVID-19 is a disease caused by a new coronavirus called SARS-CoV-2. WHO first learned of this new virus on 31 December 2019, following a report of a cluster of cases of 'viral pneumonia' in Wuhan, People's Republic of China (World Health Organization, 2020). As of 9 December 2020, more than 68.4 million cases have been confirmed, with more than 1.56 million deaths attributed to COVID-19. 

Separated by 274 miles San Antonio and Dallas are two cities that share both comparable population sizes and SVI's. Cities will be evaluated separately and then together for comparison.  

### Deliverables
1. Model to predict COVID 19 symptomatic infection by census tract in San Antonio and Dallas, TX.
2. Subgroup per community identified for focused support
2. Clean and reproducable notebook documenting worflow and findings
3. 5-10 min presentation

### Acknowledgments
Thank you to the Codeup faculty and staff that have helped us every step of the way.  Also thank you to our friends and family who have supported us on this 22 week journey.  

## Data Dictionary   
   
[Click on link to see full dictionary](https://github.com/SVI-Capstone/svi_capstone/blob/main/data_dictionary.md)   
   
## Initial Thoughts & Hypotheses   
   
1. Is the average number of COVID cases per 100k is the same across CDC SVI Range Categories?

2. Is there a correlation between SVI and the number of cases per 100k?

3. Is SVI a useful feature for predicting the number of cases per 100k?

4. Are the individual components of SVI better at predicting COVID cases, then the rank score?

5. How has SVI changed over time over time?

## Project Steps
### Acquire
We acquired SVI data from the CDC's website and downloaded COVID data for San Antonio and Dallas from the cities' respective COVID data web portals. We developed programmatic solutions to translate federal FIPS codes into discernible local Zip Codes to merge the data.  The HUD crosswalk provided a guide to transforming the data; however, HUD info is complicated as many census tracts may be in one Zip Code or overlap into multiple other Zip Code areas. To programmatically solve this problem, we found the Zip Code that accounted for the highest percentage of addresses within the tract and assigned that as the tract's sole Zip Code. This allowed us to merge the tables by matching them to a tract. Then, tract links all of the data together in a single data frame for preparation. The ratio of addresses for the census tract was then used to calculate cases per 100K measure for each tract.

### Wrangle
Twenty-nine features were selected and renamed for clarity to prepare the data for exploration. Four observations associated with military bases were removed from the data frame. Binning, the raw SVI score created a bin_svi column and a rank_svi column. The bin_svi column returns a label (low, low-mod, mod-high, high) in relation to the raw_svi score, while the rank_svi column is a numeric representation of SVI (1 representing a high score, four representing a low score). Prepared data was split into train and test for later modeling with cross-validation.  Numeric columns with values that are greater than four were scaled using sklearn's MinMaxScaler.  Six data frames were returned at the end of wrangle, including *train_explore* for exploration and individual scaled data frames for modeling  *X_train_scaled, y_train, X_test_scaled, y_test*.

### Explore
Exploration focused on answering questions regarding the relationship between the CDC's range category SVI score and cases of COVID-19 per 100k. After identifying that an average number of instances per 100k individuals appeared to be distinct, statistical testing was performed to validate this observation. Variability in the data set required us to perform an ANOVA test for San Antonio and a Kruskal test for Dallas. The test was performed at a confidence interval of 99% and returned a p-value less than alpha, requiring us to reject the null hypothesis and accept the alternate hypothesis that the average number of COVID-19 cases per 100k is significantly different across all CDC SVI range categories. This conclusion then prompted us to verify our initial assumption and test for a statistically significant correlation between the raw_svi score and cases per 100k. This verification was performed using Pearson's correlation coefficient test. The test was performed at a confidence interval of 99% and returned an r-value (correlation coefficient) of .55 for San Antonio and .29 for Dallas. This test also returned a p-value less than alpha, allowing us to reject the null hypothesis and accept the alternate hypothesis that there * a statistically significant difference* between raw_svi and cases per 100K.

As we were able to infer with 99% confidence that there is a significant difference between CDC SVI range categories, we explored the distribution of cases and SVI scores. When viewed with hue = svi_cat distinct boundaries were observed separating range categories. Dispersed clustering within categories was observed, with the most significant variation occurring in the 'low' SVI vulnerability category. Several observations within this category were located outside the IQR and identified as outliers. It was decided to leave this data alone but to further investigate the census tracts and zip codes associated in the next iteration. Also, examine was the relationships between the distribution of cases and the number of specific SVI flags. In San Antonio, a wide distribution of flags under the 'high' vulnerability category was observed. This came as an unexpected observation and suggested the need to identify subgroups inside identified communities that need more focused assistance or support. Similar trends were observed in the Dallas dataset. The most significant difference is that flags' specific distribution under the 'high' vulnerability category change.  This unique distribution of flags per city suggests that while cities may use SVI to identify vulnerable communities, they may need alternate tools to key in on particularly vulnerable subgroups. 

After it was identified that unique flags might be used to better identify groupings within the data, an exploratory clustering analysis was performed. We hoped to use clustering to generate a new feature that could then be fed into a regression model.  Raw SVI data were examined, and three columns were identified as closely related to the prediction of COVID counts per 100k.  These features were e_pov (the estimate of persons below poverty), ep_pov (the percentage of persons below poverty), and spl_theme1 (the sum of features associated with socioeconomic themes).  These features were combined into a new feature identified as poverty_cluster.  This feature, along with centroids, were added to the data frame for modeling.   

### Model
Two rounds of modeling were performed during this investigation.  The first being a regression model that assessed SVI's impact on the prediction of COVID cases per 100k, and the second being a classification model that assessed SVI's impact on case rank (defined as low, low_mod, mod_high, high). 

*Regression Model*     
   
The mean value for COVID cases per 100k was identified as the baseline for modeling. We used cross-validation instead of a three-way split into train, validate, and test datasets due to the dataset's limited size. The size of the dataset is limited by the number of census tracts in each city. Five different linear regression algorithms were used to evaluate multiple combinations of feature selection. Of these, the Tweedie had the least MAE (mean absolute error) when using all of the possible features and was run on the out of sample (test) data. The MAE of a model is the mean of the individual prediction errors' absolute values over all instances in the test set. We chose to assess model performance in terms of MAE due to its ease of interpretation. San Antonio's top-performing model was a TweedieRegressor, using Top 4 features as identified by RFE. The model demonstrated a 21% improvement over baseline. Dallas' top-performing model was also a TweedieRegressor, using Top 4 features as determined by RFE. The model showed a 3% improvement over baseline.  

*Classification Model*      
   
For the classification model, we ran a series of classification models using the Random Forest and KNN algorithms, which sought to use SVI components as features in the models to predict the severity of COVID cases based on our constructed rankings of point counts: Low, Low-Moderate, Moderate-High, and High bins. The mean value of the most common ranked bin was identified as the baseline for modeling. We found that our classification models' most useful features were also similar to the features we found as applicable in regression modeling. Using the Random Forest algorithm and using the top 4 RFE features, our best model yielded an accuracy result of 55%, which is an improvement over the baseline of 7% (or an increase in accuracy of 14.5%).

### Conclusions
**1. Is the average number of COVID cases per 100k is the same across CDC SVI Range Categories?**
- Based on the ANOVOA and Kruskal tests, we are 99% confident that there is a significant difference between the average number of cases across the CDC SVI range categories in San Antonio and Dallas. This suggests that SVI is useful in predicting vulnerable communities during this pandemic and that SVI values should be examined as modeling features.    

**2. Is there a correlation between raw_svi and the number of cases per 100k?**
- Based on a Pearson R correlation test, we are 99% confident that there is a correlation between raw_svi and the number of cases per 100k in San Antonio and Dallas.  This correlation does not suggest causation yet describes a linear relationship that exists between the two features.  This relationship is characterized by a strong correlation in San Antonio (0.54) and a weaker yet still significant correlation in Dallas (0.19).   

**3. Is SVI a useful feature for predicting the number of cases per 100k?**   
   
- *Yes, in San Antonio*. Using a TweedieRegressor with SVI as a feature, our model can predict the number of cases per 100k 15% better than baseline.    
   
- *Not really, in Dallas*.  Using a TweedieRegressor with SVI as a feature, our model can predict the number of cases per 100k only 1% better than baseline.   
   
This observation suggests that while there are statistically significant correlations between SVI and cases per 100k in San Antonio and Dallas, the SVI's predictive power is more significant in San Antonio.  Further investigation is necessary to explain this disparity.     

**4. Are the individual components of SVI better at predicting COVID cases than the rank score?**    
- *Yes, in San Antonio*, individual SVI components are better than the index at predicting cases per 100k.  Three of the top features were derived from looking at how the SVI score has changed over time.  In San Antonio, the top four features identified as necessary in predicting count per 100k included total socioeconomic themes, the change in SVI for minority and language subgroups, the change in SVI status between 2014 – 2018, and the average SVI between 2014 -2018.     
   
- *Yes, in Dallas*, individual SVI components are better than the index at predicting cases per 100k.  In Dallas, the top four features identified as necessary in predicting count per 100k included the centroids of persons below poverty, percent of persons below poverty, and the scaled count of total socioeconomic themes.  The only significant factor that was not derived from clustering was derived from the SVI change over time, the difference in SVI for minority and language subgroup.     
    
These observations suggest that socioeconomic themes and changes in SVI have a considerable influence on how our models perform.  Local governments should consider the focusing on socioeconomic need scores when thinking about the distribution of aid.  
    
**5. How has SVI changed over time over time?**    

- Almost half (45%) of the communities (tracts) in San Antonio are becomming more voulnerable year over year from 2014 to 2018 vs. 39% of the communities in Dallas   
   
- Only 8% of the areas in San Antonio are seeing a year over year improvement in SVI vs. 13% of communities in Dallas   
   
   - key grouping in San Antonio that is getting worse year over year is socioeconomic subgroup    
        San Antonio 38% getting worse vs. 30% in Dallas   
        San Antonio only 13% of areas are improving vs 19% in Dallas   
   
   - Additional key group difference household composition   
        San Antonio 30% getting worse vs. 25% in Dallas   
        San Antonio 23% getting better vs. 28% getting better in Dallas   
   
- Things that might be impacting this   
   - Dallas might have programs in place that San Antonio does not to assist these at risk areas   
   - Redlining history and continued impact in San Antonio may be disproportionately effecting or stagnating improvements in areas in San Antonio   
   
In San Antonio and Dallas, COVID cases per 100k are greatest in communities where most residents are of minority status and lack educational opportunities (individuals >25 and no diploma).  What is different about these two cities is the predictive ability SVI has on COVID case count.  In San Antonio, resources would be well allocated using the SVI, but in Dallas, that correlation does not hold. For two cities with the same approximate SVI and population, this is an interesting observation that will require further research to understand better. Regardless we hope that our work helps to better inform our local government about specific sub-populations where aid allocation should be prioritized.   
   
## How to Reproduce
[Click here for a step by step tutorial](https://github.com/SVI-Capstone/svi_capstone/blob/main/Instructions_to_reproduce.ipynb) 

## Sources
[CDC's Social Vulnerability Index (SVI)](https://www.atsdr.cdc.gov/placeandhealth/svi/index.html)   
- Centers for Disease Control and Prevention/ Agency for Toxic Substances and Disease Registry/ Geospatial Research, Analysis, and Services Program. CDC Social Vulnerability Index 2018 Database Texas. Accessed on 12-8-2020.

[City of San Antonio: Bexar County Covid Data](https://cosacovid-cosagis.hub.arcgis.com/datasets/bexar-county-covid-19-data-by-zip-code/data?geometry=-100.416%2C29.018%2C-96.502%2C29.855&showData=true)   
- Daily COVID counts by ZIP  

[Dallas County COVID Data](https://www.dallascounty.org/covid-19/)   
- Daily COVID counts by ZIP   

[HUD](https://www.huduser.gov/portal/datasets/usps_crosswalk.html)
- Schema to correlate ZIP to Census tracts, FIPS codes, and CBSA's

[World Health Organization: COVID-19](https://www.who.int/emergencies/diseases/novel-coronavirus-2019)
- WHO website and resource on COVID - 19

## Creators

[Corey Solitaire](https://github.com/CSolitaire)  
[Ryvyn Young](https://github.com/RyvynYoung)   
[Luke Becker](https://github.com/lukewbecker)   
