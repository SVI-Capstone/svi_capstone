## Data Dictionary
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
  
