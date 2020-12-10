# Importing libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# import acquire

# Importing dataframe from acquire (will replace with function that loads dataframe by running acquire functions):

full_df = pd.read_csv('full_san_antonio.csv', index_col = 0)

# Columns we want to keep (can be changed as needed):

columns_to_keep = ['st_abbr',
                    'county',
                    'tract',
                    'area_sqmi',
                    'rpl_themes',
                    'f_pov',
                    'f_unemp',
                    'f_pci',
                    'f_nohsdp',
                    'f_theme1',
                    'f_age65',
                    'f_age17',
                    'f_disabl',
                    'f_sngpnt',
                    'f_theme2',
                    'f_minrty',
                    'f_limeng',
                    'f_theme3',
                    'f_munit',
                    'f_mobile',
                    'f_crowd',
                    'f_noveh',
                    'f_groupq',
                    'f_theme4',
                    'f_total',
                    'zip',
                    'population',
                    'positive',
                    'casesp100000'
                  ]

# Using list comprehension to create a dataframe. Because there are more columns we want to remove than we want to keep, I simply iterated thru the list made above and in essence dropped all columns we didn't want to keep. Easier than using pd.drop.

df = full_df[[c for c in full_df.columns if c in columns_to_keep]]

# Renaming the columns:
df.rename(columns = {'rpl_themes': 'raw_svi', 
                     "f_theme1": "f_soci_total", 
                     "f_theme2": "f_comp_total", 
                     "f_theme3": "f_status_total", 
                     "f_theme4": "f_trans_total", 
                     "f_total": "all_flags_total",
                     "f_pov": "f_pov_soci",
                     "f_unemp": "f_unemp_soci", 
                     "f_pci": "f_pci_soci", 
                     "f_nohsdp": "f_nohsdp_soci", 
                     "f_age65": "f_age65_comp", 
                     "f_age17": "f_age17_comp", 
                     "f_disabl": "f_disabl_comp", 
                     "f_sngpnt": "f_sngpnt_comp", 
                     "f_minrty": "f_minrty_status", 
                     "f_limeng": "f_limeng_status", 
                     "f_munit": "f_munit_trans", 
                     "f_mobile": "f_mobile_trans", 
                     "f_crowd": "f_crowd_trans", 
                     "f_noveh": "f_noveh_trans", 
                     "f_groupq": "f_groupq_trans"}, inplace = True)

# Columns should now be in the format that we want: every column name now is attached to it's sub-theme as derived from the CDC's thematic grouping:
# soci == socio-economic theme
# comp == household composition theme
# status == minority status/language theme
# trans == housing type / transportation


# Dropping rows:

# There are 4 rows which are military bases according to the tract information. These rows all return -999 for all the flags and svi score, thus they are not useful for our analysis. Since 4 rows only 1% of our total rows, we opted to simply drop those 4 rows.

df = df[df.raw_svi > 0]