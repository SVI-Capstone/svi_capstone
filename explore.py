import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
from scipy.stats import f_oneway, kruskal
from math import sqrt
from sklearn.cluster import KMeans
import statsmodels.api as sm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'


############################################################ Graphing Functions ############################################################
# def set_plotting_defaults():
#     # plotting defaults
#     plt.rc('figure', figsize=(13, 7))
#     plt.style.use('seaborn-whitegrid')
#     plt.rc('font', size=16)

def sns_boxplot(train_exp):
    '''create boxplot for hypothesis test exploration
    '''
    #axes1.margins(x=0.05)
    plt.figure(figsize=(12,8))
    plt.rc('font', size=16)
    sns.boxplot(data=train_exp, x='bin_svi', y='tract_cases_per_100k')
    plt.title('December 8th COVID-19 Cases per 100K by SVI Range Category')
    plt.xlabel("CDC's SVI Range Category",labelpad=20)
    plt.ylabel("Cases by Tract per 100k",labelpad=20)
    plt.tight_layout()
    plt.show()

def joint_plot_index(x,y,df,hue):
    """
    Function that produces a joint plot and examines data by categorical variable
    """
    ax = sns.jointplot(x = x, 
                       y = y, 
                       data = df, 
                       hue = hue, 
                       height = 10,
                       space = 0)
    ax.ax_joint.set_xlabel("SVI Index Value", fontweight='bold', fontsize = 14, labelpad=20)
    ax.ax_joint.set_ylabel('Cases by Tract per 100k', fontweight='bold', fontsize = 14, labelpad=20)
    ax.fig.suptitle("Distribution of Cases and SVI Score", fontweight='bold', fontsize = 20)    
    ax.fig.tight_layout()
    plt.show()

def joint_plot_flag(x,y,df,hue):
    ax2 = sns.jointplot(x = x, 
                        y = y, 
                        data = df, 
                        hue = hue, 
                        height = 10,
                        space = 0,
                        xlim =(-1,10),)
    ax2.ax_joint.set_xlabel('Total Number of Flags', fontweight='bold', fontsize = 14,labelpad=20)
    ax2.ax_joint.set_ylabel('Cases by Tract per 100k', fontweight='bold', fontsize = 14,labelpad=20)
    ax2.fig.suptitle("Distribution of Cases and SVI Flags", fontweight='bold', fontsize = 20)

    ax2.fig.tight_layout()
    plt.show()

def hist_case(series):
    plt.figure(figsize=(12,8))
    plt.rc('font', size=16)
    plt.hist(x = series, bins = 10, color = 'gray', edgecolor='k', alpha=0.45)
    plt.title('Distribution of Cases in San Antonio, TX: December 8th 2020', y=1.02)
    plt.xlabel('Number of Cases per 100,000',labelpad=20)
    plt.ylabel('Case Count',labelpad=20)
    plt.axvline(series.mean(), color = 'tab:orange', linestyle='dashed', linewidth=5)
    min_ylim_v, max_ylim_v = plt.ylim()
    plt.text(series.mean()*1.05, max_ylim_v*0.9, 'Mean: {:.2f}'.format(series.mean()))
    # plt.axvline(series.median(), color = 'darkgreen', linestyle='dashed', linewidth=5)
    # plt.text(series.median()*.25, max_ylim_v*0.9, 'Median: {:.2f}'.format(series.median()))
    plt.grid(b = True, alpha = .45)
    # plt.figure(figsize = (16, 9))
    plt.tight_layout()
    plt.show()


def hist_case_title(series, title):
    '''
    This function will take in a series and produce a hisogram of the series' distribution. 
    You must also provide a title for the chart produced by this function, which should be a separate variable outside of the function.
    '''
    plt.figure(figsize=(12,8))
    plt.rc('font', size=16)
    plt.hist(x = series, bins = 10, color = 'gray', edgecolor='k', alpha=0.45)
    plt.title(title, y=1.02)
    plt.xlabel('Number of Cases per 100,000',labelpad=20)
    plt.ylabel('Case Count',labelpad=20)
    plt.axvline(series.mean(), color = 'tab:orange', linestyle='dashed', linewidth=5)
    min_ylim_v, max_ylim_v = plt.ylim()
    plt.text(series.mean()*1.05, max_ylim_v*0.9, 'Mean: {:.2f}'.format(series.mean()))
    # plt.axvline(series.median(), color = 'darkgreen', linestyle='dashed', linewidth=5)
    # plt.text(series.median()*.25, max_ylim_v*0.9, 'Median: {:.2f}'.format(series.median()))
    plt.grid(b = True, alpha = .45)
    # plt.figure(figsize = (16, 9))
    plt.tight_layout()
    plt.show()


# Function to produce a jointplot:

def my_plotter(df, col_x, col_x_title, col_y, col_y_title, hue = "bin_svi"):
    '''
    This function will return a jointplot with formatting. 
    You must input the x and y columns, x and y column names, the source dataframe, and the column by which the data has been binned.
    Note that col_x, col_y, col_x_title and col_y_title MUST BE STRINGS.
    '''
        
    p = sns.jointplot(x = col_x, y = col_y, data = df, hue = hue, height = 8, size = 10)
    p.fig.suptitle(f'{col_x_title} vs {col_y_title}', fontsize = 20, fontweight = 'bold')
    p.set_axis_labels(f'{col_x_title}', f'{col_y_title}', fontweight = 'bold')
    p.fig.tight_layout()
    # p.fig.subplots_adjust(top=0.95) # Reduce plot to make room 
    plt.savefig("jointplot_columns")
    plt.show()
    return p

############################################################ Hypothesis Testing ############################################################

def kruskal_test(avg_var1, avg_var2, avg_var3, avg_var4, null, alternate, alpha):
    '''
    Runs non parametric ANOVA when p-value from levene test(variance) is < 0.05
    '''
    alpha = alpha
    f, p = kruskal(avg_var1,avg_var2,avg_var3,avg_var4)
    print('f=', f)
    print('p=', p)
    print('\n')
    if p < alpha:
        print("We reject the null that: \n", null)
        print('\n')
        print("We move forward with the alternative hypothesis that: \n", alternate)
    else:
        print("We fail to reject the null")
        print("Evidence does not support the claim that smoking status and time of day are dependent/related")
    
def pearson(continuous_var1, continuous_var2, null, alternate, alpha):
    '''
    runs pearson r test on 2 continuous variables
    '''
    alpha = alpha
    r, p = stats.pearsonr(continuous_var1, continuous_var2)
    print('r=', r)
    print('p=', p)
    print('\n')
    if p < alpha:
        print("We reject the null that: \n", null)
        print('\n')
        print("We move forward with the alternative hypothesis that: \n", alternate)
    else:
        print("We fail to reject the null")
        print("Evidence does not support the claim that smoking status and time of day are dependent/related")

    