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
    plt.ylim([-100, 10000])
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
    ax.ax_joint.set_ylim(-500,12000)  
    ax.fig.tight_layout()
    plt.show()

def joint_plot_flag(x,y,df,hue):
    ax2 = sns.jointplot(x = x, 
                        y = y, 
                        data = df, 
                        hue = hue, 
                        height = 10,
                        space = 0,
                        xlim = (-1,12),
                        ylim = (-500,12800))
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
    plt.ylabel('Tract Count',labelpad=20)
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
    plt.ylabel('Tract Count',labelpad=20)
    plt.axvline(series.mean(), color = 'tab:orange', linestyle='dashed', linewidth=5)
    min_ylim_v, max_ylim_v = plt.ylim()
    plt.text(series.mean()*1.05, max_ylim_v*0.9, 'Mean: {:.2f}'.format(series.mean()))
    # plt.axvline(series.median(), color = 'darkgreen', linestyle='dashed', linewidth=5)
    # plt.text(series.median()*.25, max_ylim_v*0.9, 'Median: {:.2f}'.format(series.median()))
    plt.grid(b = True, alpha = .45)
    # plt.figure(figsize = (16, 9))
    plt.ylim([0, 100])
    plt.xlim([0, 12000])
    plt.tight_layout()
    plt.show()


# Function to produce a jointplot:

def my_plotter(df, col_x, col_x_title, col_y, col_y_title, hue = "bin_svi"):
    '''
    This function will return a jointplot with formatting. 
    You must input the x and y columns, x and y column names, the source dataframe, and the column by which the data has been binned.
    Note that col_x, col_y, col_x_title and col_y_title MUST BE STRINGS.
    '''
    p = sns.jointplot(x = col_x, y = col_y, data = df, hue = hue, height = 8, size = 10, xlim = (-1,12),ylim = (-500,12800))
    p.fig.suptitle(f'{col_x_title} vs {col_y_title}', fontsize = 20, fontweight = 'bold')
    p.set_axis_labels(f'{col_x_title}', f'{col_y_title}', fontweight = 'bold',labelpad=20)
    p.fig.tight_layout()
    # p.fig.subplots_adjust(top=0.95) # Reduce plot to make room 
    plt.savefig("jointplot_columns")
    plt.show()
    return p

def cluster_scatter(df, title, col_x, col_x_title, col_y, col_y_title, hue = "bin_svi"):
    '''
    This function will return a scatterplot with formatting. 
    You must input the x and y columns, x and y column names, the source dataframe, and the column by which the data has been binned.
    Note that col_x, col_y, col_x_title and col_y_title MUST BE STRINGS.
    '''
    plt.figure(figsize=(12,8))
    plt.rc('font', size=16)
    sns.scatterplot(x= col_x, 
                    y= col_y,
                    data = df, hue= hue,
                    legend = True)
    plt.rc('font', size=16)
    plt.title(title)
    plt.ylabel(col_y_title,labelpad=20)
    plt.xlabel(col_x_title,labelpad=20)
    plt.ylim([-1000, 15000])
    plt.tight_layout()
    plt.show()

def sns_boxplot_hypothesis(dfx, dfy, xlabel, ylabel, title):
    '''create boxplot for hypothesis test exploration
    '''
    plt.figure(figsize=(12,8))
    plt.rc('font', size=16)
    sns.boxplot(x= dfx, y=dfy)
    plt.title(title)
    plt.xlabel(xlabel,labelpad=20)
    plt.ylabel(ylabel,labelpad=20)
    plt.ylim([-500, 12000])
    plt.tight_layout()
    plt.show()
    
############################################################ Hypothesis Testing ############################################################

def kruskal_test(avg_var1, avg_var2, avg_var3, avg_var4, null, alternate, alpha):
    '''
    Runs non parametric ANOVA when p-value from levene test(variance) is greater than 0.05
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

def anova_test(avg_var1, avg_var2, avg_var3, avg_var4, null, alternate, alpha):
    '''
    Runs ANOVA when p-value from levene test(variance) is less than 0.05
    '''
    alpha = alpha
    f, p = stats.f_oneway(avg_var1,avg_var2,avg_var3,avg_var4)
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

####################################################### Clustering Functions ##############################################

def r2(x, y):
    '''
    Takes in x and y and returns pearsons correlation coefficent
    '''
    return stats.pearsonr(x, y)[0] ** 2

##################################################################################################################

def elbow_plot(X_train_scaled, cluster_vars):
    '''
    Given X_train and cluster variables plots an elbow_plot
    '''
    # elbow method to identify good k for us
    ks = range(1,10)
    
    # empty list to hold inertia (sum of squares)
    sse = []

    # loop through each k, fit kmeans, get inertia
    for k in ks:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X_train_scaled[cluster_vars])
        # inertia
        sse.append(kmeans.inertia_)

    print(pd.DataFrame(dict(k=ks, sse=sse)))

    # plot k with inertia
    plt.figure(figsize=(12,8))
    plt.rc('font', size=16)
    plt.plot(ks, sse, 'bx-')
    plt.xlabel('k',labelpad=20)
    plt.ylabel('SSE',labelpad=20)
    plt.title('Elbow Method to Find Optimal k')
    plt.tight_layout()
    plt.show()

##################################################################################################################

def run_kmeans(X_train, X_train_scaled, k, cluster_vars, cluster_col_name):
    '''
    Creates a kemeans object and creates a dataframe with cluster information
    '''
    # create kmeans object
    kmeans = KMeans(n_clusters = k, random_state = 13)
    kmeans.fit(X_train_scaled[cluster_vars])
    # predict and create a dataframe with cluster per observation
    train_clusters = \
        pd.DataFrame(kmeans.predict(X_train_scaled[cluster_vars]),
                              columns=[cluster_col_name],
                              index=X_train.index)
    
    return train_clusters, kmeans

##################################################################################################################

def kmeans_transform(X_scaled, kmeans, cluster_vars, cluster_col_name):
    '''
    Takes in a dataframe and returns custers that have been predicted on that dataframe
    '''
    kmeans.transform(X_scaled[cluster_vars])
    trans_clusters = \
        pd.DataFrame(kmeans.predict(X_scaled[cluster_vars]),
                              columns=[cluster_col_name],
                              index=X_scaled.index)
    
    return trans_clusters

##################################################################################################################

def get_centroids(cluster_vars, cluster_col_name, kmeans):
    '''
    Takes in kmeans and cluster variables to produce centroids
    '''
    centroid_col_names = ['centroid_' + i for i in cluster_vars]

    centroids = pd.DataFrame(kmeans.cluster_centers_, 
             columns=centroid_col_names).reset_index().rename(columns={'index': cluster_col_name})
    
    return centroids

##################################################################################################################

def add_to_train(train_clusters, centroids, X_train_scaled, cluster_col_name):
    '''
    Takes in a datafrme, clusters, centroids and returns a new dataframe with all information concated together
    '''
    # concatenate cluster id
    X_train2 = pd.concat([X_train_scaled, train_clusters], axis=1)

    # join on clusterid to get centroids
    X_train2 = X_train2.merge(centroids, how='left', 
                            on=cluster_col_name).\
                        set_index(X_train_scaled.index)
    
    return X_train2

##################################################################################################################