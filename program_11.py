#!/bin/env python
# Add your own header comments

# template created by Dr. Cherkauer; updated by wagne216
# ClipData and ReadData functions copied from program_10.py

#This script will import data(.txt) included in folder and make nice figures
# -5-yr stream flow (daily, yearly), CoV, TQmean, RBIndex

# modules:
import pandas as pd
import scipy.stats as stats
import numpy as np
from scipy.stats import skew 
import matplotlib.pyplot as m # plots
import pylab as p # plot save
import dateutil.parser as parser

def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    availabiel.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
    
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')
    
    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""

    # INPUT: DF, 2 dates
    # redefine MV to make it local since it's not a direct input to this function
    MissingValues = DataDF["Discharge"].isna().sum() 
    
    # given the dict, want to apply clip to each dataframe
    b = DataDF.index < startDate # before
    a = DataDF.index > endDate # after
    DataDF = DataDF[startDate:endDate] # clip
    # OUTPUT 1: Dataframe (clipped)
    # must add to missing val's based on filename, but then also redefine as the output
    MissingValues = MissingValues + b.sum() + a.sum() # add count clipped
    # OUTPUT 2: add to mv dictionary for specific file

    return( DataDF, MissingValues )

def ReadMetrics( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    the metrics from the assignment on descriptive statistics and 
    environmental metrics.  Works for both annual and monthly metrics. 
    Date column should be used as the index for the new dataframe.  Function 
    returns the completed DataFrame."""
    
    # INPUT: NAME
    # use row titles as index, columns for diff sites labeled as header
    # - want to create diff series based on station name
    tmpDF = pd.read_csv(fileName,index_col=0,delimiter=',',header=0)
    
    # possible to automcatally sort if riverName = station in dataset:
    DataDF = tmpDF[tmpDF['Station'].str.match(river)]
    # OUTPUT: DF
    
    return( DataDF )

# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define full river names as a dictionary so that abbreviations are not used in figures
    riverName = { "Wildcat": "Wildcat Creek",
                  "Tippe": "Tippecanoe River" }
    
    fileName = { "Wildcat":"WildcatCreek_Discharge_03335000_19540601-20200315.txt",\
                "Tippe":"TippecanoeRiver_Discharge_03331500_19431001-20200315.txt"}
    
    fileMetrics = ["Monthly_Metrics.csv",\
                   "Annual_Metrics.csv"]

    # define blank dictionaries (these will use the same keys as fileName)
    DataDF = {}
    MissingValues = {}
    MonthlyMetricsDF = {}
    AnnualMetricsDF= {}  
    
    # import annual & monthly metrics file from program 10, each river = series in respective dataframe
    for river in riverName.keys():
        # import metrics generated in program_10
        MonthlyMetricsDF[river] = ReadMetrics(fileMetrics[0]) # index = rowname
        AnnualMetricsDF[river] = ReadMetrics(fileMetrics[1]) # index = month of year
    
    # import full datafiles: 
    for file in fileName:
            DataDF[file], MissingValues[file] = ReadData(fileName[file])
            # clip to consistent period
            DataDF[file], MissingValues[file] = ClipData( DataDF[file], '1969-10-01', '2019-09-30' )
# %%
    # PLOTS: (plot both rivers on same axis for easi)
    # 1. daily flow for last 5 years
    m.figure(figsize=(10,6)) # makes larger to fit ppt slide
    m.rc('font',size=20) # adjust all font
    m.rc('legend', fontsize=15) # reduce legend font size
    m.tight_layout() # so nothing is cut off
    DataDF['Wildcat']['Discharge']['2015':'2019'].plot(label='Wildcat River',style='orange')
    DataDF['Tippe']['Discharge']['2015':'2019'].plot(label='Tippecanoe River',style='blue')
    m.legend()
    m.ylim(0,14000)
    m.title('Daily Flow 2015-2020')
    loc, lbls = m.yticks()
    m.yticks(loc,labels=loc/1000) # to reduce space the numbers take up
#    m.ticklabel_format(axis='y',style='plain')
    m.ylabel('Flow (ft$^{3}$/s)x10$^{3}$')
    p.savefig('DailyStreamflow.png',format='png',dpi=100) # 
    m.show() 
# %%
    # 2. annual CoV
    m.figure(figsize=(10,6)) # makes larger to fit ppt slide
    m.rc('font',size=20) # adjust all font
    m.rc('legend', fontsize=15) # reduce legend font size
    m.tight_layout() # so nothing is cut off
    AnnualMetricsDF['Wildcat']['Coeff Var'].plot(label='Wildcat River',color='orange',marker='o')
    AnnualMetricsDF['Tippe']['Coeff Var'].plot(label='Tippecanoe River',color='blue',marker='o')
    m.legend()
    m.title('Annual Coefficient of Variation')
    m.ylabel('CoV')
    # extract years from time start & end
    x1=parser.parse(np.min(AnnualMetricsDF['Wildcat']['R-B Index'].index)).year
    x2=parser.parse(np.max(AnnualMetricsDF['Wildcat']['R-B Index'].index)).year+1
    m.xticks(np.arange(1,50, step=8),\
             labels=np.arange(x1,x2, step=8)) # set yticks
    p.savefig('CoV.png',format='png',dpi=100) # 
    m.show() 
    
    # 3. TQ-mean
    m.figure(figsize=(10,6)) # makes larger to fit ppt slide
    m.rc('font',size=20) # adjust all font
    m.rc('legend', fontsize=15) # reduce legend font size
    m.tight_layout() # so nothing is cut off
    AnnualMetricsDF['Wildcat']['Tqmean'].plot(label='Wildcat River',color='orange',marker='o')
    AnnualMetricsDF['Tippe']['Tqmean'].plot(label='Tippecanoe River',color='blue',marker='o')
    m.legend()
    m.title('Annual TQ Means')
    m.ylabel('TQ Mean')
    # extract years from time start & end
    x1=parser.parse(np.min(AnnualMetricsDF['Wildcat']['R-B Index'].index)).year
    x2=parser.parse(np.max(AnnualMetricsDF['Wildcat']['R-B Index'].index)).year+1
    m.xticks(np.arange(1,50, step=8),\
             labels=np.arange(x1,x2, step=8)) # set yticks
    p.savefig('TQ.png',format='png',dpi=100) # 
    m.show() 
    
    # 4. RB-Index
    m.figure(figsize=(10,6)) # makes larger to fit ppt slide
    m.rc('font',size=20) # adjust all font
    m.rc('legend', fontsize=15) # reduce legend font size
    m.tight_layout() # so nothing is cut off
    AnnualMetricsDF['Wildcat']['R-B Index'].plot(label='Wildcat River',color='orange',marker='o')
    AnnualMetricsDF['Tippe']['R-B Index'].plot(label='Tippecanoe River',color='blue',marker='o')
    m.legend()
    m.title('Annual R-B Indices')
    m.yticks(np.arange(0,.4, step=0.05)) # set yticks
    m.ylabel('RB Index')
    # extract years from time start & end
    x1=parser.parse(np.min(AnnualMetricsDF['Wildcat']['R-B Index'].index)).year
    x2=parser.parse(np.max(AnnualMetricsDF['Wildcat']['R-B Index'].index)).year+1
    m.xticks(np.arange(1,50, step=8),\
             labels=np.arange(x1,x2, step=8)) # set yticks
    p.savefig('RB.png',format='png',dpi=100) # 
    m.show() 
    
    # 5. Mean Annual monthly flow- 1. avg each month; avg of those for each year
    mean_yr_mo = {}
    for river in riverName.keys():
        # 1. take mean of eeach month for fulll period
        mean_mo =  DataDF[river]['Discharge'].resample('MS').mean()
        # 2. then mean of months for each water year (starting oct)
        mean_yr_mo[river] = mean_mo.resample('AS-OCT').mean()
    
    m.figure(figsize=(10,6)) # makes larger to fit ppt slide
    m.rc('font',size=20) # adjust all font
    m.rc('legend', fontsize=15) # reduce legend font size
    m.tight_layout() # so nothing is cut off
    mean_yr_mo['Wildcat'].plot(label='Wildcat River',color='orange',marker='o')
    mean_yr_mo['Tippe'].plot(label='Tippecanoe River',color='blue',marker='o')
    m.legend()
    m.title('Annual Monthly Mean Flow')
    m.yticks(np.arange(0, 2000, step=400)) # set yticks
    m.ylabel('Flow (ft$^{3}$/s)')
    p.savefig('MeanMoYr.png',format='png',dpi=100) # 
    m.show() 

    # 6. Annual peak returns
    # sort annual peak flow (L to H)
    sW= AnnualMetricsDF['Wildcat'].sort_values(['Peak Flow'],ascending=False)
    sT= AnnualMetricsDF['Tippe'].sort_values(['Peak Flow'],ascending=False)
    # add row of N's correponding to the order of flows then make it the index
    pW = pd.DataFrame(sW['Peak Flow'])
    pW['N'] = np.arange(1,51)
    pW = pW.set_index('N')
    pT = pd.DataFrame(sT['Peak Flow'])
    pT['N'] = np.arange(1,51)
    pT = pT.set_index('N')
    # plotting positions: P(x)=m(x)/N+1 ; m = rank; N = # obs
    # initialize P column: 
    pW['P'] = 0 
    pT['P'] = 0
    for row in pW.index:
        pW['P'].loc[row]= row/(1+50)
        pT['P'].loc[row]= row/(1+50)
    # scatterplot of P vs Peak Flow for each 
    # reindex for easier plottin'
    pW = pW.set_index('P')
    pT = pT.set_index('P')
    # plot
    m.figure(figsize=(10,6)) # makes larger to fit ppt slide
    m.rc('font',size=20) # adjust all font
    m.rc('legend', fontsize=15) # reduce legend font size
    m.tight_layout() # so nothing is cut off
    pW['Peak Flow'].plot(linestyle='',marker='o',style='blue',label='Wildcat River')
    pT['Peak Flow'].plot(linestyle='',marker='o',style='orange',label='Tippecanoe River')
    m.legend()
    m.title('Return period of annual peak flow events')
    m.xticks(np.arange(0, 1.2, step=0.2))
    m.xlabel('Exceedance Proabability')
    m.yticks(np.arange(0, 25000, step=5000)) # set yticks
    m.ylabel('Flow (ft$^{3}$/s)')
    m.rc('font',size=20) # adjust all font
    m.rc('legend', fontsize=15) # reduce legend font size
    m.tight_layout() # so nothing is cut off
    p.savefig('peak.png',format='png',dpi=100) # 
    m.show() 
 
        
    
    
    
    
    
    