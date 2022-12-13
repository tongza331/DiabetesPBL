import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

## Display all row
def printAll(cmd):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(cmd)

## LabelEncoder
def LabelEncoder(df):
    le = LabelEncoder()
    d = df.select_dtypes(include=['object'])
    cols = d.columns
    for col in cols:
        df[col] = le.fit_transform(df[col])
    return df

## Function to plot
def BarPlot(df,x,y):
    plt.bar(df[x],df[y])
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

def BarPlotCategorical(df,x):
    pd.value_counts(df[x]).plot.bar()
    plt.xticks(rotation=0, ha='right')
    plt.xlabel(x)
    plt.ylabel('count')
    
def ScatterTrend(df,a,b):
    plt.scatter(df[a],df[b]) 
    x = df[a]
    y = df[b]
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),"r--")
    plt.xlabel(a)
    plt.ylabel(b)
    plt.show()

def DealOut(df,x):
    mu = np.average(df[x])
    sigma = np.std(df[x])
    LL = mu - 2*sigma # Lower limit 
    UL = mu + 2*sigma # Upper limit
    df[x]=df[x].clip(LL, UL)