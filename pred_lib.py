import argparse
import csv
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn import datasets,linear_model,preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier


#return a pandas data frame of the given csv file (local path)
def CreateDF(filePath):
    return pd.read_csv(filePath,header=None)

#takes in a pandas dataframe and replaces all NaN values with the given value (default of 0)
def RemoveNA(df,value=0):
    df.fillna(value,inplace=True)
    
#takes in a pandas dataframe and converts non linear columns into mulitple columns of indicator features. Takes an array of column indices
def IndicatorFeatures(df,colList):
    return pd.get_dummies(df,columns=colList)


#takes in a pnadas dataframe and converts object columns to integers using an autogenerated map. Without this a string will break the scikit library. i.e insert consistent but arbitrary numbers for ordinal data
#NOTE if we need to go back later we can store the labelenoder object in a map and use le.inverse_transform
def LinearizeFeatures(df):
    for column in df:
        if df[column].dtype==object:
            #le = preprocessing.LabelEncoder()
            df[column]=preprocessing.LabelEncoder().fit_transform(df[column])
    return df

#For some reason we need to reshape our 1D results vector for scikit
def reshape_results(results):
    return results.reshape(-1,1)
#trains a linear regression model based on the inputs
#returns the model to be used for predictions
#NOTE this needs reshaped results vector

def getTrainedLinearModel(features,results):
    regr = linear_model.LinearRegression()
    regr.fit(features,results)
    return regr

def getTrainedSGDRegressorModel(features,results):
    regr = linear_model.SGDRegressor(loss="squared_loss",penalty=None)
    regr.n_iter = np.ceil(10**6/len(results))#deprecated
    #need to scale the features    
    regr.fit(features,results)
    return regr

def getTrainedSGDClassifierModel(features,results):
    regr = linear_model.SGDClassifier(loss="hinge",penalty=None)
    regr.n_iter = np.ceil(10**6/len(results))#deprecated
    regr.fit(features,results)
    return regr

def getTrainedRandomForestModel(features,results):
    regr = RandomForestRegressor(n_jobs=2,random_state=0)
    regr.fit(features,results)
    return regr

def getTrainedRandomForestClassifier(features,results):
    regr = RandomForestClassifier(n_jobs=2,random_state=0)
    regr.fit(features,results)
    return regr
#def trainPolyModel(features,results):

#def trainRandomForestModel(features,results):

def plotResults(predictions,actual,title=None):
    fig,ax =plt.subplots()
    if title != None:
        plt.title(title)
    else:
        plt.title('Predicted Values vs Actual')
        
    ax.plot(predictions,actual,'bo')
    ax.set_xlabel("Predictions")
    ax.set_ylabel("Actual")
    fig.show()    
    plt.show()

def printStats(predictions,actual):
    diff = 0
    predDelayed=0
    realDelayed=0
    predNotDelayed=0
    realNotDelayed=0
    falsePositives=0
    falseNegatives=0
    correctPositive=0
    correctNegative=0

    for x in range (0,len(predictions)):
        diff += abs(predictions[x]-actual[x])
        if predictions[x]<=0:
            predNotDelayed+=1
        else:
            predDelayed+=1
            
        if actual[x]<=0:
            realNotDelayed+=1
        else:
            realDelayed+=1

        if predictions[x]>0 and actual[x]<=0:
            falsePositives+=1
        if predictions[x]<=0 and actual[x]>0:
            falseNegatives+=1
        if predictions[x]<=0 and actual[x]<=0:
            correctNegative+=1
        if predictions[x]>0 and actual[x]>0:
            correctPositive+=1

    print "The R2 Score from the regression is {}".format(r2_score(actual,predictions))
        
    print "The average difference between real and predicted is {}".format(diff/len(predictions))
    if realDelayed>0:
        print "correct positive predictions {} correctPos/real {}".format(correctPositive,correctPositive /(1.0*realDelayed))
    if realNotDelayed>0:
        print "correct negative predictions {} correctNeg/real {}".format(correctNegative,correctNegative/(1.0*realNotDelayed))

    if predDelayed >0:
        print "false positives (predicted to happen but didnt) {} percent of predicted that were wrong {}".format(falsePositives,falsePositives/(1.0*predDelayed))

    if predNotDelayed>0:
        print "false negatives actually happend but predicted to be ok) {} percent of not predicted that were wrong {}".format(falseNegatives,falseNegatives/(1.0*predNotDelayed))

    return
        
