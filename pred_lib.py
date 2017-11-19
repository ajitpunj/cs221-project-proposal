import argparse
import csv
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn import datasets,linear_model,preprocessing,naive_bayes
from sklearn.model_selection import cross_val_predict, GridSearchCV
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

def MatchTrainingToTest(trainingFeat,testingFeat):
    for column in testingFeat:
        if column not in trainingFeat:
            #these are all indicator, so set to 0
            trainingFeat.insert(0,column,0.0)

    for column in trainingFeat:
        if column not in testingFeat:
            testingFeat.insert(0,column,0.0)

    #reorder all of them so they match
    trainingFeat=trainingFeat.reindex_axis(sorted(trainingFeat.columns),axis=1)
    testingFeat=testingFeat.reindex_axis(sorted(testingFeat.columns),axis=1)
    return (trainingFeat,testingFeat)
    
#For some reason we need to reshape our 1D results vector for scikit
def reshape_results(results):
    return results.reshape(-1,1)

def getTrainedNaiveBayes(features,results,gs=0):
    regr=naive_bayes.GaussianNB()
    regr.fit(features,results)
    return regr
def getTrainedBayesianRidge(features,results,gs=0):
#    regr = linear_model.BayesianRidge(lambda_1=1,n_iter=100000)
    regr = linear_model.BayesianRidge()
    regr.fit(features,results)
    return regr

#trains a linear regression model based on the inputs
#returns the model to be used for predictions
#NOTE this needs reshaped results vector

def getTrainedLinearModel(features,results):
    regr = linear_model.LinearRegression()
    regr.fit(features,results)
    return regr

def getTrainedSGDRegressorModel(features,results,gs=0):
    param_grid={
        'loss':['squared_loss','huber','squared_epsilon_insensitive'],
        'penalty':['none','l2','l1'],
        'learning_rate':['constant','optimal','invscaling']
    }
    regr = linear_model.SGDRegressor(loss='squared_loss',penalty='l1',learning_rate='invscaling',eta0=.0001)
#    regr = linear_model.SGDRegressor(loss='squared_loss',penalty=None,eta0=.00001)
    regr.n_iter = np.ceil(10**6/len(results))#deprecated
    if gs:
        CVregr = GridSearchCV(estimator = regr,param_grid=param_grid)
        CVregr.fit(features,results)
        print "best parameters"
        print CVregr.best_params_
    #need to scale the features    
    regr.fit(features,results)
    return regr

def getTrainedSGDClassifierModel(features,results,gs=0):
    param_grid={
        'loss':['squared_loss','huber','squared_epsilon_insensitive'],
        'penalty':['none','l2','l1'],
        'learning_rate':['constant','optimal','invscaling']
    }
    regr = linear_model.SGDClassifier(loss="hinge",penalty=None,eta0=.00001)
    if gs:
        CVregr = GridSearchCV(estimator = regr,param_grid=param_grid)
        CVregr.fit(features,results)
        print "best parameters"
        print CVregr.best_params_
    regr.n_iter = np.ceil(10**6/len(results))#deprecated
    regr.fit(features,results)
    return regr

def getTrainedRandomForestModel(features,results,gs=0):
    param_grid ={
        'n_estimators': [200,500,700,1000],
        'max_features':['auto','sqrt','log2'],
        'criterion' : ['mse','mae']
    }    
    regr = RandomForestRegressor(n_jobs=-1,random_state=0)
    if gs:
        CVregr = GridSearchCV(estimator = regr,param_grid=param_grid)
        CVregr.fit(features,results)
        print "best parameters"
        print CVregr.best_params_
    regr.fit(features,results)
    return regr

def getTrainedRandomForestClassifier(features,results,gs=0):
    regr = RandomForestClassifier(n_jobs=2,random_state=0)
    param_grid ={
        'n_estimators': [200,500,700,1000],
        'max_features':['auto','sqrt','log2'],
        'criterion' : ['mse','mae']
    }
    if gs:
        CVregr = GridSearchCV(estimator = regr,param_grid=param_grid)
        CVregr.fit(features,results)
        print "best parameters"
        print CVregr.best_params_
    regr.fit(features,results)
    return regr
#def trainPolyModel(features,results):

#def trainRandomForestModel(features,results):

def plotResults(predictions,actual,title=None):
    #we want the axis to be the same as we are looking for linear so
    predMax = max(predictions)
    predMin = min(predictions)
    actMax = max(actual)
    actMin = min(actual)

    axisMax = max([predMax,actMax])
    axisMin = max([predMin,actMin])
    fig,ax =plt.subplots()
    ax.set_xlim([axisMin,axisMax])
    ax.set_ylim([axisMin,axisMax])
    
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
    mse = 0
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
        mse += (predictions[x]-actual[x]) *(predictions[x]-actual[x])
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

    print "The MSE between real and predicted is {}".format(mse/len(predictions))
    print "The average difference in minutes between real and predicted is {}".format(diff/len(predictions))
    
    if realDelayed>0:
        print "correct positive predictions {} correctPos/real {}".format(correctPositive,correctPositive /(1.0*realDelayed))
    if realNotDelayed>0:
        print "correct negative predictions {} correctNeg/real {}".format(correctNegative,correctNegative/(1.0*realNotDelayed))

    if predDelayed >0:
        print "false positives (predicted to happen but didnt) {} percent of predicted that were wrong {}".format(falsePositives,falsePositives/(1.0*predDelayed))

    if predNotDelayed>0:
        print "false negatives actually happend but predicted to be ok) {} percent of not predicted that were wrong {}".format(falseNegatives,falseNegatives/(1.0*predNotDelayed))

    return diff, predDelayed,realDelayed,predNotDelayed,realNotDelayed,falsePositives,falseNegatives,correctPositive,correctNegative, len(predictions)
        
def printAggregateStats(diff, predDelayed,realDelayed,predNotDelayed,realNotDelayed,falsePositives,falseNegatives,correctPositive,correctNegative,predictionLen):
    print "The average difference between real and predicted is {}".format(diff/predictionLen)
    if realDelayed>0:
        print "correct positive predictions {} correctPos/real {}".format(correctPositive,correctPositive /(1.0*realDelayed))
    if realNotDelayed>0:
        print "correct negative predictions {} correctNeg/real {}".format(correctNegative,correctNegative/(1.0*realNotDelayed))

    if predDelayed >0:
        print "false positives (predicted to happen but didnt) {} percent of predicted that were wrong {}".format(falsePositives,falsePositives/(1.0*predDelayed))

    if predNotDelayed>0:
        print "false negatives actually happend but predicted to be ok) {} percent of not predicted that were wrong {}".format(falseNegatives,falseNegatives/(1.0*predNotDelayed))

