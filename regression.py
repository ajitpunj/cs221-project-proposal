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
import pred_lib
from sklearn.preprocessing import StandardScaler




def runLinearModel(features,results,input_args):
    results = pred_lib.reshape_results(results)
    regr = pred_lib.getTrainedLinearModel(features,results)

    predictions = regr.predict(features)
    
    #average difference
    Xval=[]
    Yval=[]
    #There has got to be a better way to convert the predictions vector that gets returned to be clean for plotting....
    for x in range (0,len(predictions)):
        if input_args.plot:
            Xval.append(predictions[x][0])
            Yval.append(results[x][0])


    pred_lib.printStats(predictions,results)

    if input_args.plot:
        pred_lib.plotResults(Xval,Yval)

def runSGDModel(features,results,input_args):
    results = pred_lib.reshape_results(results)
    #need to scale the input features for SGD, algorithm sensitive
    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)
    
    if input_args.classifier:
        regr = pred_lib.getTrainedSGDClassifierModel(features,results.ravel())
    else:
        regr = pred_lib.getTrainedSGDRegressorModel(features,results.ravel())

    predictions = regr.predict(features)
    
    
    #Print the R2 Score

    #average difference
    Xval=[]
    Yval=[]
    #There has got to be a better way to convert the predictions vector that gets returned to be clean for plotting....
    for x in range (0,len(predictions)):
        if input_args.plot:
            Xval.append(predictions[x])
            Yval.append(results[x])


    pred_lib.printStats(predictions,results)

    if input_args.plot:
        pred_lib.plotResults(Xval,Yval)

def runRandomForestModel(features,results,input_args):
    results = pred_lib.reshape_results(results)
    if input_args.classifier:
        regr = pred_lib.getTrainedRandomForestClassifier(features,results.ravel())
    else:
        regr = pred_lib.getTrainedRandomForestModel(features,results.ravel())

    predictions = regr.predict(features)
    
    
    #Print the R2 Score

    #average difference
    Xval=[]
    Yval=[]
    #There has got to be a better way to convert the predictions vector that gets returned to be clean for plotting....
    for x in range (0,len(predictions)):
        if input_args.plot:
            Xval.append(predictions[x])
            Yval.append(results[x])


    pred_lib.printStats(predictions,results)

    if input_args.plot:
        pred_lib.plotResults(Xval,Yval)

    
def runRBFModel(features,results,input_args):
    regr_ply = SVR(kernel='poly',C=1e3,degree=2)
#    regr_rbf = SVR(kernel='rbf',C=1e3,gamma=0.1)
    results = results.reshape(-1,1)
    y_ply=regr_ply.fit(features,results.ravel())
#    y_rbf=regr_rbf.fit(features,results.ravel())
    predictions = regr_ply.predict(features)    
    #Print the R2 Score
    print "The R2 Score from the regression is {}".format(r2_score(results,predictions))
    index=[]
    yVal_ply=[]
    xVal_ply=[]
    correctNegatives=0
    falseNegatives=0
    for x in range(0,len(results)):
        xVal_ply.append(predictions[x])
        yVal_ply.append(results[x])
        if predictions[x]<=0 and results[x] <=0:
            correctNegatives+=1
        if predictions[x]<=0 and results[x] >0:
            falseNegatives+=1

    print "correct neg {}".format(correctNegatives)
    print "false neg {}".format(falseNegatives)
            
        
    if input_args.plot:
        pred_lib.plotResults(xVal_ply,yVal_ply)
    
def run(input_args):
    #read the inputs into numpy arrays
    featureDF=pred_lib.CreateDF(input_args.feature_file)
    #turn all 'NaN' values to 0
    pred_lib.RemoveNA(featureDF,0)
    #print featureDF
    #if the flag is set for linearize, the do that
    if input_args.linearize:
        featureDF = pred_lib.LinearizeFeatures(featureDF)
    #otherwise we split nonlinear columns out into one hot 
    else:
        #nonlinear are dayofmonth,dayofweek,airline,origin,dest
        nonLinearColumns=[0,1,4,6,7]
        featureDF = pred_lib.IndicatorFeatures(featureDF,nonLinearColumns)

    #print featureDF

    #read in all the results
    resultsDF = pred_lib.CreateDF(input_args.result_file)
    pred_lib.RemoveNA(resultsDF,0)

    #convert to matrices for scikit
    features = featureDF.as_matrix()    
    results = resultsDF.as_matrix()

    
#    if input_args.rbf:
#        runRBFModel(features,results,input_args)
    if input_args.sgd:
        runSGDModel(features,results,input_args)
    elif input_args.randomforest:
        runRandomForestModel(features,results,input_args)
    else:
        runLinearModel(features,results,input_args)
        

    #k-means
    #shape of fit_predict input is n_samples,n_features
    #http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    #TODO can we put the k-means stuff in its own function or something? or maybe its own script that produces multiple csv files we can run regressions on?
    print 'k-means'
    results = results.reshape(-1,1)
    y_pred = KMeans(5, 'random').fit(results,features)
    print y_pred.labels_

    return

parser = argparse.ArgumentParser()
parser.add_argument("feature_file", help= "the local path to the feature file. A 2D array of feature vectors")
parser.add_argument("result_file", help= "the local path to the file containing the 1D vector of results pertaining to the features")
parser.add_argument("-l","--linearize", help= "flag to linearize object column data instead of splitting columns out into indicator feature vectors",action="store_true")
#parser.add_argument("-o","--output_file", help= "If you want to print results to an output file, give a path")
parser.add_argument("-p","--plot", help= "plot predicted vs actual",action="store_true")
#parser.add_argument("-r","--rbf", help= "use an RBF kernel regression model instead of linear",action="store_true")
parser.add_argument("-s","--sgd", help= "use SGD regression with squared loss instead of generic linear",action="store_true")
parser.add_argument("-rf","--randomforest", help= "use random forest regression",action="store_true")
parser.add_argument("-c","--classifier", help= "use a classifier instead of a regressor. Either sgd or random forest must also be specified and should be used for cancellations not delays",action="store_true")
args = parser.parse_args()
run(args)
