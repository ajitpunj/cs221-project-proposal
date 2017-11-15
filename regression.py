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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

globaldiff=0
globalpredDelayed=0
globalrealDelayed=0
globalpredNotDelayed=0
globalrealNotDelayed=0
globalfalsePositives=0
globalfalseNegatives=0
globalcorrectPositive=0
globalcorrectNegative=0
globalpredictionLen=0
def updateGlobals(diff, predDelayed,realDelayed,predNotDelayed,realNotDelayed,falsePositives,falseNegatives,correctPositive,correctNegative,predictionLen):
    global globaldiff
    global globalpredDelayed
    global globalrealDelayed
    global globalpredNotDelayed
    global globalrealNotDelayed
    global globalfalsePositives
    global globalfalseNegatives
    global globalcorrectPositive
    global globalcorrectNegative
    global globalpredictionLen

    globaldiff += diff
    globalpredDelayed+=predDelayed
    globalrealDelayed+=realDelayed
    globalpredNotDelayed+=predNotDelayed
    globalrealNotDelayed+=realNotDelayed
    globalfalsePositives+=falsePositives
    globalfalseNegatives+=falseNegatives
    globalcorrectPositive+=correctPositive
    globalcorrectNegative+=correctNegative
    globalpredictionLen+=predictionLen

def runKMeans(features,results,n_samples,input_args):
    #k-means
    #shape of fit_predict input is n_samples,n_features
    #http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    print 'k-means'
    X = np.matrix(zip(results,features))
    y_pred = KMeans(n_samples, 'random').fit(features)
    totalPred = []
    totalRes = []
    for n in range(0,n_samples):
        print 'run number ',n
        feat = []
        res = []
        for y in range(len(y_pred.labels_)):
            if n == y_pred.labels_[y]:
                feat.append(features[y])
                res.append(results[y])
        runLinearModel(np.matrix(feat), np.matrix(res),input_args)
    print "AGGREGATE STATS FOR K MEANS"
    pred_lib.printAggregateStats(globaldiff,globalpredDelayed,globalrealDelayed,globalpredNotDelayed,globalrealNotDelayed,globalfalsePositives,globalfalseNegatives,globalcorrectPositive,globalcorrectNegative,globalpredictionLen)

    #add cluster number to features vector:
    # labels = y_pred.labels_
    # newFeatures = np.matrix(zip(features,labels))

def runLinearModel(features,results,testFeatures,testResults,input_args):
    regr = pred_lib.getTrainedLinearModel(features,results)

    if input_args.test:
        predictions = regr.predict(testFeatures)
        results = testResults
    else:
        predictions = regr.predict(features)
    
    #average difference
    Xval=[]
    Yval=[]
    #There has got to be a better way to convert the predictions vector that gets returned to be clean for plotting....
    for x in range (0,len(predictions)):
        if input_args.plot:
            Xval.append(predictions[x])
            Yval.append(results[x])


    diff, predDelayed,realDelayed,predNotDelayed,realNotDelayed,falsePositives,falseNegatives,correctPositive,correctNegative,predictionLen = pred_lib.printStats(predictions,results)
    updateGlobals(diff, predDelayed,realDelayed,predNotDelayed,realNotDelayed,falsePositives,falseNegatives,correctPositive,correctNegative,predictionLen)

    if input_args.plot:
        pred_lib.plotResults(Xval,Yval,input_args.plot_title)

def runSGDModel(features,results,testFeatures,testResults,input_args):
    #need to scale the input features for SGD, algorithm sensitive
    
    if input_args.classifier:
        regr = pred_lib.getTrainedSGDClassifierModel(features,results.ravel(),input_args.grid_search)
    else:
        regr = pred_lib.getTrainedSGDRegressorModel(features,results.ravel(),input_args.grid_search)

    if input_args.test:
        predictions = regr.predict(testFeatures)
        results = testResults
    else:
        predictions = regr.predict(features)
    
    
    #average difference
    Xval=[]
    Yval=[]
    #There has got to be a better way to convert the predictions vector that gets returned to be clean for plotting....
    for x in range (0,len(predictions)):
        if input_args.plot:
            Xval.append(predictions[x])
            Yval.append(results[x])


    diff, predDelayed,realDelayed,predNotDelayed,realNotDelayed,falsePositives,falseNegatives,correctPositive,correctNegative,predictionLen = pred_lib.printStats(predictions,results)
    updateGlobals(diff, predDelayed,realDelayed,predNotDelayed,realNotDelayed,falsePositives,falseNegatives,correctPositive,correctNegative,predictionLen)

    if input_args.plot:
        pred_lib.plotResults(Xval,Yval,input_args.plot_title)

def runRandomForestModel(features,results,testFeatures,testResults,input_args):

    if input_args.classifier:
        regr = pred_lib.getTrainedRandomForestClassifier(features,results,input_args.grid_search)        
    else:
        regr = pred_lib.getTrainedRandomForestModel(features,results,input_args.grid_search)
    
    if input_args.test:
        predictions = regr.predict(testFeatures)
        results = testResults        
    else:
        predictions = regr.predict(features)

    #average difference
    Xval=[]
    Yval=[]
    #There has got to be a better way to convert the predictions vector that gets returned to be clean for plotting....
    for x in range (0,len(predictions)):
        if input_args.plot:
            Xval.append(predictions[x])
            Yval.append(results[x])


    diff, predDelayed,realDelayed,predNotDelayed,realNotDelayed,falsePositives,falseNegatives,correctPositive,correctNegative,predictionLen = pred_lib.printStats(predictions,results)
    updateGlobals(diff, predDelayed,realDelayed,predNotDelayed,realNotDelayed,falsePositives,falseNegatives,correctPositive,correctNegative,predictionLen)

    if input_args.plot:
        pred_lib.plotResults(Xval,Yval,input_args.plot_title)

    
def runRBFModel(features,results,input_args):
    regr_ply = SVR(kernel='poly',C=1e3,degree=2)
#    regr_rbf = SVR(kernel='rbf',C=1e3,gamma=0.1)
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
        pred_lib.plotResults(xVal_ply,yVal_ply,input_args.plot_title)
    
def run(input_args):
    #nonlinear are dayofmonth,dayofweek,airline,origin,dest
    nonLinearColumns=[0,1,4,6,7]
    
    #read all the features into a dataframe
    featureDF=pred_lib.CreateDF(input_args.feature_file)    
    
    #if the flag is set for linearize, the do that
    if input_args.linearize:
        featureDF = pred_lib.LinearizeFeatures(featureDF)        
    else:#otherwise we split nonlinear columns out into one hot 
        featureDF = pred_lib.IndicatorFeatures(featureDF,nonLinearColumns)

    #scale all of the features#
    scaler=StandardScaler()
#    scaler=MinMaxScaler()
    featureDF_scaled=scaler.fit_transform(featureDF)
    featureDF = pd.DataFrame(featureDF_scaled)
    
    #read all the results into a dataframe
    resultsDF = pred_lib.CreateDF(input_args.result_file)   
    
    #pred_lib.RemoveNA(resultsDF,0)
    #turn 1xn into nx1
    resultsDF = resultsDF.transpose()

    #add the results as a column at the end  
    featureDF[len(featureDF.columns)] = resultsDF      
    #shuffle everything
    featureDF = featureDF.sample(frac=1,random_state=500).reset_index(drop=True)

    #TODO can edit this when doing validation to split differently
    cutoff = int(len(featureDF) *.9)
    trainingDF = featureDF[:cutoff]
    testDF = featureDF[cutoff:]

    resultsCol = len(trainingDF.columns)-1
    results = trainingDF[resultsCol]
    features = trainingDF.drop(trainingDF.columns[resultsCol],axis=1)

    results = results.as_matrix()
    features = features.as_matrix()

    if input_args.test:
        resultsCol = len(testDF.columns)-1
        testRes = testDF[resultsCol]
        testFeat =  testDF.drop(testDF.columns[resultsCol],axis=1)
        testRes = testRes.as_matrix()
        testFeat = testFeat.as_matrix()
    else:
        testRes=None
        testFeat=None

    #Everything needs the results reshaped so far
    results = pred_lib.reshape_results(results).ravel()
    if input_args.test:
        testRes = pred_lib.reshape_results(testRes).ravel()

    if input_args.sgd:
        runSGDModel(features,results,testFeat,testRes,input_args)
    elif input_args.randomforest:
        runRandomForestModel(features,results,testFeat,testRes,input_args)
    elif input_args.kmeans:
        #TODO get n_samples another way
        runKMeans(features,results,10,input_args)
    else:
        runLinearModel(features,results,testFeat,testRes,input_args)

    return

parser = argparse.ArgumentParser()
parser.add_argument("feature_file", help= "the local path to the feature file. A 2D array of feature vectors")
parser.add_argument("result_file", help= "the local path to the file containing the 1D vector of results pertaining to the features")
parser.add_argument("-l","--linearize", help= "flag to linearize object column data instead of splitting columns out into indicator feature vectors",action="store_true")
#parser.add_argument("-o","--output_file", help= "If you want to print results to an output file, give a path")
parser.add_argument("-p","--plot", help= "plot predicted vs actual",action="store_true")
parser.add_argument("-k","--kmeans", help= "use kmeans then linear regression on clusters",action="store_true")
parser.add_argument("-s","--sgd", help= "use SGD regression with squared loss instead of generic linear",action="store_true")
parser.add_argument("-rf","--randomforest", help= "use random forest regression",action="store_true")
parser.add_argument("-c","--classifier", help= "use a classifier instead of a regressor. Either sgd or random forest must also be specified and should be used for cancellations not delays",action="store_true")
parser.add_argument("-pt","--plot_title", help= "title for generated plot")
parser.add_argument("-t","--test", help= "predict test data against trained data",action="store_true")
parser.add_argument("-gt","--grid_search",help="use grid search to optimize parameters for the given model and feature set")                    
args = parser.parse_args()
run(args)
