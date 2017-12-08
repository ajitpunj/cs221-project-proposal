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

def runKMeans(features,results,testFeatures,testResults,n_samples,input_args):
    #k-means
    #shape of fit_predict input is n_samples,n_features
    #http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    print 'k-means'
    results = pred_lib.reshape_results(results)
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
        runLinearModel(np.matrix(feat), np.matrix(res),testFeatures,testResults,input_args)
    print "AGGREGATE STATS FOR K MEANS"
    pred_lib.printAggregateStats(globaldiff,globalpredDelayed,globalrealDelayed,globalpredNotDelayed,globalrealNotDelayed,globalfalsePositives,globalfalseNegatives,globalcorrectPositive,globalcorrectNegative,globalpredictionLen)

    #add cluster number to features vector:
    # labels = y_pred.labels_
    # newFeatures = np.matrix(zip(features,labels))

def TrainPredict(features,results,testFeatures,testResults,input_args):
    if input_args.sgd:
        if input_args.classifier:
            regr = pred_lib.getTrainedSGDClassifierModel(features,results.ravel(),input_args.grid_search)
        else:
            regr = pred_lib.getTrainedSGDRegressorModel(features,results.ravel(),input_args.grid_search)
    elif input_args.randomforest:
        if input_args.classifier:
            regr = pred_lib.getTrainedRandomForestClassifier(features,results,input_args.grid_search)
        else:
            regr = pred_lib.getTrainedRandomForestModel(features,results,input_args.grid_search)
    elif input_args.bayes:
        if input_args.classifier:
            regr = pred_lib.getTrainedNaiveBayes(features,results,input_args.grid_search)
        else:
            regr = pred_lib.getTrainedBayesianRidge(features,results,input_args.grid_search)
    else: #linear regression
        if input_args.classifier:
            print "YOU PROBABLY DIDN'T MEAN TO CLASSIFY USING THE NORMAL REGRESSION"
        regr = pred_lib.getTrainedLinearModel(features,results)

    predictions = regr.predict(testFeatures)
    results = testResults
    
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
def run(input_args):
    #nonlinear are dayofmonth,dayofweek,airline,origin,dest
    nonLinearColumns=[0,1,2,3,4,6,7]
    
    #read all the features into a dataframe
    featureDF=pred_lib.CreateDF(input_args.feature_file)

    #add a column that is just a constant 1 feature, as linear requires x1,x2,1
#    featureDF['const']=1
    
    if input_args.omit_features:
        #remove extraneous features #NOTE this seems to help random forest delay predictions
        featureDF = featureDF.drop(3,1) #drop arrival time
        featureDF = featureDF.drop(8,1) #drop distance
        featureDF = featureDF.drop(0,1) #drop day of month
        featureDF = featureDF.drop(1,1) #drop day of week
        
        nonLinearColumns=[2,4,6,7]
        
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
    if input_args.validation: #for validation, train on first 70% and test on next 20%. save 10% for test
        trainingStart =0
        trainingEnd=int(len(featureDF) *.7)
        testStart = trainingEnd
        testEnd=int(len(featureDF) *.9)
    elif input_args.test: #train on first 90% and then test on last 10%
        trainingStart = 0
        trainingEnd= int(len(featureDF) *.9)
        testStart = trainingEnd
        testEnd=int(len(featureDF))
    else: #else we are just doing training on training, and they are the same
        trainingStart = 0
        trainingEnd= int(len(featureDF) *.7)
        testStart=trainingStart
        testEnd=trainingEnd

    #portion out the training and test parts vectors based on the slices set above
    print "Training Range {}:{}".format(trainingStart,trainingEnd)
    print "Predicting Range {}:{}".format(testStart,testEnd)

    trainingDF = featureDF[trainingStart:trainingEnd]
    testDF = featureDF[testStart:testEnd]

    resultsCol = len(trainingDF.columns)-1
    results = trainingDF[resultsCol]
    features = trainingDF.drop(trainingDF.columns[resultsCol],axis=1)


    resultsCol = len(testDF.columns)-1
    testRes = testDF[resultsCol]
    testFeat =  testDF.drop(testDF.columns[resultsCol],axis=1)

    results = results.as_matrix()
    features = features.as_matrix()
    testRes = testRes.as_matrix()
    testFeat = testFeat.as_matrix()
    
    #Everything needs the results reshaped so far
    results = pred_lib.reshape_results(results).ravel()
    testRes = pred_lib.reshape_results(testRes).ravel()
    if input_args.baseline_oracle:
        pred_lib.BaselineOracle(testRes,input_args.classifier)
    elif input_args.kmeans:
        runKMeans(features,results,testFeat,testRes,5,input_args)
    else:
        TrainPredict(features,results,testFeat,testRes,input_args)

    return

parser = argparse.ArgumentParser()
parser.add_argument("feature_file", help= "the local path to the feature file. A 2D array of feature vectors")
parser.add_argument("result_file", help= "the local path to the file containing the 1D vector of results pertaining to the features")
parser.add_argument("-l","--linearize", help= "flag to linearize object column data instead of splitting columns out into indicator feature vectors",action="store_true")
parser.add_argument("-o","--omit_features", help= "remove the black listed (hard coded) features from the DF",action="store_true")
parser.add_argument("-bo","--baseline_oracle", help= "stats for the baseline and oracle on given data range",action="store_true")
parser.add_argument("-p","--plot", help= "plot predicted vs actual",action="store_true")
parser.add_argument("-k","--kmeans", help= "use kmeans then linear regression on clusters",action="store_true")
parser.add_argument("-s","--sgd", help= "use SGD regression with squared loss instead of generic linear",action="store_true")
parser.add_argument("-b","--bayes", help= "Use bayesian ridge predictor or naive guassian bayesian classifier",action="store_true")
parser.add_argument("-rf","--randomforest", help= "use random forest regression",action="store_true")
parser.add_argument("-c","--classifier", help= "use a classifier instead of a regressor. Either sgd or random forest must also be specified and should be used for cancellations not delays",action="store_true")
parser.add_argument("-pt","--plot_title", help= "title for generated plot")
parser.add_argument("-t","--test", help= "predict test data against trained data (including validation",action="store_true")
parser.add_argument("-v","--validation", help= "predict validation data against trained data",action="store_true")
parser.add_argument("-gt","--grid_search",help="use grid search to optimize parameters for the given model and feature set")                    
args = parser.parse_args()
run(args)
