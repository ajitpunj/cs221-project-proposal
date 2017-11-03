31 #import matplotlib.pyplot as plt
import argparse
import csv
import pandas as pd
import numpy as np
import os
from sklearn import datasets,linear_model,preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans

def run(input_args):
    #read the inputs into numpy arrays
    #convert all the non numerical values
    data= pd.read_csv(input_args.feature_file,header=None)
    #turn all 'NaN' values to 0
    data.fillna(0,inplace=True)
    #split out columns into one hot
    #nonlinear are dayofmonth,dayofweek,airline,origin,dest
    nonLinearColumns=[0,1,4,6,7]
    print data
    data= pd.get_dummies(data,columns=nonLinearColumns)
    print data

    features = data.as_matrix()
    #results should all be numbers except for nan
    data = pd.read_csv(input_args.result_file,header=None)
    data.fillna(0,inplace=True)
    results = data.as_matrix()
    #reshape for skikit (NO IDEA WHY but ok...)
    results=results.reshape(-1,1)

    #create linear regression object
    regr = linear_model.LinearRegression()

    #Train the model using the training set
    regr.fit(features,results)

    predictions = regr.predict(features)
    print (predictions)
    print (regr.coef_)

    #k-means
    #shape of fit_predict input is n_samples,n_features
    #http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    print 'k-means'
    y_pred = KMeans(5, 'random').fit(results,features)
    print y_pred.labels_

    #print "coefficients {}".format(regr.coef_)
    return

parser = argparse.ArgumentParser()
parser.add_argument("feature_file", help= "the local path to the feature file. A 2D array of feature vectors")
parser.add_argument("result_file", help= "the local path to the file containing the 1D vector of results pertaining to the features")
args = parser.parse_args()
run(args)
