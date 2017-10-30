#import matplotlib.pyplot as plt
import argparse
import csv
import pandas as pd
import numpy as np
import os
from sklearn import datasets,linear_model,preprocessing
from sklearn.metrics import mean_squared_error, r2_score

def run(input_args):
    #read the inputs into numpy arrays
    #convert all the non numerical values
    data= pd.read_csv(input_args.feature_file,header=None)
    #turn all 'NaN' values to 0
    data.fillna(0,inplace=True)
    #encode every string as a numberical value for columns that have strings (like airline codes etc)
    for column in data:
        le = preprocessing.LabelEncoder()
        if data[column].dtype == object:
            le.fit(data[column])
            data[column]=le.transform(data[column])

    features = data.as_matrix()
    
    #results should all be numbers except for nan
    data = pd.read_csv(input_args.result_file,header=None)
    data.fillna(0,inplace=True)
    results = data.as_matrix()
    #reshape for skikit (NO IDEA WHY but ok...)
    results=results.reshape(-1,1)

#    print type(features)
#    print type(results)
#    print features
#    print features.shape
#    print results.shape

    #create linear regression object
    regr = linear_model.LinearRegression()

    #Train the model using the training set
    regr.fit(features,results)

    predictions = regr.predict(features)
    print (predictions)
    print (regr.coef_)

    #print "coefficients {}".format(regr.coef_)
    return

parser = argparse.ArgumentParser()
parser.add_argument("feature_file", help= "the local path to the feature file. A 2D array of feature vectors")
parser.add_argument("result_file", help= "the local path to the file containing the 1D vector of results pertaining to the features")
args = parser.parse_args()
run(args)
