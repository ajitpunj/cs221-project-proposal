31 #import matplotlib.pyplot as plt
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

def runLinearModel(features,results,input_args):
    regr = linear_model.LinearRegression()
    #reshape for skikit (NO IDEA WHY but ok...)
    results = results.reshape(-1,1)
    regr.fit(features,results)
    #Train the model using the training set
    predictions = regr.predict(features)
    
    #Print the R2 Score
    print "The R2 Score from the regression is {}".format(r2_score(results,predictions))
    #average difference
    diff=0
    Xval=[]
    Yval=[]
    predDelayed=0
    realDelayed=0
    predNotDelayed=0
    realNotDelayed=0
    falsePositives=0
    falseNegatives=0
    correctPositive=0
    correctNegative=0

    if input_args.cutoff:
        cutoff=input_args.cutoff
    else:
        cutoff=0
        
    for x in range (0,len(predictions)):
        diff += abs(predictions[x]-results[x])
        if predictions[x]<=cutoff:
            predNotDelayed+=1
        else:
            predDelayed+=1
            
        if results[x]<=0:
            realNotDelayed+=1
        else:
            realDelayed+=1
                    
        if predictions[x]>cutoff and results[x]<=0:
            falsePositives+=1
        if predictions[x]<=cutoff and results[x]>0:
            falseNegatives+=1
        if predictions[x]<=cutoff and results[x]<=0:
            correctNegative+=1
        if predictions[x]>cutoff and results[x]>0:
            correctPositive+=1
        
        if input_args.plot:
            Xval.append(results[x][0])
            Yval.append(predictions[x][0])
            
    print "The average difference between real and predicted is {}".format(diff/len(predictions))

    print "correct positive predictions {} correctPos/real {}".format(correctPositive,correctPositive /(1.0*realDelayed))
    
    print "correct negative predictions {} correctNeg/real {}".format(correctNegative,correctNegative/(1.0*realNotDelayed))

    print "false positives (predicted to happen but didnt) {} percent of predicted that were wrong {}".format(falsePositives,falsePositives/(1.0*predDelayed))

    print "false negatives actually happend but predicted to be ok) {} percent of not predicted that were wrong {}".format(falseNegatives,falseNegatives/(1.0*predNotDelayed))

    if input_args.plot:        
        fig,ax =plt.subplots()
        ax.plot(Xval,Yval,'bo')
        ax.set_xlabel("Measured")
        ax.set_ylabel("Predicted")
        fig.show()

        plt.show()

    
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
        plt.scatter(xVal_ply,yVal_ply,color='darkorange',label='data')
        
        plt.xlabel('predicted')
        plt.ylabel('actual')
        plt.legend()
        plt.show()
    
def run(input_args):
    #read the inputs into numpy arrays
    featureDF=CreateDF(input_args.feature_file)
    #turn all 'NaN' values to 0
    RemoveNA(featureDF,0)
    #print featureDF
    #if the flag is set for linearize, the do that
    if input_args.linearize:
        featureDF = LinearizeFeatures(featureDF)
    #otherwise we split nonlinear columns out into one hot 
    else:
        #nonlinear are dayofmonth,dayofweek,airline,origin,dest
        nonLinearColumns=[0,1,4,6,7]
        featureDF = IndicatorFeatures(featureDF,nonLinearColumns)

    #print featureDF

    #read in all the results
    resultsDF = CreateDF(input_args.result_file)
    RemoveNA(resultsDF,0)

    #convert to matrices for scikit
    features = featureDF.as_matrix()    
    results = resultsDF.as_matrix()


    #k-means
    #shape of fit_predict input is n_samples,n_features
    #http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    print 'k-means'
    y_pred = KMeans(5, 'random').fit(results,features)
    print y_pred.labels_

    if input_args.rbf:
        runRBFModel(features,results,input_args)
    else:
        runLinearModel(features,results,input_args)
        
    return

parser = argparse.ArgumentParser()
parser.add_argument("feature_file", help= "the local path to the feature file. A 2D array of feature vectors")
parser.add_argument("result_file", help= "the local path to the file containing the 1D vector of results pertaining to the features")
parser.add_argument("-l","--linearize", help= "flag to linearize object column data instead of splitting columns out into indicator feature vectors",action="store_true")
#parser.add_argument("-o","--output_file", help= "If you want to print results to an output file, give a path")
parser.add_argument("-p","--plot", help= "plot predicted to actual",action="store_true")
parser.add_argument("-r","--rbf", help= "use an RBF kernel regression model instead of linear",action="store_true")
parser.add_argument("-c","--cutoff", help= "cutoff value, anything over this value is predicted as delayed / canceled. 0 if not specified",type=float)
args = parser.parse_args()
run(args)
