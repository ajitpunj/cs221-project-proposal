import csv
import argparse
import glob
import collections
import enum

class Row(enum.Enum):
    Year = 0
    Month = 1
    DayOfMonth=2
    DayOfWeek=3
    DepTime=4
    CRSDepTime=5
    ArrTime=6
    CRSArrTime=7
    UniqueCarrier=8
    FlightNum=9
    TailNum=10
    ActualElapsedTime=11
    CRSElapsedTime=12
    AirTime=13
    ArrDelay=14
    DepDelay=15
    Origin=16
    Dest=17
    Distance=18
    TaxiIn=19
    TaxiOut=20
    Cancelled=21
    CancellationCode=22
    Diverted=23
    CarrierDelay=24
    WeatherDelay=25
    NASDelay=26
    SecurityDelay=27
    LateAircraftDelay=28


def run(input_args):
    #open the input file
    #inFile = glob.glob(input_args.input_file)
    #create all the output files to write to
    stats=0
    cancelVect=[]
    delayVect=[]
    featLine=[]
    featFileName = input_args.output_file_base + "_features.csv"
    cancelFileName = input_args.output_file_base + "_cancels.csv"
    delayFileName = input_args.output_file_base + "_delays.csv"
    
    featFile = open(featFileName,'wb')
    cancelFile = open(cancelFileName,'wb')
    delayFile = open(delayFileName,'wb')
    
    featWriter = csv.writer(featFile)
    cancelWriter = csv.writer(cancelFile)
    delayWriter = csv.writer(delayFile)
    #keep track of stats if specified
    if input_args.stats!=None and input_args.stats!=0:
        stats=1
        statDict = collections.defaultdict(int)
        carrierDict = collections.defaultdict(int)
    #line by line input file processing\
    with open(input_args.input_file, 'rb') as csvFile:
        lines=csv.reader(csvFile,delimiter=',')
        for line in lines:
            featLine=\
            line[:Row.DepTime.value]\
            +line[Row.CRSDepTime.value:Row.ArrTime.value]\
            +line[Row.CRSArrTime.value:Row.ActualElapsedTime.value]\
            +line[Row.CRSElapsedTime.value:Row.ArrDelay.value]\
            +line[Row.Origin.value:Row.TaxiIn.value]
            if stats:
                statDict['cancelled']+=int(line[Row.Cancelled.value])
                statDict['total']+=1
                if line[Row.ArrDelay.value]!='NA':                    
                    if int(line[Row.ArrDelay.value])>0:
                        statDict['delayed']+=1
                if len(line[Row.Origin.value])==2:
                    statDict['twoCharOrigin']+=1
                elif len(line[Row.Origin.value])==3:
                    statDict['threeCharOrigin']+=1
                else:
                    statDict['otherCharOrigin']+=1
                carrierDict[line[Row.UniqueCarrier.value]]+=1
            featWriter.writerow(featLine)
            #add to cancel vector to write at end
            cancelVect.append(line[Row.Cancelled.value])
            #add to delay vector to write at end
            delayVect.append(line[Row.ArrDelay.value])
        #print the 1 D cancel and delay values out 
        cancelWriter.writerow(cancelVect[0])
        delayWriter.writerow(delayVect[0])

    if stats:
        print "total flight count is {}".format(statDict['total'])
        print "number of delayed flights is {}".format(statDict['delayed'])        
        print "number of cancelled flights is {}".format(statDict['cancelled'])
        print "number of unique carriers is {}".format(len(carrierDict))
        for carrier in carrierDict:
            print "{} flights for carrier {}".format(carrierDict[carrier],carrier)
        print "number of origin airports with 2 chars is {}".format(statDict['twoCharOrigin'])
        print "number of origin airports with 3 chars is {}".format(statDict['threeCharOrigin'])
        print "number of origin airports with other chars is {}".format(statDict['otherCharOrigin'])
                                                         
    featFile.close()
    cancelFile.close()
    delayFile.close()
    return
    
        
    
parser = argparse.ArgumentParser()
parser.add_argument("input_file", help= "the local path to the input file to act on")
parser.add_argument("output_file_base", help= "the local path to the output base name to write to")
parser.add_argument("-s","--stats", help= "option to print out statistics of the input file",type=int)
args = parser.parse_args()
run(args)
