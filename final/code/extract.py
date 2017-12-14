
import csv
import argparse
import glob
import collections
import enum

class Row(enum.Enum):
    Year = 0
    Month = 1
    DayOfMonth=2 #start here, all months are the same and year is irrelevant
    DayOfWeek=3
    DepTime=4 #dont take actual
    CRSDepTime=5 
    ArrTime=6 #dont take actual
    CRSArrTime=7
    UniqueCarrier=8
    FlightNum=9
    TailNum=10
    ActualElapsedTime=11 #dont take actual
    CRSElapsedTime=12
    AirTime=13
    ArrDelay=14 #output for delay value
    DepDelay=15
    Origin=16
    Dest=17
    Distance=18
    TaxiIn=19
    TaxiOut=20
    Cancelled=21 #output for cancel
    CancellationCode=22
    Diverted=23
    CarrierDelay=24
    WeatherDelay=25
    NASDelay=26
    SecurityDelay=27
    LateAircraftDelay=28

#so we are taking
#day of month
#day of week
#scheduled departure time
#scheduled arrival time
#airline
#schedule air time
#origin
#dest
#distance 
def run(input_args):
    if input_args.test:
        maxRow = input_args.test
    else:
        maxRow=float("+inf")

    if input_args.filter:
        filt =1
    else:
        filt =0
        
    #dictionary for fast look up of top 10 airports
    topAirports = collections.defaultdict(int)
    topAirports['ATL']=0
    topAirports['LAX']=0
    topAirports['ORD']=0
    topAirports['DFW']=0
    topAirports['JFK']=0
#    topAirports['DEN']=0
#    topAirports['SFO']=0
#    topAirports['LAS']=0
#    topAirports['CLT']=0
#    topAirports['SEA']=0
    
    stats=0
    invalidCount=0
    cancelVect=[]
    delayVect=[]
    featLine=[]
    delayFeatFileName = input_args.output_file_base + "_delayfeatures.csv"
    cancelFeatFileName = input_args.output_file_base + "_cancelfeatures.csv"
    cancelFileName = input_args.output_file_base + "_cancels.csv"
    delayFileName = input_args.output_file_base + "_delays.csv"
    
    delayFeatFile = open(delayFeatFileName,'wb')
    cancelFeatFile = open(cancelFeatFileName,'wb')
    cancelFile = open(cancelFileName,'wb')
    delayFile = open(delayFileName,'wb')
    #create all the output files to write to
    delayFeatWriter = csv.writer(delayFeatFile)
    cancelFeatWriter = csv.writer(cancelFeatFile)
    cancelWriter = csv.writer(cancelFile)
    delayWriter = csv.writer(delayFile)
    #keep track of stats if specified
    if input_args.stats!=None and input_args.stats!=0:
        stats=1
        statDict = collections.defaultdict(int)
        carrierDict = collections.defaultdict(int)
    counter=0
    #line by line input file processing\
    with open(input_args.input_file, 'rb') as csvFile:
        lines=csv.reader(csvFile,delimiter=',')
        for line in lines:
            if filt:
                if line[Row.Origin.value] not in topAirports or line[Row.Dest.value] not in topAirports:
                    continue
            if input_args.airline:
                if line[Row.UniqueCarrier.value]!=input_args.airline:
                    continue

            featLine=\
            line[Row.DayOfMonth.value:Row.DepTime.value]\
            +line[Row.CRSDepTime.value:Row.ArrTime.value]\
            +line[Row.CRSArrTime.value:Row.FlightNum.value]\
            +line[Row.CRSElapsedTime.value:Row.AirTime.value]\
            +line[Row.Origin.value:Row.TaxiIn.value]

            if 'NA' in featLine:
                invalidCount+=1
                continue
            
            #Convert the departure time to hour in day
            if int(featLine[2])%100 >= 30:
                featLine[2] = int(featLine[2])/100 + .5
            else:
                featLine[2] = int(featLine[2])/100
                
            #featLine[2]= int(featLine[2])/100 
            #round down arrival time to hour in the day
            if int(featLine[3])%100 >= 30:
                featLine[3] = int(featLine[3])/100 + .5
            else:
                featLine[3] = int(featLine[3])/100
            #featLine[3]= int(featLine[3])/100
            
            #Cancellations take all the flights that dont have NA for a feature or the cancellation value
            if line[Row.Cancelled.value] != 'NA':
                cancelFeatWriter.writerow(featLine)
                cancelVect.append(line[Row.Cancelled.value])
            else: #if the cancel value is NA then we don't consider it, ignore it
                continue

            #For delays we take all the flights that dont have NA values in the features
            if line[Row.ArrDelay.value] == 'NA' and int(line[Row.Cancelled.value]) != 1:
                continue 
            elif line[Row.ArrDelay.value] == 'NA':
                delayVect.append(250)
            else:
                delayVect.append(line[Row.ArrDelay.value])
            delayFeatWriter.writerow(featLine)

            #update stats
            counter+=1
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
            if counter>=maxRow:
                break
            
        #print the 1 D cancel and delay values out
        cancelWriter.writerow(cancelVect)
        delayWriter.writerow(delayVect)

    if stats:
        print "total flight count is {}".format(statDict['total'])
        print "number of invalid flight features is {}".format(invalidCount)
        print "number of elements in the cancel vector is {}".format(len(cancelVect))
        print "number of elements in the delay vector is {}".format(len(delayVect))
        print "number of delayed flights is {}".format(statDict['delayed'])        
        print "number of cancelled flights is {}".format(statDict['cancelled'])
        print "number of unique carriers is {}".format(len(carrierDict))
        for carrier in carrierDict:
            print "{} flights for carrier {}".format(carrierDict[carrier],carrier)
        print "number of origin airports with 2 chars is {}".format(statDict['twoCharOrigin'])
        print "number of origin airports with 3 chars is {}".format(statDict['threeCharOrigin'])
        print "number of origin airports with other chars is {}".format(statDict['otherCharOrigin'])
        
                                                         
    delayFeatFile.close()
    cancelFeatFile.close()
    cancelFile.close()
    delayFile.close()
    return
    
        
    
parser = argparse.ArgumentParser()
parser.add_argument("input_file", help= "the local path to the input file to act on")
parser.add_argument("output_file_base", help= "the local path to the output base name to write to")
parser.add_argument("-s","--stats", help= "option to print out statistics of the input file",action="store_true")
parser.add_argument("-t","--test", help= "test with only the given number of rows",type=int)
parser.add_argument("-f","--filter", help= "filter for only the top airports",action="store_true")
parser.add_argument("-a","--airline", help= "If specified, only flights provided by the given airline are included")
args = parser.parse_args()
run(args)
