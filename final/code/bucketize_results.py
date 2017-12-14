import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("input_file", help= "local path to csv of delay values")
parser.add_argument("output_file", help= "the local path for the output file to be written to")
parser.add_argument("-b","--binary", help= "Classify as over/under 30 minutes, otherwise 30 minute increments",action="store_true")

args = parser.parse_args()
outFile = open(args.output_file,'wb')
writer = csv.writer(outFile)

newCSV=[]
with open(args.input_file,'rb') as inFile:
    lines = csv.reader(inFile,delimiter=',')
    for line in lines: #should just be one .....
        for entry in line:
            if args.binary:
                if entry == 'NA':
                    newCSV.append(0)
                elif int(entry) >=15:
                    newCSV.append(1)
                else:
                    newCSV.append(0)
            else: #round down to 30 minute increments
                newval = int(entry)/30
                newCSV.append(newval)

writer.writerow(newCSV)
