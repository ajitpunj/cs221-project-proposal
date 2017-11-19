import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_file", help= "the local path to the input file to act on")
parser.add_argument("output_file", help= "the local path to the output file name to write to")
args = parser.parse_args()

outFile = open(args.output_file,'wb')
outWriter = csv.writer(outFile)
outVect=[]
with open(args.input_file,'rb') as csvFile:
    lines = csv.reader(csvFile,delimiter=',')
    for line in lines:
        for val in line:
            if int(val)>180:
                outVect.append(180)
            else:
                outVect.append(int(val))

outWriter.writerow(outVect)
outFile.close()
