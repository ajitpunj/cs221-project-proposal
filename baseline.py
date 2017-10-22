import csv
import glob
import random

CSVfiles = glob.glob("refined_data_*.csv")

yearCol = 0
monthCol = 1
dayCol = 2
depDelayCol = 15
cancelledCol = 21

cancelProb = .0218
delayProb = .217 

avgBaselineDelay = 57

iterationVal = 1000

for file in CSVfiles:
	totalCancel = 0
	totalDelay = 0
	for i in range(iterationVal):
		with open(file, 'rb') as csvfile:
			delayPredictError = 0
			correctCancelPredict = 0
			totalRows = 0

			currentYear = csv.reader(csvfile, delimiter=',')

			for row in currentYear:
				#get cancellation and delay data from csv
				realCancelled = int(row[cancelledCol])
				realDelay = 0
				baselineCancel = 0
				baselineDelay = 0
				#only get realDelay if cancelled column is 0, if it is 1 then delay will be NA
				if realCancelled == 0:
					realDelay = int(row[depDelayCol])

				#predict cancel with certain probability, else predict a delay of 57 min
				if random.random() < cancelProb:
					baselineCancel = 1
				elif random.random() < delayProb:
					baselineDelay = avgBaselineDelay
				#check if actual matches baseline predictions:
				if baselineCancel == realCancelled:
					correctCancelPredict += 1
				if baselineDelay != realDelay:
					delayPredictError += (abs(baselineDelay-realDelay))
				totalRows += 1
		totalCancel += correctCancelPredict/totalRows
		totalDelay += float(delayPredictError)/float(totalRows)
		print file, 'iteration ',i,' delay error: ', float(delayPredictError)/float(totalRows), ' correct cancellation prediction: ', correctCancelPredict/totalRows
	print 'avg over all runs: delay - ', totalDelay/iterationVal, 'cancel - ',totalCancel/iterationVal