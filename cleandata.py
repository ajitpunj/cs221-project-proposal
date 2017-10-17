import csv
import glob

CSVfiles = glob.glob("csv/*")

#parameters needed, initialize unknowns to 0
requiredMonth = 12
yearCol = 0
monthCol = 0


for file in CSVfiles:
	filename = file.split('/')
	filename1 = 'refined_data_'+filename[-1]
	#TODO fix naming and clean up
	writer = csv.writer(open(filename1, 'wb'))
	
	with open(file, 'rb') as csvfile:
		currentYear = csv.reader(csvfile, delimiter=',')
		firstLine = next(currentYear)
		#store the header/first row and gather important parameters
		for i in range(len(firstLine)):
			line = firstLine[i]
			if line == "Year":
				yearCol = i
			elif line == "Month":
				monthCol = i

		for row in currentYear:
			if row[monthCol] == str(requiredMonth):
				writer.writerow(row)