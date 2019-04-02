import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import svm
from sklearn import cross_validation
from sklearn import model_selection as ms
from mpl_toolkits.mplot3d import Axes3D



# read the data
df = pandas.read_csv('techsectordatareal.csv')
ndxtdf = pandas.read_csv('GSPC.csv')
#ndxtdf.set_index('Date',inplace = True)

daysAhead = 270

# calculate price volatility of previous n1 days
def calcPriceVolatility(numDays, priceArray):
	global daysAhead
	# make price volatility array
	volatilityArray = []
	movingVolatilityArray = []
	for i in range(1, numDays+1):
		percentChange = 100 * (priceArray[i] - priceArray[i-1]) / priceArray[i-1]
		movingVolatilityArray.append(percentChange)
	volatilityArray.append(np.mean(movingVolatilityArray))
	for i in range(numDays + 1, len(priceArray) - daysAhead):
		del movingVolatilityArray[0]
		percentChange = 100 * (priceArray[i] - priceArray[i-1]) / priceArray[i-1]
		movingVolatilityArray.append(percentChange)
		volatilityArray.append(np.mean(movingVolatilityArray))

	return volatilityArray

# calculate momentum of previous n1 days
def calcMomentum(numDays, priceArray):
	global daysAhead
	# now calculate momentum
	momentumArray = []
	movingMomentumArray = []
	for i in range(1, numDays + 1):
		movingMomentumArray.append(1 if priceArray[i] > priceArray[i-1] else -1)
	momentumArray.append(np.mean(movingMomentumArray))
	for i in range(numDays+1, len(priceArray) - daysAhead):
		del movingMomentumArray[0]
		movingMomentumArray.append(1 if priceArray[i] > priceArray[i-1] else -1)
		momentumArray.append(np.mean(movingMomentumArray))

	return momentumArray

def TrainAndPredict(permno, numDays, sectorVolatility, sectorMomentum):
	global df
	global daysAhead
	# get price volatility and momentum for this company
	#companyData = df[df['PERMNO'] == permno]
	companyPrices = list(df[df['PERMNO'] == permno]['PRC'])

	volatilityArray = calcPriceVolatility(numDays, companyPrices)
	momentumArray = calcMomentum(numDays, companyPrices)


	# since they are different lengths, find the min length
	if len(volatilityArray) > len(sectorVolatility):
		difference = len(volatilityArray) - len(sectorVolatility)
		del volatilityArray[:difference]
		del momentumArray[:difference]

	elif len(sectorVolatility) > len(volatilityArray):
		difference = len(sectorVolatility) - len(volatilityArray)
		del sectorVolatility[:difference]
		del sectorMomentum[:difference]

	# create the feature vectors X
	X = np.transpose(np.array([volatilityArray, momentumArray, sectorVolatility, sectorMomentum]))

	# create the feature vectors Y
	Y = []
	for i in range(numDays, len(companyPrices) - daysAhead):
		Y.append(1 if companyPrices[i+daysAhead] > companyPrices[i] else -1)
	#print(len(Y))

	# fix the length of Y if necessary
	if len(Y) > len(X):
		difference = len(Y) - len(X)
		del Y[:difference]

	# split into training and testing sets
	X_train, X_test, y_train, y_test = ms.train_test_split(X, Y, test_size=0.3)

	# fit the model and calculate its accuracy
	rbf_svm = svm.SVC(kernel='rbf')
	rbf_svm.fit(X_train, y_train)
	score = rbf_svm.score(X_test, y_test)
	#print(score)
	return score
def syncdata():
	global df
	global ndxtdf
	
	
	#minDateIndex = ndxtdf['Date'].min()
	#maxDateIndex = ndxtdf['Date'].max()
	#minDateStocks = df['date'].min()
	#maxDateStocks = df['date'].max()
	ndxtdf = ndxtdf[ndxtdf['Date'] >= '2007-01-03']
	ndxtdf = ndxtdf[ndxtdf['Date'] <= '2014-12-31']


def main():
	global df
	global ndxtdf
	syncdata()

	# find the list of companies
	permnoList = sorted(set(list(df['PERMNO'])))
	companiesNotFull = [12084, 13407, 14542, 93002, 15579] # companies without full dates
	# read the tech sector data
	ndxtPrices = list(ndxtdf['Close'])
	# we want to predict where it will be on the next day based on X days previous
	numDaysArray = [5, 10, 20, 90, 270] # day, week, month, quarter, year
	predictionDict = {}
	acc = []
	n1 = []
	n2 = []
	# iterate over combinations of n_1 and n_2 and find prediction accuracies
	for numDayIndex in numDaysArray:
		ndxtVolatilityArray = calcPriceVolatility(numDayIndex, ndxtPrices)
		ndxtMomentumArray = calcMomentum(numDayIndex, ndxtPrices)
		for numDayStock in numDaysArray:
			#print("n1", numDayIndex,"\tn2",numDayStock)
			predictionForGivenNumDaysDict = {}

			for permno in permnoList:
				if permno in companiesNotFull:
					continue
				#print(permno)
				percentage = TrainAndPredict(permno,numDayStock,ndxtVolatilityArray,ndxtMomentumArray)
				predictionForGivenNumDaysDict[permno] = percentage
				#print("permno", permno, " accuracy ", percentage)


			predictionAccuracies = list(predictionForGivenNumDaysDict.values())
			meanAccuracy = np.mean(predictionAccuracies)
			n1.append(numDayIndex)
			n2.append(numDayStock)
			acc.append(meanAccuracy)
			#maxIndex = max(predictionForGivenNumDaysDict, key=predictionForGivenNumDaysDict.get)
			#maxAccuracy = (maxIndex, predictionForGivenNumDaysDict[maxIndex])
			#minIndex = min(predictionForGivenNumDaysDict, key=predictionForGivenNumDaysDict.get)
			#minAccuracy = (minIndex, predictionForGivenNumDaysDict[minIndex])
			#median = np.median(predictionAccuracies)
			#print("accuracy", meanAccuracy)
			#numDaysTuple = (numDayIndex, numDayStock)
			#predictionDict[numDaysTuple] = {'mean':meanAccuracy, 'max':predictionForGivenNumDaysDict[maxIndex], 'min':predictionForGivenNumDaysDict[minIndex], 'median':median }


	data = {'Accuracy' : acc, 'N_1' : n1, 'N_2' : n2}
	graph = pandas.DataFrame(data)
	graph.to_csv('graph.csv')
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	#ax.bar3d(acc, n1, n2, c='r')
	ax.scatter(acc, n1, n2, c='r', marker='o')

	ax.set_xlabel('Accuracy')
	ax.set_ylabel('n_1')
	ax.set_zlabel('n_2')

	plt.show()

if __name__ == "__main__": 
	main()
