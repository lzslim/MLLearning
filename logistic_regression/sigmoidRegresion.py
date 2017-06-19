import numpy as np
import os
import math
import pdb
import matplotlib.pyplot as plt
def sigmoid(x):
	return 1/(1 + np.exp(-x))
	
def loadData(loc):
	dataSet = []
	labelSet = []
	try:
		file = open(loc)
	except Exception as e:
		print 'open file failed...'
		raise e
	for line in file.readlines():
		singleLine = line.strip().split()
		dataSet.append([1.0, float(singleLine[0]), float(singleLine[1])])
		labelSet.append(int(singleLine[2]))
	file.close()
	return np.mat(dataSet), np.mat(labelSet)
	
def training(fileLoc,times):
	dataSet,lableSet = loadData(fileLoc)
	lableSet = lableSet.transpose()
	m,n = np.shape(dataSet)
	weight = np.ones((n,1))
	step = 0.01
	for i in range(times):
		#pdb.set_trace()
		h = sigmoid(dataSet * weight)
		error = lableSet - h
		weight = weight + step * dataSet.transpose() * error
	return weight

	
def draw(times):
	weights =  training(os.path.dirname(os.path.abspath(__file__)) + '\\testSet.txt',times)
	dataMat, labelMat = loadData(os.path.dirname(os.path.abspath(__file__)) + '\\testSet.txt')
	n = np.shape(dataMat)[0]
	xcord1 = []
	ycord1 = []
	xcord2 = []
	ycord2 = []
	for i in range(n):
		if int(labelMat[0,i]) == 1:
			xcord1.append(dataMat[i,1])
			ycord1.append(dataMat[i,2])
		else:
			xcord2.append(dataMat[i,1])
			ycord2.append(dataMat[i,2])
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1,ycord1,s=30,c='red',marker = 's')
	ax.scatter(xcord2,ycord2,s=30,c='blue')
	x = np.arange(-3.0,3.0,0.1)
	y = (-weights[0,0]-weights[1,0]*x)/weights[2,0]
	ax.plot(x,y)
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.show()
	
def drawTrend(times):
	x0 = []
	x1 = []
	x2 = []
	for i in range(1,times + 1):
		weights =  training(os.path.dirname(os.path.abspath(__file__)) + '\\testSet.txt',i)
		x0.append(weights[0,0])
		x1.append(weights[1,0])
		x2.append(weights[2,0])
	fig = plt.figure()
	x0f = fig.add_subplot(3,1,1)
	x1f = fig.add_subplot(3,1,2)
	x2f = fig.add_subplot(3,1,3)
	x = range(1,times + 1)
	x0f.plot(x,x0)
	x1f.plot(x,x1)
	x2f.plot(x,x2)
	
	plt.show()
if __name__ == "__main__":
	#drawTrend(500)
	draw(2000)
	#print training(os.path.dirname(os.path.abspath(__file__)) + '\\testSet.txt',1000000)