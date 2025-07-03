#coding:utf-8

from numpy import *
import os
import re
import pdb
import random


class BayesClassifier:

	def __init__(self, filePath):   	#filePath ：{'type1':'path1','type2':'path2'...}
		self.filePath =filePath 		#specified as file folder paths of different classification path
		
		
	def vocabularyGen(self):
		voc = set()
		for key in self.filePath:
			fileList = os.listdir(self.filePath[key])
			for fileName in fileList:
				fullFileName = self.filePath[key] + '\\' + fileName
				file = open(fullFileName)
				wordStr = file.read()
				file.close()
				wordList = re.split('\\W*',wordStr)  #remove the non-alphabet words
				voc = voc | set([x for x in wordList])
		self.vocabulary = list(voc)
		
	def matAndLableGen(self):
		self.dataMat = []
		self.lable = []
		for key in self.filePath:
			fileList = os.listdir(self.filePath[key])
			for fileName in fileList:
				self.lable.append(key)
				fullFileName = self.filePath[key] + '\\' + fileName
				file = open(fullFileName)
				wordStr = file.read()
				file.close()
				wordList = re.split('\\W*',wordStr)  #remove the non-alphabet words
				dataVec = [0]*len(self.vocabulary)
				for word in wordList:
					dataVec[self.vocabulary.index(word)] = 1
				self.dataMat.append(dataVec)
				
	def NBTrain(self,dataMat,lable):
	#def NBTrain(self):
		#dataMat = self.dataMat
		#lable = self.lable
		pVec = {}
		pType = {}
		totalWordCount = {}
		types = list(set(lable))
		for i in range(0,len(dataMat)):
			if not lable[i] in pVec.keys():
				pVec[lable[i]] = zeros(len(self.vocabulary))
				totalWordCount[lable[i]] = 0
				pType[lable[i]] = 0
			pVec[lable[i]] += array(dataMat[i])										#add up each sample vector
			totalWordCount[lable[i]] += float(sum(dataMat[i]))						#computing total words in each 
			pType[lable[i]] += 1
		for type in types:
			pVec[type] = log((pVec[type] + 1)/float(totalWordCount[type] + len(self.vocabulary)))	#computing probability for each word in vocabulary with laplace smooth
			pType[type] = log(pType[type]/float(len(lable)))
		self.pVec = pVec
		self.pType = pType
	
	def prediction(self,dataVec):
		result = {}
		for type in self.pType.keys():
			if not type in result.keys():
				result[type] = 0
			result[type] = sum(dataVec*self.pVec[type]) + self.pType[type]
		resultType = result.keys()[0]
		for type in result.keys():
			if result[type] > result[resultType]:
				resultType = type
		return type
			

	
	def TestError(self):
		self.vocabularyGen()			#preparation works
		self.matAndLableGen()		#preparation works
		
		testPartion = 0.1
		fullDataMat = list(self.dataMat)
		fullLable = list(self.lable)
		trainDataMat = []
		testDataMat = []
		trainLable = []
		testLable = []
		numForTest = int(testPartion*len(self.lable))
		for i in range(0,numForTest):
			indexForTest = random.randint(0,len(fullLable) - 1)
			testDataMat.append(fullDataMat[indexForTest])
			testLable.append(fullLable[indexForTest])
			del fullDataMat[indexForTest]
			del fullLable[indexForTest]
		trainDataMat = fullDataMat
		trainLable = fullLable
		self.NBTrain(trainDataMat, trainLable)
		error = 0
		for j in range(0,len(testDataMat)):
			predictionType = self.prediction(array(testDataMat[j]))
			if predictionType != testLable[j]:
				error += 1
		print('Number of test sample: %d. Error portion: %f.'%(numForTest, error))
			
if __name__ == '__main__':
	input = {}
	input[1] = os.path.dirname(os.path.abspath(__file__)) + '\\email\\spam'
	input[0] = os.path.dirname(os.path.abspath(__file__)) + '\\email\\ham'
	b = BayesClassifier(input)
	b.TestError()
	#print b.vocabulary