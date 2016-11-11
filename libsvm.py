import sys
import numpy as np
import scipy.io as sio
from numpy.linalg import pinv
sys.path.append('libsvm-3.21/python')
#from libsvm-3.21.python.svmutil import *
from svmutil import *
import time



def svm():
	rawTrainingMat = sio.loadmat('phishing-train.mat')
	rawTrainY = rawTrainingMat['label']
	rawTrainX = np.array(rawTrainingMat['features'])
	rawTrainXfeatureDomainSize = []
	for i in range(0,rawTrainX.shape[1]):
		column = rawTrainX[:,i]
		rawTrainXfeatureDomainSize.append(np.unique(column).size)
	rawTrainXfeatureDomainSize = np.array(rawTrainXfeatureDomainSize)
	trainX = []
	for i in range(0,rawTrainX.shape[1]):
		if(rawTrainXfeatureDomainSize[i] == 2):
			column = rawTrainX[:,i]
			column[column == -1] = 0
			trainX.append(list(column))
		elif(rawTrainXfeatureDomainSize[i] == 3):
			column = rawTrainX[:,i]
			column[column != -1] = 0
			column[column == -1] = 1
			trainX.append(list(column))

			column = rawTrainX[:,i]
			column[column != 1] = 0
			column[column == 1] = 1
			trainX.append(list(column))

			column = rawTrainX[:,i]
			column[column == 1] = -1
			column[column == 0] = 1
			column[column == -1] = 0
			trainX.append(list(column))
	trainY = rawTrainY[0]
	#trainY[trainY == -1] = 0
	trainY = list(trainY)
	trainX = list(np.transpose(np.array(trainX)))
	for i in range(0,len(trainX)):
		trainX[i] = list(trainX[i])
	trainX = list(trainX)
	C = 4096
	gamma = 0.25
	prob  = svm_problem(trainY, trainX)
	param = svm_parameter('-t 2 -c {} -g {} '.format(C,gamma))
	m = svm_train(prob, param)


	rawTestingMat = sio.loadmat('phishing-test.mat')
	rawTestY = rawTestingMat['label']
	rawTestX = np.array(rawTestingMat['features'])
	rawTestXfeatureDomainSize = []
	for i in range(0,rawTestX.shape[1]):
		column = rawTestX[:,i]
		rawTestXfeatureDomainSize.append(np.unique(column).size)
	rawTestXfeatureDomainSize = np.array(rawTestXfeatureDomainSize)
	TestX = []
	for i in range(0,rawTestX.shape[1]):
		if(rawTestXfeatureDomainSize[i] == 2):
			column = rawTestX[:,i]
			column[column == -1] = 0
			TestX.append(list(column))
		elif(rawTestXfeatureDomainSize[i] == 3):
			column = rawTestX[:,i]
			column[column != -1] = 0
			column[column == -1] = 1
			TestX.append(list(column))

			column = rawTestX[:,i]
			column[column != 1] = 0
			column[column == 1] = 1
			TestX.append(list(column))

			column = rawTestX[:,i]
			column[column == 1] = -1
			column[column == 0] = 1
			column[column == -1] = 0
			TestX.append(list(column))
	TestY = rawTestY[0]
	#TestY[TestY == -1] = 0
	TestY = list(TestY)
	TestX = list(np.transpose(np.array(TestX)))
	for i in range(0,len(TestX)):
		TestX[i] = list(TestX[i])
	TestX = list(TestX)


	p_label, p_acc, p_val = svm_predict(TestY, TestX, m)
	#print p_label, p_acc, p_val

svm()
