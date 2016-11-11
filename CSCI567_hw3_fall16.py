import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from numpy.linalg import pinv
sys.path.append('libsvm-3.21/python')
from svmutil import *
import time

gpowerMax = 5


def f(x):
	return 2*x*x


def getPhi(dataset,maxPower):
	Phi = np.ones([dataset.size,maxPower+1])
	for p in range(1,maxPower+1):
		Phi[:,p] = np.power(dataset,p)
	return Phi

def getW(phi,y,lamb):
	return np.dot(np.dot(pinv(np.add(np.dot(np.transpose(phi),phi) , lamb * y.size * np.identity(phi.shape[1]))),np.transpose(phi)),y)	


def getYPred(phi,y):
	return np.dot(phi,w)

def getMSE(phi,y,w,lamb):
	z = np.dot(phi,w) - y
	reg = lamb * np.dot(np.transpose(w),w)
	return np.add(np.dot(np.transpose(z),z)/(z.size) , reg)	

def getMSEConstG(y,ypred):
	return np.dot(np.transpose(y-ypred),(y-ypred))/y.size


def main():
	print "Bias Variance Trade-off\n\n"
	vectorized_f = np.vectorize(f)
	complete_data1_X = np.random.uniform(-1,1,[100,10])
	complete_data1_Y = vectorized_f(complete_data1_X) + np.random.normal(0,0.316,[100,10])
	MSE1 = np.ones([gpowerMax,100])
	MSECG1 = np.ones([1,100])
	EH1 = []
	
	print "PART A) 100 datsets with 10 sample each:\n"
	print "func\tvariance\tbias^2"
	for idx in range(0,100):
		y = np.transpose(complete_data1_Y[idx])
		ypred = 1
		MSECG1[0][idx] = getMSEConstG(y,ypred)
		
	histParams = MSECG1[0]
	plt.hist(histParams,bins = 10)
	plt.title('Part A g1(x)')
	plt.xlabel('MSE bucket')
	plt.ylabel('bucket count')
	plt.show()
	plt.clf()
	variance1CG = 0
	bias_sq1CG = 0
	ypred = np.array([1 for i in range(0,10)])
	expectedypred = np.array([1 for i in range(0,10)])
	for idx in range(0,100):	
		y = np.transpose(complete_data1_Y[idx])
		x = complete_data1_X[idx]
		variance1CG += float(np.dot(np.transpose(ypred - expectedypred),(ypred - expectedypred)))/1000
		bias_sq1CG += float(np.dot(np.transpose(expectedypred - vectorized_f(x)),(expectedypred - vectorized_f(x))))/1000
	print "g1(x)\t",format(variance1CG,'.8f'),"\t",format(bias_sq1CG,'.8f')
	for gpower in range(0,gpowerMax):
		for idx in range(0,100):
			x = complete_data1_X[idx]
			y = np.transpose(complete_data1_Y[idx])
			phi = getPhi(x,gpower)
			w = getW(phi,y,0)
			MSE1[gpower][idx] = getMSE(phi,y,w,0)
			if idx == 0:
				EH1.append(w/100)
			else:
				EH1[-1] += w/100
		histParams = MSE1[gpower]
		plt.hist(histParams,bins = 10)
		plt.title('Part A g'+str(gpower+2)+'(x)')
		plt.xlabel('MSE bucket')
		plt.ylabel('bucket count')
		plt.show()
		plt.clf()
	variance1 = [0 for i in range(0,gpowerMax)]
	for gpower in range(0,gpowerMax):
		for idx in range(0,100):
			x = complete_data1_X[idx]
			y = np.transpose(complete_data1_Y[idx])
			phi = getPhi(x,gpower)
			w = getW(phi,y,0)
			ypred = np.dot(phi,w)
			avgW = EH1[gpower]
			expectedypred = np.dot(phi,avgW)
			z = float(np.dot(np.transpose(ypred - expectedypred),(ypred - expectedypred)))/1000
			variance1[gpower] += z
	bias_sq1 = [0 for i in range(0,gpowerMax)]
	for gpower in range(0,gpowerMax):
		for idx in range(0,100):
			x = complete_data1_X[idx]
			phi = getPhi(x,gpower)
			avgW = EH1[gpower]
			expectedypred = np.dot(phi,avgW)
			z = float(np.dot(np.transpose(expectedypred - vectorized_f(x)),(expectedypred - vectorized_f(x))))/1000
			bias_sq1[gpower] += z
	for i in range(0,len(variance1)):
		print "g"+str(i+2)+"(x)\t",format(variance1[i],'.8f'),"\t",format(bias_sq1[i],'.8f')
	
	axes = plt.gca()
	#axes.set_ylim([0,0.1])
	plt.plot([variance1CG] + variance1)
	plt.suptitle('variance')
	plt.legend(['variance'],loc = 'upper right')
	plt.xlabel('complexity')
	plt.ylabel('variance')
	plt.show()
	plt.clf()
	
	axes = plt.gca()
	#axes.set_ylim([0,0.1])
	plt.plot([bias_sq1CG] + bias_sq1)	
	plt.suptitle('bias^2')
	plt.legend(['bias^2'],loc = 'upper right')
	plt.xlabel('complexity')
	plt.ylabel('bias^2')
	plt.show()
	plt.clf()

	complete_data2_X = np.random.uniform(-1,1,[100,100])
	complete_data2_Y = vectorized_f(complete_data2_X) + np.random.normal(0,0.316,[100,100])
	print "PART B) 100 datsets with 100 sample each:\n"
	lambList = [0,0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
	for lamb in lambList:
		print "lambda = ",lamb
		MSE2 = np.ones([gpowerMax,100])
		MSECG2 = np.ones([1,100])
		for idx in range(0,100):
			y = np.transpose(complete_data2_Y[idx])
			ypred = 1
			MSECG2[0][idx] = getMSEConstG(y,ypred)
		if lamb == 0 :
			histParams = MSECG2[0]
			plt.hist(histParams,bins = 10)
			plt.title('Part B g1(x)')
			plt.xlabel('MSE bucket')
			plt.ylabel('bucket count')
			plt.show()
			plt.clf()
		variance2CG = 0
		bias_sq2CG = 0
		ypred = np.array([1 for i in range(0,100)])
		expectedypred = np.array([1 for i in range(0,100)])
		for idx in range(0,100):	
			y = np.transpose(complete_data2_Y[idx])
			x = complete_data2_X[idx]
			variance2CG += float(np.dot(np.transpose(ypred - expectedypred),(ypred - expectedypred)))/10000
			bias_sq2CG += float(np.dot(np.transpose(expectedypred - vectorized_f(x)),(expectedypred - vectorized_f(x))))/10000
		print "g1(x)\t",format(variance2CG,'.8f'),"\t",format(bias_sq2CG,'.8f')

		EH2 = []
		for gpower in range(0,gpowerMax):
			for idx in range(0,100):
				x = complete_data2_X[idx]
				y = np.transpose(complete_data2_Y[idx])
				phi = getPhi(x,gpower)
				w = getW(phi,y,lamb)
				MSE2[gpower][idx] = getMSE(phi,y,w,lamb)
				if idx == 0:
					EH2.append(w/100)
				else:
					EH2[-1] += w/100
			if lamb == 0:
				histParams = MSE2[gpower]
				plt.hist(histParams,bins = 10)
				plt.title('Part B g'+str(gpower+2)+'(x)')
				plt.xlabel('MSE bucket')
				plt.ylabel('bucket count')
				plt.show()
				plt.clf()
		variance2 = [0 for i in range(0,gpowerMax)]
		for gpower in range(0,gpowerMax):
			for idx in range(0,100):
				x = complete_data2_X[idx]
				y = np.transpose(complete_data2_Y[idx])
				phi = getPhi(x,gpower)
				w = getW(phi,y,lamb)
				ypred = np.dot(phi,w)
				avgW = EH2[gpower]
				expectedypred = np.dot(phi,avgW)
				z = float(np.dot(np.transpose(ypred - expectedypred),(ypred - expectedypred)))/10000
				variance2[gpower] += z
		
		bias_sq2 = [0 for i in range(0,gpowerMax)]
		for gpower in range(0,gpowerMax):
			for idx in range(0,100):
				x = complete_data2_X[idx]
				phi = getPhi(x,gpower)
				avgW = EH2[gpower]
				expectedypred = np.dot(phi,avgW)
				z = float(np.dot(np.transpose(expectedypred - vectorized_f(x)),(expectedypred - vectorized_f(x))))/10000
				bias_sq2[gpower] += z
		for i in range(0,len(variance2)):
			print "g"+str(i+2)+"(x)\t",format(variance2[i],'.8f'),"\t",format(bias_sq2[i],'.8f')
		plt.plot([variance2CG] + variance2)

		plt.suptitle('variance')
		plt.legend(['variance'],loc = 'upper right')
		plt.xlabel('complexity')
		plt.ylabel('variance')
		plt.show()
		plt.clf()
	
		plt.plot([bias_sq2CG] + bias_sq2)	

		plt.suptitle('bias^2')
		plt.legend(['bias^2'],loc = 'upper right')
		plt.xlabel('complexity')
		plt.ylabel('bias^2')
		plt.show()
		plt.clf()

	

def svm():
	print "Linear and Kernel SVM"
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
	trainY = list(trainY)
	trainX = list(np.transpose(np.array(trainX)))
	for i in range(0,len(trainX)):
		trainX[i] = list(trainX[i])
	trainX = list(trainX)
	prob  = svm_problem(trainY, trainX)
	C = pow(4,-6)
	count = 0
	total_time = 0
	print "Linear SVM\n"
	while C <= pow(4,2):

		print "*********C = ",C,"**********"
		param = svm_parameter('-t 0 -v 3 -c {} -q'.format(C))
		start_time = time.time()
		m = svm_train(prob, param)
		end_time = time.time()
		C = C*4
		count = count+1
		total_time += (end_time - start_time)
	average_time = total_time / count
	print "average training time = ",average_time,"sec"
		
	


	C = pow(4,-3)
	count = 0
	total_time = 0
	best_m = 0
	best_kernel = "polynomial Kernel"
	params = [1,1]
	print "\n\nPolynomial kernel\n"
	while C <= pow(4,7):
		for degree in range(1,4):
			print "*********C = ",C,"and degree =",degree,"**********"
			param = svm_parameter('-t 1 -v 3 -c {} -d {} -q'.format(C,degree))
			start_time = time.time()
			m = svm_train(prob, param)
			end_time = time.time()
			if m > best_m:
				best_m = m
				best_kernel = "polynomial Kernel"
				params = [C,degree]	
			count = count+1
			total_time += (end_time - start_time)
		C = C*4
	average_time = total_time / count
	print "average training time = ",average_time,"sec"


	C = pow(4,-3)
	count = 0
	total_time = 0
	
	print "\n\nRBF kernel\n"
	while C <= pow(4,7):
		gamma = pow(4,-7)
		while gamma <= pow(4,-1):
			print "*********C = ",C,"and gamma =",gamma,"**********"
			param = svm_parameter('-t 2 -v 3 -c {} -g {} -q'.format(C,gamma))
			start_time = time.time()
			m = svm_train(prob, param)
			end_time = time.time()
			if m > best_m:
				best_m = m
				best_kernel = "RBF kernel"
				params = [C,gamma]
			count = count+1
			total_time += (end_time - start_time)
			gamma = gamma*4
		C = C*4
	average_time = total_time / count
	print "average training time = ",average_time,"sec"

	print "best kernel=",best_kernel
	print "best param=",params
	print "best accuracy=",best_m



main()
svm()

