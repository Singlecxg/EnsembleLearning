#coding=utf-8
from optparse import OptionParser
import functools
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from random import randrange
import math

def printPredictResult(predict, y_test, options):
	accuracy = metrics.accuracy_score(y_test, predict)
	precision = metrics.precision_score(y_test, predict)
	recall = metrics.recall_score(y_test, predict)
	print 'ensemble: %s, classifier: %s, features: %s, normalize: %d, T: %d. accuracy: %.8f, precision: %.8f, recall: %.8f'%(options.ensemble, options.classifier, options.features, options.normalize, options.T, accuracy, precision, recall)

def Bagging(classifierModel, X_train, y_train, X_test, T):
	vote = np.zeros((len(X_test),), dtype=np.int)
	n_sample = len(X_train)
	n_train = len(X_train)
	for i in xrange(T):
		X_sample = list()
		y_sample = list()
		for j in xrange(n_sample):
			index = randrange(n_train)
			X_sample.append(X_train[index])
			y_sample.append(y_train[index])
		classifierModel.fit(X_sample, y_sample)
		predict_label = classifierModel.predict(X_test)
		for j in xrange(len(X_test)):
			vote[j] = vote[j] + predict_label[j]

	predict_result = list()
	# for i in xrange(len(X_test)):
	# 	if vote[i] >= 0:
	# 		predict_result.append(1)
	# 	else:
	# 		predict_result.append(-1)
	for i in xrange(len(X_test)):
		predict_result.append(float(vote[i])/float(T))	
	return predict_result

def AdaBoostM1(classifierModel, X_train, y_train, X_test, T):
	len_train = len(X_train)
	weight = np.full((len_train,), (1.0/len_train))
	vote = np.zeros((len(X_test),))
	for i in xrange(T):
		classifierModel.fit(X_train, y_train, sample_weight = weight)
		E = 0.0
		label = classifierModel.predict(X_train)
		for j in xrange(len_train):
			if label[j] != y_train[j]:
				E = E + weight[j]
		beta = E / (1.0 - E)
		sum_weight = 0.0
		for j in xrange(len_train):
			if label[j] == y_train[j]:
				weight[j] = weight[j] * beta
			sum_weight = sum_weight + weight[j]
		weight = weight * (1.0 / sum_weight)
		predict_label = classifierModel.predict(X_test)
		vote_weight = math.log(1.0 / beta)
		for j in xrange(len(X_test)):
			vote[j] = vote[j] + predict_label[j] * vote_weight

	predict_result = list()
	# for i in xrange(len(X_test)):
	# 	if vote[i] >= 0:
	# 		predict_result.append(1)
	# 	else:
	# 		predict_result.append(-1)

	for i in xrange(len(X_test)):
		predict_result.append(vote[i])
	return predict_result

if __name__ == '__main__':
	#解析参数
	parser = OptionParser()
	parser.add_option('-e', '--ensemble',		default="None",		help="ensemble learning algorithm(Bagging,AdaBoostM1,None).",	action='store', type='string', 	dest='ensemble')
	parser.add_option('-c', '--classifier',		default="SVM",  	help="base classifier(SVM,DTree,KNN,NB).",						action='store', type='string', 	dest='classifier')
	parser.add_option('-f', '--features', 		default="all",  	help="kinds of features(content,link,all).",					action='store', type='string', 	dest='features')
	parser.add_option('-n', '--normalize',		default=0,  		help="normalize(0,1,2).",										action='store', type='int', 	dest='normalize')
	parser.add_option('-T', '--T', 				default=20,  		help="T.",														action='store', type='int',		dest='T')
	(options, args) = parser.parse_args()

	if options.ensemble == "AdaBoostM1" and options.classifier == "KNN":
		exit()

	DTreeClassifier = functools.partial(DecisionTreeClassifier, min_samples_leaf=5)
	classifiersMap = {'SVM':LinearSVC, 'DTree':DTreeClassifier, 'KNN':KNeighborsClassifier, 'NB':BernoulliNB}

	# 读取数据，划分数据集
	data1 = np.loadtxt('Test_train_vector.txt')
	X_train, y_train = data1[:, :-1], data1[:, -1:].reshape(-1, ).astype("int32")
	X_test = np.loadtxt('Test_validation_review_vector.txt')

	# 根据normalize参数，进行归一化
	if options.normalize == 1:
		X_train = preprocessing.normalize(X_train, norm='l1')
		X_test = preprocessing.normalize(X_test, norm='l1')
	elif options.normalize == 2:
		X_train = preprocessing.scale(X_train)
		X_test = preprocessing.scale(X_test)

	# 根据classifier参数，创建分类器模型，之后作为参数传递，无须繁琐判断
	classifierModel = classifiersMap[options.classifier]()

	# 根据ensemble参数进入不同的集成学习函数，函数统一返回预测结果predict
	if options.ensemble == 'None':
		classifierModel.fit(X_train, y_train)
		predict = classifierModel.predict(X_test)
	elif options.ensemble == 'Bagging':
		predict = Bagging(classifierModel, X_train, y_train, X_test, options.T)
	else:
		predict = AdaBoostM1(classifierModel, X_train, y_train, X_test, options.T)
		
	# 函数统一返回predict，然后在printPredictResult函数中评估结果，进行输出
	# printPredictResult(predict, y_test, options)
	
	fw = open('Test_predict.txt','w')
	fw.write('id,label')
	cnt = 0
	for i in predict:
		fw.write('\n')
		cnt += 1
		fw.write(str(cnt))
		fw.write(',')
		fw.write(str(i))
	fw.close()
