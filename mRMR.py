import os
from pandas.io.parsers import read_csv
import numpy as np
import time

#my moudles
from LoadData import load
from DataPre import DataPreprocessing
from MICriterion import mRMR_sel

#classifiers
from svm import *
from svmutil import *
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn import cross_validation

#logging setting
import logging
if not os.path.exists('./log'):
	os.makedirs('./log')
timestr = time.strftime("%Y%m%d_%H%M%S")
log_file = "./log/mRMR_"+timestr+".log"
log_level = logging.DEBUG
logger = logging.getLogger("mRMR")
handler = logging.FileHandler(log_file, mode='w')
formatter = logging.Formatter("[%(levelname)s][%(funcName)s]\
[%(asctime)s]%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(log_level)

#Parameters
dir_path = './Dataset/'
dataset = 'ARR' #suppoort: HDR ARR NCI LYM
dir_path = dir_path + dataset + '/'
datafile = dir_path + dataset + '.csv'
MAX_FEANUM = 50
clf_name = 'LDA' #NB SVM LDA
clf_package = 'sklearn' #libsvm sklearn
if clf_package == 'libsvm' and not clf_name == 'SVM':
	print 'libsvm only suppoort SVM classifer'
	exit()
logger.info("dataset")
logger.info(dataset)
logger.info("clf_name")
logger.info(clf_name)
logger.info("clf_package")
logger.info(clf_package)

#Read data set from file
X,y = load(datafile, True if dataset == 'HDR' else False)

if dataset == 'ARR':
	y = [ 1 if yi==1 else -1 for yi in y]# 1 means normal, other cases are abnormal

logger.debug('X')
logger.debug(X)
logger.debug('y')
logger.debug(y)

#Setting of classifer
if clf_name == 'NB':
	clf = GaussianNB()
elif clf_name == 'SVM':
	clf = SVC(kernel='linear', C=1)
elif clf_name == 'LDA':
	clf = LDA()

def Wrapper(feature_index, X, y, sel = "forward"):
	f_index = feature_index
	num_featind = f_index.shape[0]
	optimal_set = [] if sel == "forward" else X[:,f_index]

	if sel == "forward":
		scores = []
		error_mean = []
		tmp_min = 0

		for inc_ind in range(num_featind):
			for index in f_index:
				if index not in optimal_set:
					cur_index = np.append(optimal_set,index)
					cur_x = X[:,cur_index]
					scores = cross_validation.cross_val_score(clf, cur_x, y, cv=10)	
					scores = 1-scores	
					error_mean.append(scores.mean())

			min_value = min(error_mean)
			min_index= error_mean.index(min_value)			

			if inc_ind == 0 or (inc_ind > 0 and min_value <= tmp_min):
				tmp_min = min_value
				np.append(optimal_set, f_index[min_index])
				np.delete(f_index, min_index, None)
			elif min_value > tmp_min:
				return optimal_set

		return optimal_set
	else:
		scores = []
		error_mean = []
		tmp_min = 0

		for inc_ind in range(num_featind):
			for index in optimal_set:				
				cur_index = np.delete(optimal_set,optimal_set.index(index))
				cur_x = X[:,cur_index]
				scores = cross_validation.cross_val_score(clf, cur_x, y, cv=10)	
				scores = 1-scores	
				error_mean.append(scores.mean())

			min_value = min(error_mean)
			min_index= error_mean.index(min_value)			

			if inc_ind == 0 or (inc_ind > 0 and min_value <= tmp_min):
				tmp_min = min_value				
			elif min_value > tmp_min:
				return optimal_set

		return optimal_set

if __name__ == "__main__":
	#Data preprocessing
	X = DataPreprocessing(X, dataset)
	#Run mRMR algorithm	
	error_mean = []
	feat_ind = []
	costtime = []
	for i in range(MAX_FEANUM):
		print "Select %d features from X" % (i+1)
		scores = []
		t0 = time.clock()
		feat_ind = mRMR_sel(X, y, feat_ind)	
		print feat_ind
		t1 = time.clock()-t0
		costtime.append(t1)
		print t1, 'seconds'
		mRMR_X = X[:,feat_ind]

		if clf_package == 'libsvm':
			#libsvm package
			mRMR_X = mRMR_X.tolist()
			prob = svm_problem(y, mRMR_X)
			param = svm_parameter('-s 3 -c 5 -h 0')
			m = svm_train(y, mRMR_X, '-c 5')
			m = svm_train(prob, '-t 2 -c 5')
			m = svm_train(prob, param)
			scores = svm_train(y, mRMR_X, '-v 10')		
			error_mean.append(1-scores/100)
		elif clf_package == 'sklearn':
			#sklearn package
			scores = cross_validation.cross_val_score(clf, mRMR_X, y, cv=10)
			scores = 1-scores
			error_mean.append(scores.mean())

		print "error mean %f" % error_mean[i]
		
	logger.info('feat_ind')
	logger.info(feat_ind)
	
	#save mean error value to file
	fmRMR = open('./log/mRMR_error_mean_'+clf_name+'_'+dataset+'_'+timestr+'.csv', 'w')
	for i in range(len(error_mean)):
		fmRMR.write("indexnum_"+str(i+1)+',')
	fmRMR.write('\n')
	for i in range(len(error_mean)):
		fmRMR.write(str(error_mean[i])+',')
	fmRMR.close()

	ftime = open('./log/cost_time_'+clf_name+'_'+dataset+'_'+timestr+'.csv', 'w')
	for i in range(len(costtime)):
		ftime.write("t"+str(i+1)+',')
	ftime.write('\n')
	for i in range(len(costtime)):
		ftime.write(str(costtime[i])+',')
	ftime.close()
