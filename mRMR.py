import os
import sys
import time
import numpy as np

# My modules
from LoadData import load
from DataPre import DataPreprocessing
from MICriterion import Mutual_Info, mRMR_sel, MaxRel_sel

# Classifiers
# from svm import *
# from svmutil import *
# from sklearn.lda import LDA
# from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut

# logging setting
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

if __name__ == "__main__":
	# args
	if not len(sys.argv) == 5:
		print('worng argument size, expected:4, current input:', len(sys.argv)-1)
		exit()

	clf_name = sys.argv[1]
	if not (clf_name == 'NB' or clf_name == 'SVM' or clf_name == 'LDA'):
		print('first argument is classfier name')
		exit()

	dataset = sys.argv[2]
	if not (dataset == 'HDR' or dataset == 'ARR' or dataset == 'NCI' or dataset == 'LYM'):
		print('second argument is data set name')
		exit()

	algthm_name = sys.argv[3]
	if not (algthm_name == 'mRMR' or algthm_name == 'MaxRel'):
		print('third argument is selection of algorithm')
		exit()

	# clf_package = sys.argv[4]
	# if not (clf_package == 'sklearn' or clf_package == 'libsvm'):
	# 	print('fourth argument is a package name')
	# 	exit()
	# if clf_package == 'libsvm' and not clf_name == 'SVM':
	# 	print('libsvm only suppoort SVM classifer')
	# 	exit()

	dir_path = './Dataset/' + dataset + '/'
	datafile = dir_path + dataset + '.csv'
	MAX_FEANUM = 50

	# Setting of classifer
	if clf_name == 'NB':
		# clf = GaussianNB()
		# clf = MultinomialNB(fit_prior=False)
		clf = BernoulliNB()
	elif clf_name == 'SVM':
		clf = SVC(kernel='linear', C=1)
	elif clf_name == 'LDA':
		clf = LDA()
	else:
		raise Exception('Incorrect setting of classifer: {}'.format(clf_name))
	logger.info('dataset')
	logger.info(dataset)
	logger.info('clf_name')
	logger.info(str(clf))
	# logger.info('clf_package')
	# logger.info(clf_package)

	# Read data set from file
	X, y = load(datafile, is_shuffle=False if dataset == 'ARR' else True)
	if dataset == 'ARR':
		y = [ 1 if yi == 1 else -1 for yi in y]  # 1 means normal, other cases are abnormal
	logger.debug('X')
	logger.debug(X)
	logger.debug('y')
	logger.debug(y)

	# Data preprocessing
	X = DataPreprocessing(X, dataset)
	logger.debug('X after preprocessing')
	logger.debug(X)
	n_sample = X.shape[0]

	# Run mRMR algorithm
	error_mean = []
	feat_ind = []
	costtime = []
	num_feat = X.shape[1]
	rel_array = np.zeros(num_feat)
	red_array = np.zeros(num_feat)
	for ith_feat in range(num_feat):
		print("Adding relevence of %dth features" % (ith_feat+1))
		xi = X[:,ith_feat]
		rel_array[ith_feat] = (Mutual_Info(xi, y))
	# mRMR_list.append(Mutual_Info(xi, y))

	for i in range(MAX_FEANUM):
		scores = 0
		t0 = time.clock()
		if i == 0:
			print("Select 1st features from X")
			feat_ind.append(np.argsort(rel_array)[-1])
		else:
			print("Select %d features from X" % (i+1))
			if algthm_name == 'mRMR':
				feat_ind = mRMR_sel(X, feat_ind, rel_array, red_array)
			elif algthm_name == 'MaxRel':
				feat_ind = MaxRel_sel(X, y, feat_ind, rel_array)
		# feat_ind = mRMR_sel(X, y, feat_ind, mRMR_list)
		# feat_ind = mRMR_sel(X, y, feat_ind)
		t1 = time.clock()-t0
		costtime.append(t1)
		print(feat_ind)
		print(t1, 'seconds')
		feat_sel_X = X[:, feat_ind]

		if dataset == 'HDR' or dataset == 'ARR':
			scores = cross_val_score(clf, feat_sel_X, y, cv=10)
			scores = 1-scores
			error_mean.append(scores.mean())
		elif dataset == 'NCI' or dataset == 'LYM':
			loo = LeaveOneOut()
			scores = 0
			for train, test in loo.split(feat_sel_X):
				ith_test = feat_sel_X[test,:]
				ith_train = feat_sel_X[train,:]
				ith_predict = y[test]
				ith_label = np.delete(y,test)
				clf.fit(ith_train,ith_label)
				scores += clf.score(ith_test,ith_predict)
			error_mean.append(1-scores/n_sample)
		'''
		cv_fold = 10 if dataset == 'HDR' or dataset == 'ARR' else n_sample
		if clf_package == 'libsvm':
			#libsvm package
			feat_sel_X = feat_sel_X.tolist()
			prob = svm_problem(y, feat_sel_X)
			param = svm_parameter('-s 3 -c 5 -h 0')
			m = svm_train(y, feat_sel_X, '-c 5')
			m = svm_train(prob, '-t 2 -c 5')
			m = svm_train(prob, param)
			scores = svm_train(y, feat_sel_X, '-v '+str(cv_fold))
			error_mean.append(1-scores/100)
		elif clf_package == 'sklearn':
			#sklearn package
			scores = cross_validation.cross_val_score(clf, feat_sel_X, y, cv=10)
			scores = 1-scores
			error_mean.append(scores.mean())
		'''

		print("error mean %f" % error_mean[i])

	logger.info('feat_ind')
	logger.info(feat_ind)

	# save mean error value to file
	fname = './log/'+algthm_name+'_error_mean_'+clf_name+'_'+dataset+'_'+timestr+'.csv'
	with open(fname, 'w+') as fmRMR:
		for i in range(len(error_mean)):
			fmRMR.write("indexnum_"+str(i+1)+',')
		fmRMR.write('\n')
		for i in range(len(error_mean)):
			fmRMR.write(str(error_mean[i])+',')

	# save cost time to file
	fname = './log/'+algthm_name+'_cost_time_'+clf_name+'_'+dataset+'_'+timestr+'.csv'
	with open(fname, 'w+') as ftime:
		for i in range(len(costtime)):
			ftime.write("t"+str(i+1)+',')
		ftime.write('\n')
		for i in range(len(costtime)):
			ftime.write(str(costtime[i])+',')

	# save feature index to file
	save_index_path = './Dataset/'+dataset+'/index/'
	if not os.path.exists(save_index_path):
		os.makedirs(save_index_path)
	fname = save_index_path+algthm_name+'_'+clf_name+'.csv'
	with open(fname, 'w+') as findex:
		for i in range(len(feat_ind)):
			findex.write("ind"+str(i+1)+',')
		findex.write('\n')
		for i in range(len(feat_ind)):
			findex.write(str(feat_ind[i])+',')
