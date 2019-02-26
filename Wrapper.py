import os
import sys
from pandas.io.parsers import read_csv
import numpy as np
import time

# My moudles
from LoadData import load
from DataPre import DataPreprocessing

# classifiers
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut


def Wrapper(feature_index, X, y, dataset, clf, sel='forward'):
	f_index = feature_index
	num_featind = f_index.shape[0]
	f_index = f_index.tolist()
	print(f_index)

	optimal_set = [] if sel == 'forward' else f_index[:]
	scores = 0
	tmp_min = 0
	n_sample = X.shape[0]
	valid_range = -1
	final_error_mean = []

	for inc_ind in range(num_featind):
		error_mean = []
		candt_ind_list = []
		print('current optimal set')
		print(optimal_set)
		print('%s %dth feature index' % ('Adding' if sel == 'forward' else 'Deleting',inc_ind+1))
		for f_ind in f_index:
			candt_ind_list = optimal_set[:]

			if (sel == 'forward' and f_ind not in candt_ind_list) or \
				(sel == 'backward' and f_ind in candt_ind_list):

				if sel == 'forward':
					candt_ind_list.append(f_ind)
				elif sel == 'backward':
					candt_ind_list.remove(f_ind)

				cur_x = X[:, candt_ind_list]

				if dataset == 'HDR' or dataset == 'ARR':
					scores = cross_val_score(clf, cur_x, y, cv=10)
					scores = 1-scores
					error_mean.append(scores.mean())
				elif dataset == 'NCI' or dataset == 'LYM':
					loo = LeaveOneOut()
					scores = 0
					for train, test in loo.split(cur_x):
						ith_test = cur_x[test,:]
						ith_train = cur_x[train,:]
						ith_predict = y[test]
						ith_label = np.delete(y,test)
						clf.fit(ith_train,ith_label)
						scores += clf.score(ith_test,ith_predict)
					error_mean.append(1-scores/n_sample)
			else:
				error_mean.append(float('inf'))

		min_value = min(error_mean)
		min_index = error_mean.index(min_value)
		min_index = f_index[min_index]
		print('index', min_index)

		if inc_ind == 0 or (inc_ind > 0 and min_value <= tmp_min):
			if min_value == tmp_min and sel == 'forward':
				valid_range -= 1
			else:
				valid_range = -1

			print('find better value %f, add to index list' % min_value)
			tmp_min = min_value
			final_error_mean.append(min_value)

			if sel == 'forward':
				optimal_set.append(min_index)
			elif sel == 'backward':
				optimal_set.remove(min_index)

		elif min_value > tmp_min:
			print('no better, value: %f, end' % min_value)
			if sel == 'forward':
				optimal_set = optimal_set[:valid_range]
			return [optimal_set, final_error_mean]

	if sel == 'forward':
		optimal_set = optimal_set[:valid_range]
	return [optimal_set, final_error_mean]


if __name__ == "__main__":
	timestr = time.strftime("%Y%m%d_%H%M%S")

	# args
	if not len(sys.argv) == 5:
		print('wrong argument size, expected:4, current input:', len(sys.argv)-1)
		exit()

	clf_name = sys.argv[1]
	if not (clf_name == 'NB' or clf_name == 'SVM' or clf_name == 'LDA'):
		print('first argument is a classifier name')
		exit()

	dataset = sys.argv[2]
	if not (dataset == 'HDR' or dataset == 'ARR' or dataset == 'NCI' or dataset == 'LYM'):
		print('second argument is a data set name')
		exit()

	algthm_name = sys.argv[3]
	if not (algthm_name == 'mRMR' or algthm_name == 'MaxRel'):
		print('third argument is a selection of algorithm')
		exit()

	# clf_package = sys.argv[4]
	# if not (clf_package == 'sklearn' or clf_package == 'libsvm'):
	# 	print('fourth argument is package name')
	# 	exit()
	# if clf_package == 'libsvm' and not clf_name == 'SVM':
	# 	print('libsvm only support SVM classifer')
	# 	exit()

	method = sys.argv[4]
	if not (method == 'forward' or method == 'backward'):
		print('fourth argument is a method')
		exit()

	index_df = read_csv(os.path.expanduser('./Dataset/'+dataset+'/index/'+algthm_name+'_'+clf_name+'.csv'))
	feat_ind = index_df[index_df.columns[:-1]].values[0]

	# Read data set from file
	dir_path = './Dataset/' + dataset + '/'
	datafile = dir_path + dataset + '.csv'
	X,y = load(datafile, True if dataset == 'HDR' else False)
	if dataset == 'ARR':
		y = [ 1 if yi==1 else -1 for yi in y]  # 1 means normal, other cases are abnormal

	# Data pre-processing
	X = DataPreprocessing(X, dataset)

	# Setting of classifier
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

	# Run Wrapper
	[optimal_set, error_mean] = Wrapper(feat_ind, X, y, dataset, clf, method)

	# save mean error value to file
	fname = './log/optimal_set_'+algthm_name+'_error_mean_'+clf_name+'_'+dataset+'_'+method+'_'+timestr+'.csv'
	with open(fname, 'w+') as fopt:
		for i in range(len(error_mean)):
			fopt.write("err"+str(i+1)+',')
		fopt.write('\n')
		for i in range(len(error_mean)):
			fopt.write(str(error_mean[i])+',')

	# save feature index to file
	save_index_path = './Dataset/'+dataset+'/index/'
	if not os.path.exists(save_index_path):
		os.makedirs(save_index_path)
	fname = save_index_path+algthm_name+'_'+clf_name+'_'+method+'.csv'
	with open(fname, 'w+') as fopt_ind:
		for i in range(len(optimal_set)):
			fopt_ind.write("ind"+str(i+1)+',')
		fopt_ind.write('\n')
		for i in range(len(optimal_set)):
			fopt_ind.write(str(optimal_set[i])+',')
