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
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.lda import LDA
from sklearn import cross_validation

def Wrapper(feature_index, X, y, dataset, clf, packname ='sklearn', sel = 'forward'):
	f_index = feature_index	
	num_featind = f_index.shape[0]
	f_index = f_index.tolist()
	print f_index
	
	optimal_set = [] if sel == 'forward' else f_index[:]
	scores = 0
	tmp_min = 0

	final_error_mean = []
	for inc_ind in range(num_featind):
		error_mean = []
		pass_ind_list = []
		print 'current optimal set', optimal_set
		print '%s %dth feature index' % ('Adding' if sel == 'forward' else 'Deleting',inc_ind+1)
		for f_ind in f_index:			
			cur_index = optimal_set[:]

			if 	(sel == 'forward' and f_ind not in cur_index) or\
				(sel == 'backward' and f_ind in cur_index):				
				pass_ind_list.append(f_ind)

				if sel == 'forward':
					cur_index.append(f_ind)
				elif sel == 'backward':
					cur_index.remove(f_ind)

				cur_x = X[:,cur_index]
				cv_fold = 10 if dataset == 'HDR' or dataset == 'ARR' else n_sample
				if packname == 'libsvm':
					#libsvm package
					cur_x = cur_x.tolist()
					prob = svm_problem(y, cur_x)
					param = svm_parameter('-s 3 -c 5 -h 0')
					m = svm_train(y, cur_x, '-c 5')
					m = svm_train(prob, '-t 2 -c 5')
					m = svm_train(prob, param)
					scores = svm_train(y, cur_x, '-v '+str(cv_fold))
					error_mean.append(1-scores/100)
				elif packname == 'sklearn':
					#sklearn package
					scores = cross_validation.cross_val_score(clf, cur_x, y, cv=cv_fold)
					scores = 1-scores
					error_mean.append(scores.mean())

				#print error_mean[-1]

		min_value = min(error_mean)
		min_index = error_mean.index(min_value)
		min_index = pass_ind_list[min_index]
		#print 'min value this round', min_value
		print 'min value index', min_index

		if inc_ind == 0 or (inc_ind > 0 and min_value <= tmp_min):
			print 'find better value, add to index list'				
			tmp_min = min_value
			final_error_mean.append(min_value)

			if sel == 'forward':
				optimal_set.append(min_index)
			elif sel == 'backward':
				optimal_set.remove(min_index)

		elif min_value > tmp_min:
			print 'no better, end'
			return [optimal_set, final_error_mean]

	return [optimal_set, error_mean]		

if __name__ == "__main__":
	timestr = time.strftime("%Y%m%d_%H%M%S")

	#Parameters
	if not len(sys.argv) == 6:
		print 'worng argument size, expected:4, current input:', len(sys.argv)-1
		exit()
	
	clf_name = sys.argv[1]
	if not (clf_name == 'NB' or clf_name == 'SVM' or clf_name == 'LDA'):
		print 'first argument is classfier name'
		exit()
	
	dataset = sys.argv[2]
	if not (dataset == 'HDR' or dataset == 'ARR' or dataset == 'NCI' or dataset == 'LYM'):
		print 'second argument is data set name'
		exit()

	algthm_name = sys.argv[3]
	if not (algthm_name == 'mRMR' or algthm_name == 'MaxRel'):
		print 'third argument is selection of algorithm'
		exit()
	
	clf_package = sys.argv[4]
	if not (clf_package == 'sklearn' or clf_package == 'libsvm'):
		print 'fourth argument is package name'
		exit()
	if clf_package == 'libsvm' and not clf_name == 'SVM':
		print 'libsvm only suppoort SVM classifer'
		exit()

	method = sys.argv[5]
	if not (method == 'forward' or method == 'backward'):
		print 'fifth argument is selecting method'
		exit()

	index_df = read_csv(os.path.expanduser('./Dataset/'+dataset+'/index/'+algthm_name+'_'+clf_name+'.csv'))
	feat_ind = (index_df[index_df.columns[:-1]].values)[0]

	#Read data set from file
	dir_path = './Dataset/' + dataset + '/'
	datafile = dir_path + dataset + '.csv'
	X,y = load(datafile, True if dataset == 'HDR' else False)
	if dataset == 'ARR':
		y = [ 1 if yi==1 else -1 for yi in y]# 1 means normal, other cases are abnormal
	#mRMR_X = X[:,feat_ind]
	
	#Setting of classifer
	if clf_name == 'NB':
		clf = GaussianNB()
		#clf = MultinomialNB(fit_prior=False)
		#clf = BernoulliNB()
	elif clf_name == 'SVM':
		clf = SVC(kernel='linear', C=1)
	elif clf_name == 'LDA':
		clf = LDA()

	[optimal_set, error_mean] = Wrapper(feat_ind, X, y, dataset, clf, clf_package, method)

	fopt = open('./log/optimal_set_'+algthm_name+'_error_mean_'+clf_name+'_'+dataset+'_'+timestr+'.csv', 'w')
	for i in range(len(error_mean)):
		fopt.write("indexnum_"+str(i+1)+',')
	fopt.write('\n')
	for i in range(len(error_mean)):
		fopt.write(str(error_mean[i])+',')
	fopt.close()

	fopt_ind = open('./log/optimal_set_index_'+clf_name+'_'+dataset+'_'+timestr+'.csv', 'w')
	for i in range(len(optimal_set)):
		fopt_ind.write("indexnum_"+str(i+1)+',')
	fopt_ind.write('\n')
	for i in range(len(optimal_set)):
		fopt_ind.write(str(optimal_set[i])+',')
	fopt_ind.close()
	