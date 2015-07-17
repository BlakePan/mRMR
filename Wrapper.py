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

def Wrapper(feature_index, X, y, dataset, clf, packname ='sklearn', sel = "forward"):
	f_index = feature_index	
	num_featind = f_index.shape[0]
	f_index = f_index.tolist()
	print f_index
	
	optimal_set = [] if sel == "forward" else X[:,f_index]

	if sel == "forward":
		scores = 0
		tmp_min = 0

		for inc_ind in range(num_featind):
			error_mean = []
			print 'current optimal set', optimal_set
			print 'Adding %dth feature index' % (inc_ind+1)
			for index in f_index:				
				cur_index = optimal_set
				if index not in cur_index:
					print 'testing index' ,index					
					cur_index.append(index)
					cur_x = X[:,cur_index]
					#exit()
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

			min_value = min(error_mean)
			min_index= error_mean.index(min_value)
			print 'min value this round', min_value

			if inc_ind == 0 or (inc_ind > 0 and min_value <= tmp_min):
				print 'find better value, add to index list'
				tmp_min = min_value
				np.append(optimal_set, f_index[min_index])
				#np.delete(f_index, min_index, None)
			elif min_value > tmp_min:
				print 'no better, end'
				return [optimal_set, error_mean]

		return [optimal_set, error_mean]
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
	timestr = time.strftime("%Y%m%d_%H%M%S")
	packname = sys.argv[1]

	dataset = 'ARR'
	clf_name = 'SVM'
	index_df = read_csv(os.path.expanduser('./Dataset/'+dataset+'/index/'+clf_name+'.csv'))
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

	[optimal_set, error_mean] = Wrapper(feat_ind, X, y, dataset, clf, packname, sel = "forward")

	fopt = open('./log/optimal_set_error_mean_'+clf_name+'_'+dataset+'_'+timestr+'.csv', 'w')
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
	