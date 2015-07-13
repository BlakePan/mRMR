import os
import numpy as np

#my moudles
from LoadData import load
from DataPre import DataPreprocessing
from MICriterion import mRMR_sel,Build_Minfo_table

#classifiers
from sklearn.svm import SVC
from sklearn import cross_validation

#logging setting
import logging
log_file = "./log/mRMR.log"
log_level = logging.DEBUG
logger = logging.getLogger("mRMR")
handler = logging.FileHandler(log_file, mode='w')
formatter = logging.Formatter("[%(levelname)s][%(funcName)s]\
[%(asctime)s]%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(log_level)

#Parameters
dir_path = "./Dataset/"
dataset = "ARR" #suppoort: HDR ARR NCI LYM
dir_path = dir_path + dataset + '/'
file = dir_path + dataset + ".csv"

#Read data set from file
X,y = load(file)

if dataset == "ARR":
	y = [ 1 if yi==1 else -1 for yi in y]# 1 means normal, other cases are abnormal

logger.debug('X')
logger.debug(X)
logger.debug('y')
logger.debug(y)

#Setting of classifer
clf = SVC(kernel='linear', C=1)

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

	#build mutual info table
	MAX_FEANUM = 50
	Total_feanum = X.shape[1]
	Rel_table = np.zeros(Total_feanum)
	Red_table = np.zeros(Total_feanum*(Total_feanum-1)/2)
	[Rel_table, Red_table] = Build_Minfo_table(X, y, Rel_table, Red_table)
	logger.debug('Rel_table')
	logger.debug(Rel_table)
	logger.debug('Red_table')
	logger.debug(Red_table)
	
	#Run mRMR algorithm
	fmRMR = open('./log/mRMR_error_mean_SVM_'+dataset+'.csv', 'w')
	error_mean = []
	
	for i in range(MAX_FEANUM):
		scores = []
		feat_ind = mRMR_sel(X, y, i+1, Rel_table, Red_table)		
		mRMR_X = X[:,feat_ind]		
		scores = cross_validation.cross_val_score(clf, mRMR_X, y, cv=10)
		scores = 1-scores
		error_mean.append(scores.mean())
	
	for i in range(len(error_mean)):
		fmRMR.write(str(error_mean[i])+',')

	fmRMR.close()	