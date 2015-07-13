import os
from pandas.io.parsers import read_csv
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
MAX_FEANUM = 50

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
	Total_feanum = X.shape[1]
	MAX_FEANUM = Total_feanum if MAX_FEANUM > Total_feanum else MAX_FEANUM
	Rel_table = np.zeros(Total_feanum)
	Red_table = np.zeros(Total_feanum*(Total_feanum-1)/2)
	
	#save to files
	Rel_table_fname = dir_path + "Rel_table.csv"
	Red_table_fname = dir_path + "Red_table.csv"

	if os.path.isfile(Rel_table_fname) and os.path.isfile(Rel_table_fname):
		#if files already exit, read tables from files
		Rel_table_df = read_csv(os.path.expanduser(Rel_table_fname))
		Rel_table = (Rel_table_df[Rel_table_df.columns[:-1]].values)[0]
		Red_table_df = read_csv(os.path.expanduser(Red_table_fname))
		Red_table = (Red_table_df[Red_table_df.columns[:-1]].values)[0]
	else:
		Rel_table = np.zeros(Total_feanum)
		Red_table = np.zeros(Total_feanum*(Total_feanum-1)/2)
		[Rel_table, Red_table] = Build_Minfo_table(X, y, Rel_table, Red_table)
		f_rel = open(Rel_table_fname, 'w')
		f_red = open(Red_table_fname, 'w')
		for i in range(len(Rel_table)):
			f_rel.write('Rel'+str(i)+',')
		f_rel.write('\n')
		for i in range(len(Rel_table)):
			f_rel.write(str(Rel_table[i])+',')
		for i in range(len(Red_table)):
			f_red.write('Red'+str(i)+',')
		f_red.write('\n')
		for i in range(len(Red_table)):
			f_red.write(str(Red_table[i])+',')
		f_rel.close()
		f_red.close()	

	logger.debug('Rel_table')
	logger.debug(Rel_table)
	logger.debug('Red_table')
	logger.debug(Red_table)
		
	#Run mRMR algorithm	
	error_mean = []	
	feat_ind = []
	for i in range(MAX_FEANUM):
		print "Select %d features from X" % (i+1)
		scores = []
		feat_ind = mRMR_sel(X, y, Rel_table, Red_table, feat_ind)	
		print feat_ind	
		mRMR_X = X[:,feat_ind]		
		scores = cross_validation.cross_val_score(clf, mRMR_X, y, cv=10)
		scores = 1-scores
		error_mean.append(scores.mean())
		print "error mean %f" % error_mean[i]
	
	#save mean error value to file
	fmRMR = open('./log/mRMR_error_mean_SVM_'+dataset+'.csv', 'w')
	for i in range(len(error_mean)):
		fmRMR.write("indexnum_"+str(i+1)+',')
	fmRMR.write('\n')
	for i in range(len(error_mean)):
		fmRMR.write(str(error_mean[i])+',')

	fmRMR.close()	