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
clf_SVM = SVC(kernel='linear', C=1)

if __name__ == "__main__":
	#exit()
	#Data preprocessing
	X = DataPreprocessing(X, dataset)

	#build mutual info table
	MAX_FEANUM = 50
	#test
	'''
	x1 = np.array([0,0,0,0,0,0,1,0,1,0])
	x2 = np.array([0,0,0,0,0,1,0,0,1,0])
	x3 = np.array([0,0,0,0,1,0,0,0,1,0])
	x4 = np.array([0,0,0,1,0,0,0,0,1,0])
	x5 = np.array([0,0,1,0,0,0,1,0,1,0])
	x6 = np.array([0,1,0,0,0,1,0,0,1,0])
		
	S = np.array([x1,x2,x3,x4,x5,x6])
	X = S.T
	y = np.array([0,1,2,2,0,0,0,1,2,0])
	'''
	#test code end

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

	'''
	error_mean = [1,2,4]
	for i in range(len(error_mean)):
		fmRMR.write(str(error_mean[i])+',')
	fmRMR.close()

	from pandas.io.parsers import read_csv
	dff = read_csv(os.path.expanduser('./log/mRMR_error_mean_SVM_'+dataset+'.csv'))
	print dff
	print dff.columns[0:-1].values
	print dff[dff.columns[0:-1]].values
	exit()
	'''
	
	for i in range(MAX_FEANUM):
		scores = []
		feat_ind = mRMR_sel(X, y, i+1, Rel_table, Red_table)
		#print feat_ind
		mRMR_X = X[:,feat_ind]
		#print mRMR_X.T
		scores = cross_validation.cross_val_score(clf_SVM, mRMR_X, y, cv=10)
		#fmRMR.write(str(scores))
		error_mean.append(scores.mean())
	#print error_mean
	for i in range(len(error_mean)):
		fmRMR.write(str(error_mean[i])+',')

	fmRMR.close()

	#Run Max-Dep algorithm
	#Run Max-Rel algorithm
	#plot comparision chart