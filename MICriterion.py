import sys
import numpy as np
import time
from scipy.stats import itemfreq

def Mutual_Info(x, C): #for single mutual information, one x vs one C
	catg_x = np.unique(x)
	catg_C = np.unique(C)
	prob_x = (itemfreq(x)[:,1])/len(x)
	prob_C = (itemfreq(C)[:,1])/len(C)

	M_table = np.zeros((len(catg_x),len(catg_C)))

	for i in range(len(x)):
		currnt_x = x[i]
		match_C = C[i]		
		x_index = np.where(catg_x==currnt_x)
		C_index = np.where(catg_C==match_C)
		x_index = int(x_index[0])
		C_index = int(C_index[0])
		M_table[x_index][C_index]+=1

	M_table/=M_table.sum()

	M_info = 0
	for x in catg_x:
		for C in catg_C:
			x_index = int(np.where(catg_x==x)[0])
			C_index = int(np.where(catg_C==C)[0])
			prob_xC = M_table[x_index][C_index]

			if prob_xC > 0:
				M_info += prob_xC * np.log10(prob_xC / (prob_x[x_index] * prob_C[C_index]))

	return M_info

def Cal_Dep(S, C):
	[M, m] = S.shape if S.ndim >1 else [S.shape[0],1]
	#M : number of features
	#m : number of data

	D = 0	
	for xi in S:
		for fea_ind in range(S.shape[1]):
			if (S[:,fea_ind]==xi).mean() == 1:
				xi_index = fea_ind
				break
		D += Mutual_Info(xi, C)* Cal_Dep(np.delete(S, xi_index, 0),C)

	return 0

def Cal_Rel(S, C):
	[M, m] = S.shape if S.ndim >1 else [S.shape[0],1]
	#M : number of features
	#m : number of data
	
	D = 0
	for xi in S:
		D += Mutual_Info(xi, C)
	
	return D/M

def Cal_Red(S):
	[M, m] = S.shape if S.ndim >1 else [S.shape[0],1]
	#M : number of features
	#m : number of data
	if M == 1:
		return 0

	R = 0
	for xi in S:		
		for xj in S:
			if not np.array_equal(xi,xj):
				R += Mutual_Info(xi,xj)
	return R/(M*M)

def mRMR(S, C):
	D = Cal_Rel(S, C)
	R = Cal_Red(S)
	return (D-R)

'''
def mRMR_sel(X, C, cur_featind):
	num_feat = X.shape[1]	
	eval_list = []

	for ith_feat in range(num_feat):
		subset = X[:,cur_featind].T if len(cur_featind) > 0 else []
		if ith_feat not in cur_featind:
			#get one feature vector which did not pick before
			xi = X[:,ith_feat]

			#Calculation for mRMR method
			tmp_rel = Mutual_Info(xi, C)

			tmp_red = 0
			if subset == []:
				tmp_red = 0
			else:
				for xj in subset:
					tmp_red += Mutual_Info(xi,xj)
				tmp_red /= subset.shape[0]

			#store the value and below find which one is the max
			eval_list.append(tmp_rel-tmp_red)
		else:
			#Give the smallest value to those features who picked before
			eval_list.append(-sys.maxint-1)

	max_value = max(eval_list)
	max_index = eval_list.index(max_value)
	cur_featind.append(max_index)		
	return cur_featind
'''
def mRMR_sel(X, C, cur_featind, rel_array, red_array):
	num_feat = X.shape[1]
	num_sel_feat = len(cur_featind)
	mRMR_array = np.ones(num_feat)*(-sys.maxint-1)
	xj = X[:,cur_featind[-1]] #the last append feature

	for ith_feat in range(num_feat):
		if ith_feat not in cur_featind:
			xi = X[:,ith_feat]
			red_array[ith_feat] += Mutual_Info(xi,xj)# record redundancy
			mRMR_array[ith_feat] = rel_array[ith_feat] - red_array[ith_feat]/num_sel_feat
	max_index = np.argsort(mRMR_array)[-1]
	cur_featind.append(max_index)
	return cur_featind