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

def Build_Minfo_table(X, C, Rel_table, Red_table):
	[M, m] = X.shape if X.ndim >1 else [1,X.shape[0]]
	#M : number of data 
	#m : number of features

	for i in range(m):
		Rel_table[i] = Mutual_Info(X[:,i],C)

	for i in range(m-1):
		for j in range(m):
			if j > i:
				print "i=%d, j=%d" % (i,j)
				offset = 0

				if i > 0:
					for offcnt in range(i):
						offset += (m-offcnt-1)
				print "index %d" % ((j-i)+offset-1)
				Red_table[(j-i)+offset-1] = Mutual_Info(X[:,i],X[:,j])

	return [Rel_table,Red_table]

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

#use below lookup table to save computing time
def Lookup_Rel(S, X, Rel_table):
	[M, m] = S.shape if S.ndim >1 else [S.shape[0],1]
	#M : number of features
	#m : number of data

	D = 0

	if S.ndim == 1:
		for fea_ind in range(X.shape[1]):
				if (X[:,fea_ind]==S).mean() == 1:
					rel_index = fea_ind
					break
		D += Rel_table[rel_index]
	else:
		for xi in S:
			for fea_ind in range(X.shape[1]):
				if (X[:,fea_ind]==xi).mean() == 1:
					rel_index = fea_ind
					break
			D += Rel_table[rel_index]
	
	return D/M

def Lookup_Red(S, X, Red_table):
	[M, m] = S.shape if S.ndim >1 else [S.shape[0],1]
	if M == 1:
		return 0

	R = 0

	xi = S[0]
	xj = S[1]

	for fea_ind in range(X.shape[1]):
		if (X[:,fea_ind]==xi).mean() == 1:
			xi_index = fea_ind
			break
	for fea_ind in range(X.shape[1]):
		if (X[:,fea_ind]==xj).mean() == 1:
			xj_index = fea_ind
			break

	if xi_index > xj_index:
		xi_index, xj_index = xj_index, xi_index

	offset = 0
	if xi_index > 0:
		for offcnt in range(xi_index):
			offset += (M-offcnt-1)

	R = Red_table[(xj_index-xi_index)+offset-1]
	return R/(M*M)

	'''
	for xi in S:		
		for xj in S:
			if not np.array_equal(xi,xj):
				#find index
				for fea_ind in range(X.shape[1]):
					if (X[:,fea_ind]==xi).mean() == 1:
						xi_index = fea_ind
						break
				for fea_ind in range(X.shape[1]):
					if (X[:,fea_ind]==xj).mean() == 1:
						xj_index = fea_ind
						break

				if xi_index > xj_index:
					xi_index, xj_index = xj_index, xi_index

				offset = 0
				if xi_index > 0:
					for offcnt in range(xi_index):
						offset += (M-offcnt-1)

				R += Red_table[(xj_index-xi_index)+offset-1]
	return R/(M*M)
	'''

'''
def mRMR_table(S, X, Rel_table, Red_table):
	D = Lookup_Rel(S, X, Rel_table)
	R = Lookup_Red(S, X, Red_table)
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

def mRMR_sel_table(X, C, Rel_table, Red_table, cur_featind):
	num_feat = X.shape[1]	
	eval_list = []

	for ith_feat in range(num_feat):
		subset = X[:,cur_featind].T if len(cur_featind) > 0 else []
		if ith_feat not in cur_featind:
			#get one feature vector which did not pick before
			cur_x = X[:,ith_feat]
			#subset = np.array(cur_x) if subset == [] else np.vstack((subset,cur_x))

			#Calculation for mRMR method
			tmp_rel = Lookup_Rel(cur_x, X, Rel_table)

			tmp_red = 0
			if subset == []:
				#tmp_red = Lookup_Red(np.array([cur_x,subset]), X, Red_table)
				tmp_red = 0
			else:
				for xj in subset:
					tmp_red += Lookup_Red(np.array([cur_x,xj]), X, Red_table)
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
if __name__ == "__main__":
	x0 = np.array([0,1,0,0,0,1,0,0,1,0])
	x1 = np.array([0,0,0,0,0,0,1,0,1,0])
	x2 = np.array([0,0,0,0,0,1,0,0,1,0])
	x3 = np.array([0,0,0,0,1,0,0,0,1,0])
	x4 = np.array([0,0,0,1,0,0,0,0,1,0])
	x5 = np.array([0,0,1,0,0,0,1,0,1,0])	
		
	S = np.array([x0,x1,x2,x3,x4,x5])
	X = S.T
	y = np.array([0,1,2,2,0,0,0,1,2,0])

	Total_feanum = X.shape[1]
	Rel_table = np.zeros(Total_feanum)
	Red_table = np.zeros(Total_feanum*(Total_feanum-1)/2)
	print Build_Minfo_table(X, y, Rel_table, Red_table)
'''

