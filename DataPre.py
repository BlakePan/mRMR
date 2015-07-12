import numpy as np

def discretize(data):
	X = np.array(data)
	[n_sample, n_feature] = X.shape	
	for ith_feat in range(n_feature):
		xi = X[:,ith_feat]
		mean = xi.mean()
		std = xi.std()

		for ith_sample in range(n_sample):
			X[ith_sample,ith_feat] = 1 if X[ith_sample,ith_feat] > mean+std \
			else -1 if X[ith_sample,ith_feat] < mean-std\
			else 0
	return X

def DataPreprocessing(data, dataset_name):
	if dataset_name == "ARR":
		data = discretize(data)
	return data

if __name__ == "__main__":
	#below is test code
	from LoadData import load
	X,y = load("./Dataset/ARR/ARR.csv")
	print X
	X = DataPreprocessing(X,"ARR")
	print "After preprocessing"
	print X