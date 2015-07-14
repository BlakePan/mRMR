import numpy as np

def discretize(data, dataset_name):
	X = np.array(data)
	[n_sample, n_feature] = X.shape

	for ith_feat in range(n_feature):
		xi = X[:,ith_feat]
		mean = xi.mean()
		std = xi.std()

		for ith_sample in range(n_sample):
			if dataset_name == 'HDR':
				X[ith_sample,ith_feat] = 1 if X[ith_sample,ith_feat] > mean \
				else -1
			elif dataset_name == 'ARR':
				X[ith_sample,ith_feat] = 1 if X[ith_sample,ith_feat] > mean+std \
				else -1 if X[ith_sample,ith_feat] < mean-std\
				else 0
	return X

def DataPreprocessing(data, dataset_name):
	if dataset_name == "HDR" or dataset_name == 'ARR':
		data = discretize(data, dataset_name)
	return data