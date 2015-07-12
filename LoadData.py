import os
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

def load(fname,shf=False):
	df = read_csv(os.path.expanduser(fname))  	# load pandas dataframe
	df = df.dropna()  							# drop all rows that have missing values in them
	X = df[df.columns[1:]].values
	y = df['Class'].values
	if shf:
		X, y = shuffle(X, y, random_state=26)  	# shuffle train data(optional)
	
	return X, y