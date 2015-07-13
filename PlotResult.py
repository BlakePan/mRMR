import os
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt

clf_name = 'SVM'
dataset = 'ARR'
filepath = './log/mRMR_error_mean_'+clf_name+'_'+dataset+'.csv'
result_df = read_csv(os.path.expanduser(filepath))
result = (result_df[result_df.columns[:-1]].values)[0]
plt.plot(result)
plt.axis([0, 50, 0.15, 0.35])
plt.show()