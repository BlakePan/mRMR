import os
import sys
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt

if __name__ == "__main__":
	task = sys.argv[1]
	if not (task == 'single' or task == 'compare'):
		exit()

	while 1:
		filename = [f for f in os.listdir('./log') if f.endswith(".csv")]
		filename.sort()
		for index, f in enumerate(filename):
			print "%d. %s" % (index+1, f)
		user_input = int(raw_input('Select one csv file (-1 to exit): '))
		if user_input == -1:
			exit()
		filename = filename[user_input-1]
		result_df = read_csv(os.path.expanduser('./log/'+filename))
		result = (result_df[result_df.columns[:-1]].values)[0]
		n_selfeat_f1 = result.shape[0]
		plt.plot(result, '-bs', label=filename[:-20])
		plt.grid(True)

		n_selfeat_f2  = 0
		if task == 'compare':
			comparefile = [f for f in os.listdir('./log') if f.endswith(".csv")]
			comparefile.sort()
			for index, f in enumerate(comparefile):
				if not f == filename:
					print "%d. %s" % (index+1, f)
			user_input = int(raw_input('Select another one csv file (-1 to exit): '))
			if user_input == -1:
				exit()
			comparefile = comparefile[user_input-1]
			result_df = read_csv(os.path.expanduser('./log/'+comparefile))
			result = (result_df[result_df.columns[:-1]].values)[0]
			n_selfeat_f2 = result.shape[0]
			plt.plot(result, '-r*', label=comparefile[:-20])

		y1 = float(raw_input('y_min: '))
		y2 = float(raw_input('y_max: '))		
		plt.ylabel('error rate')
		plt.axis([0, max(n_selfeat_f1, n_selfeat_f2), y1, y2])
		plt.legend()
		plt.show()