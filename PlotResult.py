import os
import sys
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt

if __name__ == "__main__":
	while 1:
		filename = [f for f in os.listdir('./log') if f.endswith(".csv")]
		filename.sort()
		for index, f in enumerate(filename):
			print "%d. %s" % (index+1, f)
		user_input = int(raw_input('Select one logging file (-1 for exit): '))
		if user_input == -1:
			exit()
		filename = filename[user_input-1]
		result_df = read_csv(os.path.expanduser('./log/'+filename))
		result = (result_df[result_df.columns[:-1]].values)[0]
		plt.plot(result)
		plt.grid(True)
		y1 = float(raw_input('y1: '))
		y2 = float(raw_input('y2: '))
		print user_input
		plt.axis([0, 50, y1, y2])
#		plt.axis([0, 50, 0.15, 0.35])
		plt.show()