#data_subname = ['mfeat-fac', 'mfeat-fou', 'mfeat-kar', 'mfeat-mor', 'mfeat-pix', 'mfeat-zer']
data_subname = ['mfeat-fac']

for sub in data_subname:
	fsub = open(sub, 'r')
	sub_vec = []
	for fvec in fsub:
		#print '==========================================='
		#print fvec
		#print '*******************************************'

		trans_vec = []
		tmp = str('')
		for i in fvec:
			#print i			
			if i == ' ' and tmp == '':
				tmp = str('')			
			elif (i == ' ' and not tmp == '') or i == '\n':
				trans_vec.append(tmp)
				tmp = str('')
			elif not i == ' ':
				tmp += i
		#trans_vec.append(tmp)
		#print trans_vec
		sub_vec.append(trans_vec)
	print sub_vec