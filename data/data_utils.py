import csv
def loadTechnical(input_csv_path,n=5,input_size=7):
	"""
	input_csv_path: path to csv
	output: Tensor of sive (seq_len,batch_size,input_size=7)
	"""

	# with open(input_csv_path,'r') as f: 
	# 	data = [row for row in csv.reader(f.read().splitlines())]

	with open(input_csv_path,'r') as csvfile:
		reader = csv.reader(csvfile,delimiter=",")
		data = []
		for row in reader:
			data.append(row)
			# print(','.join(row))
		# return data
		data_dict = {} #dictionary containing header-> timesequence

		keys  = data[0]
		for index in range(len(keys)): 
			new_timeseq = []
			for row in data[1:]:
				if keys[index] != 'Date':
					new_timeseq.append(float(row[index]))
				else:
					new_timeseq.append(row[index])
			data_dict[keys[index]] = new_timeseq
		# return data_dict

		#calculate Stoch_K 
		data_length = len(data_dict['High'])
		new_timeseq = []
		for i in range(data_length):
			C_t = data_dict['Close'][i]
			HH_n = 0
			LL_n = 0
			if i < 1: 
				HH_n = data_dict['High'][i]
				LL_n = data_dict['Low'][i]
			elif i < n: 
				# print(i)
				# print(data_dict['High'][1:i])
				assert len(data_dict['High'][:i]) < n
				HH_n = max(data_dict['High'][:i])
				LL_n = max(data_dict['Low'][:i])
			else: 
				length = len(data_dict['High'][i-n:i])
				assert length == n
				HH_n = max(data_dict['High'][i-n:i])
				LL_n = max(data_dict['Low'][i-n:i])
			new_timeseq.append((C_t -LL_n)/(HH_n-LL_n))
		data_dict['Stoch_K'] = new_timeseq
		assert len(data_dict['Stoch_K'])==data_length

		#calculate Stoch_D
		new_timeseq = []
		for t in range(data_length):
			sum_val = 0
			if t < 1: 
				sum_val  = data_dict['Stoch_K'][t]
			elif t < n: 
				sum_val = sum(data_dict['Stoch_K'][:t])
			else: 
				sum_val= sum(data_dict['Stoch_K'][t-(n-1):t+1])
				length = len(data_dict['Stoch_K'][t-(n-1):t+1])
				assert length == n
			new_timeseq.append(sum_val)
		data_dict['Stoch_D'] = new_timeseq




		return data_dict

		# for row in data: 


		# for (highs,lows) in zip(data_dict['high'],data_dict['low'])




