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
		data.reverse()
		# return data
		data_dict = {} #dictionary containing header-> timesequence

		keys = data[-1]
		for index in range(len(keys)): 
			new_timeseq = []
			for row in data[:-1]:
				if keys[index] != 'Date':
					new_timeseq.append(float(row[index]))
				else:
					new_timeseq.append(row[index])
			data_dict[keys[index]] = new_timeseq
		# return data_dict

		#calculate Stoch_K 
		data_length = len(data_dict['High'])
		new_timeseq = []
		for t in range(data_length):
			C_t = data_dict['Close'][t]
			HH_n = 0
			LL_n = 0
			if t < 1: 
				HH_n = data_dict['High'][t]
				LL_n = data_dict['Low'][t]
			elif t < n: 
				# print(i)
				# print(data_dict['High'][1:i])
				assert len(data_dict['High'][:t]) < n
				HH_n = max(data_dict['High'][:t])
				LL_n = max(data_dict['Low'][:t])
			else: 
				length = len(data_dict['High'][t-n:t])
				assert length == n
				HH_n = max(data_dict['High'][t-n:t])
				LL_n = max(data_dict['Low'][t-n:t])
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
			new_timeseq.append(sum_val / n)
		data_dict['Stoch_D'] = new_timeseq
		assert len(data_dict['Stoch_D'])==data_length


		# calculate momentum
		new_timeseq = []
		for t in range(data_length):
			momentum = 0
			if t < n:
				first = data_dict['Close'][0]
				momentum = data_dict['Close'][t] - first
			else:
				before = data_dict['Close'][t-4]
				momentum = data_dict['Close'][t] - before
			new_timeseq.append(momentum)
		data_dict['Momentum'] = new_timeseq
		assert len(data_dict['Momentum'])==data_length

		# calculate rate of change
		new_timeseq = []
		for t in range(data_length):
			roc = 0
			if t < n:
				first = data_dict['Close'][0]
				momentum = (data_dict['Close'][t] / first) * 100
			else:
				before = data_dict['Close'][t-n]
				roc = (data_dict['Close'][t] / before) * 100
			new_timeseq.append(roc)
		data_dict['ROC'] = new_timeseq
		assert len(data_dict['ROC'])==data_length

		
		# calculate William's %R
		new_timeseq = []
		for t in range(data_length):
			C_t = data_dict['Close'][t]
			HH_n = 0
			LL_n = 0
			if t < 1: 
				HH_n = data_dict['High'][t]
				LL_n = data_dict['Low'][t]
			elif t < n: 
				# print(i)
				# print(data_dict['High'][1:i])
				assert len(data_dict['High'][:t]) < n
				HH_n = max(data_dict['High'][:t])
				LL_n = max(data_dict['Low'][:t])
			else: 
				length = len(data_dict['High'][t-n:t])
				assert length == n
				HH_n = max(data_dict['High'][t-n:t])
				LL_n = max(data_dict['Low'][t-n:t])
			new_timeseq.append(100*(HH_n - C_t)/(HH_n-LL_n))
		data_dict['WillR'] = new_timeseq
		assert len(data_dict['WillR'])==data_length

		# calculate A/D oscillator
		new_timeseq = []
		for t in range(data_length):
			H_t = data_dict['High'][t]
			L_t = data_dict['Low'][t]
			C_tprev = 0
			if t < 1:
				C_tprev = data_dict['Close'][t]
			else:
				C_tprev = data_dict['Close'][t-1]
			new_timeseq.append((H_t - C_tprev) / (H_t - L_t))

		data_dict['AD'] = new_timeseq
		assert len(data_dict['AD'])==data_length

		# calculate Disparity 5
		new_timeseq = []
		for t in range(data_length):
			C_t = data_dict['Close'][t]
			MA = 0
			if t < 1:
				MA = data_dict['Close'][t]
			elif t < 5:
				MA = sum(data_dict['Close'][:t]) / len(data_dict['Close'][:t])
			else:
				assert len(data_dict['Close'][t-5:t]) == 5
				MA = sum(data_dict['Close'][t-5:t]) / 5
			new_timeseq.append(100 * C_t / MA)

		data_dict['Disp'] = new_timeseq
		assert len(data_dict['Disp'])==data_length
		return data_dict