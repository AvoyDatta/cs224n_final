import csv
import torch
import numpy as np
import gensim
import nltk
from collections import defaultdict
from functools import partial
nltk.download('punkt')

def loadTechnical(input_csv_path,n=5,input_size=7):
	"""
	input_csv_path: path to csv
	output: Tensor of size (seq_len,batch_size,input_size=7)
	"""

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
			for row in data[:-2]:
				if keys[index] != 'Date':
					new_timeseq.append(float(row[index]))
				else:
					new_timeseq.append(row[index])
			data_dict[keys[index]] = new_timeseq


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
		# print(data_length)
		tensor = np.array([data_dict['Stoch_K'],data_dict['Stoch_D'],data_dict['Momentum'],data_dict['ROC'],data_dict['WillR'],data_dict['AD'],data_dict['Disp']])
		# print(tens.shape)
		# print(tens.shape)
		sequence_len = tensor.shape[1]
		stack = []
		for i in range(n,sequence_len): 
		    value = tensor[:,np.newaxis,i-n:i]
		    stack.append(value)
		new_tensor = np.concatenate(stack,1)
		# print(new_tensor.shape)
		new_tensor = torch.Tensor(new_tensor)

		new_tensor = new_tensor.permute(2,1,0)
		# print(new_tensor.shape)
		return new_tensor

def loadTitle(input_csv_path):
	"""
	input: input_csv_path
	output: Tuple(Tensor of size (batch_size,channels=num_titles,seq_len=300),targets)
	"""
	with open(input_csv_path,'r') as csvfile:
		reader = csv.reader(csvfile,delimiter=",")
		data = []
		for row in reader:
			data.append(row)
	keys = data[0]
	# sanitize input
	for i in range(len(data)): 
		# print(len(data[i]))
		for j in range(len(data[i])):
			if data[i][j][0] == 'b':
				data[i][j] = data[i][j][1:]

	targets = []
	for i in range(1,len(data)-1):
		targets.append(data[i+1][1])
	# print(len(targets))
	data = data[1:-1]

	assert(len(targets) == len(data))

	#ToDo Only look up models if word2vec.csv doesn't exist 
	# model = gensim.models.KeyedVectors.load_word2vec_format('./lexvec.pos.vectors', binary=True)
	# vocab = model.vocab

	#TODO: Fill in this with an actual model, currently using dummy default dict of correct size
	model = defaultdict(lambda: np.zeros(300))
	vocab = model
	print(model['a'].shape)
	#convert titles to word2vec
	word2vec_data = []

	print("\n\nHit this")
	for i in range(len(data)):
		# for element in data[i][2:]:
		# 	for word in nltk.word_tokenize(element):
		# 		if word in vocab:
		# 			print(word)

		#TODO: add back if word in vocab
		sentence_list = [ np.mean(np.concatenate([np.array(model[word])[:,np.newaxis]for word in nltk.word_tokenize(element) ],axis=1),axis=1) for element in data[i][2:]]
		sentence_list = np.stack(sentence_list,axis=-1)

		if sentence_list.shape != (300,25):
			# print("oops")
			# print(sentence_list.shape)
			_,m = sentence_list.shape
			n = 25
			a = np.zeros((300,25))
			a[:,:m]= sentence_list
			sentence_list = a 
			# print(sentence_list.shape)
		word2vec_data.append(sentence_list)

	#batch_size,300,25

	word2vec_data = np.stack(word2vec_data,axis=0)
	new_tensor = torch.Tensor(word2vec_data)

	#batch_size,25,300
	new_tensor = new_tensor.permute(0,2,1)


	return word2vec_data





