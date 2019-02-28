import torch
import torch.nn as nn 

class Config(): 
	def __init__(self, 
				num_titles = 25, 
				title_dim = 300, 
				n_tech_indicators = 7,
				n_hidden_LSTM_titles = 128, 
				n_hidden_LSTM_tech = 50, 
				n_outputs = 2,
				batch_sz = 128,
				pool_sz = 2,
				filter_sz = 4,
				n_filters = 64, 
				window_len = 5,
				p_drop = 0.5
				):

		self.num_titles = num_titles
		self.title_dim = title_dim
		self.n_hidden_LSTM_titles = n_hidden_LSTM_titles
		self.n_hidden_LSTM_tech = n_hidden_LSTM_tech
		self.input_dim_1 = filter_sz
		self.input_dim_2 = n_tech_indicators

		self.n_outputs = n_outputs
		self.window_len = window_len
		self.filter_sz = filter_sz,
		self.n_filters = n_filters,
		self.pool_sz = pool_sz,
		self.batch_sz = batch_sz
		self.p_drop = p_drop

"""
Currently skeleton code, fill with actual models as desired
"""
class RCNN(nn.Module):
	def __init__(self, config):
		''' 
		input_dim: Dimensionality of input corpus
		hidden_dim: Hidden layer dimensionality
		output_dim: Dimensionality of output of model (should be scalar)
		'''
		super(RCNN,self).__init__()
		self.config = config

		self.cnn = nn.Conv1d(config.title_dim, config.n_filters, config.filter_sz)
		self.max_pool = nn.MaxPool1d(config.pool_sz, stride = 1, padding = 1) #Dim increases by 1
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(config.p_drop)
		self.lstm1 = nn.LSTM(config.input_dim, config.n_hidden_LSTM_titles)
		self.lstm2 = nn.LSTM(hidden_dim, config.n_hidden_LSTM_tech)

		#Linearly project final hidden states from LSTM to sigmoid output
		self.map_to_out = nn.Linear(config.n_hidden_LSTM_titles + config.n_hidden_LSTM_tech, 
									config.n_outputs)
		self.softmax = nn.Softmax() #MIGHT NEED TO EDIT DIM LATER
		
	"""
	Forward pass for the RCNN.
	Inputs: 
		titles: Batched Tensor of news embedding vectors for a day. Each batch element has L titles, each title has dim m 
		tech_indicators: Batched tensor of tech indicators. Each batch element has indicators (dim 7) for a window of size n = 5. Expected dims: (seq_len, batch, input_dim)
	"""

	def forward(self, titles, tech_indicators):
		
		#Input: (batch_sz, sent_embed_sz, num_titles), i.e. (batch, m, L) in paper
		conv_out = self.cnn(titles) #Out: (batch, num_filters, L - R + 1) , R is window length
		pool_out = self.max_pool(conv_out) #Out: (batch, num_filters, L - R + 2)

		relud_pool = self.relu(pool_out) #(batch, num_filters, L - R + 2)

		dropped_relu = self.dropout(relud_pool) #(batch, num_filters, L - R + 2) #NOT SURE ABOUT DIMENSION DROPPED

		relud_pool_reshaped = relud_pool.permute(2, 0, 1)

		#Input : (seq_len, batch, input_dim)
		_, (last_hidden_1, last_cell_1) = self.lstm1(relud_pool_reshaped) #Both hidden & cell are (1, batch, hidden_size)
		last_hidden_1, last_cell_1 = last_hidden_1.squeeze(0), last_cell_1.squeeze(0) # Both are now (batch, hidden_size)
		

		_, (last_hidden_1, last_cell_1) = self.lstm2(tech_indicators)
		last_hidden_2, last_cell_2 = last_hidden_2.squeeze(0), last_cell_2.squeeze(0) # Both are now (batch, hidden_size)
		

		concatenated = torch.cat((last_hidden_1, last_cell_1, last_hidden_2, last_cell_2), 1) #(batch, 2*h_dim_1 + 2*h_dim_2)

		output = self.softmax(self.map_to_out(concatenated)) #(batch, 2)

		return output

	

