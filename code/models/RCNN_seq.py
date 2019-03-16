import torch
import torch.nn as nn
from graphviz import Digraph
import re
import torch
from torch.autograd import Variable
import torchviz


class Config_seq(): 
	def __init__(self, 
				num_titles = 25, 
				title_dim = 300, 
				n_tech_indicators = 7,
				n_hidden_LSTM_titles = 128, 
				n_hidden_LSTM_tech = 50, 
				n_outputs = 2,
				batch_sz = 128,
				pool_sz = 2,
				filter_sz_title = 4,
				filter_sz_day = 4,
				n_filters_day= 64,
				n_filters_title = 128, 
				window_len_titles = 5,
				window_len_tech = 5, 
				p_drop = 0.5,
				num_batches = 1000,
				num_conv = 5,
				num_LSTM_layers = 2
				):

		self.title_dim = title_dim
		#self.n_hidden_LSTM_titles = n_hidden_LSTM_titles
		#self.n_hidden_LSTM_tech = n_hidden_LSTM_tech
		#self.input_dim_1 = n_filters
		#self.input_dim_2 = n_tech_indicators

		self.n_outputs = n_outputs
		self.window_len_titles = window_len_titles
		self.window_len_tech = window_len_tech
		self.filter_sz_day = filter_sz_day
		self.filter_sz_title = filter_sz_title

		self.n_filters_day = n_filters_day
		self.n_filters_title = n_filters_title
		self.pool_sz = pool_sz
		self.batch_sz = batch_sz
		self.num_batches = num_batches
		self.p_drop = p_drop
		self.num_conv = num_conv
		self.num_titles = num_titles
		self.input_dim_LSTM = n_filters_day + n_tech_indicators
		self.n_hidden_LSTM = n_hidden_LSTM_titles
		self.max_title_len = 56 #Hard-coded
		self.num_LSTM_layers = num_LSTM_layers



"""
RCNN_seq WITHOUT ATTENTION
"""
class RCNN_seq(nn.Module):
	def __init__(self, config):

		super(RCNN_seq,self).__init__()
		self.config = config

		##print("RCNN_seq config: ", config.__dict__)
		self.conv_title = nn.Conv1d(config.title_dim, config.n_filters_title, config.filter_sz_title)

		self.max_pool_title = nn.MaxPool1d(config.max_title_len - config.filter_sz_title + 1)
		self.relu_title = nn.ReLU()
		self.relu_day = nn.ReLU()

		#self.dropout = nn.Dropout(config.p_drop)
		self.conv_day = nn.Conv1d(config.n_filters_title, config.n_filters_day, config.filter_sz_day)
		self.max_pool_day = nn.MaxPool1d(config.num_titles - config.filter_sz_day + 1)

		self.lstm = nn.LSTM(config.input_dim_LSTM, config.n_hidden_LSTM, num_layers = config.num_LSTM_layers)

		#self.lstm2 = nn.LSTM(config.input_dim_2, config.n_hidden_LSTM_tech)

		self.map_down = nn.Linear(2 * config.num_LSTM_layers * config.n_hidden_LSTM, 64) #Maps down the concatenated hidden state inputs
		#Linearly project final hidden states from LSTM to sigmoid output
		self.map_to_out = nn.Linear(64, config.n_outputs)

		self.log_softmax = nn.LogSoftmax(dim = 1) 
		self.criterion = nn.NLLLoss(reduction = True, reduce = 'mean')
		
	"""
	Forward pass for the RCNN.
	Inputs: 
		titles: Batched Tensor of news embedding vectors. Shape: (Batch, window_len_days, num_titles_per_day, embed_size, max_words_in_title)
		tech_indicators: Batched tensor of tech indicators. Each batch element has indicators (dim 7) for a window of size n = 5. Expected dims: (seq_len, batch, input_dim)
	"""
	def forward(self, titles, tech_indicators):
		batch_sz = titles.size(0)

		# print("Input titles shape: ", titles.shape)
		# print("Input tech indicators: ", tech_indicators.shape)

		titles_reshaped = titles.contiguous().view(batch_sz * titles.size(1) * titles.size(2), titles.size(3), titles.size(4)) 
		#(batch * window_len_days * num_titles_day, num_filters_title, words_title - filter_sz_title + 1)
		#print("reshape # 1:", titles_reshaped.shape)

		conv_out_titles = self.conv_title(titles_reshaped) #Out: (batch, window_len_days, num_titles_day, num_filters_title, words_title - filter_sz_title + 1)
		#print("conv_out_titles: ", conv_out_titles.shape)


		conv_title_pool_out = self.max_pool_title(conv_out_titles) #Out: (batch * window_len_days * num_titles_day, num_filters_title, 1)
		
		conv_title_pool_out = conv_title_pool_out.contiguous().view(batch_sz, titles.size(1), titles.size(2), self.config.n_filters_title, 1) 
		#(batch, window_len_days, num_titles_day, num_filters_title, 1)
		#print("reshape back # 1:", conv_title_pool_out.shape)


		conv_title_pool_out = conv_title_pool_out.squeeze(-1) #Out: (batch, window_len_days, num_titles_day, num_filters_title)

		#print("Title pool: ",conv_title_pool_out.shape)

		relud_pool_title = self.relu_title(conv_title_pool_out) #(batch, window_len_days, num_titles_day, num_filters_title)
		# ##print(relud_pool.shape)
		# dropped_relu = self.dropout(relud_pool) #(batch, num_filters, L - R + 2) #NOT SURE ABOUT DIMENSION DROPPED
		# ##print(dropped_relu.shape)
		relud_pool_title_reshaped = relud_pool_title.permute(0, 1, 3, 2) #(batch, window_len_days, num_filters_title, num_titles_day)

		#print(relud_pool_title_reshaped.shape)
		relud_pool_title_reshaped = relud_pool_title_reshaped.contiguous().view(batch_sz * self.config.window_len_titles, 
																				self.config.n_filters_title, self.config.num_titles) #(batch * window_len_days, num_filters_title, num_titles_day)

		#print("Conv day input :", relud_pool_title_reshaped.shape)

		conv_out_day = self.conv_day(relud_pool_title_reshaped) #Out: (batch * window_len_days, num_filters_day, num_titles_day - kernel_sz_day + 1)
		
		#print("Conv_out_day: ", conv_out_day.shape)
		
		conv_day_pool_out = self.max_pool_title(conv_out_day) #Out: (batch * window_len_days, num_filters_day, 1)
		#print("Conv day pool out: ", conv_day_pool_out.shape)
		
		conv_day_pool_out = conv_day_pool_out.contiguous().view(batch_sz, titles.size(1), self.config.n_filters_day, 1)
		#(batch, window_len_days, num_filters_day, 1)
		#print("reshape back # 2:", conv_day_pool_out.shape)

		conv_day_pool_out = conv_day_pool_out.squeeze(-1) #Out: (batch, window_len_days, num_filters_day)

		relud_pool_day = self.relu_day(conv_day_pool_out) #Out: (batch, window_len_days, num_filters_day)
		#print("Relud_pool_day: ", relud_pool_day.shape)

		# dropped_relu = self.dropout(relud_pool) #(batch, num_filters, L - R + 2) #NOT SURE ABOUT DIMENSION DROPPED
		##print(dropped_relu.shape)
		relud_pool_day_reshaped = relud_pool_day.permute(1, 0, 2) #Out: (window_len_days, batch, num_filters_day)

		#print("tech_indicators shape: ", tech_indicators.shape, "relud_pool_day_reshaped shape: ", relud_pool_day_reshaped.shape)
		concat_input = torch.cat((relud_pool_day_reshaped, tech_indicators), dim = 2) #Size: (window_len_days==seq_len_tech, batch, input_dim_tech + num_filters_day)

		#print("LSTM inputs: ", concat_input.shape)

		lstm_outputs, (last_hidden, last_cell) = self.lstm(concat_input) #outputs shape: (seq_len, batch, hidden_sz). Both hidden & cell are (2, batch, hidden_size)

		lstm_outputs_reshaped = lstm_outputs.permute(1, 2, 0) #(batch, hidden_sz, seq_len)

		# print(lstm_outputs_reshaped.shape)
		last_hidden = last_hidden.view(batch_sz, -1) #out: (batch, 2 * hidden_size)
		last_cell = last_cell.view(batch_sz, -1)

		attn_proj = torch.matmul(lstm_outputs_reshaped, self.attn_vector)
		attn_proj = attn_proj.squeeze(-1)
		# print(self.attn_vector.data)
		# print(self.attn_vector.data.grad)
		# print("Attn proj: ", attn_proj.shape)

		#print("Concatenated end states: ", end_states_concatenated.shape)

		mapped_down = self.relu_day(self.map_down(concat_end_states))

		output = self.log_softmax(self.map_to_out(mapped_down)) #(batch, 2)
		#print(output.shape)
		return output

"""
RCNN_seq WITH ATTENTION
"""
class RCNN_seq_attn(nn.Module):
	def __init__(self, config):

		super(RCNN_seq_attn,self).__init__()
		self.config = config
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print("Running on device: ", str(self.device))

		##print("RCNN_seq config: ", config.__dict__)
		self.conv_title = nn.Conv1d(config.title_dim, config.n_filters_title, config.filter_sz_title)

		self.max_pool_title = nn.MaxPool1d(config.max_title_len - config.filter_sz_title + 1)
		self.relu_title = nn.ReLU()
		self.relu_day = nn.ReLU()
		self.relu_attn = nn.ReLU()

		#self.dropout = nn.Dropout(config.p_drop)
		self.conv_day = nn.Conv1d(config.n_filters_title, config.n_filters_day, config.filter_sz_day)
		self.max_pool_day = nn.MaxPool1d(config.num_titles - config.filter_sz_day + 1)

		self.lstm = nn.LSTM(config.input_dim_LSTM, config.n_hidden_LSTM, num_layers = config.num_LSTM_layers)

		#self.lstm2 = nn.LSTM(config.input_dim_2, config.n_hidden_LSTM_tech)

		self.map_down = nn.Linear(config.n_hidden_LSTM, 16) #Maps down the attn proj 
		#Linearly project final hidden states from LSTM to sigmoid output
		self.map_to_out = nn.Linear(16, config.n_outputs)

		self.log_softmax = nn.LogSoftmax(dim = 1) 
		self.criterion = nn.NLLLoss(reduction = True, reduce = 'mean')
		#self.attn_vector = torch.autograd.Variable(torch.randn(config.window_len_titles, 1, device = self.device), requires_grad=True) #Shape: (window_len, 1)
		#self.attn_vector = nn.Parameter(torch.randn(config.window_len_titles, 1, device = self.device, requires_grad = True), requires_grad = True)  #Shape: (window_len, 1)
		
		self.attn_layer = nn.Linear(self.config.window_len_titles, 1, bias = False)
		print(self.attn_layer.weight)
	"""
	Forward pass for the RCNN_seq_attn.
	Inputs: 
		titles: Batched Tensor of news embedding vectors. Shape: (Batch, window_len_days, num_titles_per_day, embed_size, max_words_in_title)
		tech_indicators: Batched tensor of tech indicators. Each batch element has indicators (dim 7) for a window of size n = 5. Expected dims: (seq_len, batch, input_dim)
	"""
	def forward(self, titles, tech_indicators):
		batch_sz = titles.size(0)

		# print("Input titles shape: ", titles.shape)
		# print("Input tech indicators: ", tech_indicators.shape)

		titles_reshaped = titles.contiguous().view(batch_sz * titles.size(1) * titles.size(2), titles.size(3), titles.size(4)) 
		#(batch * window_len_days * num_titles_day, num_filters_title, words_title - filter_sz_title + 1)
		#print("reshape # 1:", titles_reshaped.shape)

		conv_out_titles = self.conv_title(titles_reshaped) #Out: (batch, window_len_days, num_titles_day, num_filters_title, words_title - filter_sz_title + 1)
		#print("conv_out_titles: ", conv_out_titles.shape)


		conv_title_pool_out = self.max_pool_title(conv_out_titles) #Out: (batch * window_len_days * num_titles_day, num_filters_title, 1)
		
		conv_title_pool_out = conv_title_pool_out.contiguous().view(batch_sz, titles.size(1), titles.size(2), self.config.n_filters_title, 1) 
		#(batch, window_len_days, num_titles_day, num_filters_title, 1)
		#print("reshape back # 1:", conv_title_pool_out.shape)


		conv_title_pool_out = conv_title_pool_out.squeeze(-1) #Out: (batch, window_len_days, num_titles_day, num_filters_title)

		#print("Title pool: ",conv_title_pool_out.shape)

		relud_pool_title = self.relu_title(conv_title_pool_out) #(batch, window_len_days, num_titles_day, num_filters_title)
		# ##print(relud_pool.shape)
		# dropped_relu = self.dropout(relud_pool) #(batch, num_filters, L - R + 2) #NOT SURE ABOUT DIMENSION DROPPED
		# ##print(dropped_relu.shape)
		relud_pool_title_reshaped = relud_pool_title.permute(0, 1, 3, 2) #(batch, window_len_days, num_filters_title, num_titles_day)

		#print(relud_pool_title_reshaped.shape)
		relud_pool_title_reshaped = relud_pool_title_reshaped.contiguous().view(batch_sz * self.config.window_len_titles, 
																				self.config.n_filters_title, self.config.num_titles) #(batch * window_len_days, num_filters_title, num_titles_day)

		#print("Conv day input :", relud_pool_title_reshaped.shape)

		conv_out_day = self.conv_day(relud_pool_title_reshaped) #Out: (batch * window_len_days, num_filters_day, num_titles_day - kernel_sz_day + 1)
		
		#print("Conv_out_day: ", conv_out_day.shape)
		
		conv_day_pool_out = self.max_pool_title(conv_out_day) #Out: (batch * window_len_days, num_filters_day, 1)
		#print("Conv day pool out: ", conv_day_pool_out.shape)
		
		conv_day_pool_out = conv_day_pool_out.contiguous().view(batch_sz, titles.size(1), self.config.n_filters_day, 1)
		#(batch, window_len_days, num_filters_day, 1)
		#print("reshape back # 2:", conv_day_pool_out.shape)

		conv_day_pool_out = conv_day_pool_out.squeeze(-1) #Out: (batch, window_len_days, num_filters_day)

		relud_pool_day = self.relu_day(conv_day_pool_out) #Out: (batch, window_len_days, num_filters_day)
		#print("Relud_pool_day: ", relud_pool_day.shape)

		# dropped_relu = self.dropout(relud_pool) #(batch, num_filters, L - R + 2) #NOT SURE ABOUT DIMENSION DROPPED
		##print(dropped_relu.shape)
		relud_pool_day_reshaped = relud_pool_day.permute(1, 0, 2) #Out: (window_len_days, batch, num_filters_day)

		#print("tech_indicators shape: ", tech_indicators.shape, "relud_pool_day_reshaped shape: ", relud_pool_day_reshaped.shape)
		concat_input = torch.cat((relud_pool_day_reshaped, tech_indicators), dim = 2) #Size: (window_len_days==seq_len_tech, batch, input_dim_tech + num_filters_day)

		#print("LSTM inputs: ", concat_input.shape)

		lstm_outputs, (last_hidden, last_cell) = self.lstm(concat_input) #outputs shape: (seq_len, batch, hidden_sz). Both hidden & cell are (2, batch, hidden_size)

		lstm_outputs_reshaped = lstm_outputs.permute(1, 2, 0) #(batch, hidden_sz, seq_len)

		# print(lstm_outputs_reshaped.shape)
		# last_hidden = last_hidden.view(batch_sz, -1) #out: (batch, 2 * hidden_size)
		# last_cell = last_cell.view(batch_sz, -1)

		# attn_proj = torch.matmul(lstm_outputs_reshaped, self.attn_vector)
		attn_proj = self.attn_layer(lstm_outputs_reshaped) #(batch, hidden_sz, 1)
		attn_proj = attn_proj.squeeze(-1)
		# print(self.attn_vector.data)
		# print(self.attn_vector.data.grad)
		print("Attn proj: ", attn_proj.shape)
		print(self.attn_layer.weight)

		# print("Attn proj: ", attn_proj.shape)

		#print("Concatenated end states: ", end_states_concatenated.shape)

		mapped_down = self.relu_attn(self.map_down(attn_proj))

		output = self.log_softmax(self.map_to_out(mapped_down)) #(batch, 2)
		#print(output.shape)
		return output

	def backprop(self,optimizer, logits, labels):

		optimizer.zero_grad()
		loss = self.criterion(logits,labels)
		# print(loss)
		loss.backward()
		optimizer.step()
		return loss


if __name__ == "__main__":
	config = Config_seq()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	config.batch_sz = 32
	model = RCNN_seq_attn(config).to(device)
	# for name, param in model.named_parameters():
	# 	if param.requires_grad:
	# 		print (name, param.data)
	# batch_sz, sent_embed_sz, num_titles)
	tech_indicators = torch.randn(5, 32,7).to(device)
	titles = torch.randn(32, 5, 25, config.title_dim, 56).to(device)
	# inputs = titles,tech_indicators
	y = model.forward(Variable(titles),Variable(tech_indicators))

	dot = torchviz.make_dot(y.mean(),params=dict(model.named_parameters()))
	#dot.view()

	y =torch.randint(0,2,(config.batch_sz,1))
	y = torch.squeeze(y)
	# ##print(y)
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
	num_iters=10000
	for t in range(num_iters):
	    # Forward pass: Compute predicted y by passing x to the model
	    y_pred = model.forward(titles, tech_indicators)
	    # ##print(y_pred)
	    # y_pred = torch.randn(config.batch_sz,1)
	    ###print("y_pred shape: {}".format(y_pred.size()))
	    # Compute and ##print loss
	    loss = model.backprop(optimizer, y_pred, y)
	    if t % 10== 0: print(t, loss.item())
	#
	#     # Zero gradients, perform a backward pass, and update the weights.