import torch
import torch.nn as nn 

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


class RCNN_seq(nn.Module):
	def __init__(self, config):

		super(RCNN_seq,self).__init__()
		self.config = config

		#print("RCNN_seq config: ", config.__dict__)
		self.conv_title = nn.Conv1d(config.title_dim, config.n_filters_title, config.filter_sz_title)

		self.max_pool_title = nn.MaxPool1d(config.max_title_len - config.filter_sz_title + 1)
		self.relu_title = nn.ReLU()
		self.relu_day = nn.ReLU()

		#self.dropout = nn.Dropout(config.p_drop)
		self.conv_day = nn.Conv1d(config.n_filters_title, config.n_filters_day, config.filter_sz_day)
		self.max_pool_day = nn.MaxPool1d(config.num_titles - config.filter_sz_day + 1)

		self.lstm = nn.LSTM(config.input_dim_LSTM, config.n_hidden_LSTM, num_layers = config.num_LSTM_layers)

		#self.lstm2 = nn.LSTM(config.input_dim_2, config.n_hidden_LSTM_tech)

		self.map_down = nn.Linear(2 * config.num_LSTM_layers * config.n_hidden_LSTM, 64)
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
		
		tech_indicators = tech_indicators.permute(1,0,2)

		#print("Input titles shape: ", titles.shape)
		titles_reshaped = titles.contiguous().view(titles.size(0) * titles.size(1) * titles.size(2), titles.size(3), titles.size(4)) 
		#(batch * window_len_days * num_titles_day, num_filters_title, words_title - filter_sz_title + 1)
		#print("reshape # 1:", titles_reshaped.shape)

		conv_out_titles = self.conv_title(titles_reshaped) #Out: (batch, window_len_days, num_titles_day, num_filters_title, words_title - filter_sz_title + 1)
		#print("conv_out_titles: ", conv_out_titles.shape)


		conv_title_pool_out = self.max_pool_title(conv_out_titles) #Out: (batch * window_len_days * num_titles_day, num_filters_title, 1)
		
		conv_title_pool_out = conv_title_pool_out.contiguous().view(titles.size(0), titles.size(1), titles.size(2), self.config.n_filters_title, 1) 
		#(batch, window_len_days, num_titles_day, num_filters_title, 1)
		#print("reshape back # 1:", conv_title_pool_out.shape)


		conv_title_pool_out = conv_title_pool_out.squeeze(-1) #Out: (batch, window_len_days, num_titles_day, num_filters_title)

		#print("Title pool: ",conv_title_pool_out.shape)

		relud_pool_title = self.relu_title(conv_title_pool_out) #(batch, window_len_days, num_titles_day, num_filters_title)
		# #print(relud_pool.shape)
		# dropped_relu = self.dropout(relud_pool) #(batch, num_filters, L - R + 2) #NOT SURE ABOUT DIMENSION DROPPED
		# #print(dropped_relu.shape)
		relud_pool_title_reshaped = relud_pool_title.permute(0, 1, 3, 2) #(batch, window_len_days, num_filters_title, num_titles_day)

		relud_pool_title_reshaped = relud_pool_title_reshaped.contiguous().view(self.config.batch_sz * self.config.window_len_titles, 
																				self.config.n_filters_title, self.config.num_titles) #(batch * window_len_days, num_filters_title, num_titles_day)

		#print("Conv day input :", relud_pool_title_reshaped.shape)

		conv_out_day = self.conv_day(relud_pool_title_reshaped) #Out: (batch * window_len_days, num_filters_day, num_titles_day - kernel_sz_day + 1)
		
		#print("Conv_out_day: ", conv_out_day.shape)
		
		conv_day_pool_out = self.max_pool_title(conv_out_day) #Out: (batch * window_len_days, num_filters_day, 1)
		#print("Conv day pool out: ", conv_day_pool_out.shape)
		
		conv_day_pool_out = conv_day_pool_out.contiguous().view(titles.size(0), titles.size(1), self.config.n_filters_day, 1)
		#(batch, window_len_days, num_filters_day, 1)
		#print("reshape back # 2:", conv_day_pool_out.shape)

		conv_day_pool_out = conv_day_pool_out.squeeze(-1) #Out: (batch, window_len_days, num_filters_day)

		relud_pool_day = self.relu_day(conv_day_pool_out) #Out: (batch, window_len_days, num_filters_day)
		#print("Relud_pool_day: ", relud_pool_day.shape)

		# dropped_relu = self.dropout(relud_pool) #(batch, num_filters, L - R + 2) #NOT SURE ABOUT DIMENSION DROPPED
		# #print(dropped_relu.shape)
		relud_pool_day_reshaped = relud_pool_day.permute(1, 0, 2) #Out: (window_len_days, batch, num_filters_day)

		concat_input = torch.cat((relud_pool_day_reshaped, tech_indicators), dim = 2) #Size: (window_len_days==seq_len_tech, batch, input_dim_tech + num_filters_day)

		#print("LSTM inputs: ", concat_input.shape)

		lstm_outputs, (last_hidden, last_cell) = self.lstm(concat_input) #outputs shape: (seq_len, batch, hidden_sz). Both hidden & cell are (2, batch, hidden_size)

		last_hidden = last_hidden.view(config.batch_sz, -1) #out: (batch, 2 * hidden_size)
		last_cell = last_cell.view(config.batch_sz, -1)

		lstm_hidden_concat = torch.cat((last_hidden, last_cell), 1)

		end_states_concatenated = torch.cat((last_hidden, last_cell), dim = 1)

		#print("Concatenated end states: ", end_states_concatenated.shape)

		mapped_down = self.map_down(end_states_concatenated)

		output = self.log_softmax(self.map_to_out(mapped_down)) #(batch, 2)
		# #print(output.shape)
		return output

	def backprop(self,optimizer, logits, labels):

		optimizer.zero_grad()
		loss = self.criterion(logits,labels)
		loss.backward()
		optimizer.step()
		return loss


if __name__ == "__main__":
	config = Config_seq()

	model = RCNN_seq(config)
	# batch_sz, sent_embed_sz, num_titles)
	tech_indicators = torch.randn(config.batch_sz, 5,7)
	titles = torch.randn(config.batch_sz, 5, 25, config.title_dim, 56)

	y =torch.randint(0,2,(config.batch_sz,1))
	y = torch.squeeze(y)
	# #print(y)
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
	num_iters=10000
	for t in range(num_iters):
	    # Forward pass: Compute predicted y by passing x to the model
	    y_pred = model.forward(titles,tech_indicators.permute(1,0,2))
	    # #print(y_pred)
	    # y_pred = torch.randn(config.batch_sz,1)
	    ##print("y_pred shape: {}".format(y_pred.size()))
	    # Compute and #print loss
	    loss = model.backprop(optimizer, y_pred, y)
	    if t % 10== 0: print(t, loss.item())

	    # Zero gradients, perform a backward pass, and update the weights.



# class conv_pool_relu_dropout(nn.Module):
#     def __init__(self, title_dim, n_filters,filter_sz,pool_sz,p_drop,num_conv):
#         super(conv_pool_relu_dropout,self).__init__()
#         # self.conv = nn.Sequential(
#         #     nn.Conv1d(title_dim, n_filters, , padding=1),
#         #     nn.MaxPool1d(pool_sz,stride=1,padding=1)
#         #     nn.reLU(),
#         # )
#         modules = []
#         modules.append(nn.Conv1d(title_dim,n_filters,filter_sz))
#         modules.append(nn.MaxPool1d(pool_sz,stride=1,padding=1))
#         modules.append(nn.ReLU())
#         modules.append(nn.Dropout(p_drop))
#         for i in range(num_conv):
#         	modules.append(nn.Conv1d(n_filters, n_filters,filter_sz, padding=1))
#         	modules.append(nn.MaxPool1d(pool_sz,stride=1,padding=1))
#         	modules.append(nn.ReLU())
#         	modules.append(nn.Dropout(p_drop))

#         self.conv = nn.Sequential(*modules)

#     def forward(self, x):
#         x = self.conv(x)
#         return x

		############Previous dims. For reference only. ################
		#Input: (batch_sz, sent_embed_sz, num_titles), i.e. (batch, m, L) in paper
		#Out conv: (batch, num_filters, L - R + 1) , R is window length
		################################################################