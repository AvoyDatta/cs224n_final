import torch
import torch.nn as nn 

class Config_base(): 
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
				p_drop = 0.5,
				num_batches = 1000,
				num_conv = 5
				):

		self.title_dim = title_dim
		self.n_hidden_LSTM_titles = n_hidden_LSTM_titles
		self.n_hidden_LSTM_tech = n_hidden_LSTM_tech
		self.input_dim_1 = n_filters
		self.input_dim_2 = n_tech_indicators

		self.n_outputs = n_outputs
		self.window_len = window_len
		self.filter_sz = filter_sz
		self.n_filters = n_filters
		self.pool_sz = pool_sz
		self.batch_sz = batch_sz
		self.num_batches = num_batches
		self.p_drop = p_drop
		self.num_conv = num_conv

class conv_pool_relu_dropout(nn.Module):
    def __init__(self, title_dim, n_filters,filter_sz,pool_sz,p_drop,num_conv):
        super(conv_pool_relu_dropout,self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv1d(title_dim, n_filters, , padding=1),
        #     nn.MaxPool1d(pool_sz,stride=1,padding=1)
        #     nn.reLU(),
        # )
        modules = []
        modules.append(nn.Conv1d(title_dim,n_filters,filter_sz))
        modules.append(nn.MaxPool1d(pool_sz,stride=1,padding=1))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(p_drop))
        for i in range(num_conv):
        	modules.append(nn.Conv1d(n_filters, n_filters,filter_sz, padding=1))
        	modules.append(nn.MaxPool1d(pool_sz,stride=1,padding=1))
        	modules.append(nn.ReLU())
        	modules.append(nn.Dropout(p_drop))

        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        x = self.conv(x)
        return x

"""
Currently skeleton code, fill with actual models as desired
"""

class RCNN_base(nn.Module):
	def __init__(self, config):
		''' 
		input_dim: Dimensionality of input corpus
		hidden_dim: Hidden layer dimensionality
		output_dim: Dimensionality of output of model (should be scalar)
		'''
		super(RCNN_base,self).__init__()
		self.config = config

		print(config.__dict__.values())
		# self.cnn = nn.Conv1d(config.title_dim, config.n_filters, config.filter_sz)
		# self.max_pool = nn.MaxPool1d(config.pool_sz, stride = 1, padding = 1) #Dim increases by 1
		# self.relu = nn.ReLU()
		# self.dropout = nn.Dropout(config.p_drop)

		self.cnn = conv_pool_relu_dropout(config.title_dim,config.n_filters,config.filter_sz,config.pool_sz,config.p_drop,config.num_conv)

		self.lstm1 = nn.LSTM(config.input_dim_1, config.n_hidden_LSTM_titles)
		self.lstm2 = nn.LSTM(config.input_dim_2, config.n_hidden_LSTM_tech)

		#Linearly project final hidden states from LSTM to sigmoid output
		self.map_to_out = nn.Linear(2 * (config.n_hidden_LSTM_titles + config.n_hidden_LSTM_tech), 
									config.n_outputs)

		self.softmax = nn.LogSoftmax() #MIGHT NEED TO EDIT DIM LATER
		self.criterion = nn.NLLLoss(reduction = True, reduce = 'mean')
		
	"""
	Forward pass for the RCNN.
	Inputs: 
		titles: Batched Tensor of news embedding vectors for a day. Each batch element has L titles, each title has dim m 
		tech_indicators: Batched tensor of tech indicators. Each batch element has indicators (dim 7) for a window of size n = 5. Expected dims: (seq_len, batch, input_dim)
	"""

	def forward(self, titles, tech_indicators):
		
		#Input: (batch_sz, sent_embed_sz, num_titles), i.e. (batch, m, L) in paper
		# print(titles.shape)
		conv_out = self.cnn(titles) #Out: (batch, num_filters, L - R + 1) , R is window length
		# print(conv_out.shape)
		# pool_out = self.max_pool(conv_out) #Out: (batch, num_filters, L - R + 2)
		# print(pool_out.shape)
		# relud_pool = self.relu(pool_out) #(batch, num_filters, L - R + 2)
		# print(relud_pool.shape)
		# dropped_relu = self.dropout(relud_pool) #(batch, num_filters, L - R + 2) #NOT SURE ABOUT DIMENSION DROPPED
		# print(dropped_relu.shape)
		relud_pool_reshaped = conv_out.permute(2, 0, 1)

		#Input : (seq_len, batch, input_dim)
		# print(relud_pool_reshaped.shape)
		_, (last_hidden_1, last_cell_1) = self.lstm1(relud_pool_reshaped) #Both hidden & cell are (1, batch, hidden_size)
		last_hidden_1, last_cell_1 = last_hidden_1.squeeze(0), last_cell_1.squeeze(0) # Both are now (batch, hidden_size)
		
		# print(last_hidden_1.shape)
		# print(tech_indicators.shape)
		_, (last_hidden_2, last_cell_2) = self.lstm2(tech_indicators.permute(1,0,2))
		last_hidden_2, last_cell_2 = last_hidden_2.squeeze(0), last_cell_2.squeeze(0) # Both are now (batch, hidden_size)
		
		# print(last_hidden_2.shape)
		concatenated = torch.cat((last_hidden_1, last_cell_1, last_hidden_2, last_cell_2), 1) #(batch, 2*h_dim_1 + 2*h_dim_2)

		output = self.softmax(self.map_to_out(concatenated)) #(batch, 2)
		# print(output.shape)
		return output

	def backprop(self,optimizer, logits, labels):

		optimizer.zero_grad()
		loss = self.criterion(logits,labels)
		loss.backward()
		optimizer.step()
		return loss


if __name__ == "__main__":
	config = Config_base()

	model = RCNN_base(config)
	# batch_sz, sent_embed_sz, num_titles)
	tech_indicators = torch.randn(config.batch_sz, 5,7)
	titles = torch.randn(config.batch_sz,config.title_dim,config.num_titles)

	y =torch.randint(0,2,(config.batch_sz,1))
	y = torch.squeeze(y)
	# print(y)
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
	num_iters=10000
	for t in range(num_iters):
	    # Forward pass: Compute predicted y by passing x to the model
	    y_pred = model.forward(titles,tech_indicators)
	    # print(y_pred)
	    # y_pred = torch.randn(config.batch_sz,1)
	    #print("y_pred shape: {}".format(y_pred.size()))
	    # Compute and print loss
	    loss = model.backprop(optimizer, y_pred, y)
	    if t % 10== 0: print(t, loss.item())

	    # Zero gradients, perform a backward pass, and update the weights.



