import torch.nn as nn
import torch


class Config_concat():
	def __init__(self,
				num_titles = 25,
				title_dim = 300,
				n_tech_indicators = 7,
				n_hidden_LSTM_titles = 128,
				n_hidden_LSTM_tech = 128,
				n_outputs = 2,
				batch_sz = 128,
				pool_sz = 2,
				filter_sz = 4,
				n_filters = 64,
				window_len = 5,
				p_drop = 0.5,
				num_batches = 1000):

		self.num_titles = num_titles
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


class RCNN_concat_outputs(nn.Module):
	def __init__(self, config):
		'''
		input_dim: Dimensionality of input corpus
		hidden_dim: Hidden layer dimensionality
		output_dim: Dimensionality of output of model (should be scalar)
		'''
		super(RCNN_concat_outputs, self).__init__()
		self.config = config

		print(config.__dict__.values())
		self.cnn = nn.Conv1d(config.title_dim, config.n_filters, config.filter_sz)
		self.max_pool = nn.MaxPool1d(config.pool_sz, stride=1, padding=1)  # Dim increases by 1
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(config.p_drop)
		self.lstm1 = nn.LSTM(config.input_dim_1, config.n_hidden_LSTM_titles)
		self.lstm2 = nn.LSTM(config.input_dim_2, config.n_hidden_LSTM_tech)

		# Linearly project outputs of LSTM_titles and LSTM_technical
		self.map_to_out = nn.Linear(config.n_hidden_LSTM_titles,
									config.n_outputs)

		self.softmax = nn.LogSoftmax(dim=1)  # MIGHT NEED TO EDIT DIM LATER
		self.criterion = nn.NLLLoss(reduction=True, reduce='mean')


	def forward(self, titles, tech_indicators):
		# Input: (batch_sz, sent_embed_sz, num_titles), i.e. (batch, m, L) in paper
		# print(titles.shape)
		conv_out = self.cnn(titles)  # Out: (batch, num_filters, L - R + 1) , R is window length
		# print(conv_out.shape)
		# pool_out = self.max_pool(conv_out) #Out: (batch, num_filters, L - R + 2)
		# print(pool_out.shape)
		# relud_pool = self.relu(pool_out) #(batch, num_filters, L - R + 2)
		# print(relud_pool.shape)
		# dropped_relu = self.dropout(relud_pool) #(batch, num_filters, L - R + 2) #NOT SURE ABOUT DIMENSION DROPPED
		# print(dropped_relu.shape)
		relud_pool_reshaped = conv_out.permute(2, 0, 1)

		# Input : (seq_len, batch, input_dim)
		# print(relud_pool_reshaped.shape)
		output_1, (last_hidden_1, last_cell_1) = self.lstm1(relud_pool_reshaped)  # Both hidden & cell are (1, batch, hidden_size)
		last_hidden_1, last_cell_1 = last_hidden_1.squeeze(0), last_cell_1.squeeze(0)  # Both are now (batch, hidden_size)

		# print(last_hidden_1.shape)
		# print(tech_indicators.shape)
		output_2, (last_hidden_2, last_cell_2) = self.lstm2(tech_indicators.permute(1, 0, 2))
		last_hidden_2, last_cell_2 = last_hidden_2.squeeze(0), last_cell_2.squeeze(0)  # Both are now (batch, hidden_size)

		output_1,output_2 = output_1.squeeze(0),output_2.squeeze(0)

		# print(output_1.shape)
		# print(output_2.shape)
		# print(last_hidden_2.shape)
		concatenated = torch.cat((output_1,output_2),0)
		# print(concatenated.shape)
		concatenated = torch.max(concatenated,0)[0]
		# print(concatenated.shape)

		output = self.softmax(self.map_to_out(concatenated))  # (batch, 2)
		# print(output.shape)
		return output

	def backprop(self, optimizer, logits, labels):
		# print("got to backprop")
		optimizer.zero_grad()
		# print('optimizer zero grads')
		# print("shape logits: {} shape labels: {} ".format(logits.shape,labels.shape))
		loss = self.criterion(logits, labels)
		# print("got to loss")
		loss.backward()
		optimizer.step()
		return loss


if __name__ == "__main__":
	config = Config_concat()

	model = RCNN_concat_outputs(config)
	# batch_sz, sent_embed_sz, num_titles)
	tech_indicators = torch.randn(config.batch_sz, 5,7)
	titles = torch.randn(config.batch_sz,config.title_dim,config.num_titles)

	y = torch.randn(config.batch_sz)
	y[y > 0.5 ] = 1
	y[y <= 0.5] = 0
	# print(y)
	y = y.type(torch.LongTensor)
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
	num_iters=10000
	for t in range(num_iters):
		# Forward pass: Compute predicted y by passing x to the model
		y_pred = model.forward(titles,tech_indicators)
		# print(y_pred.shape)
		# y_pred = torch.randn(config.batch_sz,1)
		#print("y_pred shape: {}".format(y_pred.size()))
		# Compute and print loss
		loss = model.backprop(optimizer, y_pred, y)
		# print(loss)
		if t % 10== 0: print(t, loss.item())
