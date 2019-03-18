import torch
import torch.nn as nn
from graphviz import Digraph
import re
import torch
from torch.autograd import Variable
import torchviz


class Config_v2(): 
	def __init__(self, 
				num_titles = 25, 
				title_dim = 300, 
				n_tech_indicators = 7,
				n_hidden_LSTM_titles_sentence = 256, 
				n_hidden_LSTM_titles_window = 128, 

				n_hidden_LSTM_tech = 128, 

				n_outputs = 2,
				batch_sz = 128,
				filter_sz_day = 4,
				n_filters_day= 256,
				window_len_titles = 3,
				window_len_tech = 3, 
				p_drop = 0.5,
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

		self.n_filters_day = n_filters_day
		self.batch_sz = batch_sz
		self.p_drop = p_drop
		self.num_titles = num_titles
		self.n_hidden_LSTM_tech = n_hidden_LSTM_tech
		self.n_hidden_LSTM_titles_sentence = n_hidden_LSTM_titles_sentence
		self.n_hidden_LSTM_titles_window = n_hidden_LSTM_titles_window
		self.max_title_len = 56 #Hard-coded
		self.num_LSTM_layers = num_LSTM_layers

		self.n_tech_indicators = n_tech_indicators

"""
RCNN_v2 WITH ATTENTION.
Titles:
Bi-LSTM layers are used to run over over each sentence and encode information into a hidden state. 
This embedding (for each title) in the same day is passed with all other titles in the same day into a CNN layer that performs ...D convolution to generate a final embedding for the day, for each of window_len days
The embedding is concatenated with a tech_indicators vector and passed into an LSTM.

"""
class RCNN_v2(nn.Module):
	def __init__(self, config, attn = False):

		super(RCNN_v2,self).__init__()
		self.config = config

		# ##print("RCNN_seq config: ", config.__dict__)
		self.conv_titles_day = nn.Conv1d(4 * config.n_hidden_LSTM_titles_sentence, config.n_filters_day, config.filter_sz_day)
		self.max_pool_day = nn.MaxPool1d(config.num_titles - config.filter_sz_day + 1)

		self.lstm_titles_sentence = nn.LSTM(config.title_dim, config.n_hidden_LSTM_titles_sentence, batch_first = True, num_layers = config.num_LSTM_layers)
		self.lstm_titles_window = nn.LSTM(config.n_filters_day, config.n_hidden_LSTM_titles_window, batch_first = False, num_layers = config.num_LSTM_layers)
		self.lstm_tech = nn.LSTM(config.n_tech_indicators, config.n_hidden_LSTM_tech, batch_first = False, num_layers = config.num_LSTM_layers)

		self.map_titles_down = nn.Linear(4 * config.n_hidden_LSTM_titles_window, 16)
		self.map_titles_out = nn.Linear(16, config.n_outputs)

		self.map_tech_down = nn.Linear(4 *config.n_hidden_LSTM_tech, 16)
		self.map_tech_out = nn.Linear(16, config.n_outputs)

		self.softmax = nn.Softmax(dim = 1)
		self.relu = nn.ReLU()
		
		self.log_softmax = nn.LogSoftmax(dim = 1) 
		self.criterion = nn.NLLLoss(reduction = True, reduce = 'mean')
		


		self.aggregate = nn.Linear(2 * config.n_outputs, config.n_outputs)
	"""
	Forward pass for the RCNN.
	Inputs: 
		titles: Batched Tensor of news embedding vectors. Shape: (Batch, window_len_days, num_titles_per_day, embed_size, max_words_in_title)
		tech_indicators: Batched tensor of tech indicators. Each batch element has indicators (dim 7) for a window of size n = 5. Expected dims: (seq_len, batch, input_dim)
	"""
	def forward(self, titles, tech_indicators):
		batch_sz = titles.size(0)

		#print("Input titles shape: ", titles.shape)
		#print("Input tech indicators: ", tech_indicators.shape)
		#########################Titles#################################################################

		titles_reshaped = titles.contiguous().view(batch_sz * titles.size(1) * titles.size(2), titles.size(4), titles.size(3)) 
		#(batch * window_len_days * num_titles_day, max_words, embed_size)
		#print("reshape initial:", titles_reshaped.shape)

		_, (sents_h, sents_c) = self.lstm_titles_sentence(titles_reshaped) #sents_h: (2, new_batch, n_hidden)
		#print("Sentence embeddings: ", sents_h.shape)
		cat_sents = torch.cat((sents_h[0, :, :].squeeze(0), sents_h[1, :, :].squeeze(0), sents_c[0, :, :].squeeze(0), sents_c[1, :, :].squeeze(0)), 1) #(batch * window_len_days * num_titles_day, 4*n_hidden)
		#print("sents cat:", cat_sents.shape)

		cat_sent_embeds = cat_sents.contiguous().view(batch_sz * self.config.window_len_titles, self.config.num_titles, 4 * self.config.n_hidden_LSTM_titles_sentence) 
		#(batch * window_days,  num_titles, 4*n_hidden_sents)

		cat_sent_embeds = cat_sent_embeds.permute(0, 2, 1) #(batch * window_days,  4*n_hidden_sents, num_titles)

		conv_day_out = self.conv_titles_day(cat_sent_embeds) #(batch * window_days, n_filters_day, num_titles - filter_sz_day + 1)
				
		#print("Conv_day_out: ", conv_day_out.shape)
		
		conv_day_pool_out = self.max_pool_day(conv_day_out) #Out: (batch * window_len_titles, n_filters_day, 1)
		#print("Conv day pool out: ", conv_day_pool_out.shape)
		
		conv_day_pool_out = conv_day_pool_out.contiguous().view(batch_sz, self.config.window_len_titles, self.config.n_filters_day, 1)
		#(batch, window_len_titles, num_filters_day, 1)
		#print("reshape after conv pool:", conv_day_pool_out.shape)

		conv_day_pool_out = conv_day_pool_out.squeeze(-1) #Out: (batch, window_len_titles, n_filters_day)

		relud_pool_day = self.relu(conv_day_pool_out) #Out: (batch, window_len_titles, num_filters_day)
		#print("Relud_pool_day: ", relud_pool_day.shape)

		# dropped_relu = self.dropout(relud_pool) #(batch, num_filters, L - R + 2) #NOT SURE ABOUT DIMENSION DROPPED
		##print(dropped_relu.shape)
		relud_pool_day_reshaped = relud_pool_day.permute(1, 0, 2) #Out: (window_len_days, batch, num_filters_day)


		_, (titles_h, titles_c) = self.lstm_titles_window(relud_pool_day_reshaped) #titles_h: (2, batch, n_hidden_titles_window)

		cat_titles_days = torch.cat((titles_h[0, :, :].squeeze(0), titles_h[1, :, :].squeeze(0), titles_c[0, :, :].squeeze(0), titles_c[1, :, :].squeeze(0)), 1) #(batch, 4 * n_hidden_titles_window)
		#print("cat titles day", cat_titles_days.shape)

		mapped_down_titles = self.relu(self.map_titles_down(cat_titles_days)) #(batch, 16)
		titles_out = self.relu(self.map_titles_out(mapped_down_titles))  #(batch, 2)
		#print(titles_out.data)
		#print("Titles out: ", titles_out.shape)
		#########################Tech Indicators################################################

		_, (tech_h, tech_c) = self.lstm_tech(tech_indicators) #titles_h: (2, batch, n_hidden_tech)

		cat_tech = torch.cat((tech_h[0, :, :].squeeze(0), tech_h[1, :, :].squeeze(0), tech_c[0, :, :].squeeze(0), tech_c[1, :, :].squeeze(0)), 1) #(batch, 4 * n_hidden_titles_window)
		#print("cat_tech", cat_tech.shape)

		mapped_down_tech = self.relu(self.map_tech_down(cat_tech)) #(batch, 16)
		tech_out = self.relu(self.map_tech_out(mapped_down_tech))  #(batch, 2)
		#print(tech_out.data)
		#print("tech_out", tech_out.shape)
		###############################Combined##################################################
		output = self.aggregate(torch.cat((titles_out, tech_out), 1)) #(batch, 2) 

		#print(output.data)
		# attn_proj = torch.matmul(lstm_outputs_reshaped, self.attn_vector)
		# attn_proj = attn_proj.squeeze(-1)

		output = self.log_softmax(output) #(batch, 2)
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
	config = Config_v2()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	config.batch_sz = 32
	model = RCNN_v2(config).to(device)
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
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
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