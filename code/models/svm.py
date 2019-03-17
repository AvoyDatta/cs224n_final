from sklearn import svm
from sklearn.metrics import classification_report
from docopt import docopt

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time
import os
import sys
import torch.utils as utils
import pickle
# sys.path.append(os.path.join(os.getcwd(),'..\\..\\data'))
# sys.path.append(os.path.join(os.getcwd(),'..\\..\\data\\glove.6B'))
sys.path.append('../../data/')
sys.path.append('../../data/glove.6B/')
print(sys.path)

import data_utils
from RCNN_seq import Config_seq, RCNN_seq

from RCNN_base import Config_base, RCNN_base
from RCNN import Config, RCNN
from RCNN_concat_outputs import Config_concat,RCNN_concat_outputs

from torch.utils.data import DataLoader

main_model_path = "../../trained_models/RCNN_seq/RCNN_seq.pt"


batch_sz = 128
config = Config_seq()
model = RCNN_seq(config)
load_path = main_model_path

if (load_path != None):  # If model is retrained from saved ckpt


	print("Loading model from {}".format(load_path))
	if  torch.cuda.is_available():
		checkpoint = torch.load(load_path)
	else:
		checkpoint = torch.load(load_path,map_location='cpu')
	model.load_state_dict(checkpoint['model_state_dict'])
	# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	# epoch = checkpoint['epoch']
	# loss = checkpoint['loss']
	print("Model successfully loaded from {}".format(load_path))

# store
data = data_utils.DJIA_Dataset('../../data/DJIA_table.csv', '../../data/Combined_News_DJIA.csv', start=None, end=None, randomize_sz=None)

q1 = 1600  # originally 1600
q2 = 1800  # 1800
q3 = 1980  # 1980

data_train = utils.data.Subset(data, [i for i in range(q1)])
# dataloader_train = DataLoader(data_train, batch_size = int(config.batch_sz))

data_val = utils.data.Subset(data,[i for i in range(q1+1,q2)])
# dataloader_val = DataLoader(data_val,batch_size=int(batch_sz))

data_test = utils.data.Subset(data, [i for i in range(q2+1, q3)])

# dataloader_test = DataLoader(data_test, batch_size = batch_sz)


def run_svm():
	# train
	train_input = []
	train_labels = []
	print("Loading training data for SVM...")
	# with tqdm(total = len(dataloader_train)) as pbar: # 13
	# 	for index, sample in enumerate(dataloader_train):
	with tqdm(total = len(data_train))as pbar:
		for i in range(len(data_train)):
			titles, tech_indicators, movement = data_train[i]['titles'],data_train[i]['tech_indicators'],data_train[i]['movement']
			titles = titles.unsqueeze(0)
			tech_indicators = tech_indicators.unsqueeze(0).permute(1, 0, 2)
			combined = np.concatenate((np.reshape(titles[:,4,:,:,:],-1),np.reshape(tech_indicators,-1)),axis=0)
			# combined = concat_titles_tech(titles, tech_indicators).detach().numpy() #(5,128,71)
			# combined = np.reshape(combined[4],newshape=-1)
			# print(combined.shape)
			train_input.append(combined)
			train_labels.append(movement.numpy())
			pbar.update(1)
	train_input = np.stack(train_input,axis=0)
	train_input = np.squeeze(train_input) # causes issues
	train_labels = np.squeeze(np.array(train_labels))

	print("train input shape: ",train_input.shape )
	print("Training started...")
	svc = svm.SVC(kernel='rbf')
	svc.fit(train_input, train_labels)
	print("Training finished... making predictions...")
	train_pred = svc.predict(train_input)
	print("Training accuracy: ", np.mean(train_pred == train_labels))

	# val
	val_input = []
	val_labels = []
	print("Loading validation data...")
	with tqdm(total = len(data_val))as pbar:
		for i in range(len(data_val)):
			titles, tech_indicators, movement = data_val[i]['titles'],data_val[i]['tech_indicators'],data_val[i]['movement']
			titles = titles.unsqueeze(0)
			tech_indicators = tech_indicators.unsqueeze(0).permute(1, 0, 2)
			combined = np.concatenate((np.reshape(titles[:, 4, :, :, :], -1), np.reshape(tech_indicators, -1)), axis=0)
			# combined = concat_titles_tech(titles, tech_indicators).detach().numpy() #(5,128,71)
			# combined = np.reshape(combined[4],newshape=-1)
			val_input.append(combined)
			val_labels.append(movement.numpy())
			pbar.update(1)

	val_input = np.stack(val_input,axis=0)
	val_input = np.squeeze(val_input)
	val_labels = np.squeeze(np.array(val_labels))

	print("Validation started...")
	# svc.fit(val_input, val_labels)
	print("Validation finished... making predictions...")
	val_pred = svc.predict(val_input)
	print("Validation accuracy: ", np.mean(val_pred == val_labels))


	# test
	test_input = []
	test_labels = []
	print("Loading test data...")
	with tqdm(total = len(data_test))as pbar:
		for i in range(len(data_test)):
			titles, tech_indicators, movement = data_test[i]['titles'],data_test[i]['tech_indicators'],data_test[i]['movement']
			titles = titles.unsqueeze(0)
			tech_indicators = tech_indicators.unsqueeze(0).permute(1, 0, 2)
			combined = np.concatenate((np.reshape(titles[:, 4, :, :, :], -1), np.reshape(tech_indicators, -1)), axis=0)
			# combined = concat_titles_tech(titles, tech_indicators).detach().numpy() #(5,128,71)
			# combined = np.reshape(combined[4],newshape=-1)
			test_input.append(combined)
			test_labels.append(movement.numpy())
			pbar.update(1)

	test_input = np.stack(test_input,axis=0)
	test_input = np.squeeze(test_input)
	test_labels = np.squeeze(np.array(test_labels))

	test_pred = svc.predict(test_input)
	print("Test accuracy: ", np.mean(test_pred == test_labels))
	print(classification_report(test_labels, test_pred))


def concat_titles_tech(titles, tech_indicators):
	batch_sz = titles.size(0)

	# print("Input titles shape: ", titles.shape)
	# print("Input tech indicators: ", tech_indicators.shape)

	titles_reshaped = titles.contiguous().view(batch_sz * titles.size(1) * titles.size(2), titles.size(3), titles.size(4)) 
	conv_out_titles = model.conv_title(titles_reshaped) #Out: (batch, window_len_days, num_titles_day, num_filters_title, words_title - filter_sz_title + 1)

	conv_title_pool_out = model.max_pool_title(conv_out_titles) #Out: (batch * window_len_days * num_titles_day, num_filters_title, 1)
	
	n_filters_title = 128
	conv_title_pool_out = conv_title_pool_out.contiguous().view(batch_sz, titles.size(1), titles.size(2), 128, 1) 

	conv_title_pool_out = conv_title_pool_out.squeeze(-1) #Out: (batch, window_len_days, num_titles_day, num_filters_title)
	relud_pool_title = model.relu_title(conv_title_pool_out) #(batch, window_len_days, num_titles_day, num_filters_title)
	relud_pool_title_reshaped = relud_pool_title.permute(0, 1, 3, 2) #(batch, window_len_days, num_filters_title, num_titles_day)
	relud_pool_title_reshaped = relud_pool_title_reshaped.contiguous().view(batch_sz * config.window_len_titles, 
																			config.n_filters_title, config.num_titles) #(batch * window_len_days, num_filters_title, num_titles_day)
	conv_out_day = model.conv_day(relud_pool_title_reshaped) #Out: (batch * window_len_days, num_filters_day, num_titles_day - kernel_sz_day + 1)
	
	conv_day_pool_out = model.max_pool_title(conv_out_day) #Out: (batch * window_len_days, num_filters_day, 1)
	
	conv_day_pool_out = conv_day_pool_out.contiguous().view(batch_sz, titles.size(1), config.n_filters_day, 1)

	conv_day_pool_out = conv_day_pool_out.squeeze(-1) #Out: (batch, window_len_days, num_filters_day)

	relud_pool_day = model.relu_day(conv_day_pool_out) #Out: (batch, window_len_days, num_filters_day)

	relud_pool_day_reshaped = relud_pool_day.permute(1, 0, 2) #Out: (window_len_days, batch, num_filters_day)

	concat_input = torch.cat((relud_pool_day_reshaped, tech_indicators), dim = 2)
	return concat_input


run_svm()
