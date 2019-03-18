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
data = data_utils.DJIA_Dataset('../../data/DJIA_table.csv', '../../data/Combined_News_DJIA.csv', start=1, end=510, randomize_sz=None)

q1 = 400  # originally 1600
q2 = 451  # 1800
q3 = 502  # 1980

data_train = utils.data.Subset(data, [i for i in range(q1)])

data_test = utils.data.Subset(data,[i for i in range(q1+1,q2)])

data_extended_test = utils.data.Subset(data, [i for i in range(q2+1, q3)])

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

	print("Test started...")
	print("Test finished... making predictions...")
	test_pred = svc.predict(test_input)
	print("Test accuracy: ", np.mean(test_pred == test_labels))


	# extended test
	extended_test_input = []
	extended_test_labels = []
	print("Loading extended_test data...")
	with tqdm(total = len(data_extended_test))as pbar:
		for i in range(len(data_extended_test)):
			titles, tech_indicators, movement = data_extended_test[i]['titles'],data_extended_test[i]['tech_indicators'],data_extended_test[i]['movement']
			titles = titles.unsqueeze(0)
			tech_indicators = tech_indicators.unsqueeze(0).permute(1, 0, 2)
			combined = np.concatenate((np.reshape(titles[:, 4, :, :, :], -1), np.reshape(tech_indicators, -1)), axis=0)
			# combined = concat_titles_tech(titles, tech_indicators).detach().numpy() #(5,128,71)
			# combined = np.reshape(combined[4],newshape=-1)
			extended_test_input.append(combined)
			extended_test_labels.append(movement.numpy())
			pbar.update(1)

	extended_test_input = np.stack(extended_test_input,axis=0)
	extended_test_input = np.squeeze(extended_test_input)
	extended_test_labels = np.squeeze(np.array(extended_test_labels))

	extended_test_pred = svc.predict(extended_test_input)
	print("extended_Test accuracy: ", np.mean(extended_test_pred == extended_test_labels))
	print(classification_report(extended_test_labels, extended_test_pred))



run_svm()
