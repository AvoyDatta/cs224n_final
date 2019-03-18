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
# from RCNN_seq import Config_seq, RCNN_seq

from RCNN_base import Config_base, RCNN_base
from RCNN import Config, RCNN
from RCNN_concat_outputs import Config_concat,RCNN_concat_outputs

from torch.utils.data import DataLoader

main_model_path = "../../trained_models/RCNN_seq/RCNN_seq.pt"


batch_sz = 128
# config = Config_seq()
# model = RCNN_seq(config)
# load_path = main_model_path
#
# if (load_path != None):  # If model is retrained from saved ckpt
#
#
# 	print("Loading model from {}".format(load_path))
# 	if  torch.cuda.is_available():
# 		checkpoint = torch.load(load_path)
# 	else:
# 		checkpoint = torch.load(load_path,map_location='cpu')
# 	model.load_state_dict(checkpoint['model_state_dict'])
# 	# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# 	# epoch = checkpoint['epoch']
# 	# loss = checkpoint['loss']
# 	print("Model successfully loaded from {}".format(load_path))

# store


def run_svm(data_train,data_test,data_extended_test):
	# train
	train_input = []
	train_labels = []
	print("Loading training data for SVM...")
	with tqdm(total = len(data_train))as pbar:
		for i in range(len(data_train)):
			titles, tech_indicators, movement = data_train[i]['titles'],data_train[i]['tech_indicators'],data_train[i]['movement']
			titles = titles.unsqueeze(0)
			tech_indicators = tech_indicators.unsqueeze(0).permute(1, 0, 2)
			combined = np.concatenate((np.reshape(titles,-1),np.reshape(tech_indicators,-1)),axis=0)
			train_input.append(combined)
			train_labels.append(movement.numpy())
			pbar.update(1)
	train_input = np.stack(train_input,axis=0)
	train_input = np.squeeze(train_input) # causes issues
	train_labels = np.squeeze(np.array(train_labels))

	print("train input shape: ",train_input.shape )
	print("Training started...")
	svc = svm.SVC(kernel='rbf',verbose=True)
	svc.fit(train_input, train_labels)
	print("Training finished... making predictions...")
	train_pred = svc.predict(train_input)
	print("Training accuracy: ", np.mean(train_pred == train_labels))

	# test
	test_input = []
	test_labels = []
	print("Loading validation (test) data...")
	with tqdm(total = len(data_test))as pbar:
		for i in range(len(data_test)):
			titles, tech_indicators, movement = data_test[i]['titles'],data_test[i]['tech_indicators'],data_test[i]['movement']
			titles = titles.unsqueeze(0)
			tech_indicators = tech_indicators.unsqueeze(0).permute(1, 0, 2)
			combined = np.concatenate((np.reshape(titles, -1), np.reshape(tech_indicators, -1)), axis=0)
			# combined = np.reshape(tech_indicators,-1)
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
	print("Loading extended test data...")
	with tqdm(total = len(data_extended_test))as pbar:
		for i in range(len(data_extended_test)):
			titles, tech_indicators, movement = data_extended_test[i]['titles'],data_extended_test[i]['tech_indicators'],data_extended_test[i]['movement']
			titles = titles.unsqueeze(0)
			tech_indicators = tech_indicators.unsqueeze(0).permute(1, 0, 2)
			combined = np.concatenate((np.reshape(titles, -1), np.reshape(tech_indicators, -1)), axis=0)
			extended_test_input.append(combined)
			extended_test_labels.append(movement.numpy())
			pbar.update(1)

	extended_test_input = np.stack(extended_test_input,axis=0)
	extended_test_input = np.squeeze(extended_test_input)
	extended_test_labels = np.squeeze(np.array(extended_test_labels))

	extended_test_pred = svc.predict(extended_test_input)

	train_accuracy = np.mean(train_pred == train_labels)
	test_accuracy =  np.mean(test_pred == test_labels)
	extended_test_accuracy =  np.mean(extended_test_pred == extended_test_labels)
	print("Extended test accuracy : ", extended_test_accuracy)
	print(classification_report(extended_test_labels, extended_test_pred))

	return train_accuracy,test_accuracy,extended_test_accuracy


if __name__ == "__main__":
	data = data_utils.DJIA_Dataset('../../data/DJIA_table.csv', '../../data/Combined_News_DJIA.csv', start=None, end=None, randomize_sz=None)

	# q1 = 1600  # originally 1600
	# q2 = 1800  # 1800
	# q3 = 1980  # 1980
	train_split = 0.8
	test_split = 0.1
	extended_test_split = 0.1

	##chunk dataset
	window_sz = 25
	chunk_sz = len(data)/window_sz
	chunk_end_idxs = np.linspace(0,len(data),chunk_sz,dtype=int)
	print(chunk_end_idxs)
	chunk_train_accs = []
	chunk_test_accs = []
	chunk_extended_test_accs = []
	for i in range(1,len(chunk_end_idxs)): #dummy for loop
		chunk = utils.data.Subset(data,[j for j in range(chunk_end_idxs[i-1],chunk_end_idxs[i])])
		chunk_len = len(chunk)
		# print("chunk len: ",chunk_len)
		# print("chunk idxs: ",chunk_end_idxs[i-1],chunk_end_idxs[i])

		max_idx_train = int(train_split*len(chunk))
		max_idx_test = max_idx_train + int(test_split*len(chunk))
		max_idx_extended_test = max_idx_test + int(test_split*len(chunk))
		data_train = utils.data.Subset(chunk, [i for i in range(max_idx_train)])
		# dataloader_train = DataLoader(data_train, batch_size = int(config.batch_sz))

		data_test = utils.data.Subset(chunk,[i for i in range(max_idx_train,max_idx_test)])
		# print("data_CHANGE!_test length: ",len(data_CHANGE!_test))
		# dataloader_CHANGE!_test = DataLoader(data_CHANGE!_test,batch_size=int(batch_sz))

		data_extended_test = utils.data.Subset(chunk, [i for i in range(max_idx_test, max_idx_extended_test)])
		# print("data_test length: ",len(data_test))

		# dataloader_test = DataLoader(data_test, batch_size = batch_sz)

		train_acc, test_acc,extended_test_acc = run_svm(data_train,data_test,data_extended_test)
		chunk_train_accs.append(train_acc)
		chunk_test_accs.append(test_acc)
		chunk_extended_test_accs.append(extended_test_acc)
	print("avg training accuracy on windowed set: ",np.mean(chunk_train_accs))
	print("avg test accuracy on windowed set: ",np.mean(chunk_test_accs))
	print("avg extended_test accuracy on windowed set",np.mean(chunk_extended_test_accs))