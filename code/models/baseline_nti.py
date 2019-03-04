#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline
baseline.py

Usage:
	baseline.py train --batch_sz=<int> --num_batches=<int> --print_every=<int> --save_every=<int> --num_epochs=<int>
	baseline.py test --batch_sz=<int>

Options:
	--num_batches=<int>               Number of minibatches per epoch[default:1000]
	--print_every=<int>               Specifies frequency of epochs with which metrics are printed [default: 1]
	--save_every=<int>                Specifies frequency of epochs with which model is saved [default: 5]
	--data_path=<file>                Path to data
	--num_epochs=<int>                Number of epochs to train for
	--batch_sz=<int>                  Batch size [default: 128]
"""
from docopt import docopt

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time
import os
import sys
import torch.utils as utils
# sys.path.append(os.path.join(os.getcwd(),'..\\..\\data'))
# sys.path.append(os.path.join(os.getcwd(),'..\\..\\data\\glove.6B'))
sys.path.append('../../data/')
sys.path.append('../../data/glove.6B/')
print(sys.path)

import data_utils

from RCNN_nti import Config, RCNN_nti 
from torch.utils.data import DataLoader


baseline_model_path = "../../trained_models/baseline/baseline_nti.pt"

def backprop(optimizer, logits, labels):

	optimizer.zero_grad()
	loss = nn.NLLLoss(logits, labels, reduce = True, reduction = 'mean')
	loss.backward()
	optimizer.step()

	return loss

def get_accuracy(logits, labels):

	return torch.mean(torch.eq(torch.argmax(logits, dim = 1), labels).float())
	
def train(args, config):
	"""
	Train baseline model
	"""

	print("Training initiated...")

	baseline_step = 0.1
	baseline_momentum = 0.9
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print_every = int(args['--print_every']) 
	num_epochs = int(args['--num_epochs'])
	save_every = int(args['--save_every']) 
	save_path = baseline_model_path 
	num_batches = int(args['--num_batches'])


	#Stores hyperparams for model
	

	model = RCNN_nti(config)
	model.to(device)

	optimizer = torch.optim.SGD(model.parameters(), lr= baseline_step, momentum = baseline_momentum)

	# train_data_path = args['--data_path'] 
	# if train_data_path == None: 
	# 	raise Exception("Training data path not fed in.")

	# print("Loading training data from {}".format(train_data_path))


	##############LOAD TRAIN DATA and initiate train

	data = data_utils.DJIA_Dataset('../../data/DJIA_table.csv', '../../data/Combined_News_DJIA.csv')
	data_train = utils.data.Subset(data, [i for i in range(1800)])

	dataloader_train = DataLoader(data_train, batch_size = int(config.batch_sz))

	#print("Finished loading training data from {}".format(train_data_path))


	##Read in data from train_data_path
	## Parse news titles into averages of word embeddings over title

	train_accs = []
	train_losses = []
	train_ctr = 0

	init_epoch = 0

	# if (load_path != 'None'):  #If model is retrained from saved ckpt
		
	# 	checkpoint = torch.load(load_path)
	# 	model.load_state_dict(checkpoint['model_state_dict'])
	# 	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	# 	epoch = checkpoint['epoch']
	# 	loss = checkpoint['loss']

	model.train()

	try: 
		for epoch in range(init_epoch, num_epochs):
			start = time.time()

			#INITIATE dataloader_train
			with tqdm(total = num_batches * config.batch_sz) as pbar:
				for index, sample in enumerate(dataloader_train):

					titles, tech_indicators, movement = sample['titles'], sample['tech_indicators'], sample['movement']
					titles.to(device)
					tech_indicators.to(device)
					movement.to(device)

					logits = model.forward(titles)
					loss = model.backprop(optimizer, logits, movement)
					train_ctr += 1
					accuracy = get_accuracy(logits, movement) #Accuracy over entire mini-batch
					
					train_losses.append(loss)
					train_accs.append(accuracy)

			if epoch % print_every == 0: 
				print ("Epoch: {}, Training iter: {}, Time since start: {}, Loss: {}, Training Accuracy: {}".format(epoch, train_ctr, (time.time() - start), loss, accuracy))

			if epoch % save_every == 0:
				print ("Saving model to {}".format(save_path))
				torch.save({'epoch': epoch,
							'model_state_dict': model.state_dict(), 
							'optimizer_state_dict': optimizer.state_dict(), 
							'loss': loss}, 
							save_path)
				print ("Saved successfully to {}".format(save_path))

	except KeyboardInterrupt:
		print("Training interupted...")
		print ("Saving model to {}".format(save_path))
		torch.save({'epoch': epoch,
					'model_state_dict': model.state_dict(), 
					'optimizer_state_dict': optimizer.state_dict(), 
					'loss': loss}, 
					save_path)
		print ("Saved successfully to {}".format(save_path))
			

	print("Training completed.")

	return (train_losses, train_accs, loss, accuracy) 

def test(args, config):
	#Get test data & parse

	#Initiate dataloader_test

	#Load saved model

	# test_data_path = args['--data_path'] 
	# if test_data_path == None: 
	# 	raise Exception("Test data path not fed in.")

	# print("Loading test data from {}".format(train_data_path))


	##############LOAD TEST DATA and initiate dataloader as dataloader_test

	data = data_utils.DJIA_Dataset('../../data/DJIA_table.csv', '../../data/Combined_News_DJIA.csv')

	data_test = utils.data.Subset(data, [i for i in range(1800, 1980)])

	dataloader_test = DataLoader(data_test, batch_size = config.batch_sz)

	#print("Finished loading training data from {}".format(train_data_path))

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	load_path = baseline_model_path 

	model = RCNN_nti(config)
	model.to(device)

	if (load_path != None):  #If model is retrained from saved ckpt
		
		print("Loading model from {}".format(load_path))
		checkpoint = torch.load(load_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		#epoch = checkpoint['epoch']
		#loss = checkpoint['loss']
		print("Model successfully loaded from {}".format(load_path))

	model.eval()

	start = time.time()
	print("Testing ...")

	test_loss = []
	test_accuracy = []

	#INITIATE dataloader_test
	# len(dataloader_test) should be int(len(test_data) / test_batch_sz)

	with tqdm(total = len(dataloader_test)) as pbar:
		for index, sample in enumerate(dataloader_test):

			titles, tech_indicators, movement = sample['titles'], sample['tech_indicators'], sample['movement']
			titles.to(device)
			tech_indicators.to(device)
			movement.to(device)

			logits = model.forward(titles)
			temp_criterion = nn.NLLLoss(reduce = True, reduction = 'mean')
			loss = temp_criterion(logits, movement)
			accuracy = get_accuracy(logits, movement) #Accuracy over entire mini-batch
			print(loss, accuracy)

			test_loss.append(loss.detach().numpy())

			test_accuracy.append(accuracy.detach().numpy())
			#Training step

	print(test_loss, test_accuracy)
	test_loss = np.mean(np.array(test_loss))
	test_accuracy = np.mean(np.array(test_accuracy))

	return (test_loss, test_accuracy)


def main():
	args = docopt(__doc__)

	config = Config(batch_sz = int(args['--batch_sz']))

	if args['--num_batches']: config.num_batches = int(args['--num_batches'])

	if args['train']:

		train_losses, train_accs, loss, accuracy = train(args, config)
		np.save('train_accs.npy', np.array(train_accs))
		np.save('train_losses.npy', np.array(train_losses))
		print("Final training loss: {}, Final Training Accuracy: {}".format(loss, accuracy))

	elif args['test']:
		loss, accuracy = test(args, config)
		print("Test loss: {}, Test accuracy: {}".format(loss, accuracy))


if __name__ == "__main__":

	main()
