#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main model
run.py

Usage:
	run.py train --batch_sz=<int> --lr=<float> --print_every=<int> --save_every=<int> --num_epochs=<int> --randomize_sz=<int>
	run.py test --batch_sz=<int>

Options:
	--lr=<float>               		  Learning rate [default: 0.01]
	--print_every=<int>               Specifies frequency of epochs with which metrics are printed [default: 1]
	--save_every=<int>                Specifies frequency of epochs with which model is saved [default: 5]
	--data_path=<file>                Path to data
	--num_epochs=<int>                Number of epochs to train for
	--batch_sz=<int>                  Batch size [default: 128]
	--randomize_sz=<int>			  Number of randomized titles to choose per day [default:25]
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

from RCNN_base import Config_base, RCNN_base
from RCNN_seq import Config_seq, RCNN_seq, RCNN_seq_attn
from RCNN_v2 import Config_v2, RCNN_v2
from RCNN_v2_ti import Config_v2_ti, RCNN_v2_ti
from RCNN import Config, RCNN
from RCNN_concat_outputs import Config_concat,RCNN_concat_outputs

from torch.utils.data import DataLoader



main_model_path = "../../trained_models/RCNN_v2/RCNN_v2_ti.pt"

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
	Train main model
	"""

	print("Training initiated...")

	optimizer_step = float(args['--lr'])
	optimizer_momentum = 0.9
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print_every = int(args['--print_every']) 
	num_epochs = int(args['--num_epochs'])
	save_every = int(args['--save_every']) 
	save_path = main_model_path 


	#Stores hyperparams for model
	

	model = RCNN_v2_ti(config)
	model.to(device)

	optimizer = torch.optim.SGD(model.parameters(), lr= optimizer_step, momentum = optimizer_momentum)

	# train_data_path = args['--data_path'] 
	# if train_data_path == None: 
	# 	raise Exception("Training data path not fed in.")

	# print("Loading training data from {}".format(train_data_path))


	##############LOAD TRAIN DATA and initiate train

	randomize_sz = None if int(args['--randomize_sz']) == -1 else int(args['--randomize_sz'])
	data = data_utils.DJIA_Dataset('../../data/DJIA_table.csv', '../../data/Combined_News_DJIA.csv',randomize_sz=None)
	data_train = utils.data.Subset(data, [i for i in range(1600)])

	#dataset_val = data_utils.DJIA_Dataset('../../data/DJIA_table.csv', '../../data/Combined_News_DJIA.csv',randomize_sz=None)
	data_val = utils.data.Subset(data,[i for i in range(1800,1980)])

	dataloader_train = DataLoader(data_train, batch_size = int(config.batch_sz))
	dataloader_val = DataLoader(data_val,batch_size=int(config.batch_sz))

	#print("Finished loading training data from {}".format(train_data_path))


	##Read in data from train_data_path
	## Parse news titles into averages of word embeddings over title

	train_accs = []
	train_losses = []
	val_accs = []
	val_losses = []
	train_ctr = 0

	init_epoch = 0

	# if (load_path != 'None'):  #If model is retrained from saved ckpt
		
	# 	checkpoint = torch.load(load_path)
	# 	model.load_state_dict(checkpoint['model_state_dict'])
	# 	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	# 	init_epoch = checkpoint['epoch']
	# 	loss = checkpoint['loss']

	best_val_acc = 0
	model.train()

	start = time.time()
	print("Model parameters: {}".format(model.config.__dict__))
	try: 
		for epoch in range(1, num_epochs + 1):

			#INITIATE dataloader_train
			with tqdm(total = len(dataloader_train)) as pbar:
				for index, sample in enumerate(dataloader_train):

					model.train()

					titles, tech_indicators, movement = sample['titles'], sample['tech_indicators'], sample['movement']
					tech_indicators = tech_indicators.permute(1, 0, 2)
					#if index == 0:
					# print("tech_indicators sample dim: ", tech_indicators.shape, "titles sample dim: ", titles.shape)


					titles = titles.to(device)
					tech_indicators = tech_indicators.to(device)
					movement = movement.to(device)

					logits = model.forward(titles, tech_indicators)
					
					if index == 1:
						print("Predictions: ", torch.argmax(logits, dim = 1), "Labels: ", movement)

					loss = model.backprop(optimizer, logits, movement)
					train_ctr += 1
					accuracy = get_accuracy(logits, movement) #Accuracy over entire mini-batch

					#run on validation
					avg_val_accuracy = []
					avg_val_loss = []
					model.eval()
					with torch.no_grad():
						for index,sample in enumerate(dataloader_val):
							titles, tech_indicators, movement = sample['titles'], sample['tech_indicators'], sample['movement']
							tech_indicators = tech_indicators.permute(1, 0, 2)

							titles = titles.to(device)
							tech_indicators = tech_indicators.to(device)
							movement = movement.to(device)

							logits = model.forward(titles,tech_indicators)

							loss_fn = nn.NLLLoss(reduce = True, reduction = 'mean')
							loss_val = loss_fn(logits, movement)
							accuracy = get_accuracy(logits,movement)
							avg_val_accuracy.append(accuracy.cpu().numpy())
							avg_val_loss.append(loss_val.cpu().numpy())

					avg_val_accuracy = np.mean(avg_val_accuracy)
					avg_val_loss = np.mean(avg_val_loss)						
					val_accs.append(avg_val_accuracy)
					val_losses.append(avg_val_loss)


					
					train_losses.append(loss)
					train_accs.append(accuracy)

					pbar.update(1)

			if epoch % print_every == 0: 
				print ("Epoch: {}, Training iter: {}, Time since start: {}, Loss: {}, Training Accuracy: {}".format(epoch, train_ctr, (time.time() - start), loss, accuracy))
				print ("Epoch: {}, Avg Val Loss: {},Avg Val Accuracy: {}".format(epoch, avg_val_loss, avg_val_accuracy))

			if epoch % save_every == 0 and best_val_acc < avg_val_accuracy:
				best_val_acc = avg_val_accuracy
				print("new best Avg Val Accuracy: {}".format(best_val_acc))
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

	return (train_losses, train_accs, loss,val_losses,val_accs, best_val_acc) 

def test(args, config):
	#Get test data & parse

	#Initiate dataloader_test

	#Load saved model

	# test_data_path = args['--data_path'] 
	# if test_data_path == None: 
	# 	raise Exception("Test data path not fed in.")

	# print("Loading test data from {}".format(train_data_path))


	##############LOAD TEST DATA and initiate dataloader as dataloader_test

	data = data_utils.DJIA_Dataset('../../data/DJIA_table.csv', '../../data/Combined_News_DJIA.csv',randomize_sz=None)

	data_test = utils.data.Subset(data, [i for i in range(1600, 1800)])

	dataloader_test = DataLoader(data_test, batch_size = config.batch_sz)

	#print("Finished loading training data from {}".format(train_data_path))

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	load_path = main_model_path 

	model =  RCNN_v2_ti(config)
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
			tech_indicators = tech_indicators.permute(1, 0, 2)
			titles = titles.to(device)
			tech_indicators = tech_indicators.to(device)
			movement = movement.to(device)

			logits = model.forward(titles, tech_indicators)
			temp_criterion = nn.NLLLoss(reduce = True, reduction = 'mean')
			loss = temp_criterion(logits, movement)

			accuracy = get_accuracy(logits, movement) #Accuracy over entire mini-batch
			
			test_loss.append(loss.detach().cpu().numpy())

			test_accuracy.append(accuracy.detach().cpu().numpy())
			#Training step


	test_loss, test_accuracy = np.mean(np.array(test_loss)), np.mean(np.array(test_accuracy))

	str_to_save = "Test_loss: " + str(test_loss) + " , Test Accuracy: " + str(test_accuracy)

	with open("test_results_randomize_sz.txt", 'a') as test_writer:
		test_writer.write(str_to_save + "\n")

	return (test_loss, test_accuracy)


def main():
	args = docopt(__doc__)

	config = Config_v2_ti(batch_sz = int(args['--batch_sz']))

	if args['train']:

		train_losses, train_accs, loss,val_losses,val_accs, accuracy = train(args, config)
		np.save('train_accs.npy', np.array(train_accs))
		np.save('train_losses.npy', np.array(train_losses))
		np.save('val_accs.npy', np.array(val_accs))
		np.save('val_losses.npy', np.array(val_losses))
		print("Final training loss: {}, Best Validation Accuracy: {}".format(loss, accuracy))

	elif args['test']:
		loss, accuracy = test(args, config)
		print("Test loss: {}, Test accuracy: {}".format(loss, accuracy))


if __name__ == "__main__":

	main()
