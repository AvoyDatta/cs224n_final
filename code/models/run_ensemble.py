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
from RCNN_seq import Config_seq, RCNN_seq, RCNN_seq_attn, RCNN_seq_2
from RCNN_v2 import Config_v2, RCNN_v2
from RCNN import Config, RCNN
from RCNN_concat_outputs import Config_concat,RCNN_concat_outputs

from torch.utils.data import DataLoader



main_model_path = "../../trained_models/RCNN_seq_attn/RCNN_seq_attn"
max_ensemble = 5

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
	

	model = RCNN_seq_attn(config)
	model.to(device)

	optimizer = torch.optim.SGD(model.parameters(), lr= optimizer_step, momentum = optimizer_momentum)

	# train_data_path = args['--data_path'] 
	# if train_data_path == None: 
	# 	raise Exception("Training data path not fed in.")

	# print("Loading training data from {}".format(train_data_path))


	##############LOAD TRAIN DATA and initiate train

	randomize_sz = None if int(args['--randomize_sz']) == -1 else int(args['--randomize_sz'])
	data = data_utils.DJIA_Dataset('../../data/DJIA_table.csv', '../../data/Combined_News_DJIA.csv',randomize_sz=None, start=1, end=250)
	data_train = utils.data.Subset(data, [i for i in range(100)])

	#dataset_val = data_utils.DJIA_Dataset('../../data/DJIA_table.csv', '../../data/Combined_News_DJIA.csv',randomize_sz=None)
	data_val = utils.data.Subset(data,[i for i in range(100,200)])

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

					logits,_ = model.forward(titles, tech_indicators)
					
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

						for index_val,sample_val in enumerate(dataloader_val):
							titles_val, tech_indicators_val, movement_val = sample_val['titles'], sample_val['tech_indicators'], sample_val['movement']
							tech_indicators_val = tech_indicators_val.permute(1, 0, 2)

							titles_val = titles_val.to(device)
							tech_indicators_val = tech_indicators_val.to(device)
							movement_val = movement_val.to(device)

							logits_val,_ = model.forward(titles_val,tech_indicators_val)

							loss_fn = nn.NLLLoss(reduce = True, reduction = 'mean')
							loss_val = loss_fn(logits_val, movement_val)
							accuracy_val = get_accuracy(logits_val,movement_val)
							avg_val_accuracy.append(accuracy_val.cpu().numpy())
							avg_val_loss.append(loss_val.cpu().numpy())

					avg_val_accuracy = np.mean(avg_val_accuracy)
					avg_val_loss = np.mean(avg_val_loss)						
					#val_accs.append(avg_val_accuracy)
					val_losses.append(avg_val_loss)


					
					train_losses.append(loss)
					train_accs.append(accuracy)

					pbar.update(1)

			if epoch % print_every == 0: 
				print ("Epoch: {}, Training iter: {}, Time since start: {}, Loss: {}, Training Accuracy: {}".format(epoch, train_ctr, (time.time() - start), loss, accuracy))
				print ("Epoch: {}, Avg Val Loss: {},Avg Val Accuracy: {}".format(epoch, avg_val_loss, avg_val_accuracy))

			# if epoch % save_every == 0 and best_val_acc < avg_val_accuracy:
			# 	best_val_acc = avg_val_accuracy
			# 	print("new best Avg Val Accuracy: {}".format(best_val_acc))
			# 	print ("Saving model to {}".format(save_path))
			# 	torch.save({'epoch': epoch,
			# 				'model_state_dict': model.state_dict(), 
			# 				'optimizer_state_dict': optimizer.state_dict(), 
			# 				'loss': loss}, 
			# 				save_path)
			# 	print ("Saved successfully to {}".format(save_path))


			if epoch % save_every == 0:
				if len(val_accs) < max_ensemble:
					val_accs.append(avg_val_accuracy)
					torch.save({'epoch': epoch,
							'model_state_dict': model.state_dict(), 
							'optimizer_state_dict': optimizer.state_dict(), 
							'loss': loss}, 
							 save_path + str(len(val_accs) - 1) + ".pt")
					print("Saved successfully to {}".format(save_path + str(len(val_accs) - 1) + ".pt"))

				else: # val accs has 5 elems  
					min_idx = val_accs.index(min(val_accs))

					if avg_val_accuracy > min(val_accs):
						min_idx = val_accs.index(min(val_accs))

						torch.save({'epoch': epoch,
							'model_state_dict': model.state_dict(), 
							'optimizer_state_dict': optimizer.state_dict(), 
							'loss': loss}, 
							 save_path + str(min_idx) + ".pt")
						print("Saved successfully to {}".format(save_path + str(min_idx) + ".pt"))
						val_accs[min_idx] = avg_val_accuracy

	except KeyboardInterrupt:
		print("Training interupted...")
		print ("Saving model to {}".format(save_path))
		torch.save({'epoch': epoch,
					'model_state_dict': model.state_dict(), 
					'optimizer_state_dict': optimizer.state_dict(), 
					'loss': loss}, 
					save_path + ".pt")
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

	model1 =  RCNN_seq_attn(config)
	model1.to(device)

	model2 =  RCNN_seq_attn(config)
	model2.to(device)

	model3 =  RCNN_seq_attn(config)
	model3.to(device)

	model4 =  RCNN_seq_attn(config)
	model4.to(device)

	model5 =  RCNN_seq_attn(config)
	model5.to(device)

	if (load_path != None):  #If model is retrained from saved ckpt
		
		print("Loading model from {}".format(load_path))
		checkpoint1 = torch.load(load_path + "0.pt")
		checkpoint2 = torch.load(load_path + "1.pt")
		checkpoint3 = torch.load(load_path + "2.pt")
		checkpoint4 = torch.load(load_path + "3.pt")
		checkpoint5 = torch.load(load_path + "4.pt")

		model1.load_state_dict(checkpoint1['model_state_dict'])
		model2.load_state_dict(checkpoint2['model_state_dict'])
		model3.load_state_dict(checkpoint3['model_state_dict'])
		model4.load_state_dict(checkpoint4['model_state_dict'])
		model5.load_state_dict(checkpoint5['model_state_dict'])


		#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		#epoch = checkpoint['epoch']
		#loss = checkpoint['loss']
		print("Models successfully loaded from {}".format(load_path))

	model1.eval()
	model2.eval()
	model3.eval()
	model4.eval()
	model5.eval()

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

			logits1,_ = model1.forward(titles, tech_indicators)
			logits2,_ = model2.forward(titles, tech_indicators)
			logits3,_ = model3.forward(titles, tech_indicators)
			logits4,_ = model4.forward(titles, tech_indicators)
			logits5,_ = model5.forward(titles, tech_indicators)


			temp_criterion = nn.NLLLoss(reduce = True, reduction = 'mean')
			loss = temp_criterion(logits1, movement)

			#accuracy = get_accuracy(logits, movement) #Accuracy over entire mini-batch
			preds1 = torch.argmax(logits1, dim = 1)
			preds2 = torch.argmax(logits2, dim = 1)
			preds3 = torch.argmax(logits3, dim = 1)
			preds4 = torch.argmax(logits4, dim = 1)
			preds5 = torch.argmax(logits5, dim = 1)

			preds = torch.gt(torch.sum(torch.cat((preds1, preds2,preds3,preds4,preds5), 1), dim=1), 2.0)

			accuracy = torch.mean(torch.eq(preds, labels).float())

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

	config = Config_seq(batch_sz = int(args['--batch_sz']))

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





