"""
Main run file for baseline model.


baseline.py can be run with the following opt arguments:
	'train' OR 'test', [MANDATORY]
	'--print_every', #Specifies frequency of epochs with which metrics are printed
	'--save_every',  #Specifies frequency of epochs with which model is saved
	'--batch_sz', #batch size of train or test
	'--num_batches'

"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time
import os

from docopt import docopt
from RCNN import Config, RCNN
from torch.utils.data import DataLoader


baseline_model_path = "../../trained_models/baseline/baseline.pt"

def backprop(optimizer, logits, labels):

	optimizer.zero_grad()
	loss = nn.NLLLoss(logits, labels, reduce = True, reduction = 'mean')
	loss.backward()
	optimizer.step()

	return loss

def get_accuracy(logits, labels)

	return torch.mean(torch.eq(torch.argmax(logits, dim = 1), labels))
	
def train(args, config):
	"""
	Train baseline model
	"""

	print("Training initiated...")

	baseline_step = 0.1
	baseline_momentum = 0.9
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print_every = 1 if !args['--print_every'] else args['--print_every']
	num_epochs = args['--num_epochs']
	save_every = 10 if !args['--save_every'] else args['--save_every']
	save_path = baseline_model_path if !args['--save_path'] else args['--save_path']
	load_path = None if !args['--load_path'] else args['--load_path']

	#Stores hyperparams for model
	

	model = RCNN(config)
	model.to(device)

	optimizer = torch.optim.SGD(model.parameters(), lr= baseline_step, momentum = baseline_momentum)

	train_data_path = args['--data_path'] else None
	if train_data_path == None: 
		raise Exception("Training data path not fed in.")

	print("Loading training data from {}".format(train_data_path))


	##############LOAD TRAIN DATA and initiate train

	data_train = ...
	dataloader_train = DataLoader(data_train, batch_size = config.batch_sz)

	print("Finished loading training data from {}".format(train_data_path))


	##Read in data from train_data_path
	## Parse news titles into averages of word embeddings over title

	train_accs = []
	train_losses = []
	train_ctr = 0

	init_epoch = 0

	if (load_path != None):  #If model is retrained from saved ckpt
		
		checkpoint = torch.load(load_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		loss = checkpoint['loss']

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

					logits = model.forward(titles, tech_indicators)
					loss = backprop(optimizer, logits, movement)
					train_ctr += 1
					accuracy = get_accuracy(logits, movement) #Accuracy over entire mini-batch
					
					train_losses.append(loss)
					train_accs.append(accuracy)

			if epoch % print_every == 0: 
				print ("Epoch: {}, Training iter: {}, Time since start: {}, Loss: , Training Accuracy: {}".format(epoch, train_ctr, (time.time() - start), loss, accuracy))

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

	test_data_path = args['--data_path'] else None
	if test_data_path == None: 
		raise Exception("Test data path not fed in.")

	print("Loading test data from {}".format(train_data_path))


	##############LOAD TEST DATA and initiate dataloader as dataloader_test

	data_test = ...
	dataloader_test = DataLoader(data_test, batch_size = config.batch_sz)

	print("Finished loading training data from {}".format(train_data_path))

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	load_path = baseline_model_path if !args['--load_path'] else args['--load_path']

	model = RCNN(config)
	model.to(device)

	if (load_path != None):  #If model is retrained from saved ckpt
		
		print("Loading model from {}".format(load_path))
		checkpoint = torch.load(load_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		loss = checkpoint['loss']
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

			logits = model.forward(titles, tech_indicators)
			loss = nn.NLLLoss(logits, movement, reduce = True, reduction = 'mean')
			accuracy = get_accuracy(logits, movement) #Accuracy over entire mini-batch
			
			test_loss.append(loss)
			test_accuracy.append(accuracy)
			#Training step

	test_loss, test_accuracy = np.mean(np.array(test_loss)), np.mean(np.array(test_accuracy))

	return (test_loss, test_accuracy)


def main():
	args = docopt(__doc__)

	config = Config(batch_sz = args['--batch_sz'], 
					num_batches = args['--num_batches']) 

	if args['train']:

        train_losses, train_accs, loss, accuracy = train(args, config, baseline_model_path)
        np.save(np.array(train_accs), 'train_accs.npy')
        np.save(np.array(train_losses), 'train_losses.npy')
        print("Final training loss: {}, Final Training Accuracy: {}".format(loss, accuracy))

    else args['test']:
    	loss, accuracy = test(args, config)
        print("Test loss: {}", "Test accuracy: {}".format(loss, accuracy))


if __name__ == "__main__":

	main()
