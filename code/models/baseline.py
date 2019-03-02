"""
Main run file for baseline model.

"""

import torch
import torch.nn as nn
from tqdm import tqdm
import time

from docopt import docopt

from RCNN import Config, RCNN

def backprop(optimizer, logits, labels):

	optimizer.zero_grad()
	loss = nn.NLLLoss(logits, labels, reduce = True, reduction = 'mean')
	loss.backward()
	optimizer.step()

	return loss


def train(args):
	"""
	Train baseline model
	"""
	print("Training initiated...")

	baseline_step = 0.1
	baseline_momentum = 0.9
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print_every = 1 if !args['--print_every'] else args['--save_every']
	print_every = 1 if !args['--print_every'] else args['--print_every']
	num_epochs = args['--num_epochs']
	save_every = 10 if !args['--save_every'] else args['--save_every']
	save_path = 

	config = Config(batch_sz = args['--train_batch_sz']) #Stores hyperparams for model

	model = RCNN(config)
	model.to(device)

	optimizer = torch.optim.SGD(model.parameters(), lr= baseline_step, momentum = baseline_momentum)



	##Read in data
	## Parse news titles into averages of word embeddings over title


	for epoch in range(num_epochs):
		start = time.time()

		with tqdm(total = len(dataloader)) as pbar:
			for index, sample in enumerate(dataloader):

				titles, tech_indicators, movement = sample['titles'], sample['tech_indicators'], sample['movement']
				titles.to(device)
				tech_indicators.to(device)
				movement.to(device)

				logits = model(titles, tech_indicators)
				loss = backprop(optimizer, logits, movement)
				accuracy = torch.mean(torch.eq(torch.argmax(logits, dim = 1), movement)) #Accuracy over entire mini-batch


				#Training step
		if epoch % print_every == 0: 
			print ("Epoch: {}, Time since start: {}, Loss: , Training Accuracy: {}".format(epoch, loss, accuracy))

		if epoch % save_every == 0:
			


	print("Training completed.")
	return 

def test(args):



def main():
	args = docopt(__doc__)

	if args['train']:
        train(args)
    else args['test']:
    	test(args)

if __name__ == "__main__":

	main()
