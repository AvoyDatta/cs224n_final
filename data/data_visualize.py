"""
Utility functions to visualize data for papers. 

Dependencies: data_utils.py

"""
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import data_utils 


plot_save_path = "tech_indicators.png"
technical_csv = "DJIA_table.csv"
title_csv = "Combined_News_DJIA.csv"


def plot_tech_indicators(days_to_plot = 100, n_days = 1):

	tech_indicators = data_utils.loadTechnical(technical_csv, n=1, input_size=7) 
	reshaped_indicators = np.squeeze(tech_indicators.numpy()) #Numpy array of shape (num_days, num_indicators)
	

	reshaped_indicators = reshaped_indicators[:days_to_plot, :]
	print(reshaped_indicators.shape)


	days = np.arange(1, 101, 1)
	labels = ["Movement", "Stoch_K", "Stoch_D", "Momentum", "ROC", "Will's%R", "A/D", "Disp5"]

	
	plt.title("Technical indicators for the first {} days".format(days_to_plot))
	plt.subplot(7, 1, 1)
	plt.plot(days, movement[:days_to_plot, indicator], 'rx-')
	plt.ylabel(labels[0])
	plt.xlabel("Stock price movement")
	for indicator in range(1, reshaped_indicators.shape[1] + 1):

		plt.subplot(7, 1, indicator + 1)
		plt.plot(days, reshaped_indicators[:days_to_plot, indicator], '.-')
		plt.ylabel(labels[indicator])

	plt.xlabel("Number of days after 8th August 2008.")
	plt.show()
	# plt.subplot(7, 1, 2)
	# plt.plot(x2, y2, '.-')
	# plt.ylabel('Undamped')

	# plt.subplot(7, 1, 3)
	# plt.plot(x1, y1, '.-')
	# plt.title('A tale of 2 subplots')
	# plt.ylabel('Damped oscillation')

	# plt.subplot(7, 1, 4)
	# plt.plot(x1, y1, '.-')
	# plt.title('A tale of 2 subplots')
	# plt.ylabel('Damped oscillation')

	# plt.subplot(7, 1, 5)
	# plt.plot(x1, y1, '.-')
	# plt.title('A tale of 2 subplots')
	# plt.ylabel('Damped oscillation')

	# plt.subplot(7, 1, 1)
	# plt.plot(x1, y1, '.-')
	# plt.title('A tale of 2 subplots')
	# plt.ylabel('Damped oscillation')

	# plt.subplot(7, 1, 1)
	# plt.plot(x1, y1, '.-')
	# plt.title('A tale of 2 subplots')
	# plt.ylabel('Damped oscillation')
	# plt.xlabel('time (s)')


def plot_val_accs():
	val_accs = np.load("val_accs.npy")
	train_accs = np.load("train_accs.npy")

	plt.title("Train and Validation Set Accuracies")
	low = 0
	high = 10
	num_points = len(val_accs)
	x = np.linspace(low,high,num_points)

	red_val = mpatches.Patch(color='red', label='Validation Accuracies')
	blue_train = mpatches.Patch(color='blue', label='Train Accuracies')
	plt.legend(handles=[red_val, blue_train])

	plt.plot(x, val_accs)
	plt.plot(x, train_accs)
	plt.show()


def plot_val_losses():
	val_losses = np.load("val_losses.npy")
	train_losses = np.load("train_losses.npy")

	plt.title("Train and Validation Set Losses")
	low = 0
	high = 10
	num_points = len(val_losses)
	x = np.linspace(low,high,num_points)

	red_val = mpatches.Patch(color='red', label='Validation Losses')
	blue_train = mpatches.Patch(color='blue', label='Train Losses')
	plt.legend(handles=[red_val, blue_train])

	plt.plot(x, val_losses)
	plt.plot(x, train_losses)
	plt.show()


if __name__== "__main__":

	plot_tech_indicators(days_to_plot = 100, n_days = 1)
	# plot_val_accs()
	# plot_val_losses()

