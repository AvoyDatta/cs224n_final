"""
Utility functions to visualize data for papers. 

Dependencies: data_utils.py

"""
from matplotlib import pyplot as plt
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
	labels = ["Stoch_K", "Stoch_D", "Momentum", "ROC", "Will's%R", "A/D", "Disp5"]

	
	plt.title("Technical indicators for the first {} days".format(days_to_plot))
	for indicator in range(reshaped_indicators.shape[1]):

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



if __name__== "__main__":

	plot_tech_indicators(days_to_plot = 100, n_days = 1)

