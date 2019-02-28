import torch
import torch.nn as nn 

"""
Currently skeleton code, fill with actual models as desired
"""
class RCNN(nn.Module):
	def __init__(self,input_dim,hidden_dim,output_dim):
		''' 
		input_dim: Dimensionality of input corpus
		hidden_dim: Hidden layer dimensionality
		output_dim: Dimensionality of output of model (should be scalar)
		'''
		super(RCNN,self).__init__()
		self.linear1 = torch.nn.Linear(input_dim,hidden_dim)
		self.linear2 = torch.nn.Linear(hidden_dim,output)
	def forward(self,input):
		output = self.linear1(input).clamp(0)
		output = self.linear2(output)

		return output


