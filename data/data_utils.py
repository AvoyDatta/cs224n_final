import csv
def loadTechnical(input_csv_path,seq=5,input_size=7):
	"""
	input_csv_path: path to csv
	output: Tensor of sive (seq_len,batch_size,input_size=7)
	"""
	with open(input_csv_path,'r') as f: 
		data = [row for row in csv.reader(read().splitlines())]
	print(data[0])

