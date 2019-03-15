import csv
import torch
from torch.utils.data import Dataset
import numpy as np
import gensim
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
from functools import partial
import os
from tqdm import tqdm
import time
import random
#nltk.download('punkt')
#nltk.download("stopwords")


# nltk.download('punkt')

class DJIA_Dataset(Dataset):
    def __init__(self, path_technical_csv, path_title_csv, randomize_sz=25):
        """
		path_technical_csv: path to technical csv
		path_title_csv: path to csv with titles of articles
		"""
        self.technical_data = loadTechnical(path_technical_csv)
        print("technical data dims: ", self.technical_data.shape)
        self.targets, self.title_data = loadTitle(path_title_csv, randomize_sz=randomize_sz)
        print("Title data dims: ", self.title_data.shape)
        print("Targets dims: ", self.targets.shape)
        self.length = self.targets.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
		index: index of element in dataset you want
		returns: tuple( technical_data(5,7), title_data(25,50),target(1)  )
		"""
        # print(self.title_data.shape)
        # print(self.technical_data.shape)
        return {"titles": self.title_data[index, :, :,:,:],
                "tech_indicators": self.technical_data[:, index, :],
                "movement": self.targets[index].type(torch.LongTensor)}


def loadTechnical(input_csv_path, n=5, input_size=7):
    """
	input_csv_path: path to csv
	output: Tensor of size (seq_len,batch_size,input_size=7)
	"""

    with open(input_csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        data = []
        for row in reader:
            data.append(row)
        # print(','.join(row))
        data.reverse()
        # return data
        data_dict = {}  # dictionary containing header-> timesequence
        keys = data[-1]
        for index in range(len(keys)):
            new_timeseq = []
            for row in data[:-2]:
                if keys[index] != 'Date':
                    new_timeseq.append(float(row[index]))
                else:
                    new_timeseq.append(row[index])
            data_dict[keys[index]] = new_timeseq

        # calculate Stoch_K
        data_length = len(data_dict['High'])  # 1988
        new_timeseq = []
        for t in range(data_length):
            C_t = data_dict['Close'][t]
            HH_n = 0
            LL_n = 0
            if t < 1:
                HH_n = data_dict['High'][t]
                LL_n = data_dict['Low'][t]
            elif t < n:
                assert len(data_dict['High'][:t]) < n
                HH_n = max(data_dict['High'][:t])
                LL_n = max(data_dict['Low'][:t])
            else:
                length = len(data_dict['High'][t - n:t])
                assert length == n
                HH_n = max(data_dict['High'][t - n:t])
                LL_n = max(data_dict['Low'][t - n:t])
            new_timeseq.append((C_t - LL_n) / (HH_n - LL_n))
        data_dict['Stoch_K'] = new_timeseq
        assert len(data_dict['Stoch_K']) == data_length

        # calculate Stoch_D
        new_timeseq = []
        for t in range(data_length):
            sum_val = 0
            if t < 1:
                sum_val = data_dict['Stoch_K'][t]
            elif t < n:
                sum_val = sum(data_dict['Stoch_K'][:t])
            else:
                sum_val = sum(data_dict['Stoch_K'][t - (n - 1):t + 1])
                length = len(data_dict['Stoch_K'][t - (n - 1):t + 1])
                assert length == n
            new_timeseq.append(sum_val / n)
        data_dict['Stoch_D'] = new_timeseq
        assert len(data_dict['Stoch_D']) == data_length

        # calculate momentum
        new_timeseq = []
        for t in range(data_length):
            momentum = 0
            if t < n:
                first = data_dict['Close'][0]
                momentum = data_dict['Close'][t] - first
            else:
                before = data_dict['Close'][t - (n - 1)]
                momentum = data_dict['Close'][t] - before
            new_timeseq.append(momentum)
        data_dict['Momentum'] = new_timeseq
        assert len(data_dict['Momentum']) == data_length

        # calculate rate of change
        new_timeseq = []
        for t in range(data_length):
            roc = 0
            if t < n:
                first = data_dict['Close'][0]
                momentum = (data_dict['Close'][t] / first) * 100
            else:
                before = data_dict['Close'][t - n]
                roc = (data_dict['Close'][t] / before) * 100
            new_timeseq.append(roc)
        data_dict['ROC'] = new_timeseq
        assert len(data_dict['ROC']) == data_length

        # calculate William's %R
        new_timeseq = []
        for t in range(data_length):
            C_t = data_dict['Close'][t]
            HH_n = 0
            LL_n = 0
            if t < 1:
                HH_n = data_dict['High'][t]
                LL_n = data_dict['Low'][t]
            elif t < n:
                assert len(data_dict['High'][:t]) < n
                HH_n = max(data_dict['High'][:t])
                LL_n = max(data_dict['Low'][:t])
            else:
                length = len(data_dict['High'][t - n:t])
                assert length == n
                HH_n = max(data_dict['High'][t - n:t])
                LL_n = max(data_dict['Low'][t - n:t])
            new_timeseq.append(100 * (HH_n - C_t) / (HH_n - LL_n))
        data_dict['WillR'] = new_timeseq
        assert len(data_dict['WillR']) == data_length

        # calculate A/D oscillator
        new_timeseq = []
        for t in range(data_length):
            H_t = data_dict['High'][t]
            L_t = data_dict['Low'][t]
            C_tprev = 0
            if t < 1:
                C_tprev = data_dict['Close'][t]
            else:
                C_tprev = data_dict['Close'][t - 1]
            new_timeseq.append((H_t - C_tprev) / (H_t - L_t))

        data_dict['AD'] = new_timeseq
        assert len(data_dict['AD']) == data_length

        # calculate Disparity 5
        new_timeseq = []
        for t in range(data_length):
            C_t = data_dict['Close'][t]
            MA = 0
            if t < 1:
                MA = data_dict['Close'][t]
            elif t < 5:
                MA = sum(data_dict['Close'][:t]) / len(data_dict['Close'][:t])
            else:
                assert len(data_dict['Close'][t - 5:t]) == 5
                MA = sum(data_dict['Close'][t - 5:t]) / 5
            new_timeseq.append(100 * C_t / MA)
        data_dict['Disp'] = new_timeseq
        assert len(data_dict['Disp']) == data_length

        # skip overr first n days
        tensor = np.array(
            [data_dict['Stoch_K'], data_dict['Stoch_D'], data_dict['Momentum'], data_dict['ROC'],
             data_dict['WillR'], data_dict['AD'], data_dict['Disp']])

        sequence_len = tensor.shape[1]
        stack = []
        for i in range(n, sequence_len):
            value = tensor[:, np.newaxis, i - n:i]
            stack.append(value)
        new_tensor = np.concatenate(stack, 1)
        # print(new_tensor.shape)
        new_tensor = torch.Tensor(new_tensor)

        new_tensor = new_tensor.permute(2, 1, 0)
        # print(new_tensor.shape)
        return new_tensor


def loadTitle(input_csv_path, n=5, randomize_sz=None):
    """
	input: input_csv_path
	input: randomize_sz: choose the number of titles to randomly choose from to incorporate into titles for a particular day
	output: Tuple(Tensor of size (batch_size,channels=num_titles,seq_len=300),targets) #modified
	"""
    if randomize_sz is not None:
        print("randomly choosing {} titles per day".format(randomize_sz))
    with open(input_csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        data = []
        for row in reader:
            data.append([s.lower() for s in row])

    keys = data[0]
    # sanitize input
    for i in range(len(data)):
        # print(len(data[i]))
        for j in range(len(data[i])):
            if data[i][j][0] == 'b':  # might accidentally remove first b..
                data[i][j] = data[i][j][1:]

    targets = []
    for i in range(1, len(data) - 1):
        targets.append(float(data[i + 1][1]))
    # print(len(targets))
    data = data[1:-1]

    assert (len(targets) == len(data))

    # ToDo Only look up models if word2vec.csv doesn't exist
    # model = gensim.models.KeyedVectors.load_word2vec_format('./lexvec.pos.vectors', binary=True)
    # model = gensim.models.KeyedVectors.load_word2vec_format('lexvec.enwiki+newscrawl.300d.W.pos.vectors', binary=True)

    ### LOAD PREVIOUSLY SAVED MODEL
    model_path = 'glove_word2vec300.model'
    if not os.path.isfile(model_path):
        glove_file = 'glove.6B.300d.txt'
        tmp_file = get_tmpfile("test_word2vec.txt")
        glove2word2vec(glove_file, tmp_file)
        model = KeyedVectors.load_word2vec_format(tmp_file)
        model.save("glove_word2vec300.model")

    model = KeyedVectors.load(model_path)
    vocab = model.vocab.keys()

    # print(len(vocab))
    set_vocab = set(vocab)
    # t1 = time.time()
    # if "time" in set_vocab:
    # 	# print("yeet")
    # 	next
    # t2 = time.time()
    # print("lookin up whether a word exists in vocab takes: {} seconds \n".format(t2-t1))

    ### LOAD MODEL FROM SCRATCH
    # https://nlp.stanford.edu/projects/glove/ : glove.6b.zip

    # (batch, n, embed size, num_titles, num_words)

    # convert titles to word2vec
    # word2vec_data = []

    embed_size = 300

    stop_words = set(stopwords.words('english'))

    max_sent_len = 0  # over all data, without stopwords or out-of-vocab words

    for i in range(n, len(data)):
        data_row = data[i][2:]
        for hline in data_row:
            toks = [tok for tok in nltk.word_tokenize(hline) if tok in set_vocab and tok not in stop_words]
            max_sent_len = max(max_sent_len, len(toks))

    print("Loading titles...")

    word2vec_data_prestack = torch.zeros((len(data) - (n), n, 25, embed_size, max_sent_len))
    print("prestack size: ", word2vec_data_prestack.shape)
    with tqdm(total=len(data) - n) as pbar:
        # skip first n days
        for i in range(n, len(data)):
            # data_window = []
            # look back on previous n days
            data_window = [data[j][2:] for j in range(i - 5, i)]
            assert len(data_window) == n

            # data_row = data[i][2:]

            # title choice randomization
            if randomize_sz is not None:
                random.shuffle(data_row for data_row in data_window)
                for row_index in range(len(data_window)):
                    data_row = data_window[row_index]
                    aux = []
                    for random_sample in range(randomize_sz):
                        if random_sample >= len(data_row): break
                        aux.append(data_row[random_sample])
                    data_window[row_index] = aux

            # print("max sent len: ", max_sent_len)
            headline_list_window = []
            for data_row in data_window:
                headline_list_day = []
                for headline in data_row:
                    # headline_words = np.concatenate(np.array([[model[word] for word in nltk.word_tokenize(headline) if word in set_vocab and word not in stop_words]]))
                    headline_words = [model[word] for word in nltk.word_tokenize(headline) if
                                      word in set_vocab and word not in stop_words]
                    # print("len of headline: ", len(headline_words))
                    # pad to max sentence length
                    if len(headline_words) < max_sent_len:
                        pad_len = max_sent_len - len(headline_words)
                        for pad_idx in range(pad_len):
                            headline_words.append(np.zeros(embed_size))
                    assert len(headline_words) == max_sent_len
                    headline_list_day.append(headline_words)
                # print("num headlines in one day:", len(headline_list_day))

                # pad to 25
                if len(headline_list_day) < 25:
                    pad_len = 25 - len(headline_list_day)
                    for i in range(pad_len):
                        headline_list_day.append([np.zeros(embed_size)] * max_sent_len)

                headline_list_day = np.stack(headline_list_day, axis=-1)
                headline_list_window.append(headline_list_day)

            headline_list_window = np.stack(headline_list_window, axis=-1)

            # headline_list_window = torch.tensor(headline_list_window)
            # torch.Size([56, 300, 25, 5])
            # print(headline_list_window.shape)
            word2vec_data_prestack[i - n, :, :, :, :] = torch.Tensor(np.transpose(headline_list_window, (3, 2, 1, 0)))
            # word2vec_data.append(headline_list_window)
            pbar.update(1)

    print("about to stack...")
    print(len(word2vec_data_prestack), len(word2vec_data_prestack[0]))
    # word2vec_data = np.stack(word2vec_data,axis=0)
    print("stacked")
    new_tensor = word2vec_data_prestack
    print("SHAPE: ", new_tensor.shape)
    # want (batch, window_len, num_titles, embed_sz, max_words)
    # new_tensor = new_tensor.permute(0, 4, 3, 2, 1)
    # new_tensor = new_tensor[n:,:,:]
    targets = torch.Tensor(targets[n:])
    print("NEW SHAPE: ", new_tensor.shape)

    print(new_tensor.shape)
    # print(new_tensor)
    print("Completed loading titles")
    return (targets, new_tensor)


# to test
if __name__ == "__main__":
    # loadTechnical('DJIA_table.csv',n=5,input_size=7)
    targets, new_tensor = loadTitle('Combined_News_DJIA.csv')
    # print(targets[0:10])
# print(new_tensor.shape)

