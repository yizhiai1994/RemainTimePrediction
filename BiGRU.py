import numpy as np
import torch
import torch.nn as nn
import gensim
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import sqrt
class BiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_size, batch_size=1, n_layer = 1, dropout = 0,
                 embedding = None, CUDA_type = True):
        super(BiGRU, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_shape = out_size
        self.embedding = embedding
        self.CUDA_type = CUDA_type
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.biType = 1
        self.dropout = dropout
        self.biType = 2
        print('Initialization BiGRU Model')
        if self.CUDA_type == True:
            self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, dropout=self.dropout,
                              num_layers=self.n_layer, bidirectional=True).cuda()
            self.out = nn.Linear(hidden_dim * self.biType, out_size).cuda()
        else:
            self.rnn = nn.GRU(input_size = embedding_dim, hidden_size = hidden_dim, dropout = self.dropout,
                               num_layers = self.n_layer, bidirectional=True)
            self.out = nn.Linear(hidden_dim * self.biType, out_size)

    def forward(self, X):
        # orignal input : [batch_size, len_seq, embedding_dim]
        input = self.embedding(X)
        # GRU input of shape (seq_len, batch, embedding_dim(input_size)):
        # LSTM input : [len_seq, batch_size, embedding_dim(input_size)]
        input = input.permute(1, 0, 2)
        #h_0 of LSTM shape (num_layers * num_directions, batch, hidden_size)
        #h_0 of GRU shape (num_layers * num_directions, batch, hidden_size):
        if self.CUDA_type == False:
            hidden_state = Variable(
                torch.randn(self.n_layer * self.biType, self.batch_size, self.hidden_dim))
        else:
            hidden_state = Variable(
                torch.randn(self.n_layer * self.biType, self.batch_size, self.hidden_dim)).cuda()

        output,final_hidden_state = self.rnn(input, hidden_state)
        hn = output[-1]
        output = self.out(hn)
        return  output # model : [batch_size, num_classes], attention : [batch_size, n_step]
