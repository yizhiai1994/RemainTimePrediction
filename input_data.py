import numpy as np
import torch
import torch.nn as nn
import gensim
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from datetime import datetime
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import sqrt
import random
random.seed=13
class InputData():
    def __init__(self,data_address, embd_dimension = 2, time_unit = 'second'):
        self.embedding = None
        self.ogrinal_data = list()
        self.orginal_trace = list()
        self.encode_trace = list()
        self.train_dataset = list()
        self.test_dataset = list()
        self.train_mixLengthData = list()
        self.test_mixLengthData = list()
        self.event2id = dict()
        self.id2event = dict()
        self.train_batch_mix = list()
        self.test_batch_mix = list()
        self.train_singleLengthData = dict()
        self.test_singleLengthData = dict()
        self.train_batch = dict()
        self.test_batch = dict()
        self.train_batch_single = dict()
        self.test_batch_single = dict()

        self.vocab_size = 0
        self.train_maxLength = 0
        self.test_maxLength = 0
        self.embd_dimension = embd_dimension
        if time_unit == 'second':
            self.time_unit = 1
        elif time_unit == 'minute':
            self.time_unit = 60
        elif time_unit == 'hour':
            self.time_unit = 60 * 60
        elif time_unit == 'day':
            self.time_unit = 24 * 60 * 60
        elif time_unit == 'month':
            self.time_unit = 30 * 24 * 60 * 60
        self.initData(data_address)
    def initData(self,data_address):
        orginal_trace = list()
        record = list()
        trace_temp = list()
        with open(data_address, 'r', encoding='utf-8') as f:
            next(f)
            lines = f.readlines()
            for line in lines:
                record.append(line)
        flag = record[0].split(',')[0]
        for line in record:
            line = line.replace('\r', '').replace('\n', '')
            line = line.split(',')
            if line[0] == flag:
                trace_temp.append([line[1], line[2]])
            else:
                flag = line[0]
                if len(trace_temp) > 0:
                    orginal_trace.append(trace_temp.copy())
                trace_temp = list()
                trace_temp.append([line[1], line[2]])
        self.ogrinal_data = record
        self.orginal_trace = orginal_trace
    def encodeEvent(self,vector_address):
        event2id = dict()
        id2event = dict()
        if vector_address == None:
            for line in self.ogrinal_data:
                line = line.replace('\r', '').replace('\n', '')
                line = line.split(',')
                try:
                    event2id[line[1]] = event2id[line[1]]
                    id2event[event2id[line[1]]] = id2event[event2id[line[1]]]
                except KeyError as ke:
                    event2id[line[1]] = len(event2id)
                    id2event[len(id2event)] = line[1]
            self.vocab_size = len(event2id)
            self.embedding = nn.Embedding(self.vocab_size + 1, self.embd_dimension, padding_idx= self.vocab_size).cuda()
        else:
            with open(vector_address, 'r', encoding='utf-8') as f:
                information = next(f)
                information = information.replace('\n','').replace('\r','').split(' ')
                self.vocab_size = int(information[0])
                self.embd_dimension = int(information[1])
                lines = f.readlines()
                for line in lines:
                    line = line.split(' ')
                    event2id[line[0]] = len(event2id)
                    id2event[len(id2event)] = line[0]
            # 加载预训练词向量
            wvmodel = gensim.models.KeyedVectors.load_word2vec_format(
                vector_address, binary=False, encoding='utf-8')
            #num_classes = len(wvmodel.index2word)
            weight = torch.zeros(self.vocab_size + 1, self.embd_dimension)
            for i in range(len(wvmodel.index2word)):
                weight[i, :] = torch.from_numpy(wvmodel.get_vector(
                    wvmodel.index2word[i]))
            self.embedding = nn.Embedding.from_pretrained(weight, padding_idx=self.vocab_size).cuda()
        self.event2id = event2id
        self.id2event = id2event
    def encodeEventByVocab(self, vocab_address, vector_address):
        v_o = open(vocab_address, 'r', encoding='utf-8')
        v_r = v_o.readlines()
        event2id = dict()
        id2event = dict()
        for line in v_r:
            line = line.replace('\r', '')
            line = line.replace('\n', '')
            line = line.split('\t')
            event2id[line[0]] = int(line[1])
            id2event[int(line[1])] = line[0]
        v_o.close()
        self.event2id = event2id
        self.id2event = id2event
        # 加载预训练词向量
        wvmodel = gensim.models.KeyedVectors.load_word2vec_format(
            vector_address, binary=False, encoding='utf-8')
        print(wvmodel.index2word)
        # num_classes = len(wvmodel.index2word)
        self.vocab_size = len(wvmodel.index2word)
        weight = torch.zeros(self.vocab_size + 1, self.embd_dimension)
        for i in range(len(wvmodel.index2word)):
            weight[i, :] = torch.from_numpy(wvmodel.get_vector(
                wvmodel.index2word[i]))
        self.embedding = nn.Embedding.from_pretrained(weight, padding_idx=self.vocab_size).cuda()
    def encodeTrace(self):
        encode_trace = list()
        max = 0
        for line in self.orginal_trace:
            trace_temp = list()
            for line2 in line:
                trace_temp.append([self.event2id[line2[0]], line2[1]])
            if len(trace_temp) > max:
                max = len(trace_temp)
            encode_trace.append(trace_temp.copy())
        self.max = max
        self.encode_trace = encode_trace
    def splitData(self,train_splitThreshold = 1):
        self.train_dataset, self.test_dataset = train_test_split(self.encode_trace, train_size=train_splitThreshold, test_size=1-train_splitThreshold)
    def initBatchData(self, time_unit, start_pos):
        if time_unit == 'second':
            time_unit = 1
        elif time_unit == 'minute':
            time_unit = 60
        elif time_unit == 'hour':
            time_unit = 60 * 60
        elif time_unit == 'day':
            time_unit = 24 * 60 * 60
        elif time_unit == 'month':
            time_unit = 30 * 24 * 60 * 60
        train_singleLengthData = dict()
        test_singleLengthData = dict()
        train_mixLengthData = list()
        test_mixLengthData = list()
        train_maxLength = 0
        test_maxLength = 0

        for line in self.train_dataset:
            train_input_temp = list()
            for line2 in line:
                train_input_temp.append(line2[0])
                if len(train_input_temp) > train_maxLength:
                    train_maxLength = len(train_input_temp)
                target_time = abs((datetime.strptime(str(line2[1]), '%Y-%m-%d %H:%M:%S') - \
                               datetime.strptime(str(line[-1][1]), '%Y-%m-%d %H:%M:%S')).total_seconds() / time_unit)
                try:
                    train_singleLengthData[len(train_input_temp)].append((train_input_temp.copy(), target_time))
                except BaseException as e:
                    train_singleLengthData[len(train_input_temp)] = list()
                    train_singleLengthData[len(train_input_temp)].append((train_input_temp.copy(), target_time))
                if len(train_input_temp) >= start_pos:
                    train_mixLengthData.append((train_input_temp.copy(), target_time))
        for line in self.test_dataset:
            test_input_temp = list()
            for line2 in line:
                test_input_temp.append(line2[0])
                if len(test_input_temp) > test_maxLength:
                    test_maxLength = len(test_input_temp)
                target_time = abs((datetime.strptime(str(line2[1]), '%Y-%m-%d %H:%M:%S') - \
                               datetime.strptime(str(line[-1][1]), '%Y-%m-%d %H:%M:%S')).total_seconds() / time_unit)
                try:
                    test_singleLengthData[len(test_input_temp)].append((test_input_temp.copy(), target_time))
                except BaseException as e:
                    test_singleLengthData[len(test_input_temp)] = list()
                    test_singleLengthData[len(test_input_temp)].append((test_input_temp.copy(), target_time))
                if len(test_input_temp) >= start_pos:
                    test_mixLengthData.append((test_input_temp.copy(), target_time))
        self.train_singleLengthData = train_singleLengthData
        self.test_singleLengthData = test_singleLengthData
        self.train_mixLengthData = train_mixLengthData
        self.test_mixLengthData = test_mixLengthData
        self.train_maxLength = train_maxLength
        self.test_maxLength = test_maxLength
    def generateSingleLengthBatch(self,batch_size,length_size):
        train_batch_single = list()
        test_batch_single = list()
        input_temp = list()
        target_temp = list()
        max_length = 0
        if length_size in self.train_batch_single:
            self.train_batch = self.train_batch_single[length_size]
            self.test_batch = self.test_batch_single[length_size]
            return 0
        for line in self.train_singleLengthData[length_size]:
            if len(input_temp) == batch_size:
                for num in range(len(input_temp)):
                    while len(input_temp[num]) < max_length:
                        input_temp[num].append(self.vocab_size)
                train_batch_single.append((input_temp.copy(),target_temp.copy()))
                max_length = 0
                input_temp = list()
                target_temp = list()
                input_temp.append(line[0])
                target_temp.append(line[1])
            input_temp.append(line[0])
            target_temp.append(line[1])
        while len(input_temp) < batch_size:
            if len(train_batch_single) ==0 and len(input_temp) == 0:
                break
            elif len(train_batch_single) ==0 and len(input_temp) != 0:
                ran1 = random.randint(0, len(input_temp) - 1)
                input_temp.append(input_temp[ran1])
                target_temp.append(target_temp[ran1])
            elif len(train_batch_single) !=0 and len(input_temp) != 0:
                ran1 = random.randint(0, len(train_batch_single)-1)
                (ran_input,ran_target) = train_batch_single[ran1]
                ran2 = random.randint(0, len(train_batch_single[ran1])-1)
                input_temp.append(ran_input[ran2])
                target_temp.append(ran_target[ran2])
        train_batch_single.append((input_temp.copy(), target_temp.copy()))
        max_length = 0
        input_temp = list()
        target_temp = list()
        for line in self.test_singleLengthData[length_size]:
            if len(input_temp) == batch_size:
                for num in range(len(input_temp)):
                    while len(input_temp[num]) < max_length:
                        input_temp[num].append(self.vocab_size)
                test_batch_single.append((input_temp.copy(),target_temp.copy()))
                max_length = 0
                input_temp = list()
                target_temp = list()
                input_temp.append(line[0])
                target_temp.append(line[1])
            input_temp.append(line[0])
            target_temp.append(line[1])
        while len(input_temp) < batch_size:
            #print(len(test_batch_single),test_batch_single)
            if len(test_batch_single) ==0 and len(input_temp) == 0:
                break
            elif len(test_batch_single) ==0 and len(input_temp) != 0:
                ran1 = random.randint(0, len(input_temp) - 1)
                input_temp.append(input_temp[ran1])
                target_temp.append(target_temp[ran1])
            elif len(test_batch_single) !=0 and len(input_temp) != 0:
                ran1 = random.randint(0, len(test_batch_single)-1)
                (ran_input,ran_target) = test_batch_single[ran1]
                ran2 = random.randint(0, len(test_batch_single[ran1])-1)
                input_temp.append(ran_input[ran2])
                target_temp.append(ran_target[ran2])
        test_batch_single.append((input_temp.copy(), target_temp.copy()))
        #print(test_batch_single)
        self.train_batch_single[length_size] = train_batch_single
        self.test_batch_single[length_size] = test_batch_single
        self.train_batch = self.train_batch_single[length_size]
        self.test_batch = self.test_batch_single[length_size]
    def generateMixLengthBatch(self, batch_size):
        train_batch = list()
        test_batch = list()
        input_temp = list()
        target_temp = list()
        max_length = 0
        if len(self.test_batch_mix) > 0:
            self.train_batch = self.train_batch_mix
            self.test_batch = self.test_batch_mix
            return 0
        for line in self.train_mixLengthData:
            if len(input_temp) == batch_size:
                for num in range(len(input_temp)):
                    while len(input_temp[num]) < max_length:
                        input_temp[num].append(self.vocab_size)
                train_batch.append((input_temp.copy(),target_temp.copy()))
                max_length = 0
                input_temp = list()
                target_temp = list()
            input_temp.append(line[0])
            target_temp.append(line[1])
            if max_length < len(line[0]):
                max_length = len(line[0])
        while len(input_temp) < batch_size:
            ran1 = random.randint(0, len(train_batch)-1)
            (ran_input,ran_target) = train_batch[ran1]
            ran2 = random.randint(0, len(train_batch[ran1])-1)
            input_temp.append(ran_input[ran2].copy())
            target_temp.append(ran_target[ran2])
        max_length = 0
        for line in input_temp:
            if max_length < len(line):
                max_length = len(line)
        for num in range(len(input_temp)):
            while len(input_temp[num]) < max_length:
                input_temp[num].append(self.vocab_size)
        train_batch.append((input_temp.copy(), target_temp.copy()))
        max_length = 0
        input_temp = list()
        target_temp = list()
        for line in self.test_mixLengthData:
            if len(input_temp) == batch_size:
                for num in range(len(input_temp)):
                    while len(input_temp[num]) < max_length:
                        input_temp[num].append(self.vocab_size)
                test_batch.append((input_temp.copy(),target_temp.copy()))
                max_length = 0
                input_temp = list()
                target_temp = list()
            input_temp.append(line[0])
            target_temp.append(line[1])
            if max_length < len(line[0]):
                max_length = len(line[0])

        while len(input_temp) < batch_size:
            ran1 = random.randint(0, len(test_batch)-1)
            (ran_input,ran_target) = test_batch[ran1]
            ran2 = random.randint(0, len(test_batch[ran1])-1)
            input_temp.append(ran_input[ran2].copy())
            target_temp.append(ran_target[ran2])
        max_length = 0
        for line in input_temp:
            if max_length < len(line):
                max_length = len(line)

        for num in range(len(input_temp)):
            while len(input_temp[num]) < max_length:
                input_temp[num].append(self.vocab_size)
        test_batch.append((input_temp.copy(), target_temp.copy()))

        self.train_batch = train_batch
        self.test_batch = test_batch
        self.train_batch_mix = train_batch
        self.test_batch_mix = test_batch





