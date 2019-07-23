#coding: utf-8
from GRU import GRU
from GRUAtt import GRUAtt
from BiGRU import BiGRU
from BiGRUAtt import BiGRUAtt
from LSTM import LSTM
from LSTMAtt import LSTMAtt
from BiLSTM import BiLSTM
from BiLSTMAtt import BiLSTMAtt
from input_data import InputData
from collections import deque
import numpy as np
import os
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
#time_unit = second minute hour day month
#train_type = single iteration mix



def train(data_address,data_name,vector_address=None, vocab_address=None, embd_dimension =2, train_splitThreshold=0.7,
          time_unit='second', batch_size=20, start_pos=3, stop_pos=10, length_size=3, prefix_minLength=0, prefix_maxLength=None,
          loss_type= 'L1Loss', optim_type= 'Adam', model_type='BiGRUAtt', hidden_dim=5,
          train_type='iteration', n_layer=1, dropout=1, max_epoch_num=500, learn_rate_min = 0.0001,
          train_record_folder='./train_record/', model_save_folder='./model/', result_save_folder='./result/' ):
    #初始化数据
    out_size = 1
    epoch = 0
    learn_rate = 0.01
    learn_rate_backup = learn_rate
    learn_rate_down = 0.001
    loss_deque = deque(maxlen=20)
    loss_change_deque = deque(maxlen=30)
    loss_chage = 0

    data = InputData(data_address, embd_dimension = embd_dimension, time_unit = 'second')
    if vector_address == None:
        data.encodeEvent(None)
    elif vector_address != None and vocab_address != None:
        data.encodeEventByVocab(vocab_address, vector_address)
    else:
        data.encodeEvent(vector_address)
    data.encodeTrace()
    data.splitData(train_splitThreshold)
    data.initBatchData(time_unit, start_pos)
    if train_type == 'mix':
        data.generateMixLengthBatch(batch_size)
    else:
        data.generateSingleLengthBatch(batch_size,start_pos)

    #初始化模型
    if model_type == 'LSTM':
        model=LSTM(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                   batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)
    elif model_type == 'LSTMAtt':
        model=LSTMAtt(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                        batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)
    elif model_type == 'BiLSTM':
        model=BiLSTM(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                       batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)
    elif model_type == 'BiLSTMAtt':
        model=BiLSTMAtt(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                          batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)
    elif model_type == 'GRU':
        model=GRU(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer ,dropout=dropout, embedding=data.embedding)
    elif model_type == 'GRUAtt':
        model=GRUAtt(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                       batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)
    elif model_type == 'BiGRU':
        model=BiGRU(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                      batch_size=batch_size,n_layer=n_layer,dropout=dropout, embedding=data.embedding)
    elif model_type == 'BiGRUAtt':
        model=BiGRUAtt(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                         batch_size=batch_size,n_layer=n_layer,dropout=dropout, embedding=data.embedding)
    if loss_type == 'L1Loss':
        criterion = nn.L1Loss().cuda()
    elif loss_type == 'MSELoss':
        criterion = nn.MSELoss().cuda()

    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    #初始化存储文件
    start_time = datetime.now().strftime('%Y-%m-%d(%H-%M-%S)')
    model_detal = 'embdDim' + str(embd_dimension) + '_loss' + loss_type + '_optim' + optim_type + '_hiddenDim' \
                  + str(hidden_dim) + '_startPos' + str(start_pos) + '_trainType' + train_type + '_nLayer' + str(n_layer) \
                  + '_dropout' + str(dropout)
    save_model_folder = model_save_folder + data_name + '/' + model_type + '/' + model_detal + '/'
    save_record_all = train_record_folder + data_name +'_sum.csv'
    save_record_single = train_record_folder + data_name + '/' + model_type + '/' + model_detal + '/'
    save_result_folder = result_save_folder + data_name + '/' + model_type + '/' + model_detal + '/'
    for folder in [save_model_folder,save_record_single]:
        if not os.path.exists(folder):
            os.makedirs(folder)
        save_record_single = save_record_single + start_time + '.csv'
    if not os.path.exists(save_record_all):
        save_record_all_open = open(save_record_all, 'a', encoding='utf-8')
        save_record_all_write = 'modelType,embdDim,lossType,optimType,hiddenDim,startPos,trainType,layerNum,' \
                                'dropout,epoch,learnRate,MSE,MAE,RMSE,meanLoss,totalLoss,modelFile,recordFile,resultFile\n'
        save_record_all_open.writelines(save_record_all_write)
        save_record_all_open.close()
    save_record_single_open = open(save_record_single,'w',encoding='utf-8')
    save_record_single_write = 'epoch,startPos,learnRate,MSE,MAE,RMSE,meanLoss,totalLoss\n'
    save_record_single_open.writelines(save_record_single_write)
    #开始训练
    if train_type == 'iteration':
        for start_pos_temp in range(start_pos,stop_pos):
            epoch = 1
            learn_rate = learn_rate_backup
            while epoch < max_epoch_num and learn_rate >= learn_rate_min:
                total_loss = torch.FloatTensor([0])
                for (input, target) in data.train_batch:
                    optimizer.zero_grad()
                    input = np.array(input)
                    target = np.array([[t] for t in target])
                    #target = np.array(target)
                    input = Variable(torch.LongTensor(input).cuda()).cuda()
                    # print(data_input.shape)
                    target = Variable(torch.LongTensor(target).cuda()).cuda()
                    #print(input, target)
                    target = target.float()
                    output = model(input)
                    loss = criterion(output, target)
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    total_loss += loss.data
                loss_deque.append(total_loss.item())
                loss_change_deque.append(total_loss.item())
                loss_change = total_loss.item() - sum(loss_deque) / len(loss_deque)
                loss_change = abs(loss_change)
                MSE, MAE, RMSE, TOTAL, MEAN = evaluate(model,data.test_batch)
                if loss_change < 10 and len(loss_deque) == 20:
                    now_time = datetime.now().strftime('%Y-%m-%d(%H-%M-%S)')
                    model_save = save_model_folder + now_time + '.pth'
                    torch.save(model, model_save)
                    result_save_file = result_save_folder + 'length' + str(start_pos_temp) + 'epoch' + str(epoch) + now_time + '.csv'
                    result_save_open = open(result_save_file,'w',encoding='utf-8')
                    result_save_write = 'epoch,startPos,prefixLength,learnRate,MSE,MAE,RMSE,meanLoss,totalLoss\n'
                    result_save_open.writelines(result_save_write)
                    result_save_write = str(epoch) + ',' + str(start_pos_temp) + ',' + 'mix' + ',' + str(learn_rate) + ',' + str(MSE) + ',' + str(MAE) \
                    + ',' + str(RMSE) + ',' + str(total_loss.item() / len(data.train_batch)) + ',' + str(
                        total_loss.item()) + '\n'
                    result_save_open.writelines(result_save_write)
                    for prefix_length in range(start_pos_temp,stop_pos + 1):
                        data.generateSingleLengthBatch(batch_size,prefix_length)
                        if len(data.test_batch) != 0:
                            MSE1, MAE1, RMSE1, TOTAL1, MEAN1 = evaluate(model, data.test_batch)
                            result_save_write = str(epoch) + ',' + str(start_pos_temp) + ',' + str(prefix_length) + ',' + str(
                                learn_rate) + ',' + str(MSE1) + ',' + str(MAE1) \
                                                + ',' + str(RMSE1) + ',' + str(
                                total_loss.item() / len(data.train_batch)) + ',' + str(
                                total_loss.item()) + '\n'
                        else:
                            result_save_write = str(epoch) + ',' + str(start_pos_temp) + ',' + str(prefix_length) + ',' + str(
                                learn_rate) + ',' + '无测试数据' + ',' + '无测试数据' \
                                                + ',' + '无测试数据' + ',' + str(
                                total_loss.item() / len(data.train_batch)) + ',' + str(
                                total_loss.item()) + '\n'
                        result_save_open.writelines(result_save_write)
                    result_save_open.close()
                    if train_type == 'mix':
                        data.generateMixLengthBatch(batch_size)
                    else:
                        data.generateSingleLengthBatch(batch_size, start_pos_temp)
                    save_record_all_open = open(save_record_all,'a',encoding='utf-8')
                    save_record_all_write = model_type +','+ str(embd_dimension) +','+ loss_type \
                                            +','+ optim_type +','+ str(hidden_dim) +','+ str(start_pos_temp) \
                                            +','+ train_type +','+ str(n_layer) +','+ str(dropout) \
                                            +','+ str(epoch) +','+ str(learn_rate) +','+ str(MSE) +','+ str(MAE)\
                                            +','+ str(RMSE) +','+ str(total_loss.item()/len(data.train_batch)) +','+ str(total_loss.item())\
                                            +','+ model_save +',' + save_record_single +',' + result_save_file + '\n'

                    save_record_all_open.writelines(save_record_all_write)
                    save_record_all_open.close()
                    if learn_rate > learn_rate_down:
                        learn_rate = learn_rate - learn_rate_down
                    else:
                        learn_rate_down = learn_rate_down * 0.1
                        learn_rate = learn_rate - learn_rate_down
                    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
                    loss_deque = deque(maxlen=20)
                    loss_deque.append(total_loss.item())
                if len(loss_change_deque) == 30 and (max(loss_change_deque) - min(loss_change_deque) < 20):
                    now_time = datetime.now().strftime('%Y-%m-%d(%H-%M-%S)')
                    model_save = save_model_folder + now_time + '.pth'
                    torch.save(model, model_save)
                    result_save_file = result_save_folder + 'length' + str(start_pos_temp) + 'epoch' + str(epoch) + now_time + '.csv'
                    result_save_open = open(result_save_file,'w',encoding='utf-8')
                    result_save_write = 'epoch,startPos,prefixLength,learnRate,MSE,MAE,RMSE,meanLoss,totalLoss\n'
                    result_save_open.writelines(result_save_write)
                    result_save_write = str(epoch) + ',' + str(start_pos_temp) + ',' + 'mix' + ',' + str(learn_rate) + ',' + str(MSE) + ',' + str(MAE) \
                    + ',' + str(RMSE) + ',' + str(total_loss.item() / len(data.train_batch)) + ',' + str(
                        total_loss.item()) + '\n'
                    result_save_open.writelines(result_save_write)
                    for prefix_length in range(start_pos_temp,stop_pos + 1):
                        data.generateSingleLengthBatch(batch_size,prefix_length)
                        if len(data.test_batch) != 0:
                            MSE1, MAE1, RMSE1, TOTAL1, MEAN1 = evaluate(model, data.test_batch)
                            result_save_write = str(epoch) + ',' + str(start_pos_temp) + ',' + str(prefix_length) + ',' + str(
                                learn_rate) + ',' + str(MSE1) + ',' + str(MAE1) \
                                                + ',' + str(RMSE1) + ',' + str(
                                total_loss.item() / len(data.train_batch)) + ',' + str(
                                total_loss.item()) + '\n'
                        else:
                            result_save_write = str(epoch) + ',' + str(start_pos_temp) + ',' + str(prefix_length) + ',' + str(
                                learn_rate) + ',' + '无测试数据' + ',' + '无测试数据' \
                                                + ',' + '无测试数据' + ',' + str(
                                total_loss.item() / len(data.train_batch)) + ',' + str(
                                total_loss.item()) + '\n'
                        result_save_open.writelines(result_save_write)
                    result_save_open.close()
                    if train_type == 'mix':
                        data.generateMixLengthBatch(batch_size)
                    else:
                        data.generateSingleLengthBatch(batch_size, start_pos_temp)
                    save_record_all_open = open(save_record_all,'a',encoding='utf-8')
                    save_record_all_write = model_type +','+ str(embd_dimension) +','+ loss_type \
                                            +','+ optim_type +','+ str(hidden_dim) +','+ str(start_pos_temp) \
                                            +','+ train_type +','+ str(n_layer) +','+ str(dropout) \
                                            +','+ str(epoch) +','+ str(learn_rate) +','+ str(MSE) +','+ str(MAE)\
                                            +','+ str(RMSE) +','+ str(total_loss.item()/len(data.train_batch)) +','+ str(total_loss.item())\
                                            +','+ model_save +',' + save_record_single + ',' + result_save_file + '\n'
                    save_record_all_open.writelines(save_record_all_write)
                    save_record_all_open.close()
                    if learn_rate > learn_rate_down:
                        learn_rate = learn_rate - learn_rate_down
                    else:
                        learn_rate_down = learn_rate_down * 0.1
                        learn_rate = learn_rate - learn_rate_down
                    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
                    loss_change_deque = deque(maxlen=30)
                    loss_change_deque.append(total_loss.item())
                print(MSE, MAE, RMSE, TOTAL, total_loss.item(),epoch,learn_rate,loss_change)
                # save_record_single_write = 'epoch'+ str(epoch) + 'startPos'+ str(start_pos_temp) +'learnRate'+ str(learn_rate) + 'MSE'+ str(MSE) +'MAE'+ str(MAE)\
                #                         +'RMSE'+ str(RMSE) + 'meanLoss'+ str(total_loss.item()/len(data.train_batch)) + 'totalLoss'+ str(total_loss.item()) + '\n'
                save_record_single_write = str(epoch) + ','+ str(start_pos_temp) +','+ str(learn_rate) + ','+ str(MSE) +','+ str(MAE)\
                                        +','+ str(RMSE) + ','+ str(total_loss.item()/len(data.train_batch)) + ','+ str(total_loss.item()) + '\n'
                save_record_single_open.writelines(save_record_single_write)
                #print(loss_change)
                epoch = epoch + 1
    save_record_single_open.close()
def evaluate(model, test_batchs):
    target_list = list()
    predict_list = list()
    for (input, target) in test_batchs:
        input = np.array(input)
        #print(input)
        input = Variable(torch.LongTensor(input).cuda()).cuda()

        prediction = model(input)
        predict_list += [pdic.item() for pdic in prediction]
        target_list += target
    MSE = computeMSE(target_list,predict_list)
    MAE = computeMAE(target_list,predict_list)
    RMSE = sqrt(MSE)
    TOTAL = computeTOTAL(target_list,predict_list)
    MEAN = computeMEAN(target_list,predict_list)
    return MSE,MAE,RMSE,TOTAL,MEAN
def computeMAE(list_a,list_b):
    MAE_temp = []
    for num in range(len(list_a)):
        MAE_temp.append(abs(list_a[num]-list_b[num]))
    MAE = sum(MAE_temp)/len(list_a)
    return MAE
def computeMSE(list_a,list_b):
    MSE_temp = []
    for num in range(len(list_a)):
        MSE_temp.append((list_a[num] - list_b[num]) * (list_a[num] - list_b[num]))
    MSE = sum(MSE_temp) / len(list_a)
    return MSE
def computeTOTAL(list_a,list_b):
    TOTAL_temp = []
    for num in range(len(list_a)):
        TOTAL_temp.append(abs(list_a[num] - list_b[num]))
    TOTAL = sum(TOTAL_temp)
    return TOTAL
def computeMEAN(list_a,list_b):
    MEAN_temp = []
    for num in range(len(list_a)):
        MEAN_temp.append(abs(list_a[num] - list_b[num]))
    MEAN = sum(MEAN_temp)/len(list_a)
    return MEAN



# train('./data/helpdesk_extend.csv',vector_address = './vector/vector.txt', vocab_address = None, train_splitThreshold = 0.7,
#       time_unit = 'second', batch_size=20, )
train('./data/helpdesk_extend.csv', data_name='helpdesk_extend',
      vector_address='./vector/helpdesk_extend_vectors_2CBoW_noTime_noEnd_Vector_vLoss_v1.txt',
      vocab_address='./vector/helpdesk_extend_2CBoW_noTime_noEnd_vocabulary.txt',
      embd_dimension=2, train_splitThreshold=0.7, time_unit='day', batch_size=20, )
#test inputdata
#1.
