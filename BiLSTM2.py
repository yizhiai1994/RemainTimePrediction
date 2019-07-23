'''
当前输入有补齐空位操作
'''
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
#import datetime
import time
#global veriabl
batch_size = 32
batch_temp = 0
torch.manual_seed(13)
torch.backends.cudnn.enabled = True

# def Embedding(file_address,vocabulary_file):
#     trace_list_orginal = []
#     trace_list_embd = []
#     trace_temp = []
#     event_list = []
#     event_embd_dict = {}
#     embd_event_dict = {}
#     max = 0
#     v_o = open(vocabulary_file,'r',encoding='utf-8')
#     v_r = v_o.readlines()
#     for line in v_r:
#         line = line.replace('\r', '')
#         line = line.replace('\n', '')
#         line = line.split('\t')
#         event_embd_dict[line[0]] = int(line[1])
#         embd_event_dict[int(line[1])] = line[0]
#     v_o.close()
#     f_o = open(file_address,'r',encoding='utf-8')
#     f_r = f_o.readlines()
#     record = []
#     flag = 0
#     for line in f_r:
#         if flag == 0:
#             flag = 1
#             continue
#         record.append(line)
#     flag = record[0][0]
#     for line in record:
#         line = line.replace('\r','').replace('\n','')
#         line = line.split(',')
#         event_list.append(line[1])
#         if line[0] == flag:
#             #print(line[1:])
#             trace_temp.append([line[1],line[2]])
#         else:
#             flag = line[0]
#             if len(trace_temp) > 2:
#                 trace_list_orginal.append(trace_temp.copy())
#             trace_temp = []
#             trace_temp.append([line[1],line[2]])
#     event_list = set(event_list)
#
#     for line in trace_list_orginal:
#         trace_temp = []
#         for line2 in line:
#             #print(line2)
#             trace_temp.append([event_embd_dict[line2[0]],line2[1]])
#         if len(trace_temp) > max:
#             max = len(trace_temp)
#         trace_list_embd.append(trace_temp.copy())
#     num_classes = len(event_list) + 1
#
#     #print(trace_list_orginal[0:3])
#     #print(trace_list_embd[0:3])
#     return num_classes,trace_list_orginal,trace_list_embd,embd_event_dict,event_embd_dict
def Embedding(embed_size,file_address,embedding_file,vocab_file):
    #加载词汇表
    v_o = open(vocab_file,'r',encoding='utf-8')
    v_r =v_o.readlines()
    event_embd_dict = {}
    embd_event_dict = {}
    for line in v_r:
        line = line.replace('\r', '')
        line = line.replace('\n', '')
        line = line.split('\t')
        event_embd_dict[line[0]] = int(line[1])
        embd_event_dict[int(line[1])] = line[0]
    v_o.close()
    #将原始活动序列进行编码
    trace_list_orginal = []
    trace_list_embd = []
    trace_temp = []
    event_list = []
    max = 0
    f_o = open(file_address, 'r', encoding='utf-8')
    f_r = f_o.readlines()
    record = []
    flag = 0
    for line in f_r:
        if flag == 0:
            flag = 1
            continue
        record.append(line)
    flag = record[0][0]
    for line in record:
        line = line.replace('\r', '').replace('\n', '')
        line = line.split(',')
        event_list.append(line[1])
        if line[0] == flag:
            # print(line[1:])
            trace_temp.append([line[1], line[2]])
        else:
            flag = line[0]
            if len(trace_temp) > 2:
                trace_list_orginal.append(trace_temp.copy())
            trace_temp = []
            trace_temp.append([line[1], line[2]])
    for line in trace_list_orginal:
        trace_temp = []
        for line2 in line:
            # print(line2)
            trace_temp.append([event_embd_dict[line2[0]], line2[1]])
        if len(trace_temp) > max:
            max = len(trace_temp)
        trace_list_embd.append(trace_temp.copy())
    #加载预训练词向量
    wvmodel = gensim.models.KeyedVectors.load_word2vec_format(
        embedding_file, binary=False, encoding='utf-8')
    num_classes = len(wvmodel.index2word)
    weight = torch.zeros(num_classes, embed_size)
    for i in range(len(wvmodel.index2word)):
        weight[i, :] = torch.from_numpy(wvmodel.get_vector(
            wvmodel.index2word[i]))
    #print(trace_list_orginal[0])
    #print(trace_list_embd[0])
    #print(event_embd_dict['1_2'])
    #print(wvmodel.vectors[event_embd_dict['1_2']])
    embedding = nn.Embedding.from_pretrained(weight,padding_idx=66)
    #print(embedding.weight[event_embd_dict['1_2']])
    # print(trace_list_orginal[0:3])
    # print(trace_list_embd[0:3])
    return num_classes, trace_list_orginal, trace_list_embd, embd_event_dict, event_embd_dict,embedding
def generateData(trace_list,threshold=0,train_test_unify = True):
    #print(trace_list[:3])
    train_data = []
    test_data = []
    train_label = []
    test_label = []
    return_train_data = []
    return_test_data = []
    return_train_label = []
    return_test_label = []
    if train_test_unify == False:
        if threshold == 0:
            for line in trace_list:
                data_temp = []
                label_temp = []
                for num in range(0, len(line) - 1):
                    data_temp.append(line[num])
                    label_temp.append(line[num + 1])
                train_data.append(data_temp.copy())
                train_label.append(label_temp.copy())
                test_data.append(data_temp.copy())
                test_label.append(label_temp.copy())
        else:
            train_temp,test_temp = train_test_split(trace_list,test_size=threshold)
            for line in train_temp:
                data_temp = []
                label_temp = []
                for num in range(0,len(line) - 1):
                    data_temp.append(line[num])
                    label_temp.append(line[num+1])
                train_data.append(data_temp.copy())
                train_label.append(label_temp.copy())
            for line in test_temp:
                data_temp = []
                label_temp = []
                for num in range(0,len(line) - 1):
                    data_temp.append(line[num])
                    label_temp.append(line[num+1])
                test_data.append(data_temp.copy())
                test_label.append(label_temp.copy())
    elif train_test_unify == True:
        if threshold == 0:
            for line in trace_list:
                train_data.append(line.copy())
                train_label.append(line.copy())
                test_data.append(line.copy())
                test_label.append(line.copy())
        else:
            train_temp,test_temp = train_test_split(trace_list,train_size=)
            for line in train_temp:
                train_data.append(line.copy())
                train_label.append(line.copy())
            for line in test_temp:
                test_data.append(line.copy())
                test_label.append(line.copy())
    #print(111111)
    #print(len(train_data),len(test_data))
    return train_data,train_label,test_data,test_label

def getData(data_list,label_list):
    global batch_temp
    global batch_size
    data_return = []
    label_return = []
    flag = True
    #print(data_list[:3])
    for num in range(batch_temp,batch_temp + batch_size):
        if num >= len(data_list):
            batch_temp = 0
            flag = False
            break
        data_return.append(data_list[num])
        #print(data_list[num])
        #print(label_list[num])
        label_return.append(label_list[num])
    if flag != False :
        batch_temp = batch_temp + batch_size
    #print(data_return)
    data_return = np.array(data_return)
    label_return = np.array(label_return)
    #print(data_return)
    data_return = Variable(torch.LongTensor(data_return))
    label_return = Variable(torch.LongTensor(label_return))
    return flag,data_return,label_return
def getData_oneLable(data_list, label_list, batch_size, posType='All', startPos=0):
    #print(label_list)
    global batch_temp
    #global batch_size
    data_return = []
    label_return = []
    length_return = []
    flag = True

    # print(data_list[:3])
    while len(data_return) < batch_size:

        if batch_temp >= len(data_list):
            batch_temp = 0
            flag = False
            break
        #print(posType)
        if posType == 'All':
            for num in range(startPos,len(data_list[batch_temp])+1):
                data_temp = data_list[batch_temp][0:num]
                #print(data_list)
                data_temp_2 = [unit[0] for unit in data_temp]
                #label_temp = label_list[num][num2]
                #print(data_temp)
                label_temp = (datetime.strptime(str(label_list[batch_temp][-1][1]),'%Y-%m-%d %H:%M:%S') - \
                             datetime.strptime(str(data_temp[-1][1]), '%Y-%m-%d %H:%M:%S')).total_seconds()
                data_return.append(data_temp_2.copy())
                length_return.append(len(data_list[batch_temp]))
                # print(0,data_list[batch_temp])
                # print(1,label_list[batch_temp])
                # print(2,data_temp)
                # print(3,label_temp)
                # print(4,str(label_list[batch_temp][-1][1]))
                # print(5,datetime.strptime(str(label_list[batch_temp][-1][1]),'%Y-%m-%d %H:%M:%S'))
                # print(6,str(data_list[batch_temp][num][1]))
                # print(7,datetime.strptime(str(data_list[batch_temp][num][1]), '%Y-%m-%d %H:%M:%S'))
                # print(8,label_list[num])
                # print(9,data_list[num])
                # print(10,num)
                label_return.append(label_temp)
        elif posType == 'Single':
            if startPos > len(data_list[batch_temp]):
                batch_temp = batch_temp + 1
                continue
            data_temp = data_list[batch_temp][0:startPos]
            # print(0,startPos)
            # print(data_list)
            data_temp_2 = [unit[0] for unit in data_temp]
            # label_temp = label_list[num][num2]
            #print(label_list[batch_temp])
            label_temp = (datetime.strptime(str(label_list[batch_temp][-1][1]), '%Y-%m-%d %H:%M:%S') - \
                          datetime.strptime(str(data_temp[-1][1]), '%Y-%m-%d %H:%M:%S')).total_seconds()/60/60/24
            # print(1,data_temp)
            # print(2,label_list[batch_temp])
            # print(3,label_temp)
            data_return.append(data_temp_2.copy())
            label_return.append(label_temp)
            length_return.append(len(data_list[batch_temp]))
        #print(batch_temp)
        batch_temp = batch_temp + 1
            #print(data_list)
        # if flag != False:
        #     batch_temp = batch_temp + batch_size
    #print(data_return)
    #print(label_return)
    return flag, data_return, label_return, length_return
def evaluate(X,Y):
    #print(X)
    #1
    MSE_temp = []
    if len(X) == 0:
        return 'len(x)=0', 'len(x)=0', 'len(x)=0', 'len(x)=0', 'len(x)=0', 'len(x)=0', 'len(x)=0', 'len(x)=0', 'len(x)=0'
    for num in range(len(X)):
        MSE_temp.append((X[num]-Y[num])*(X[num]-Y[num]))
    MSE1 = sum(MSE_temp)/len(X)
    RMSE1 = sqrt(MSE1)
    MAE_temp = []
    for num in range(len(X)):
        MAE_temp.append(abs(X[num]-Y[num]))
    MAE1 = sum(MAE_temp)/len(X)

    #2
    target = X
    prediction = Y
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值

    #print("Square Error: ", squaredError)
    #print("Absolute Value of Error: ", absError)
    MSE2 =sum(squaredError) / len(squaredError) # 均方误差MSE
    RMSE2 = sqrt(sum(squaredError) / len(squaredError))  # 均方根误差RMSE
    MAE2 = sum(absError) / len(absError) # 平均绝对误差MAE
    targetDeviation = []
    targetMean = sum(target) / len(target)  # target平均值
    for val in target:
        targetDeviation.append((val - targetMean) * (val - targetMean))
    TargetVariance =  sum(targetDeviation) / len(targetDeviation)  # 方差
    TargetStandardDeviation =  sqrt(sum(targetDeviation) / len(targetDeviation))  # 标准差
    return MSE1,MSE2,MAE1,MAE2,RMSE1,RMSE2,targetMean,TargetVariance,TargetStandardDeviation
class PredictionModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim,n_hidden,out_size,batch_size=1,n_layer = 1, dropout = 0,
                 preEmbd=None,baseModel='LSTM',attetionModel='None'):
        super(PredictionModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_hidden = n_hidden
        self.out_shape = out_size
        self.embedding = preEmbd
        self.attetionModel = attetionModel
        self.baseModel = baseModel
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.biType = 1
        self.dropout = dropout
        if baseModel == 'BiLSTM' or baseModel == 'BiGRU':
            self.biType = 2
        self.weight_W = nn.Parameter(torch.Tensor(batch_size, n_hidden * self.biType, n_hidden * self.biType).cuda()).cuda()
        self.weight_Mu = nn.Parameter(torch.Tensor(n_hidden * self.biType, n_layer).cuda()).cuda()
        print(self.weight_Mu )
        if baseModel == 'LSTM':
            print('Initialization LSTM Model')
            self.rnn = nn.LSTM(input_size = embedding_dim, hidden_size = n_hidden, dropout = self.dropout,num_layers = self.n_layer, bidirectional=False).cuda()
        elif baseModel == 'GRU':
            print('Initialization GRU Model')
            self.rnn = nn.GRU(input_size = embedding_dim, hidden_size = n_hidden, dropout = self.dropout,
                              num_layers = self.n_layer, bidirectional=False).cuda()
        elif baseModel == 'BiLSTM':
            print('Initialization BiLSTM Model')
            self.rnn = nn.LSTM(input_size = embedding_dim, hidden_size = n_hidden, dropout = self.dropout,
                               num_layers = self.n_layer, bidirectional=True).cuda()
        elif baseModel == 'BiGRU':
            print('Initialization BiGRU Model')
            self.rnn = nn.GRU(input_size = embedding_dim, hidden_size = n_hidden, dropout = self.dropout,
                               num_layers = self.n_layer, bidirectional=True).cuda()
        if self.attetionModel == 'Dot':
            print('Initialization Dot Attention')
        elif self.attetionModel == 'General':
            print('Initialization General Attention')
        elif self.attetionModel == 'DotNoDecoder':
            print('Initialization DotNoDecoder Attention')
        elif self.attetionModel == 'GeneralNoDecoder':
            print('Initialization GeneralNoDecoder Attention')
        self.out = nn.Linear(n_hidden * self.biType, out_size).cuda()

        # self.out = nn.Sequential(
        #     nn.Linear(n_hidden, out_size)
        # )

    #
    #  : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix

    def attention_net(self, rnn_output, final_state):
        #print(1111111)
        hidden = final_state.view(self.batch_size, self.n_hidden * self.biType ,
                                  self.n_layer)  # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        if self.attetionModel == 'Dot':
            #print('Initialization Dot Attention')
            attn_weights = torch.bmm(rnn_output, hidden).squeeze(2).cuda()
            #print('4',attn_weights)
        elif self.attetionModel == 'General':
            #print('Initialization General Attention')
            attn_weights = torch.bmm(rnn_output, self.weight_W).squeeze(2).cuda()  # attn_weights : [batch_size, n_step]
            attn_weights = torch.bmm(attn_weights, hidden).squeeze(2).cuda()
        elif self.attetionModel == 'DotNoDecoder':
            #print('Initialization DotNoDecoder Attention')
            #attn_weights = torch.bmm(rnn_output, self.weight_Mu).squeeze(2).cuda()
            #print(rnn_output.shape)
            #print(self.weight_Mu)
            #print(self.weight_Mu[0].item())
            attn_weights = torch.matmul(rnn_output,self.weight_Mu).cuda()
            #print('注意力矩阵', self.weight_Mu)
            #print('计算后的注意力权重attn_weights',attn_weights)
        elif self.attetionModel == 'GeneralNoDecoder':
            #print('Initialization GeneralNoDecoder Attention')
            attn_weights = torch.bmm(rnn_output, self.weight_W).squeeze(2).cuda()  # attn_weights : [batch_size, n_step]
            attn_weights = torch.bmm(attn_weights, self.weight_Mu).squeeze(2).cuda()

        #torch.nn.functional.softmax(input, dim=None, _stacklevel=3, dtype=None)
        #print(attn_weights)
        soft_attn_weights = F.softmax(attn_weights, 1)
        #print('经过softmax的注意力权重',soft_attn_weights)
        #print('soft_attn_weights',soft_attn_weights)~
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        #torch.bmm(batch1, batch2, out=None) → Tensor
        #torch.transpose(x, 0, 1) == x.transpose(0, 1)
        #transpose(dim0,dim1)返回输入矩阵input的转置。交换维度dim0和dim1。
        #************************************************************************
        #print(rnn_output.transpose(1, 2).shape)
        #print(soft_attn_weights.shape)
        context = torch.bmm(rnn_output.transpose(1, 2), soft_attn_weights).squeeze(2).cuda()
        #print(torch.bmm(rnn_output.transpose(1, 2), soft_attn_weights).shape)
        #print(torch.bmm(rnn_output.transpose(1, 2), soft_attn_weights).squeeze(2).shape)
        #print('经过计算的context',context)
        return context, soft_attn_weights.data.cpu().numpy()  # context : [batch_size, n_hidden * num_directions(=2)]

    def forward(self, X):
        # orignal input : [batch_size, len_seq, embedding_dim]
        input = self.embedding(X)
        # GRU input of shape (seq_len, batch, embedding_dim(input_size)):
        # LSTM input : [len_seq, batch_size, embedding_dim(input_size)]
        #print(input)
        #print(input.shape)
        input = input.permute(1, 0, 2)
        #print(input)
        #h_0 of LSTM shape (num_layers * num_directions, batch, hidden_size)
        #h_0 of GRU shape (num_layers * num_directions, batch, hidden_size):
        hidden_state = Variable(
            torch.randn(self.n_layer * self.biType, self.batch_size, self.n_hidden)).cuda()
        # hidden_state = Variable(
        #     torch.randn(self.n_layer * self.biType, 1, self.n_hidden)).cuda()
        # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        #c_0 of LSTM shape (num_layers * num_directions, batch, hidden_size)
        # cell_state = Variable(
        #     torch.randn(self.n_layer * self.biType, 1, self.n_hidden)).cuda()
        cell_state = Variable(
            torch.randn(self.n_layer * self.biType, self.batch_size, self.n_hidden)).cuda()
        # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        #output of LSTM shape (seq_len, batch, num_directions * hidden_size)
        #output of GRU shape (seq_len, batch, num_directions * hidden_size)
        #h_n of GRU shape (num_layers * num_directions, batch, hidden_size)
        #h_n of LSTM shape (num_layers * num_directions, batch, hidden_size)
        #c_n of LSTM shape (num_layers * num_directions, batch, hidden_size)
        if self.baseModel == 'LSTM' or self.baseModel == 'BiLSTM':
            #print(input.shape)
            #print(input)
            #print(hidden_state)
            #print(cell_state)
            output, (final_hidden_state, final_cell_state) = self.rnn(input, (hidden_state, cell_state))
        elif self.baseModel == 'GRU' or self.baseModel == 'BiGRU':
            #print(input, hidden_state.size())
            #print(input)
            output,final_hidden_state = self.rnn(input, hidden_state)
        #print('output',output)
       # print('final_hidden_state',final_hidden_state)
        if self.attetionModel == 'None':
            # s, b, h = final_hidden_state.shape
            # print('1',output)
            # output = final_hidden_state.view(s * b, h)
            # print('2', output)
            # output = self.out(output)
            # output = output.view(s, b, -1)
            #print(output)
            hn = output[-1]
            #print(hn)
            #print('刚从RNN出来的output', output[-1])
            #print('刚从RNN出来的final_hidden_state', final_hidden_state)
            #s, b, h = final_hidden_state.shape
            #output = final_hidden_state.view(s * b * h)
            #output = output.permute(1, 0, 2)
            #s, b, h = final_hidden_state.shape
            #print(s,b,h)
            #output = final_hidden_state.view(s * b, h)
            #print('转换完形状的output', output)
            #print(output.size())
            #print(s, b, h)
            output = self.out(hn)
            #print(output.size())
            #output = output.view(s, b, -1)

        elif self.attetionModel != 'None':
            #print('刚从RNN出来的output', output)
            output = output.permute(1, 0, 2)  # output : [batch_size, len_seq, n_hidden]
            #print('转换完形状的output',output)
            output, attention = self.attention_net(output, final_hidden_state)
            #print('经过attention的output',output)
            output = self.out(output)
            #print('经过linear的output',output)

        return  output # model : [batch_size, num_classes], attention : [batch_size, n_step]


def train(vocab_size, train_data, train_label, preEmbd, posType, startPos, result_folder, result_file,
          embedding_dim, n_hidden, out_size, epoch_sum, n_layer, dropout, batch_size,
          baseModel, attetionModel):
    #训练过程结果存储
    save_o = open(result_file+'_iteration_train.csv','w',encoding='utf-8')
    save_o.write('epoch,pos,trace_len,loss,predictTime,trueTime,D-value\n')
    f = open(result_file+'_iteration_all_result.csv','w',encoding='utf-8')
    f.write('Epoch,Pos,MSE1,MSE2,MAE1,MAE2,RMSE1,RMSE2,targetMean,TargetVariance,TargetStandardDeviation\n')
    target_list = []
    predict_list = []
    target_dict = {}
    predict_dict = {}
    data_remain_list = []
    target_remain_list = []
    #损失函数
    criterion = nn.MSELoss().cuda()
    #初始化模型
    model = PredictionModel(vocab_size, embedding_dim, n_hidden, out_size, batch_size, n_layer, dropout,
                 preEmbd,baseModel,attetionModel)

    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    #生成数据
    # Training
    for epoch in range(epoch_sum):
        print('epoch_sum',epoch_sum)
        print('epoch',epoch)
        target_list = []
        predict_list = []
        target_dict = {}
        predict_dict = {}
        optimizer.zero_grad()
        flag, input_batch, target_batch, length_batch = getData_oneLable(train_data, train_label,batch_size=batch_size,
                                                           posType=posType, startPos=startPos)
        print('epoch******************************************************',epoch)
        print(datetime.now())
        #print(input_batch)
        if batch_size == 1:
            while flag == True:
                #print(input_batch)
                for num in range(len(input_batch)):
                    #print(num)
                    data_input = np.array([input_batch[num]])
                    label_input = np.array([target_batch[num]])
                    #print(data_input)
                    data_input = Variable(torch.LongTensor(data_input).cuda()).cuda()
                    #print(data_input.shape)
                    label_input = Variable(torch.LongTensor([[label_input]]).cuda()).cuda()
                    label_input = label_input.float()
                    output = model(data_input)
                    optimizer.zero_grad()
                    loss = criterion(output, label_input)
                    #print(loss.item())
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    save_o.write(str(epoch)+','+str(len(input_batch[num]))+','+str(length_batch[num])+','+str(loss.item())+','+str(output.item())
                                 + ',' +str(target_batch[num])+','+str(output.item() - target_batch[num])+'\n')
                    target_list.append(label_input.item())
                    predict_list.append(output.item())
                    #f.write(str(len(input_batch[num])) + ',' + str(output.item()) + ',' + str(label_input) + '\n')
                    if len(input_batch[num]) not in target_dict:
                        target_dict[len(input_batch[num])] = []
                        predict_dict[len(input_batch[num])] = []
                        target_dict[len(input_batch[num])].append(label_input.item())
                        predict_dict[len(input_batch[num])].append(output.item())
                    elif len([input_batch[num]]) in target_dict:
                        target_dict[len(input_batch[num])].append(label_input.item())
                        predict_dict[len(input_batch[num])].append(output.item())
                flag, input_batch, target_batch, length_batch = getData_oneLable(train_data, train_label, batch_size,
                                                                   posType=posType, startPos=startPos)
            for num in range(len(input_batch)):
                data_input = np.array([input_batch[num]])
                label_input = np.array([target_batch[num]])
                data_input = Variable(torch.LongTensor(data_input).cuda()).cuda()
                label_input = Variable(torch.LongTensor([[label_input]]).cuda()).cuda()
                label_input = label_input.float()
                output = model(data_input)
                optimizer.zero_grad()
                loss = criterion(output, label_input)
                loss.backward(retain_graph=True)
                optimizer.step()
                save_o.write(str(epoch) + ',' + str(len(input_batch[num])) + ',' + str(length_batch[num]) + ',' + str(
                    loss.item()) + ',' + str(output.item())+ ',' + str(target_batch[num]) + ',' + str(output.item() - target_batch[num]) + '\n')
                target_list.append(label_input.item())
                predict_list.append(output.item())
                #f.write(str(len(input_batch[num])) + ',' + str(output.item()) + ',' + str(label_input) + '\n')
                if len(input_batch[num]) not in target_dict:
                    target_dict[len(input_batch[num])] = []
                    predict_dict[len(input_batch[num])] = []
                    target_dict[len(input_batch[num])].append(label_input.item())
                    predict_dict[len(input_batch[num])].append(output.item())
                elif len([input_batch[num]]) in target_dict:
                    target_dict[len(input_batch[num])].append(label_input.item())
                    predict_dict[len(input_batch[num])].append(output.item())
        elif batch_size != 1:
            while flag == True:
                input_batch_temp = []
                for key in input_batch:
                    while len(key) < 153:
                        key.append(66)
                    #input_batch_temp.append(key)
                #print(len(input_batch))
                #print(input_batch_temp)
                #print(target_batch)
                #print(num)
                input_batch_temp = input_batch
                target_batch_temp = target_batch
                if len(input_batch)>batch_size:
                    input_batch_temp = input_batch_temp[0:batch_size]
                    target_batch_temp = target_batch[0:batch_size]
                    for key in input_batch[batch_size:]:
                        data_remain_list.append(key)
                    for key in target_batch[batch_size:]:
                        target_remain_list.append(key)
                #print(data_input)
                data_input = np.array(input_batch_temp)
                label_input = np.array(target_batch_temp)
                data_input = Variable(torch.LongTensor(data_input).cuda()).cuda()
                #print(data_input.shape)
                label_input = Variable(torch.LongTensor(label_input).cuda()).cuda()
                label_input = label_input.float()

                output = model(data_input)
                optimizer.zero_grad()
                #print(len(output))

                loss = criterion(output, label_input)
                #print(loss.item())
                loss.backward(retain_graph=True)
                optimizer.step()
                #f.write(str(len(input_batch[num])) + ',' + str(output.item()) + ',' + str(label_input) + '\n')
                    #print('label_input',label_input[num].item())
                    #print('output',output[num].item())
                flag, input_batch, target_batch, length_batch = getData_oneLable(train_data, train_label, batch_size,
                                                                   posType=posType, startPos=startPos)
                #print(input_batch)
            #print(data_remain_list)
            if len(data_remain_list) < batch_size:
                if len(input_batch) != 0 and (len(data_remain_list) + len(input_batch)) < batch_size:
                    input_batch = input_batch + data_remain_list
                    target_batch = target_batch + target_remain_list
                    while len(input_batch) < batch_size:
                        input_batch.append(input_batch[len(input_batch)-1])
                        target_batch.append(target_batch[len(target_batch)-1])
                        length_batch.append(length_batch[len(length_batch) - 1])
                    input_batch_temp = []
                    for key in input_batch:
                        while len(key) < 153:
                            key.append(66)
                        input_batch_temp.append(key)
                    data_input = np.array(input_batch_temp)
                    label_input = np.array(target_batch)
                    data_input = Variable(torch.LongTensor(data_input).cuda()).cuda()
                    label_input = Variable(torch.LongTensor(label_input).cuda()).cuda()
                    label_input = label_input.float()
                    output = model(data_input)
                    optimizer.zero_grad()
                    loss = criterion(output, label_input)
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    for num in range(len(input_batch)):
                        if len(input_batch[num]) not in target_dict:
                            #print(len(input_batch[num]))
                            target_dict[len(input_batch[num])] = []
                            predict_dict[len(input_batch[num])] = []
                            target_dict[len(input_batch[num])].append(label_input[num].item())
                            predict_dict[len(input_batch[num])].append(output[num].item())
                        elif len([input_batch[num]]) in target_dict:
                            target_dict[len(input_batch[num])].append(label_input[num].item())
                            predict_dict[len(input_batch[num])].append(output[num].item())


        if (epoch + 1) % 1000 == 0:
            #print(loss)
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', float(loss))
    save_o.close()
    f.close()
    return model
def test(test_data, test_label,model,result_file, posType, startPos, batch_size):
    result_file = result_file + '_' + str(startPos)
    f = open(result_file+'_iteration.csv','w',encoding='utf-8')
    f.write('pos,out,label,D_value'+'\n')
    flag, input_batch, target_batch, length_batch = getData_oneLable(test_data, test_label, batch_size = batch_size
                                                       , posType=posType, startPos=startPos)
    target_list = []
    predict_list = []
    target_dict = {}
    predict_dict = {}
    if batch_size == 1:
        while flag == True:
            for num in range(len(input_batch)):

                data_input = np.array([input_batch[num]])
                label_input = np.array([target_batch[num]])
                label_input = target_batch[num]
                data_input = Variable(torch.LongTensor(data_input).cuda()).cuda()
                predict = model(data_input)
                #print(predict)
                predict = predict.item()
                target_list.append(label_input)
                predict_list.append(predict)
                f.write(str(len(input_batch[num]))+','+str(predict)+','+str(label_input)+'\n')
                if len(input_batch[num]) not in target_dict:
                    target_dict[len(input_batch[num])] = []
                    predict_dict[len(input_batch[num])] = []
                    target_dict[len(input_batch[num])].append(label_input)
                    predict_dict[len(input_batch[num])].append(predict)
                elif len([input_batch[num]]) in target_dict:
                    target_dict[len(input_batch[num])].append(label_input)
                    predict_dict[len(input_batch[num])].append(predict)
            flag, input_batch, target_batch, length_batch = getData_oneLable(test_data, test_label, batch_size = batch_size,
                                                               posType=posType, startPos=startPos)
        for num in range(len(input_batch)):
            data_input = np.array([input_batch[num]])
            label_input = np.array([target_batch[num]])
            label_input = target_batch[num]
            data_input = Variable(torch.LongTensor(data_input).cuda()).cuda()
            predict = model(data_input)
            predict = predict.item()
            f.write(str(len(input_batch[num]))+','+str(predict) + ',' + str(label_input) + ','
                    + str(predict - label_input)+'\n')
            target_list.append(label_input)
            predict_list.append(predict)
            #print(len(input_batch[num]))
            if len(input_batch[num]) not in target_dict:
                #print(len(input_batch[num]))
                target_dict[len(input_batch[num])] = []
                predict_dict[len(input_batch[num])] = []
                target_dict[len(input_batch[num])].append(label_input)
                predict_dict[len(input_batch[num])].append(predict)
            elif len(input_batch[num]) in target_dict:
                target_dict[len(input_batch[num])].append(label_input)
                predict_dict[len(input_batch[num])].append(predict)
    elif batch_size != 1:
        while flag == True:
            #print(11111111111111111111)

            # while len(input_batch) < 14:
            #     input_batch.append(79)
            input_batch_temp = []
            for key in input_batch:
                while len(key) < 153:
                    key.append(66)
                input_batch_temp.append(key)
            #print(input_batch_temp)
            data_input = np.array(input_batch)
            label_input = np.array(target_batch)
            label_input = target_batch
            data_input = Variable(torch.LongTensor(data_input).cuda()).cuda()
            predict = model(data_input)
            #print(predict)
            #predict = predict.item()
            for num in range(len(input_batch)):

                target_list.append(label_input[num])
                predict_list.append(predict[num].item())
                f.write(str(len(input_batch[num]))+','+str(predict[num].item())+','+str(label_input[num])+'\n')
                if len(input_batch[num]) not in target_dict:
                    target_dict[len(input_batch[num])] = []
                    predict_dict[len(input_batch[num])] = []
                    target_dict[len(input_batch[num])].append(label_input[num])
                    predict_dict[len(input_batch[num])].append(predict[num].item())
                elif len([input_batch[num]]) in target_dict:
                    target_dict[len(input_batch[num])].append(label_input[num])
                    predict_dict[len(input_batch[num])].append(predict[num].item())
            flag, input_batch, target_batch, length_batch = getData_oneLable(test_data, test_label, batch_size = batch_size,
                                                               posType=posType, startPos=startPos)
        if len(input_batch) != 0:
            length_temp = len(input_batch)
            while len(input_batch) < batch_size:
                #print(input_batch)
                input_batch.append(input_batch[len(input_batch) % length_temp])
                target_batch.append(target_batch[len(target_batch) % length_temp])
                length_batch.append(length_batch[len(length_batch) % length_temp])
            input_batch_temp = []
            for key in input_batch:
                while len(key) < 153:
                    key.append(66)
                input_batch_temp.append(key)
            # print(input_batch_temp)
            data_input = np.array(input_batch_temp)
            label_input = np.array(target_batch)
            label_input = target_batch
            data_input = Variable(torch.LongTensor(data_input).cuda()).cuda()
            #print(data_input)
            predict = model(data_input)
            #predict = predict.item()
            for num in range(len(input_batch)):
                f.write(str(len(input_batch[num]))+','+str(predict[num].item()) + ',' + str(label_input[num]) + ','
                        + str(predict[num].item() - label_input[num])+'\n')
                target_list.append(label_input[num])
                predict_list.append(predict[num].item())
                #print(len(input_batch[num]))

                if len(input_batch[num]) not in target_dict:
                    #print(len(input_batch[num]))
                    target_dict[len(input_batch[num])] = []
                    predict_dict[len(input_batch[num])] = []
                    target_dict[len(input_batch[num])].append(label_input[num])
                    predict_dict[len(input_batch[num])].append(predict[num].item())
                elif len(input_batch[num]) in target_dict:
                    target_dict[len(input_batch[num])].append(label_input[num])
                    predict_dict[len(input_batch[num])].append(predict[num].item())
    f.close()
    #print(target_dict)
    #print('target_list',target_list)
    #print('target_dict',target_dict)
    return target_list, predict_list, target_dict, predict_dict
def iteration(model_temp,vocab_size, train_data, train_label, preEmbd, posType, startPos, result_folder, result_file,
          embedding_dim, n_hidden, out_size, epoch_sum, n_layer, dropout, batch_size,
          baseModel, attetionModel):
    #训练过程结果存储
    result_file = result_file + '_' + str(startPos)
    save_o = open(result_file+'_iteration_train.csv','w',encoding='utf-8')
    save_o.write('epoch,pos,trace_len,loss,predictTime,trueTime,D-value\n')
    f = open(result_file+'_all_iteration_result.csv','w',encoding='utf-8')
    f.write('Epoch,Pos,MSE1,MSE2,MAE1,MAE2,RMSE1,RMSE2,targetMean,TargetVariance,TargetStandardDeviation\n')
    target_list = []
    predict_list = []
    target_dict = {}
    predict_dict = {}
    #损失函数
    criterion = nn.MSELoss()
    #初始化模型
    model = model_temp

    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    #生成数据
    # Training
    for epoch in range(epoch_sum):
        target_list = []
        predict_list = []
        target_dict = {}
        predict_dict = {}
        optimizer.zero_grad()
        flag, input_batch, target_batch, length_batch = getData_oneLable(train_data, train_label,batch_size=batch_size,
                                                           posType=posType, startPos=startPos)
        print('epoch',epoch)
        print(datetime.now())
        if batch_size == 1:
            while flag == True:
                #print(input_batch)
                for num in range(len(input_batch)):
                    #print(num)
                    data_input = np.array([input_batch[num]])
                    label_input = np.array([target_batch[num]])
                    data_input = Variable(torch.LongTensor(data_input).cuda()).cuda()
                    label_input = Variable(torch.LongTensor([[label_input]]).cuda()).cuda()
                    label_input = label_input.float()
                    output = model(data_input)
                    optimizer.zero_grad()

                    loss = criterion(output, label_input)
                    #print(loss.item())
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    save_o.write(str(epoch)+','+str(len(input_batch[num]))+','+str(length_batch[num])+','+str(loss.item())+','+str(output.item())
                                 + ',' +str(target_batch[num])+','+str(output.item() - target_batch[num])+'\n')
                    target_list.append(label_input.item())
                    predict_list.append(output.item())
                    #f.write(str(len(input_batch[num])) + ',' + str(output.item()) + ',' + str(label_input) + '\n')
                    if len(input_batch[num]) not in target_dict:
                        target_dict[len(input_batch[num])] = []
                        predict_dict[len(input_batch[num])] = []
                        target_dict[len(input_batch[num])].append(label_input.item())
                        predict_dict[len(input_batch[num])].append(output.item())
                    elif len([input_batch[num]]) in target_dict:
                        target_dict[len(input_batch[num])].append(label_input.item())
                        predict_dict[len(input_batch[num])].append(output.item())
                flag, input_batch, target_batch, length_batch = getData_oneLable(train_data, train_label, batch_size,
                                                                   posType=posType, startPos=startPos)
            for num in range(len(input_batch)):
                data_input = np.array([input_batch[num]])
                label_input = np.array([target_batch[num]])
                data_input = Variable(torch.LongTensor(data_input).cuda()).cuda()
                label_input = Variable(torch.LongTensor([[label_input]]).cuda()).cuda()
                label_input = label_input.float()
                output = model(data_input)
                optimizer.zero_grad()
                loss = criterion(output, label_input)
                loss.backward(retain_graph=True)
                optimizer.step()
                save_o.write(str(epoch) + ',' + str(len(input_batch[num])) + ',' + str(length_batch[num]) + ',' + str(
                    loss.item()) + ',' + str(output.item())+ ',' + str(target_batch[num]) + ',' + str(output.item() - target_batch[num]) + '\n')
                target_list.append(label_input.item())
                predict_list.append(output.item())
                #f.write(str(len(input_batch[num])) + ',' + str(output.item()) + ',' + str(label_input) + '\n')
                if len(input_batch[num]) not in target_dict:
                    target_dict[len(input_batch[num])] = []
                    predict_dict[len(input_batch[num])] = []
                    target_dict[len(input_batch[num])].append(label_input.item())
                    predict_dict[len(input_batch[num])].append(output.item())
                elif len([input_batch[num]]) in target_dict:
                    target_dict[len(input_batch[num])].append(label_input.item())
                    predict_dict[len(input_batch[num])].append(output.item())
        elif batch_size != 1:
            while flag == True:
                # print(input_batch)
                # print(num)
                data_input = np.array(input_batch)
                label_input = np.array(target_batch)
                # print(data_input)
                data_input = Variable(torch.LongTensor(data_input).cuda()).cuda()
                # print(data_input.shape)
                label_input = Variable(torch.LongTensor(label_input).cuda()).cuda()
                label_input = label_input.float()
                output = model(data_input)
                optimizer.zero_grad()
                loss = criterion(output, label_input)
                # print(loss.item())
                loss.backward(retain_graph=True)
                optimizer.step()
                for num in range(len(input_batch)):
                    save_o.write(
                        str(epoch) + ',' + str(len(input_batch[num])) + ',' + str(length_batch[num]) + ',' + str(
                            loss.item()) + ',' + str(output[num].item())
                        + ',' + str(target_batch[num]) + ',' + str(output[num].item() - target_batch[num]) + '\n')
                    target_list.append(label_input[num].item())
                    predict_list.append(output[num].item())
                    # f.write(str(len(input_batch[num])) + ',' + str(output.item()) + ',' + str(label_input) + '\n')
                    if len(input_batch[num]) not in target_dict:
                        # print(len(input_batch[num]))
                        target_dict[len(input_batch[num])] = []
                        predict_dict[len(input_batch[num])] = []
                        target_dict[len(input_batch[num])].append(label_input[num].item())
                        predict_dict[len(input_batch[num])].append(output[num].item())
                    elif len([input_batch[num]]) in target_dict:
                        target_dict[len(input_batch[num])].append(label_input[num].item())
                        predict_dict[len(input_batch[num])].append(output.item())
                flag, input_batch, target_batch, length_batch = getData_oneLable(train_data, train_label, batch_size,
                                                                                 posType=posType, startPos=startPos)
                # print(input_batch)
            if len(input_batch) != 0:
                while len(input_batch) < batch_size:
                    input_batch.append(input_batch[len(input_batch) - 1])
                    target_batch.append(target_batch[len(target_batch) - 1])
                    length_batch.append(length_batch[len(length_batch) - 1])
                data_input = np.array(input_batch)
                label_input = np.array(target_batch)
                data_input = Variable(torch.LongTensor(data_input).cuda()).cuda()
                label_input = Variable(torch.LongTensor(label_input).cuda()).cuda()
                label_input = label_input.float()

                output = model(data_input)
                optimizer.zero_grad()
                loss = criterion(output, label_input)
                loss.backward(retain_graph=True)
                optimizer.step()
                for num in range(len(input_batch)):
                    # print(str(epoch))
                    # print(str(len(input_batch[num])))
                    # print(str(loss.item()))
                    # print(str(output[num].item()))
                    # print(str(target_batch[num]))
                    # print( str(output[num].item() - target_batch[num]))
                    save_o.write(
                        str(epoch) + ',' + str(len(input_batch[num])) + ',' + str(length_batch[num]) + ',' + str(
                            loss.item()) + ',' + str(output[num].item())
                        + ',' + str(target_batch[num]) + ',' + str(output[num].item() - target_batch[num]) + '\n')
                    target_list.append(label_input[num].item())
                    predict_list.append(output[num].item())
                    # f.write(str(len(input_batch[num])) + ',' + str(output.item()) + ',' + str(label_input) + '\n')
                    if len(input_batch[num]) not in target_dict:
                        # print(len(input_batch[num]))
                        target_dict[len(input_batch[num])] = []
                        predict_dict[len(input_batch[num])] = []
                        target_dict[len(input_batch[num])].append(label_input[num].item())
                        predict_dict[len(input_batch[num])].append(output[num].item())
                    elif len([input_batch[num]]) in target_dict:
                        target_dict[len(input_batch[num])].append(label_input[num].item())
                        predict_dict[len(input_batch[num])].append(output[num].item())
        MSE1, MSE2, MAE1, MAE2, RMSE1, RMSE2, targetMean, TargetVariance, TargetStandardDeviation = evaluate(target_list,predict_list)
        f.write(str(epoch) + ',all' + ',' + str(MSE1) + ',' + str(MSE2) + ',' + str(MAE1) + ',' +
                str(MAE2) + ',' + str(RMSE1) + ',' + str(RMSE2) + ',' + str(targetMean) + ',' +
                str(TargetVariance) + ',' + str(TargetStandardDeviation) + '\n')
        for key in target_dict:
            MSE1, MSE2, MAE1, MAE2, RMSE1, RMSE2, targetMean, TargetVariance, TargetStandardDeviation = evaluate(
                target_dict[key], predict_dict[key])
            f.write(str(epoch) + ',' + str(key) + ',' + str(MSE1) + ',' + str(MSE2) + ',' + str(MAE1) + ',' +
                    str(MAE2) + ',' + str(RMSE1) + ',' + str(RMSE2) + ',' + str(targetMean) + ',' +
                    str(TargetVariance) + ',' + str(TargetStandardDeviation) + '\n')
        if (epoch + 1) % 1000 == 0:
            #print(loss)
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', float(loss))
    save_o.close()
    f.close()
    return model

def main(preEmbd_type=False,baseModel='LSTM',attetionModel='None',posType='All',startPos=0,
        eventlog_name='',csv_address='', result_folder='',preEmbd_file = '',vocab_file='',
         train_test_split=0,embedding_dim=10, n_hidden=5, out_size=1, batch_size=1, epoch_sum = 200, dropout=0, n_layer=1):
    #结果存储

    result_file = result_folder + eventlog_name + '_model' + baseModel + '_att' + attetionModel + '_preTrain' + \
                  str(preEmbd_type) + '_emb' + str(embedding_dim) + '_hidden' + str(n_hidden) + '_split' + \
                  str(train_test_split) + '_epoch' + str(epoch_sum) + '_posType' + posType + '_startPos' + str(startPos)
    result_file_csv = result_file + '_iteration.csv'
    result_file_txt = result_file + '_iteration_result.csv'
    #读取预训练数据
    if preEmbd_type == False:
        num_classes, trace_list_orginal, trace_list_embd, embd_event_dict, event_embd_dict = noEmbedding(csv_address)

        vocab_size = num_classes
        embedding = nn.Embedding(vocab_size, embedding_dim)
    elif preEmbd_type == True:
        num_classes, trace_list_orginal, trace_list_embd, embd_event_dict, event_embd_dict, embedding = Embedding(embedding_dim,csv_address,preEmbd_file,vocab_file)
        embedding = embedding.cuda()
    vocab_size = num_classes
    preEmbd = embedding
    #读取训练数据
    train_data, train_label, test_data, test_label = generateData(trace_list_embd,train_test_split)

    #开始训练
    model = train(vocab_size, train_data, train_label, preEmbd, posType, startPos, result_folder, result_file,
          embedding_dim, n_hidden, out_size, epoch_sum, n_layer, dropout, batch_size,
          baseModel, attetionModel)

    squaredError = []
    absError = []

    f = open(result_file_txt,'w',encoding='utf-8')
    all_result_o = open(result_folder + 'all_iteration_result.txt','a',encoding='utf-8')
    rua = open(result_folder + 'rua1'+'single'+str(startPos)+'.csv','a',encoding='utf-8')
    # all_result_o.write(result_file + '\n')
    # f.write('Iteration,Pos,MSE1,MSE2,MAE1,MAE2,RMSE1,RMSE2,targetMean,TargetVariance,TargetStandardDeviation\n')
    # all_result_o.write('Iteration\tPos\tMSE1\tMSE2\tMAE1\tMAE2\tRMSE1\tRMSE2\ttargetMean\tTargetVariance\tTargetStandardDeviation\n')
    # rua.write('startPos\tcheckPos\tMSE1\tMSE2\tMAE1\tMAE2\tRMSE1\tRMSE2\ttargetMean\tTargetVariance\tTargetStandardDeviation\n')
    # target, prediction, target_dict, predict_dict = \
    #     test(test_data, test_label, model, result_file, posType, startPos, batch_size)
    error = []
    # for i in range(len(target)):
    #     error.append(target[i] - prediction[i])
    # MSE1, MSE2, MAE1, MAE2, RMSE1, RMSE2, targetMean, TargetVariance, TargetStandardDeviation=evaluate(target,prediction)
    # all_result_o.write('1\tall\t' + '\t' + str(MSE1) + '\t' + str(MSE2) + '\t' + str(MAE1) + '\t' +
    #         str(MAE2) + '\t' + str(RMSE1) + '\t' + str(RMSE2) + '\t' + str(targetMean) + '\t' +
    #         str(TargetVariance) + '\t' + str(TargetStandardDeviation)+'\n')
    # f.write('1,all,' + ',' + str(MSE1) + ',' + str(MSE2) + ',' + str(MAE1) + ',' +
    #         str(MAE2) + ',' + str(RMSE1) + ',' + str(RMSE2) + ',' + str(targetMean) + ',' +
    #         str(TargetVariance) + ',' + str(TargetStandardDeviation)+'\n')
    # for key in target_dict:
    #     MSE1, MSE2, MAE1, MAE2, RMSE1, RMSE2, targetMean, TargetVariance, TargetStandardDeviation = evaluate(target_dict[key],
    #                                                                                                          predict_dict[key])
    #     f.write(str(key) + ',' + str(MSE1) + ',' + str(MSE2) + ',' + str(MAE1) + ',' +
    #             str(MAE2) + ',' + str(RMSE1) + ',' + str(RMSE2) + ',' + str(targetMean) + ',' +
    #             str(TargetVariance) + ',' + str(TargetStandardDeviation) + '\n')
    #     all_result_o.write(str(key) + '\t' + str(MSE1) + '\t' + str(MSE2) + '\t' + str(MAE1) + '\t' +
    #             str(MAE2) + '\t' + str(RMSE1) + '\t' + str(RMSE2) + '\t' + str(targetMean) + '\t' +
    #             str(TargetVariance) + '\t' + str(TargetStandardDeviation) + '\n')

    for pos_start_temp in range(1, 154):
        target, prediction, target_dict, predict_dict = \
            test(test_data, test_label, model, result_file, 'Single', pos_start_temp, batch_size)
        for i in range(len(target)):
            error.append(target[i] - prediction[i])
        #print(target)
        #print(target_dict)
        MSE1, MSE2, MAE1, MAE2, RMSE1, RMSE2, targetMean, TargetVariance, TargetStandardDeviation = evaluate(target,
                                                                                                             prediction)
        # rua.write(str(pos_start_temp)+'\tall\t' + '\t' + str(MSE1) + '\t' + str(MSE2) + '\t' + str(MAE1) + '\t' +
        #                    str(MAE2) + '\t' + str(RMSE1) + '\t' + str(RMSE2) + '\t' + str(targetMean) + '\t' +
        #                    str(TargetVariance) + '\t' + str(TargetStandardDeviation) + '\n')

        for key in target_dict:
            MSE1, MSE2, MAE1, MAE2, RMSE1, RMSE2, targetMean, TargetVariance, TargetStandardDeviation = evaluate(
                target_dict[key],
                predict_dict[key])
            rua.write(
                str(startPos)+'\t' + str(pos_start_temp) +  '\t' + str(MSE1) + '\t' + str(MSE2) + '\t' + str(MAE1) + '\t' +
                str(MAE2) + '\t' + str(RMSE1) + '\t' + str(RMSE2) + '\t' + str(targetMean) + '\t' +
                str(TargetVariance) + '\t' + str(TargetStandardDeviation) + '\n')

    f.close()
    all_result_o.close()
    rua.close()
# attetionModel None Dot  General  DotNoDecoder  GeneralNoDecoder
#startPos从1开始
main(preEmbd_type = True, baseModel = 'BiLSTM', attetionModel = 'None', posType = 'All', startPos = 3,
     eventlog_name = 'bpi_12_w_extend', csv_address = './data/bpi_12_w_extend.csv', result_folder = './BiLSTM2/',
     preEmbd_file = './vector/bpi_12_w_extend_vectors_2CBoW_noTime_noEnd_Vector_vLoss_v1.txt',
     vocab_file = './vector/bpi_12_w_extend_2CBoW_noTime_noEnd_vocabulary.txt',
    train_test_split = 0.3, embedding_dim = 2, n_hidden = 5, out_size = 1, batch_size = 512,
     epoch_sum = 150, dropout = 0, n_layer = 1)
# main(embedding_dim=10,preEmbd_type=True,
#      preEmbd_file = './vector/bpi_12_w_extend_vectors_10CBoW_noTime_noEnd_Vector_vLoss_v1.txt',
#      vocab_file='./vector/helpd5k_extend_10CBoW_noTime_noEnd_vocabulary.txt')