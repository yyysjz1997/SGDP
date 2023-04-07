#!/usr/bin/env python36
# -*- coding: utf-8 -*-
import argparse
import pickle
import time
import os
import numpy as np
from torch import nn
import torch
import pandas as pd
from tqdm import tqdm
import cProfile
parser = argparse.ArgumentParser()
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')

opt = parser.parse_args()
print(opt)
import numpy as np
import copy 
from collections import Counter
class TestData():
    def __init__(self):
        self.Data=[]
        self.Data_backup=[]
    def data_generate(self,method='ours'):
        if method == 'ours':
            # print(len(self.Data),'datagene')
            session = self.Data
            # print(self.Data['last_64M_key'],session)
            if len(session)<=0:
                return None
            else:
                # print(len(session))
                mask = [[1] *len(session)]
                items, n_node, A, alias_inputs = [], [], [], []
                n_node = [len(np.unique(session))]
                node = np.unique(session)
                items.append(node.tolist())
                u_A = np.zeros((n_node[0], n_node[0]))
                for i in np.arange(len(session) - 1):
                    u = np.where(node == session[i])[0][0]
                    for j in np.arange(1, len(node)-i-1):
                        v = np.where(node == session[i + j])[0][0]
                        u_A[u][v] = 1
                for i in np.arange(len(session) - 1):
                    u = np.where(node == session[i])[0][0]
                    v = np.where(node == session[i + 1])[0][0]
                    u_A[u][v] += 1  
                u_sum_in = np.sum(u_A, 0)
                u_sum_in[np.where(u_sum_in == 0)] = 1
                u_A_in = np.divide(u_A, u_sum_in)
                u_sum_out = np.sum(u_A, 1)
                u_sum_out[np.where(u_sum_out == 0)] = 1
                u_A_out = np.divide(u_A.transpose(), u_sum_out)
                u_A = np.concatenate([u_A_in, u_A_out]).transpose()
                A.append(u_A)
                alias_inputs.append([np.where(node == i)[0][0] for i in session])
                return alias_inputs, A, items, mask
        # return generated_data

    def data_append(self,new_data,isreal=True,ignore_ts = False):
        if isreal:
            self.Data_backup.append(new_data)
            self.Data_backup=self.Data_backup[-32:]
        else:
            self.Data.append(new_data)
            self.Data=self.Data[-32:]
        return 0

    def create_temp(self):
        self.Data = []
        self.Data = self.Data_backup.copy()
        # print(len(self.Data),'datatemp')
        return 0

def predict(model,data):
    alias_inputs, A, items, mask= data
    alias_inputs = torch.Tensor(alias_inputs).long().cuda()
    items = torch.Tensor(items).long().cuda()
    A = torch.Tensor(A).float().cuda()
    mask = torch.Tensor(mask).long().cuda()
    hidden = model.forward(items, A)
    seq_hidden = torch.stack([hidden[i][alias_inputs[i]] for i in torch.arange(len(alias_inputs)).long()])
    scores = model.compute_scores(seq_hidden, mask).cpu()
    prediction = np.argmax(scores.detach().numpy(),axis=1)
    return prediction

def one_step_prediction(model,test_data,method='ours'):
    processed_data = test_data.data_generate(method=method)
    # print(processed_data)
    if processed_data:
        if method=='ours':
            prediction = predict(model,processed_data)
            # print(prediction)
        return prediction
        
    else:
        return None

def roll_test_with_trace(model,test_data_trace,rollstep=1,method='ours'):
    model.eval()
    test_data = TestData()
    pred_list_list=[]
    for single_test_data in tqdm(test_data_trace):
        # print(single_test_data)
        pre_list=[]
        # print(single_test_data)
        test_data.data_append(single_test_data,isreal=True,ignore_ts=True)
        test_data.create_temp()
        for _ in range(rollstep):
            prediction = one_step_prediction(model,test_data,method=method)
            if prediction:
                prediction = prediction.item()
                test_data.data_append(prediction,isreal=False,ignore_ts=True)
                pre_list.append(prediction)
                # print(pre_list)
                # print(LBA_8K_offset,LBA_8K_offset_temp)
            else:
                break
        # print(pre_list)
        pred_list_list.append(pre_list)                                                                                                           
    return pred_list_list

def last_file_name_selecter(name_list):
    print(name_list)
    name_list_desort = np.sort(name_list)
    return name_list_desort[-1]



def dict_generate(train_trace, top_num=1000):
    train_trace['ByteOffset_Delta'] = train_trace['ByteOffset'] - \
        train_trace['ByteOffset'].shift(-1)
    train_trace['ByteOffset_Delta'] = train_trace['ByteOffset_Delta'].fillna(0)

    a = train_trace['ByteOffset_Delta'].astype(int).unique().tolist()

    operation_id_map = {}
    for i, id in enumerate(a):
        operation_id_map[id] = i
    train_trace['ByteOffset_Delta_class'] = train_trace['ByteOffset_Delta'].map(
        lambda x: operation_id_map[x])

    x = Counter(train_trace['ByteOffset_Delta_class'])
    vals = {}
    vals = x.most_common(top_num)
    bo_list = []

    for x in vals:
        bo_list.append(x[0])

    count = 0
    label_list = []
    while (count < len(train_trace)):
        x = train_trace['ByteOffset_Delta_class'].iloc[count]
        if x in bo_list:
            label_list.append(x)
        else:
            label_list.append(999999)  # no Prefetch class
        count = count + 1

    train_trace['ByteOffset_Delta_class'] = label_list
    a = train_trace['ByteOffset_Delta_class'].unique().tolist()
    bo_map = {}
    for i, id in enumerate(a):
        bo_map[id] = i
    operation_id_map_div = {v: k for k, v in operation_id_map.items()}
    operation_id_map_div[999999] = 0
    bo_map_div = {v: k for k, v in bo_map.items()}

    return bo_map, bo_map_div, operation_id_map, operation_id_map_div


def dataset2input(dataset, window_size=32, method='top', top_num=1000):
    if method == 'top':
        print('\ntrain dataset, trace name:\t', dataset, '\n')
        lba_trace = '../data/8k_lba_traces/' + dataset + '.csv'
        print('\n', lba_trace, '\n')
        names = ['ByteOffset', 'TimeStamp']
        df = pd.read_csv(lba_trace, engine='python', skiprows=1, header=None, na_values=[
                         '-1'], usecols=[0, 1], names=names)
        df = df.sort_values(by=['TimeStamp'])
        df.reset_index(inplace=True, drop=True)
        test_trace = df[int(len(df)*-opt.valid_portion) +
                        1:]['ByteOffset'].tolist()
        dicts = dict_generate(df, top_num=top_num)
        bo_map, _, operation_id_map, _ = dict_generate(df, top_num=top_num)

        keys = bo_map.keys()
        inputs = []
        #len(test_trace)-1
        for i in tqdm(range(10032)):
            diff = int(test_trace[i]-test_trace[i+1])
            if operation_id_map[diff] in keys:
                input_single= bo_map[operation_id_map[diff]]+1###
            else:
                input_single= bo_map[999999]+1###

            inputs.append(input_single)
        
        n_node = top_num + 3

        return  dicts, n_node,inputs

if __name__ == '__main__':

    dataset_col =  ['ftds_0802_1021_trace','ftds_0805_17_trace','ftds_0804_1023_trace','prxy_0','mds_0','hm_1','proj_0','src1_2']
    deviceID = 1
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(deviceID)
    device = torch.device('cuda:'+str(deviceID))

    for dataset in dataset_col:
        dicts, n_node,inputs = dataset2input(
            dataset=dataset)


        model_list = os.listdir('checkpoint/')

        last_model_path = last_file_name_selecter([model_path for model_path in model_list if model_path.startswith('model_'+dataset)])
        
        last_model_path = os.path.join('checkpoint/',last_model_path)

        model_checkpoint_list = os.listdir(last_model_path)
        last_model_checkpoint = last_file_name_selecter([model_path for model_path in model_checkpoint_list if model_path.endswith('.pt')])

        last_model_checkpoint = os.path.join(last_model_path,last_model_checkpoint)

        print(last_model_checkpoint)
        model =torch.load(last_model_checkpoint).cuda()

        predlist= roll_test_with_trace(model=model,test_data_trace=inputs)
        # cProfile.run("roll_test_with_trace(model=model,test_data_trace=inputs)")
        np.save('predlist/'+dataset+'30_list',predlist)
        print('-------------------------------------------------------')
        

