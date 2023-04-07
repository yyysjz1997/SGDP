#!/usr/bin/env python36
# -*- coding: utf-8 -*-
import argparse
import pickle
import time
import os
import torch
import pandas as pd
import numpy as np


from utils import Data, split_validation
from model import *
from tqdm import tqdm
from collections import Counter
from cache import *


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int,
                    default=128, help='input batch size')
parser.add_argument('--hiddenSize', type=int,
                    default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=5,
                    help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1,
                    help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3,
                    help='the number of steps after which the learning rate decay')
# [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--step', type=int, default=1,
                    help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=3,
                    help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true',
                    help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1,
                    help='split the portion of training set as validation set')
parser.add_argument('--draw_graph', action='store_true')
parser.add_argument('--see_ori_dataset', action='store_true')
parser.add_argument('--topn', type=int, default=20, help='top n')
parser.add_argument('--dataset_percent', type=float, default=1.0,
                    help='datasets percent for training and testing')
parser.add_argument('--window', type=int, default=32, help='window')
parser.add_argument('--topnum', type=int, default=1000, help='top n')
opt = parser.parse_args()
print(opt)


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


def trace2input(dicts, trace, window_size=32):
    bo_map, _, operation_id_map, _ = dicts
    # print(len(trace))

    keys = bo_map.keys()
    inputs = []
    targets = []
    for i in tqdm(range(len(trace)-window_size-1)):

        input_single = []
        for j in range(i, i+window_size+1):
            diff = int(trace[j]-trace[j+1])
            if operation_id_map[diff] in keys:
                input_single.append(bo_map[operation_id_map[diff]]+1)###
            else:
                input_single.append(bo_map[999999]+1)###
        inputs.append(input_single[:-1])
        targets.append(input_single[-1])
    return inputs, targets


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

        train_trace = df[:int(len(df)*-opt.valid_portion)
                         ]['ByteOffset'].tolist()
        test_trace = df[int(len(df)*-opt.valid_portion) +
                        1:]['ByteOffset'].tolist()

        dicts = dict_generate(df, top_num=top_num)

        train_data = tuple(trace2input(
            dicts, train_trace, window_size=window_size))
        test_data = tuple(trace2input(
            dicts, test_trace, window_size=window_size))

        train_data = Data(train_data, shuffle=True)
        test_data = Data(test_data, shuffle=False)

        train_silces = train_data.generate_batch(opt.batchSize)
        train_data_list = []

        for i in tqdm(train_silces):
            alias_inputs, A, items, mask, targets = train_data.get_slice(i)
            train_data_list.append((alias_inputs, A, items, mask, targets))

        test_silces = test_data.generate_batch(opt.batchSize)
        test_data_list = []

        for i in tqdm(test_silces):
            alias_inputs, A, items, mask, targets = test_data.get_slice(i)
            test_data_list.append((alias_inputs, A, items, mask, targets))

        n_node = top_num + 3

        return train_data_list, train_silces, test_data_list, test_silces, dicts, n_node, train_trace, test_trace


def single_cache_test(test_trace, all_pred, save_name, dicts):
    bo_map, bo_map_div, operation_id_map, operation_id_map_div = dicts
    hit_rate = []
    prehit_rate = []
    stats = []
    caches = {}
    maxsize = [5] + \
        [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 11)]

    for i in range(len(maxsize)):
        caches["LRU"+str(maxsize[i])] = CacheTest(maxsize[i])

    for i in tqdm(range(0, len(test_trace))):
        for name, cache in caches.items():
            cache.push_normal(test_trace[i])
            if all_pred[i][0] > 0:
                cache.push_prefetch(
                    test_trace[i] - operation_id_map_div[bo_map_div[all_pred[i][0]-1]])###

    for name, cache in caches.items():
        print(format(cache.get_hit_rate(), '.4f'), format(
            cache.get_prehit_rate(), '.4f'), '\t', name)
        hit_rate.append(cache.get_hit_rate())
        prehit_rate.append(cache.get_prehit_rate())
        stats.append(cache.get_stats())

    np.savetxt('hit_results/'+save_name+'_hit_rate.txt', hit_rate, fmt='%.4f')
    np.savetxt('hit_results/'+save_name +
               '_pre_hit_rate.txt', prehit_rate, fmt='%.4f')
    np.savetxt('hit_results/'+save_name+'_stats.txt', stats, fmt='%d')
    return 0

def score_compute(all_preds,all_targets,save_name):
    
    pre_list = []
    mmr_list = []
    for i in range(1,len(all_preds[0])):
        pre_list.append(np.mean([np.where(t in p[:i],1,0) for t,p in zip(all_targets,all_preds)]))
        mmr_list.append(np.mean([1/(np.where(p[:i]==t)[0]+1) if t in p[:i] else 0 for t,p in zip(all_targets,all_preds)]))
    np.savetxt('hit_results/'+save_name+'_pre_list.txt', pre_list, fmt='%.4f')  
    np.savetxt('hit_results/'+save_name +'_mmr_list.txt', mmr_list, fmt='%.4f')
    return pre_list,mmr_list

    
def main():
    dataset_col = ['ftds_0805_1016_trace']#'ftds_0802_1021_trace','ftds_0805_17_trace',
    deviceID = 1
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(deviceID)
    device = torch.device('cuda:'+str(deviceID))

    for dataset in dataset_col:
        train_data_list, train_silces, test_data_list, test_silces, dicts, n_node, train_trace, test_trace = dataset2input(
            dataset=dataset, window_size=opt.window, top_num=opt.topnum)

        model = trans_to_cuda(SessionGraph(opt, n_node))
        model_path = 'checkpoint/'+'model_' + \
            str(dataset)+'_'+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        folder = os.path.exists(model_path)
        if not folder:
            os.makedirs(model_path)
        print('\nModel_Path:', model_path)
        print('\n-------------------------------------------------------\n')
        for epoch in range(opt.epoch):
            print('\n-------------------------------------------------------\n')
            print('epoch: ', epoch)
            print('start training: ')
            all_pred,all_targets = train_test_pred(
                model, train_data_list, train_silces, test_data_list, test_silces)

            save_name = dataset+'_'+str(epoch)+'_epoch'
            print('start cache test: ')
            _ = single_cache_test(
                test_trace=test_trace[opt.window:-1], all_pred=all_pred, save_name=save_name, dicts=dicts)
            pre,mmr = score_compute(all_preds=all_pred,all_targets=all_targets,save_name = save_name)
            print('pre:',pre)
            print('mmr:',mmr)
            torch.save(model, os.path.join(model_path, str(epoch)+'.pt'))
        torch.cuda.empty_cache()
        print('-------------------------------------------------------')


if __name__ == '__main__':
    main()
