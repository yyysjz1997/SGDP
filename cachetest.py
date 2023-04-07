#!/usr/bin/env python36
# -*- coding: utf-8 -*-
import argparse
import pickle
import time
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from cache import CacheTest
from collections import Counter
parser = argparse.ArgumentParser()
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
import copy
opt = parser.parse_args()
print(opt)

def roll_cache_test(test_data_trace,pre_list_list,dicts,rollstep=10):
    
    hit_rate=[]
    prehit_rate=[]
    stats=[]
    caches={}
    bo_map, bo_map_div, operation_id_map, operation_id_map_div=dicts
    maxsize = [5] + [i*10 for i in range(1,10)] + [i*100 for i in range(1,11)]
    for i in range(len(maxsize)):
        caches["LRU"+str(maxsize[i])] = CacheTest(maxsize[i])
    for single_test_data,pre_list in tqdm(zip(test_data_trace[1:],pre_list_list)):
        for name, cache in caches.items():
            cache.push_normal(single_test_data) 
            single_test_data_temp = single_test_data+0
            for pred_temp in pre_list[:rollstep]:
                single_test_data_temp = single_test_data_temp- operation_id_map_div[bo_map_div[pred_temp-1]]
                cache.push_prefetch(single_test_data_temp)

        # print(cache.get_hit_rate(),end='\r')
        # break

    for name, cache in caches.items():
        print(format(cache.get_hit_rate(), '.4f'),format(cache.get_prehit_rate(), '.4f'),'\t',name)     
        hit_rate.append(cache.get_hit_rate())   
        prehit_rate.append(cache.get_prehit_rate())      
        stats.append(cache.get_stats())   
        print(cache.get_stats())                                                                                                                      
    return hit_rate,prehit_rate,stats

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
        bo_map, _, operation_id_map, _ = dicts

        keys = bo_map.keys()
        inputs = []
        for i in tqdm(range(len(test_trace)-1)):
            diff = int(test_trace[i]-test_trace[i+1])
            if operation_id_map[diff] in keys:
                input_single= bo_map[operation_id_map[diff]]+1###
            else:
                input_single= bo_map[999999]+1###

            inputs.append(input_single)
        
        n_node = top_num + 3

        return  dicts, n_node,test_trace

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

if __name__ == '__main__':
    #python cachetest.py
    dataset_col = ['ftds_0802_1021_trace','ftds_0805_17_trace','ftds_0804_1023_trace','prxy_0','mds_0','hm_1','proj_0','src1_2','web_1']

    for rollstep in range(10,11):
        for dataset in dataset_col:
            print(rollstep)
            print('\ntrace name:\t',dataset,'\n')

            dicts, n_node,test_data_trace = dataset2input(dataset=dataset)
            pre_list_list = np.load('predlist/'+dataset+'30_list.npy',allow_pickle=True)
            print(len(pre_list_list),len(test_data_trace))
            hit_rate,prehit_rate,stats = roll_cache_test(test_data_trace=test_data_trace,pre_list_list=pre_list_list,rollstep=rollstep,dicts=dicts)
            # print(predlist)
            # print(hit_rate)
            np.savetxt('roll_hit_results/'+dataset+'_'+str(rollstep)+'_hit_rate.txt',hit_rate,fmt='%.4f')
            np.savetxt('roll_hit_results/'+dataset+'_'+str(rollstep)+'_pre_hit_rate.txt',prehit_rate,fmt='%.4f')
            np.savetxt('roll_hit_results/'+dataset+'_'+str(rollstep)+'_stats.txt',stats,fmt='%d')
            print('-------------------------------------------------------')
        

