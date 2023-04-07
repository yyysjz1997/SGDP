import pandas as pd
import argparse
import numpy as np
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import csv
import pickle
import operator
import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--type')
opt = parser.parse_args()

warnings.filterwarnings('ignore')
# idx_upper_bound = 1.5e5
idx_upper_bound = 1e6


usecols_by_type = {'MSRC': [0, 1, 2, 3],
                   'HW': [2, 7, 8, 10],
                   'VDI': [0, 2, 4, 5],
                   'VDI512': [0, 2, 4, 5]}

sep_by_type = {'MSRC': ' ',
               'HW': '\\s*\\|\\s*',
               'VDI': ',',
               'VDI512': ','}

colnames_by_type = {'MSRC': ['TIME_STAMP', 'RW', 'LBA', 'iosize'],
                    'HW': ['RW', 'LBA', 'iosize', 'TIME_STAMP'],
                    'VDI': ['TIME_STAMP', 'RW', 'LBA', 'iosize'],
                    'VDI512': ['TIME_STAMP', 'RW', 'LBA', 'iosize']}

IO_type_by_type = {'MSRC': 'RS',
                   'HW': 'READ',
                   'VDI': 'R',
                   'VDI512': 'R'}

TS_order_by_type = {'MSRC': 1e5,
                    'HW': 1e-4,
                    'VDI': 1e5,
                     'VDI512': 1e5}

lba_size_by_type = {'MSRC': 512,
                    'HW': 512,
                    'VDI': 1,
                    'VDI512': 2}

def trace_to_8k(trace_path, trace_type='MSRC'):
    usecols = usecols_by_type[trace_type]
    colnames = colnames_by_type[trace_type]
    sep = sep_by_type[trace_type]
    IO_type = IO_type_by_type[trace_type]
    TS_order = TS_order_by_type[trace_type]
    lba_size = lba_size_by_type[trace_type]

    dataframe = pd.read_table(trace_path, sep=sep, usecols=usecols)

    dataframe.columns = colnames

    trace = dataframe[dataframe['RW'] == IO_type][:-1].reset_index(drop=True)
    del trace['RW']

    trace['LBA'] = trace['LBA'].astype('float').astype('int')*lba_size
    trace['iosize'] = trace['iosize'].astype('float').astype('int')*lba_size

    trace['TIME_STAMP'] = trace['TIME_STAMP'].astype('float64')
    trace['TIME_STAMP'] = trace['TIME_STAMP'] * TS_order
    trace['TIME_STAMP'] = trace['TIME_STAMP'] - min(trace['TIME_STAMP'])

    trace.dropna(inplace=True)
    
    trace = trace.sort_values(by='TIME_STAMP', ascending=True).reset_index(drop=True)
    print(len(trace))
    trace_iosize_reduce = []
    for idx, row in tqdm(trace.iterrows()):
        if idx>idx_upper_bound:
            break
        iosize = int(row['iosize'])
        lba = int(row['LBA'])
        time_adder = 1
        for lba_div in range(lba//8192, (lba+iosize)//8192):
            trace_iosize_reduce.append([lba_div, row['TIME_STAMP']+time_adder*0.01])
            time_adder += 1

    trace_8k_lba = pd.DataFrame(trace_iosize_reduce, columns=['LBA', 'TIME_STAMP'])

    trace_8k_lba['LBA_64M_id'] = trace_8k_lba['LBA']//8192
    trace_8k_lba['LBA_8K_offset'] = trace_8k_lba['LBA'] % 8192
    trace_8k_lba['LBA_8K_offset'] = trace_8k_lba['LBA_8K_offset'].astype(int)
    save_file = '../data/8k_lba_traces/'+str(trace_path.split('/')[-1].split('.')[0])+'.csv'
    trace_8k_lba.to_csv(save_file, index=None)
    return 0 

def seperate_8k_by_64M(trace_path):
    trace_8k_lba = pd.read_csv(trace_path)

    dir_64M =  os.path.join('data/64M_sep/',str(trace_path.split('/')[-1].split('.')[0]))
    if not os.path.exists(dir_64M):        
        os.makedirs(dir_64M)  

    for idx,offset in tqdm(list(trace_8k_lba.groupby('LBA_64M_id'))):
        offset.to_csv(os.path.join(dir_64M,str(idx)+'.csv'), index=None)
    return 0 

def session_generate(trace_path):
    session_list=[]
    id_64m_list=[]
    for idx in tqdm(os.listdir(trace_path)):
        # if trace_path[-5] is not 'm':
        #     return 0
        # print(idx)
        df_64m = pd.read_csv(os.path.join(trace_path,idx))

        ts_temp = df_64m['TIME_STAMP'][0] 
        session = []
        ts = []
        for _, row in df_64m.iterrows():
            if row['TIME_STAMP']<ts_temp+1e5:
                session.append(row['LBA_8K_offset'].astype(int))
                ts.append(row['TIME_STAMP'])
            else:
                if len(session)>=3:
                    for i in range(2,len(session)):
                        session_list.append([session[max(i-32,0):i],session[i],min(len(session)-i,32),session[i:min(len(session),i+32)],row['TIME_STAMP']])
                        id_64m_list.append(idx[:-4])
                session=[]
                ts = []
            ts_temp=row['TIME_STAMP']
        # break
    df = pd.DataFrame(session_list)   
    df = df.sort_values(by=4,ascending=True).reset_index()
    df = df.iloc[:,[1,2,3,4]]
    df_list = df.values.tolist()
    df_list_reshape = [[item[i] for item in df_list] for i in range(4) ]
    print('generate finish')
    pickle.dump(df_list_reshape,open(os.path.join('data/sessions/',trace_path.split('/')[-1]), 'wb'))
    id_64m_list =pd.DataFrame(id_64m_list)
    id_64m_list.to_csv(os.path.join('data/sessions/','64m_id'+trace_path.split('/')[-1]),index=None,header=None)
    return 0 

def delta_session_generate(trace_path):
    session_list=[]
    id_64m_list=[]
    for idx in tqdm(os.listdir(trace_path)):
        # if trace_path[-5] is not 'm':
        #     return 0
        # print(idx)
        df_64m = pd.read_csv(os.path.join(trace_path,idx))

        ts_temp = df_64m['TIME_STAMP'][0] 
        session = []
        ts = []
        for _, row in df_64m.iterrows():
            if row['TIME_STAMP']<ts_temp+1e5:
                session.append(row['LBA_8K_offset'].astype(int))
                ts.append(row['TIME_STAMP'])
            else:
                if len(session)>=3:
                    session = np.diff(np.array(session))
                    session = [i+8192 for i in session]
                    for i in range(1,len(session)):
                        session_list.append([session[max(i-32,0):i],session[i],min(len(session)-i,32),session[i:min(len(session),i+32)],row['TIME_STAMP']])
                        id_64m_list.append(idx[:-4])
                session=[]
                ts = []
            ts_temp=row['TIME_STAMP']
        # break
    df = pd.DataFrame(session_list)   
    df = df.sort_values(by=4,ascending=True).reset_index()
    df = df.iloc[:,[1,2,3,4]]
    df_list = df.values.tolist()
    df_list_reshape = [[item[i] for item in df_list] for i in range(4) ]
    print('generate finish')
    pickle.dump(df_list_reshape,open(os.path.join('data/delta_sessions/',trace_path.split('/')[-1]), 'wb'))
    id_64m_list =pd.DataFrame(id_64m_list)
    id_64m_list.to_csv(os.path.join('data/delta_sessions/','64m_id'+trace_path.split('/')[-1]),index=None,header=None)
    return 0 

def full_delta_session_generate(trace_path):
    session_list=[]
    id_64m_list=[]
    for idx in tqdm(os.listdir(trace_path)):
        # if trace_path[-5] is not 'm':
        #     return 0
        # print(idx)
        df_64m = pd.read_csv(os.path.join(trace_path,idx))

        ts_temp = df_64m['TIME_STAMP'][0] 
        session = []
        ts = []
        for _, row in df_64m.iterrows():
            session.append(row['LBA_8K_offset'].astype(int))
            ts.append(row['TIME_STAMP'])
        if len(session)>=3:
            session = np.diff(np.array(session))
            session = [i+8192 for i in session]
            for i in range(1,len(session)):
                session_list.append([session[max(i-32,0):i],session[i],min(len(session)-i,32),session[i:min(len(session),i+32)],row['TIME_STAMP']])
                id_64m_list.append(idx[:-4])
        session=[]
        ts = []
        ts_temp=row['TIME_STAMP']
        # break
    df = pd.DataFrame(session_list)   
    df = df.sort_values(by=4,ascending=True).reset_index()
    df = df.iloc[:,[1,2,3,4]]
    df_list = df.values.tolist()
    df_list_reshape = [[item[i] for item in df_list] for i in range(4) ]
    print('generate finish')
    pickle.dump(df_list_reshape,open(os.path.join('data/full_delta_sessions/',trace_path.split('/')[-1]), 'wb'))
    id_64m_list =pd.DataFrame(id_64m_list)
    id_64m_list.to_csv(os.path.join('data/full_delta_sessions/','64m_id'+trace_path.split('/')[-1]),index=None,header=None)
    return 0 
if __name__ == '__main__':

    if opt.type =='full' or opt.type =='8k':
        dataset_dir = '../data/datasets_by_type/'
        for type_dir in os.listdir(dataset_dir):
                print(type_dir)
                if type_dir =='VDI512':
                    dataset_type_dir = os.path.join(dataset_dir,type_dir)
                    for trace_name in os.listdir(dataset_type_dir):
                        trace_path = os.path.join(dataset_type_dir,trace_name)
                        print('processing original trace to 8klba trace, trace name:',trace_path)
                        # trace_type = type_dir.split('/')[-2]
                        trace_to_8k(trace_path=trace_path,trace_type=type_dir)

    if opt.type =='full' or opt.type =='sep':
        dataset_dir = 'data/8k_lba_traces/'
        for trace_name in os.listdir(dataset_dir):
            trace_path = os.path.join(dataset_dir,trace_name)
            print('seperating 8klba trace with 64M ids, trace name:',trace_path)
            seperate_8k_by_64M(trace_path=trace_path)

    if opt.type =='full' or opt.type =='sess':
        dataset_dir = 'data/64M_sep/'
        for trace_name in os.listdir(dataset_dir):
            trace_path = os.path.join(dataset_dir,trace_name)
            print('generating session data , trace name:',trace_path)
            session_generate(trace_path=trace_path)
