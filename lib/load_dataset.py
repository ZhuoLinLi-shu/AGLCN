import os
import numpy as np
import pandas as pd

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD4':
        # todo 这里很神奇的就是，按道理来讲应该加上..,但是加上就错了，去掉就对，我也是无语了
        data_path = os.path.join('data/PeMSD4/pems04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('data/PeMSD8/pems08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data


def load_wind_dataset(dataset):
    # output B, N, D
    # 我的方法就是  'data/wspd_120nodes.csv'
    # baseline方法的路径就是 '../data/wspd_120nodes.csv'
    data_path = os.path.join('data/wspd_40nodes.csv')  # wspd_120nodes.csv
    data = pd.read_csv(data_path)
    data = data.values  # 讲csv文件转换为numpy

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data