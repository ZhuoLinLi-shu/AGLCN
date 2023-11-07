import os
import numpy as np
import pandas as pd

def load_wind_dataset(dataset):
    data_path = os.path.join('data/wspd_40nodes.csv')  
    data = pd.read_csv(data_path)
    data = data.values

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data