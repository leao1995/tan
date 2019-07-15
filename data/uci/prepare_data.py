import os
import numpy as np
import pandas as pd
import pickle

path = './original_data'
save_path = '.'


def split(data, name):
    data = data.copy()
    np.random.shuffle(data)
    N = data.shape[0]
    train = data[:int(N * 0.6)]
    valid = data[int(N * 0.6):int(N * 0.8)]
    test = data[int(N * 0.8):]

    dataset = {
        'train': train,
        'valid': valid,
        'test': test
    }
    fname = os.path.join(save_path, '{}.p'.format(name))
    pickle.dump(dataset, open(fname, 'wb'))


# white wine
data = pd.read_csv(os.path.join(path, 'winequality-white.csv'), sep=';')
data = np.array(data)
split(data, 'white')
