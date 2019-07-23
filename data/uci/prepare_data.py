import os
import sys
sys.path.insert(0, '/nas/longleaf/home/yangli95/Workspace/TAN/tan')
import numpy as np
import pandas as pd
import pickle

from tan.data import bsds, hepmass, miniboone, power, gas, miniboone_noprune


path = '../../dataset/uci/original_data'
save_path = '.'


def split(data, name):
    data = data.copy()
    np.random.shuffle(data)
    N = data.shape[0]
    train = data[:int(N * 0.85)]
    valid = data[int(N * 0.85):N]
    test = data

    dataset = {
        'train': train,
        'valid': valid,
        'test': test
    }
    fname = os.path.join(save_path, '{}.p'.format(name))
    pickle.dump(dataset, open(fname, 'wb'))


# white wine
# data = pd.read_csv(os.path.join(path, 'winequality-white.csv'), sep=';')
# data = np.array(data)
# split(data, 'white_vaeac')

# uci
bsds.download_and_make_data(path)
# hepmass.download_and_make_data(path)
miniboone.download_and_make_data(path)
miniboone_noprune.download_and_make_data(path)
power.download_and_make_data(path)
gas.download_and_make_data(path)
