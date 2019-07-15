import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

path = './original_data'
save_path = '.'

datasets = read_data_sets(path)
train = (datasets.train.images * 255.).astype('uint8')
valid = (datasets.validation.images * 255.).astype('uint8')
test = (datasets.test.images * 255.).astype('uint8')

dataset = {
    'train': train,
    'valid': valid,
    'test': test
}
fname = os.path.join(save_path, 'mnist.p')
pickle.dump(dataset, open(fname, 'wb'))
