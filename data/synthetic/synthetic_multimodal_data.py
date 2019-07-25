import numpy as np
import pickle as p
import os

TRAIN_RAT = 0.5
VALID_RAT = 0.1
TEST_RAT = 0.4

def single_modal_data(sample_size, d=4):
    # Generate a random covariate matrix
    v = np.random.rand(d, d)
    sigma = np.matmul(v, np.transpose(v))
    mu = np.random.rand(d)
    data = np.random.multivariate_normal(mu, sigma, sample_size)

    return data, (mu, sigma)

def multi_modal_data(sample_size, d=4, num_modes=4):
    # get the weights of each gaussian
    weights = np.random.rand(num_modes)
    weights = weights/weights.sum()

    data = np.zeros((sample_size, d))
    dist_data = []

    for i in range(num_modes):
        gauss_sample, dist = single_modal_data(sample_size, d=d)
        # weight gaussian sample
        data += gauss_sample * weights[i]
        dist_data.append((weights[i], dist))
    
    return data, dist_data

def create_dataset(path, file_name, sample_size, d, num_modes, seed=2):
    np.random.seed(seed)
    # path = os.path.expanduser("~") + path
    if not os.path.isdir(path):
        os.mkdir(path)

    data, dist_data = multi_modal_data(sample_size, d=d, num_modes=num_modes)
    dataset = {
        "training": data[:int(sample_size*TRAIN_RAT)],
        "test": data[int(sample_size*TRAIN_RAT): int(sample_size*TRAIN_RAT + TEST_RAT)],
        "validation": data[int(sample_size*TRAIN_RAT + TEST_RAT):],
        "dist_data": dist_data
    }
    with open(path+"/"+file_name, 'wb') as f:
        p.dump(dataset, f)

create_dataset('./synthetic_multimodal', 'multimodal.p', 102240, 15, 4)
