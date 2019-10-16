import os
import argparse
import numpy as np
import tensorflow as tf
from tan.demos import synthetic_imputation_demo as sdemo
from tan.demos import imputation_demo as demo
from tan.demos import visualize_imputation_demo as vdemo
from tan.demos import time_imputation_demo as tdemo

home = './exp'

time_datasets = {
    '2_spirals_10': './dataset/time/2_spirals_10.p',
    '2000_spirals_10': './dataset/time/2000_spirals_10.p'
}

visualize_datasets = {
    '2spirals': './dataset/visualize/2spirals.p',
    '8gaussians': './dataset/visualize/8gaussians.p',
    'checkerboard': './dataset/visualize/checkerboard.p',
    'circles': './dataset/visualize/circles.p',
    'cos': './dataset/visualize/cos.p',
    'line': './dataset/visualize/line.p',
    'moons': './dataset/visualize/moons.p',
    'pinwheel': './dataset/visualize/pinwheel.p',
    'swissroll': './dataset/visualize/swissroll.p'
}

synthetic_datasets = {
    '2d': './dataset/synthetic/imp_2d.p',
    '2d_fix': './dataset/synthetic/imp_2d_fix.p',
    'rand': './dataset/synthetic/imp_rand.p',
    '2d_4d': './dataset/synthetic/imp_2d_4d.p',
    'm2_2d_4d': './dataset/synthetic/imp_m2_2d_4d.p',
    'm4_2d_4d': './dataset/synthetic/imp_m4_2d_4d.p'
}

datasets = {
    'mnist': './dataset/mnist/mnist.p',
    'white': './dataset/uci/white.p',
    'white_vaeac': './dataset/uci/white_vaeac.p',
    'hepmass': './dataset/uci/hepmass.p',
    'maf_bsds': './dataset/maf/maf_bsds.p',
    'maf_gas': './dataset/maf/maf_gas.p',
    'maf_hepmass': './dataset/maf/maf_hepmass.p',
    'maf_miniboone': './dataset/maf/maf_miniboone.p',
    'maf_power': './dataset/maf/maf_power.p'
}

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--ename', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

all_ll = []
all_nrmse_sample = []
all_nrmse_mean = []
for trail in range(5):
    print('#', trail)
    seed = np.random.randint(10000)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    if args.dataset in synthetic_datasets:
        sdemo.main(home, args.ename, synthetic_datasets[args.dataset])
    elif args.dataset in visualize_datasets:
        vdemo.main(home, args.ename, visualize_datasets[args.dataset])
    elif args.dataset in time_datasets:
        tdemo.main(home, args.ename, time_datasets[args.dataset])
    elif args.dataset in datasets:
        ll, nrmse_sample, nrmse_mean = demo.main(
            home, args.ename, datasets[args.dataset])
    else:
        print('not recognized dataset!!!')

    all_ll.append(ll)
    all_nrmse_sample.append(nrmse_sample)
    all_nrmse_mean.append(nrmse_mean)

print('LL:', np.mean(all_ll), np.std(all_ll))
print('NRMSE(sample):', np.mean(all_nrmse_sample), np.std(all_nrmse_sample))
print('NRMSE(mean):', np.mean(all_nrmse_mean), np.std(all_nrmse_mean))
