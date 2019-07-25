import os
import argparse
import numpy as np
import tensorflow as tf
from tan.demos import synthetic_imputation_demo as sdemo
from tan.demos import imputation_demo as demo

home = './exp'
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
np.random.seed(123)
tf.set_random_seed(123)

if args.dataset in synthetic_datasets:
    sdemo.main(home, args.ename, synthetic_datasets[args.dataset])
elif args.dataset in datasets:
    demo.main(home, args.ename, datasets[args.dataset])
else:
    print('not recognized dataset!!!')
