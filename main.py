import os
import argparse
from tan.demos import synthetic_imputation_demo as sdemo
from tan.demos import imputation_demo as demo

home = './exp'
synthetic_datasets = {
    '2d': './data/synthetic/imp_2d.p',
    '2d_fix': './data/synthetic/imp_2d_fix.p',
    'rand': './data/synthetic/imp_rand.p',
    '2d_4d': './data/synthetic/imp_2d_4d.p',
}

datasets = {
    'mnist': './data/mnist/mnist.p',
    'white': './data/uci/white.p'
}

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--ename', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.dataset in synthetic_datasets:
    sdemo.main(home, args.ename, synthetic_datasets[args.dataset])
elif args.dataset in datasets:
    demo.main(home, args.ename, datasets[args.dataset])
else:
    print('not recognized dataset!!!')
