import os
import numpy as np
import scipy.stats
import pickle
import matplotlib
matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt  # noqa
from datetime import datetime
from ..experiments import runner
from ..model import transforms as trans


def calc_ground_truth(cond, dist_data):
    '''
    calculate the conditional covariance matrix and mean vector given observed values
    Args:
        cond: [2d]
        dist_data: (mean, covariance)
    '''
    d = int(cond.shape[0] / 2)
    mu, sigma = dist_data
    xo = cond[:d]
    bitmask = cond[d:]

    nqs = np.nonzero(bitmask)[0]  # indices of observed
    qs = np.nonzero(1 - bitmask)[0]  # indices of unobserved

    def index(x, rows, cols):
        return np.array([row[cols] for row in x[rows]])

    e11, e21, e12, e22 = (
        index(sigma, qs, qs),
        index(sigma, nqs, qs),
        index(sigma, qs, nqs),
        index(sigma, nqs, nqs),
    )

    tmp1 = np.dot(e12, np.linalg.inv(e22))
    tmp2 = xo[nqs] - mu[nqs]
    mu_update = np.dot(tmp1, tmp2)
    new_mu = mu[qs] + mu_update
    new_sigma = e11 - np.dot(np.dot(e12, np.linalg.inv(e22)), e21)

    return new_mu, new_sigma


def calc_likelihood(x, cond, dist_data):
    if not isinstance(dist_data, list):
        dist_data = [dist_data]
    likel = []
    for dist in dist_data:
        mu, sigma = calc_ground_truth(cond, dist)
        d = len(mu)
        x = x[:d]
        l = scipy.stats.multivariate_normal.pdf(x, mu, sigma)
        likel.append(l)
    return np.log(np.mean(likel))


def sample(cond, dist_data):
    if not isinstance(dist_data, list):
        dist_data = [dist_data]
    samples = []
    for dist in dist_data:
        mu, sigma = calc_ground_truth(cond, dist)
        samp = np.random.multivariate_normal(mu, sigma, 500)
        samples.append(samp)
    samples = np.stack(samples, axis=-1)
    num_modes = len(dist_data)
    inds = np.random.choice(num_modes, [500])
    sel = np.zeros([500, 1, num_modes])
    for i, ind in enumerate(inds):
        sel[i, :, ind] = 1.
    x = np.sum(samples * sel, axis=-1)

    return x


def main(home, ename, datapath):
    ac = {
        'init_lr': (0.005, ),
        'lr_decay': (0.5, ),
        'max_grad_norm': (1, ),
        'train_iters': (60000, ),
        'first_do_linear_map': (False, ),
        'first_trainable_A': (False, ),
        'trans_funcs': ([
            trans.cond_linear_map,
            trans.cond_rnn_coupling,
            trans.cond_log_rescale,
            trans.cond_leaky_transformation, ], ),
        # 'trans_funcs': ([], ),
        'rnn_coupling_params': ({'units': 16, 'num_layers': 1}, ),
        'cond_func': (runner.conds.rnn_model, ),
        # 'cond_func': (runner.conds.independent_model, ),
        # 'single_margin': (True, ),
        # 'standard': (True, ),
        'rnn_params': ({'units': 16, 'num_layers': 2}, ),
        'param_nlayers': (1, ),
        'ncomps': (10,),
        'batch_size': (1024, ),
        'nsample_batches': (1, ),
        'sample_per_cond': (500, ),
        'trial': range(1),
    }
    ret_new = runner.run_experiment(
        datapath, arg_list=runner.misc.make_arguments(ac),
        home=home, experiments_name=ename)
    results = ret_new[0]['results']

    res_path = os.path.join(home, 'results', ename)
    runner.misc.make_path(res_path)
    pickle.dump(results, open(os.path.join(res_path, 'results.p'), 'wb'))

    # Get test likelihoods
    test_llks = results['test_llks']
    results['mean_test_llks'] = np.mean(test_llks)
    results['std_test_llks'] = np.std(test_llks)
    results['stderr_test_llks'] = \
        2 * results['std_test_llks'] / np.sqrt(len(test_llks))
    print('{}\nAverage Test Log-Likelihood: {} +/- {}\n'.format(
        datapath, results['mean_test_llks'], results['stderr_test_llks']))

    ########################################
    ###  test against groundtruth  #########
    ########################################
    print('Loading test data for plotting.')
    with open(datapath, 'rb') as f:
        dataset = pickle.load(f)
        test_data = dataset['test']
        test_cond = dataset['test_labels']
        dist_data = dataset['dist_data']

    # groundtruth test log likelihood
    gt_lls = []
    num = test_data.shape[0]
    for n in range(num):
        cond = test_cond[n]
        x = test_data[n]
        ll = calc_likelihood(x, cond, dist_data)
        gt_lls.append(ll)
    gt_lls = np.mean(gt_lls)
    print('Average Ground Truth Test Log-Likelihood: {}\n'.format(gt_lls))

    # mse
    samples = results['test_samples']
    samples_cond = results['test_samples_cond']
    N, n, d = samples.shape
    mask = samples_cond[:, d:]
    avg_samples = np.mean(samples, axis=1)
    sorted_mask = np.sort(1 - mask, axis=1)[:, ::-1]
    r_samples = avg_samples.copy()
    r_samples[mask == 0] = avg_samples[sorted_mask != 0]
    r_samples *= (1 - mask)

    complete_data = samples_cond[:, :d].copy()
    complete_data[mask == 0] = test_data[sorted_mask != 0]
    std = np.std(complete_data, axis=0)  # [d]
    mse = (r_samples - complete_data)**2
    mse *= (1 - mask)
    mse = np.sum(mse, axis=0)
    num = np.sum(1 - mask, axis=0)
    num = np.maximum(np.ones_like(num), num)
    mse /= num
    nrmse = np.sqrt(mse) / std  # [d]
    nrmse = np.mean(nrmse)
    mse = np.mean(mse)

    print('Average Test NRMSE: {}'.format(nrmse))
    print('Average Test MSE: {}'.format(mse))

    # show samples
    if '2d' not in datapath:
        print('not 2d dataset, samples cannot be displayed.')
        return

    for i in range(10):
        samp = samples[i, :, :2]
        cond = samples_cond[i]
        gt_samp = sample(cond, dist_data)

        fig = plt.figure()
        plt.scatter(samp[:, 0], samp[:, 1], c='red',
                    alpha=0.5, label='samples')
        plt.scatter(gt_samp[:, 0], gt_samp[:, 1], alpha=0.5,
                    c='blue', label='groundtruth')
        plt.legend()
        plt.savefig(os.path.join(res_path, 'samp_{}.png'.format(i)))
        plt.close('all')
