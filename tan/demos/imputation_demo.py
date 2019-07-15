import os
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt  # noqa
from datetime import datetime
from ..experiments import runner
from ..model import transforms as trans
from ..data.imputation_fetcher import generate_fetchers, impute


def main(home, ename, datapath):
    ac = {
        'init_lr': (0.005, ),
        'lr_decay': (0.5, ),
        'max_grad_norm': (1, ),
        'train_iters': (0, ),
        'first_do_linear_map': (False, ),
        'first_trainable_A': (False, ),
        'trans_funcs': ([
            trans.cond_leaky_transformation,
            trans.cond_log_rescale, trans.cond_rnn_coupling, trans.cond_reverse,
            trans.cond_linear_map, trans.cond_leaky_transformation,
            trans.cond_log_rescale, trans.cond_rnn_coupling, trans.cond_reverse,
            trans.cond_linear_map, trans.cond_leaky_transformation,
            trans.cond_log_rescale, trans.cond_rnn_coupling, trans.cond_reverse,
            trans.cond_linear_map, trans.cond_leaky_transformation,
            trans.cond_log_rescale, trans.cond_rnn_coupling, trans.cond_reverse,
            trans.cond_linear_map, trans.cond_leaky_transformation,
            trans.cond_log_rescale, ], ),
        'rnn_coupling_params': ({'units': 256, 'num_layers': 1}, ),
        'cond_func': (runner.conds.rnn_model, ),
        'rnn_params': ({'units': 256, 'num_layers': 2}, ),
        'param_nlayers': (2, ),
        'ncomps': (40,),
        'batch_size': (128, ),
        'nsample_batches': (1, ),
        'samp_per_cond': (5, ),
        'trial': range(1),
    }

    is_image = False
    channels = 1
    resize = 8
    if 'mnist' in datapath:
        is_image = True

    fetcher = generate_fetchers(is_image, channels, resize)

    ret_new = runner.run_experiment(
        datapath, arg_list=runner.misc.make_arguments(ac),
        home=home, experiments_name=ename,
        fetcher_class=fetcher)
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

    # Get test MSE

    #####################################################################
    if not is_image:
        print('not image, samples cannot be displayed.')
        return

    samples = results['samples']  # [N, 5, d]
    samples_cond = results['samples_cond'][0]['cond_val']  # [N, 2d]

    N, n, d = samples.shape
    s = int(np.sqrt(d / channels))
    for i in range(5):
        samp = samples[i]
        cond = samples_cond[i]
        img = []

        c = cond[:d]
        c = c.reshape([s, s, channels])
        c = np.squeeze(c, axis=-1)
        c = (c * 255).astype('uint8')
        img.append(c)
        for k in range(n):
            x = impute(samp[k], cond)
            x = x.reshape([s, s, channels])
            x = np.squeeze(x, axis=-1)
            x = (x * 255).astype('uint8')
            img.append(x)

        img = np.concatenate(img, axis=1)
        plt.imsave(os.path.join(res_path, 'samp_{}.png'.format(i)), img)
