import os
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt  # noqa
from datetime import datetime
from ..experiments import runner
from ..model import transforms as trans
from ..data.imputation_fetcher import generate_fetchers, impute, DatasetFetchers, batch_resize


def main(home, ename, datapath):
    ac = {
        'lambda_mse': (None,),
        'print_iters': (100, ),
        'init_lr': (0.005, ),
        'lr_decay': (0.5, ),
        'max_grad_norm': (1, ),
        'train_iters': (60000, ),
        'first_do_linear_map': (False, ),
        'first_trainable_A': (False, ),
        'do_init_cond_trans': (True, ),
        'do_final_cond_trans': (True, ),
        'cond_hidden_sizes': ([256, 256],),
        # make sure using even number of reverse
        'trans_funcs': ([
            trans.cond_leaky_transformation,
            trans.cond_log_rescale, trans.cond_additive_coupling, trans.cond_reverse,
            trans.cond_linear_map, trans.cond_leaky_transformation,
            trans.cond_log_rescale, trans.cond_additive_coupling, trans.cond_reverse,
            trans.cond_linear_map, trans.cond_leaky_transformation,
            trans.cond_log_rescale, trans.cond_additive_coupling, trans.cond_reverse,
            trans.cond_linear_map, trans.cond_leaky_transformation,
            trans.cond_log_rescale, trans.cond_additive_coupling, trans.cond_reverse,
            trans.cond_linear_map, trans.cond_leaky_transformation,
            trans.cond_log_rescale, ], ),
        'add_hidden_sizes': ([256, 256],),
        'cond_linear_rank': (-1,),
        'cond_linear_hids': ([256, 256],),
        'rnn_coupling_params': ({'units': 256, 'num_layers': 2}, ),
        'cond_func': (runner.conds.rnn_model, ),
        'rnn_params': ({'units': 256, 'num_layers': 2}, ),
        # 'cond_func': (runner.conds.independent_model, ),
        # 'single_margin': (True, ),
        # 'standard': (True, ),
        'param_nlayers': (2, ),
        'ncomps': (40,),
        'batch_size': (1024, ),
        'nsample_batches': (1, ),
        'samp_per_cond': (10, ),
        'trial': range(1),
    }

    is_image = False
    channels = 1
    resize = 8
    noise_std = 0.0
    standardize = True
    logit = True
    if 'mnist' in datapath:
        is_image = True

    fetcher = generate_fetchers(
        is_image, channels, resize, noise_std, standardize, logit)

    best, all_results = runner.run_experiment(
        datapath, arg_list=runner.misc.make_arguments(ac),
        home=home, experiments_name=ename,
        fetcher_class=fetcher)

    for aid, results in enumerate(all_results):
        res_path = os.path.join(home, 'results', ename, 'trail_{}'.format(aid))
        runner.misc.make_path(res_path)
        pickle.dump(results, open(os.path.join(res_path, 'results.p'), 'wb'))

        print('=' * 10, 'Trail {}'.format(aid), '=' * 10)
        # Get test likelihoods
        test_llks = results['test_llks']
        results['mean_test_llks'] = np.mean(test_llks)
        results['std_test_llks'] = np.std(test_llks)
        results['stderr_test_llks'] = \
            2 * results['std_test_llks'] / np.sqrt(len(test_llks))
        print('{}\nAverage Test Log-Likelihood: {} +/- {}\n'.format(
            datapath, results['mean_test_llks'], results['stderr_test_llks']))

        # Get test MSE
        print('Loading test data for computing MSE.')
        with open(datapath, 'rb') as f:
            dataset = pickle.load(f)
            test_data = dataset['test']  # [N, d]
        if is_image and resize > 0:
            test_data = batch_resize(test_data, resize, channels)
        samples = results['test_samples']  # [N, n, d]
        means = results['test_means']  # [N, d]
        samples_cond = results['test_samples_cond']  # [N, 2d]

        assert samples.shape[0] == test_data.shape[0]
        N, n, d = samples.shape
        mask = samples_cond[:, d:]
        bitmask = np.repeat(np.expand_dims(mask, axis=1), n, axis=1)  # [N,n,d]
        sorted_bitmask = np.sort(1 - bitmask, axis=2)[:, :, ::-1]
        r_samples = samples.copy()
        r_samples[bitmask == 0] = samples[sorted_bitmask != 0]
        r_samples *= (1 - bitmask)

        r_samples = DatasetFetchers.reverse(
            r_samples, is_image, standardize, logit)

        avg_samples = np.mean(r_samples, axis=1)  # [N,d]

        std = np.std(test_data, axis=0)  # [d]
        mse = (avg_samples - test_data)**2
        mse *= (1 - mask)
        mse = np.sum(mse, axis=0)
        num = np.sum(1 - mask, axis=0)
        num = np.maximum(np.ones_like(num), num)
        mse /= num
        nrmse = np.sqrt(mse) / std  # [d]
        nrmse = np.mean(nrmse)
        mse = np.mean(mse)

        print('Average Test NRMSE (using samples): {}'.format(nrmse))
        print('Average Test MSE (using samples): {}'.format(mse))

        sorted_mask = np.sort(1 - mask, axis=1)[:, ::-1]
        r_means = means.copy()
        r_means[mask == 0] = means[sorted_mask != 0]
        r_means *= (1 - mask)

        r_means = DatasetFetchers.reverse(
            r_means, is_image, standardize, logit)

        mse = (r_means - test_data)**2
        mse *= (1 - mask)
        mse = np.sum(mse, axis=0)
        mse /= num
        nrmse = np.sqrt(mse) / std  # [d]
        nrmse = np.mean(nrmse)
        mse = np.mean(mse)

        print('Average Test NRMSE (using means): {}'.format(nrmse))
        print('Average Test MSE (using means): {}'.format(mse))

        #####################################################################
        if not is_image:
            print('not image, samples cannot be displayed.')
        else:
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
                plt.imsave(os.path.join(
                    res_path, 'samp_{}.png'.format(i)), img)
