from ..model import transforms as trans
from ..model import conditionals as conds
from ..model import likelihoods as likel
import numpy as np
import tensorflow as tf
from ..experiments import runner
from ..utils import misc
from ..rnn import cells
import pickle as p
import os


def test_transformations(xs, cond):
    inputs_pl = tf.placeholder(
        tf.float32, (xs.shape[0], xs.shape[1]), 'inputs'
    )
    conditioning_pl = tf.placeholder(
        tf.float32, (cond.shape[0], cond.shape[1]), 'conditioning'
    )

    # cond_log_rescale
    z, det, invmap = trans.cond_log_rescale(inputs_pl, conditioning_pl)
    x_rec = invmap(z, conditioning_pl)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        inv = sess.run(x_rec,
                       feed_dict={inputs_pl: xs, conditioning_pl: cond}
                       )
        err = np.mean(np.abs(xs - inv))
        print('cond_log_rescale:', err)

    # cond_reverse
    z, det, invmap = trans.cond_reverse(inputs_pl, conditioning_pl)
    x_rec = invmap(z, conditioning_pl)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        inv = sess.run(x_rec,
                       feed_dict={inputs_pl: xs, conditioning_pl: cond}
                       )
        err = np.mean(np.abs(xs - inv))
        print('cond_reverse:', err)

    # cond_leaky
    z, det, invmap = trans.cond_leaky_transformation(
        inputs_pl, conditioning_pl)
    x_rec = invmap(z, conditioning_pl)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        inv = sess.run(x_rec,
                       feed_dict={inputs_pl: xs, conditioning_pl: cond}
                       )
        err = np.mean(np.abs(xs - inv))
        print('cond_leaky:', err)

    # cond_rnn_coupling
    rnn_class = cells.GRUCell()
    z, det, invmap = trans.cond_rnn_coupling(
        inputs_pl, conditioning_pl, rnn_class)
    x_rec = invmap(z, conditioning_pl)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        inv = sess.run(x_rec,
                       feed_dict={inputs_pl: xs, conditioning_pl: cond}
                       )
        err = np.mean(np.abs(xs - inv))
        print('cond_rnn_coupling:', err)

    # cond_trans
    z, det, invmap = trans.conditioning_transformation(
        inputs_pl, conditioning_pl, [256])
    x_rec = invmap(z, conditioning_pl)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        inv = sess.run(x_rec,
                       feed_dict={inputs_pl: xs, conditioning_pl: cond}
                       )
        err = np.mean(np.abs(xs - inv))
        print('conditioning:', err)

    # cond_linear
    inputs_pl = tf.placeholder(
        tf.float32, (1024, xs.shape[1]), 'inputs'
    )
    conditioning_pl = tf.placeholder(
        tf.float32, (1024, cond.shape[1]), 'conditioning'
    )
    z, det, invmap = trans.cond_linear_map(
        inputs_pl, conditioning_pl, cond_rank=5)
    x_rec = invmap(z, conditioning_pl)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        xs = xs[:1024]
        cond = cond[:1024]
        inv = sess.run(x_rec,
                       feed_dict={inputs_pl: xs, conditioning_pl: cond}
                       )
        err = np.mean(np.abs(xs - inv))
        print('cond_linear:', err)


def test_conditionals(xs, cond):
    inputs_pl = tf.placeholder(
        tf.float32, (xs.shape[0], xs.shape[1]), 'inputs'
    )
    conditioning_pl = tf.placeholder(
        tf.float32, (cond.shape[0], cond.shape[1]), 'conditioning'
    )

    # independent conditional
    nparams = 3 * 40
    cond_inputs, cond_targets = conds.make_in_out(inputs_pl)

    with tf.variable_scope('independent'):
        params, sampler = conds.independent_model(
            cond_inputs, nparams, single_marginal=True,
            conditioning=conditioning_pl, use_conditioning=True)
        loss, lls = likel.make_nll_loss(
            params, cond_targets, 0., conditioning=conditioning_pl)
        z_sample = sampler(128, conditioning=conditioning_pl)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            nll, ll, samp = sess.run(
                [loss, lls, z_sample],
                feed_dict={inputs_pl: xs, conditioning_pl: cond})
            print(samp.shape)

    with tf.variable_scope('independent1'):
        params, sampler = conds.independent_model(
            cond_inputs, nparams, single_marginal=False,
            conditioning=conditioning_pl, use_conditioning=True)
        loss, lls = likel.make_nll_loss(
            params, cond_targets, 0., conditioning=conditioning_pl)
        z_sample = sampler(128, conditioning=conditioning_pl)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            nll, ll, samp = sess.run(
                [loss, lls, z_sample],
                feed_dict={inputs_pl: xs, conditioning_pl: cond})
            print(samp.shape)

    # rnn conditionals
    rnn_class = cells.GRUCell()
    params, sampler = conds.rnn_model(
        cond_inputs, nparams, rnn_class, conditioning=conditioning_pl)
    loss, lls = likel.make_nll_loss(
        params, cond_targets, 0., conditioning=conditioning_pl)
    z_sample = sampler(128, conditioning=conditioning_pl)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        nll, ll, samp = sess.run(
            [loss, lls, z_sample],
            feed_dict={inputs_pl: xs, conditioning_pl: cond})
        print(samp.shape)


def main(path):
    dataset = p.load(open(path, 'rb'))
    xs = dataset['train']
    cond = dataset['train_labels']

    test_transformations(xs, cond)
    # test_conditionals(xs, cond)

    ac = {
        'init_lr': (1e-9, ),
        'lr_decay': (0.5, ),
        'first_trainable_A': (True, ),
        'trans_funcs': (
            [trans.leaky_transformation,
                trans.log_rescale, trans.cond_rnn_coupling, trans.reverse,
                trans.cond_linear_map, trans.leaky_transformation,
                trans.log_rescale, trans.cond_rnn_coupling, trans.reverse,
                trans.cond_linear_map, trans.leaky_transformation,
                trans.log_rescale, trans.cond_rnn_coupling, trans.reverse,
                trans.cond_linear_map, trans.leaky_transformation,
                trans.log_rescale, trans.cond_rnn_coupling, trans.reverse,
                trans.cond_linear_map, trans.leaky_transformation,
                trans.log_rescale, ], ),
        'cond_func': (runner.conds.independent_model, ),
        'param_nlayers': (2, ),
        'train_iters': (60000, ),
        'batch_size': (1024, ),
        'relu_alpha': (None, ),
        'trial': range(10),
    }
    # ret_new, exper = runner.run_experiment(
    #     path, arg_list=runner.misc.make_arguments(ac))


if __name__ == '__main__':
    main('./dataset/synthetic/imp_rand.p')
