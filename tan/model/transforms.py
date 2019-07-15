""" Transformation of variable component of TANs.
- Transformations are function that
  - take in:
    - an input `[N x d]`
    - (and possibly) a conditioning value `[N x p]`
  - return:
    - transformed covariates `[N x d]`
    - log determinant of the Jacobian `[N]` or scalar
    - inverse mapping `[function: N x d (, N x p) -> N x d]`
- `transformer` takes in a list of transformations and
  composes them into single transformation.
"""

import tensorflow as tf
import numpy as np
import scipy.linalg as linalg  # noqa
from . import simple_rnn as simple
from ..utils import nn


# %% Permutation functions.
#
def reverse(x, name='reverse'):
    """Reverse along last dimension."""
    with tf.variable_scope(name) as scope:
        z = tf.reverse(x, [-1])
        logdet = 0.0

    # Inverse map
    def invmap(z):
        with tf.variable_scope(scope, reuse=True):
            x = tf.reverse(z, [-1])
            return x
    return z, logdet, invmap


def cond_reverse(x, conditioning, name='cond_reverse'):
    '''
    reverse based on bitmask
    '''
    d = tf.shape(x)[1]
    with tf.variable_scope(name) as scope:
        bitmask = conditioning[:, d:]
        sorted_bitmask = tf.contrib.framework.sort(
            1. - bitmask, axis=-1, direction='DESCENDING')
        m = tf.reverse(sorted_bitmask, [-1])
        ind = tf.contrib.framework.argsort(
            m, axis=-1, direction='DESCENDING', stable=True)
        z = tf.reverse(x, [-1])
        z = tf.batch_gather(z, ind)
        logdet = 0.0

    def invmap(z, conditioning):
        with tf.variable_scope(scope, reuse=True):
            bitmask = conditioning[:, d:]
            sorted_bitmask = tf.contrib.framework.sort(
                1. - bitmask, axis=-1, direction='DESCENDING')
            m = tf.reverse(sorted_bitmask, [-1])
            ind = tf.contrib.framework.argsort(
                m, axis=-1, direction='DESCENDING', stable=True)
            x = tf.reverse(z, [-1])
            x = tf.batch_gather(x, ind)
            return x

    return z, logdet, invmap


def permute(x, perm, name='perm'):
    """Permutes according perm along last dimension."""
    with tf.variable_scope(name) as scope:
        z = tf.transpose(tf.gather(tf.transpose(x), perm))
        logdet = 0.0

    # Inverse map
    def invmap(z):
        with tf.variable_scope(scope, reuse=True):
            x = tf.transpose(tf.gather(tf.transpose(z), invperm(perm)))
            return x
    return z, logdet, invmap


def invperm(perm):
    """Returns the inverse permutation."""
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse

# %% Linear mapping functions.
#


def get_LU_map(mat_params, b):
    """Make the matrix for linear map y^t = x^t (L U) + b^t.
    Args:
        mat_params: d x d array of matrix parameters. Contains lower and upper
            matrices L, U. L has unit diagonal.
        b: d length array of biases
    Returns:
        A: the linear map matrix resulting from the multiplication of L and U.
        logdet: the log determinant of the Jacobian for this transformation.
        invmap: function that computes the inverse transformation.
    """
    with tf.variable_scope('LU'):
        with tf.variable_scope('unpack'):
            # Unpack the mat_params and U matrices
            d = int(mat_params.get_shape()[0])
            U = tf.matrix_band_part(mat_params, 0, -1)
            L = tf.eye(d) + mat_params * tf.constant(
                np.tril(np.ones((d, d), dtype=np.float32), -1),
                dtype=tf.float32, name='tril'
            )
            A = tf.matmul(L, U, name='A')
        with tf.variable_scope('logdet'):
            # Get the log absolute determinate
            logdet = tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(U))))

        # Inverse map
        def invmap(y):
            with tf.variable_scope('invmap'):
                Ut = tf.transpose(U)
                Lt = tf.transpose(L)
                yt = tf.transpose(y)
                sol = tf.matrix_triangular_solve(
                    Ut, yt - tf.expand_dims(b, -1))
                x = tf.transpose(
                    tf.matrix_triangular_solve(Lt, sol, lower=False)
                )
                return x
    return A, logdet, invmap


def linear_map(x, init_mat_params=None, init_b=None, mat_func=get_LU_map,
               trainable_A=True, trainable_b=True, irange=1e-10,
               name='linear_map'):
    """Return the linearly transformed, y^t = x^t * mat_func(mat_params) + b^t,
    log determinant of Jacobian and inverse map.
    Args:
        x: N x d real tensor of covariates to be linearly transformed.
        init_mat_params: tensor of parameters for linear map returned by
            mat_func(init_mat_params, b) (see get_LU_map above).
        init_b: d length tensor of biases.
        mat_func: function that returns matrix, log determinant, and inverse
            for linear mapping (see get_LU_map).
        trainable_A: boolean indicating whether to train matrix for linear
            map.
        trainable_b: boolean indicating whether to train bias for linear
            map.
        name: variable scope.
    Returns:
        z: N x d linearly transformed covariates.
        logdet: scalar, the log determinant of the Jacobian for transformation.
        invmap: function that computes the inverse transformation.
    """
    if irange is not None:
        initializer = tf.random_uniform_initializer(-irange, irange)
    else:
        initializer = None
    with tf.variable_scope(name, initializer=initializer):
        d = int(x.get_shape()[-1])
        if init_mat_params is None:
            # mat_params = tf.get_variable(
            #     'mat_params', dtype=tf.float32,
            #     shape=(d, d), trainable=trainable_A)
            mat_params = tf.get_variable(
                'mat_params', dtype=tf.float32,
                initializer=tf.eye(d, dtype=tf.float32),
                trainable=trainable_A)
        else:
            mat_params = tf.get_variable('mat_params', dtype=tf.float32,
                                         initializer=init_mat_params,
                                         trainable=trainable_A)
        if init_b is None:
            # b = tf.get_variable('b', dtype=tf.float32, shape=(d,),
            #                     trainable=trainable_b)
            b = tf.get_variable('b', dtype=tf.float32,
                                initializer=tf.zeros((d, ), tf.float32),
                                trainable=trainable_b)
        else:
            b = tf.get_variable('b', dtype=tf.float32, initializer=init_b,
                                trainable=trainable_b)
        A, logdet, invmap = mat_func(mat_params, b)
        z = tf.matmul(x, A) + tf.expand_dims(b, 0)


def linear_cond_values(conditioning, d, hidden_sizes=[256]):
    # run conditioning information through fully connected layer
    with tf.variable_scope("linear_conditional_matrix_param", reuse=tf.AUTO_REUSE):
        mat = nn.fc_network(
            conditioning, d, hidden_sizes=hidden_sizes, name='mlp', output_init_range=4,
        )
        mat = tf.matrix_diag(mat)
    with tf.variable_scope("linear_conditional_bias", reuse=tf.AUTO_REUSE):
        bias = nn.fc_network(
            conditioning, d, hidden_sizes=hidden_sizes, name='mlp', output_init_range=1e-2
        )
    return tf.squeeze(mat), tf.squeeze(bias)


def linear_conditional_matrix(conditioning, mat_params):
    d = int(conditioning.get_shape()[-1] / 2)
    set_size = conditioning.get_shape()[0]
    mat_cond, bias_cond = linear_cond_values(
        tf.expand_dims(conditioning, axis=0), d)
    A = tf.tile(tf.expand_dims(mat_params, axis=0),
                [set_size, 1, 1]) + mat_cond
    bitmask = 1 - conditioning[:, d:]
    order = tf.contrib.framework.argsort(
        bitmask, direction='DESCENDING', stable=True)
    t = tf.batch_gather(tf.matrix_diag(bitmask), order)
    bias_cond = tf.squeeze(tf.matmul(t, tf.expand_dims(bias_cond, -1)))
    return tf.matmul(tf.matmul(t, A), tf.transpose(t, perm=[0, 2, 1])), bias_cond

# Conditional linear transform for data imputation


def get_cond_LU_map(mat_params, b, conditioning):
    """Make the matrix for linear map y^t = x^t (L U) + b^t.
    Args:
        mat_params: d x d array of matrix parameters. Contains lower and upper
            matrices L, U. L has unit diagonal.
        b: d length array of biases
    Returns:
        A: the linear map matrix resulting from the multiplication of L and U.
        logdet: the log determinant of the Jacobian for this transformation.
        invmap: function that computes the inverse transformation.
    """
    with tf.variable_scope('LU'):
        with tf.variable_scope('unpack'):
            # Unpack the mat_params and U matrices
            d = int(mat_params.get_shape()[1])
            # get upper triangular matrix
            U = tf.matrix_band_part(mat_params, 0, -1)
            L = tf.eye(d) + mat_params - U
            A = tf.matmul(L, U, name='A')
            # add 1 to all zeros in diagonal
            U = U + tf.matrix_diag(
                tf.contrib.framework.sort(conditioning[:, d:], direction='ASCENDING')
            )
        with tf.variable_scope('logdet'):
            # Get the log absolute determinate
            logdet = tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(U))), axis=1)

    return A, logdet, U, L


def cond_linear_map(x, conditioning, mat_func=get_cond_LU_map, trainable_A=True,
                    trainable_b=True, irange=1e-10, name='cond_linear_map'):
    """Return the linearly transformed, y^t = x^t * mat_func(mat_params) + b^t,
    log determinant of Jacobian and inverse map.
    Args:
        x: N x d real tensor of covariates to be linearly transformed.
        conditioning: 
        mat_func: function that returns matrix, log determinant, and inverse
            for linear mapping (see get_LU_map).
        trainable_A: boolean indicating whether to train matrix for linear
            map.
        trainable_b: boolean indicating whether to train bias for linear
            map.
        name: variable scope.
    Returns:
        z: N x d linearly transformed covariates.
        logdet: scalar, the log determinant of the Jacobian for transformation.
        invmap: function that computes the inverse transformation.
    """
    if irange is not None:
        initializer = tf.random_uniform_initializer(-irange, irange)
    else:
        initializer = None
    with tf.variable_scope(name, initializer=initializer) as scope:
        d = int(x.get_shape()[-1])
        # create weight matrix
        mat_params = tf.get_variable(
            'mat_params', dtype=tf.float32,
            initializer=tf.eye(d, dtype=tf.float32), trainable=trainable_A
        )
        mats, b = linear_conditional_matrix(conditioning, mat_params)
        A, logdet, _, _ = mat_func(mats, b, conditioning)
        z = tf.einsum('ai,aik->ak', x, A, name="mat_mul") + b

        # Inverse map
        def invmap(z, conditioning):
           with tf.variable_scope(scope):
               mats, b = linear_conditional_matrix(conditioning, mat_params)
               _, _, U, L = mat_func(
                   mats, b, conditioning
               )
           with tf.variable_scope('invmap'):
               Ut = tf.transpose(U, perm=[0, 2, 1])
               Lt = tf.transpose(L, perm=[0, 2, 1])
               zt = tf.expand_dims(z - b, -1)
               sol = tf.matrix_triangular_solve(Ut, zt)
               x = tf.matrix_triangular_solve(Lt, sol, lower=False)

               return tf.squeeze(x)
    return z, logdet, invmap


# %% RNN transformation functions.
#
# TODO: change name, nonlinear rnn?
# TODO: make rnn cell a parameter?
# TODO: general rnn transformation/invertable function.
# TODO: make use_static_rnn false by defualt.
def simple_rnn_transform(x, state_size, alpha=None, state_activation=None,
                         use_static_rnn=True, name='simple_rnn'):
    """
    Non-linear rnn transformation based on simple RNN.
    Args:
        x: N x d tensor of covariates to transform.
        state_size: int size of the hidden state.
        alpha: scalar, alpha parameter of leaky relu.
        state_activation: activation function to use on state of simple RNN.
            Uses relu by default.
        use_static_rnn: boolean indicating whether to use static_rnn tf function
            (useful when debugging).
        name: variable scope.
    Returns:
        z: N x d rnn transformed covariates.
        logdet: N tensor of log determinant of the Jacobian for transformation.
        invmap: function that computes the inverse transformation.
    """
    with tf.variable_scope(name):
        d = int(x.get_shape()[1])
        cell = simple.Simple1dCell(state_size, alpha=alpha,
                                   state_activation=state_activation)
        # Get output from rnn cell.
        if not use_static_rnn:
            y, _ = tf.nn.dynamic_rnn(cell, tf.expand_dims(x, -1),
                                     dtype=tf.float32)
        else:
            # I think dynamic_rnn was giving trouble when using check_numerics
            rnn_input = tf.expand_dims(x, -1)
            split_rnn_input = tf.split(
                rnn_input, int(rnn_input.get_shape()[1]), 1
            )
            squeezed_rnn_input = [
                tf.squeeze(ri, 1) for ri in split_rnn_input
            ]
            outputs_list, _ = \
                tf.contrib.rnn.static_rnn(cell, squeezed_rnn_input,
                                          dtype=tf.float32)
            y = tf.concat(
                [tf.expand_dims(oi, 1) for oi in outputs_list], 1
            )
        y = tf.squeeze(y, -1)
        # log determinant, can get according to the number of negatives in
        # output.
        num_negative = tf.reduce_sum(tf.cast(tf.less(y, 0.0), tf.float32), 1)
        logdet = d * tf.log(tf.abs(cell._w_z_y)) + \
            num_negative * tf.log(cell._alpha)
        invmap = cell.inverse
    return y, logdet, invmap


def rnn_coupling(x, rnn_class, name='rnn_coupling'):
    """
    RNN coupling where the covariates are transformed as z_i = x_i + m(s_i).
    Args:
        x: N x d input covariates.
        rnn_class: function the returns rnn_cell with output of spcified size,
            e.g. rnn_class(nout).
        name: variable scope.
    Returns:
        z: N x d rnn transformed covariates.
        logdet: N tensor of log determinant of the Jacobian for transformation.
        invmap: function that computes the inverse transformation.
    """
    with tf.variable_scope(name) as scope:
        # Get RNN cell for transforming single covariates at a time.
        rnn_cell = rnn_class(1)  # TODO: change from 1 to 2 for optional scale
        # Shapes.
        batch_size = tf.shape(x)[0]
        d = int(x.get_shape()[1])
        # Initial variables.
        state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
        inp = -tf.ones((batch_size, 1), dtype=tf.float32)
        z_list = []
        for t in range(d):
            m_t, state = rnn_cell(inp, state)
            x_t = tf.expand_dims(x[:, t], -1)
            z_t = x_t + m_t
            z_list.append(z_t)
            inp = x_t
        z = tf.concat(z_list, 1)
        # Jacobian is lower triangular with unit diagonal.
        logdet = 0.0

    # inverse
    def invmap(z):
        with tf.variable_scope(scope, reuse=True):
            # Shapes.
            batch_size = tf.shape(z)[0]
            # Initial variables.
            state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
            inp = -tf.ones((batch_size, 1), dtype=tf.float32)
            x_list = []
            for t in range(d):
                m_t, state = rnn_cell(inp, state)
                z_t = tf.expand_dims(z[:, t], -1)
                x_t = z_t - m_t
                x_list.append(x_t)
                inp = x_t
            x = tf.concat(x_list, 1)
        return x
    return z, logdet, invmap


def cond_rnn_coupling(x, conditioning, rnn_class, name='cond_rnn_coupling'):
    """
    RNN coupling where the covariates are transformed as z_i = x_i + m(s_i).
    Args:
        x: N x d input covariates.
        rnn_class: function the returns rnn_cell with output of spcified size,
            e.g. rnn_class(nout).
        name: variable scope.
    Returns:
        z: N x d rnn transformed covariates.
        logdet: N tensor of log determinant of the Jacobian for transformation.
        invmap: function that computes the inverse transformation.
    """
    with tf.variable_scope(name) as scope:
        # Get RNN cell for transforming single covariates at a time.
        rnn_cell = rnn_class(1)  # TODO: change from 1 to 2 for optional scale
        # Shapes.
        batch_size = tf.shape(x)[0]
        d = int(x.get_shape()[1])
        # Initial variables.
        state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
        x_t = -tf.ones((batch_size, 1), dtype=tf.float32)
        z_list = []
        for t in range(d):
            inp = tf.concat([x_t, conditioning], axis=1)
            m_t, state = rnn_cell(inp, state)
            x_t = tf.expand_dims(x[:, t], -1)
            z_t = x_t + m_t
            z_list.append(z_t)
        z = tf.concat(z_list, 1)
        # Jacobian is lower triangular with unit diagonal.
        logdet = 0.0

    # inverse
    def invmap(z, conditioning):
        with tf.variable_scope(scope, reuse=True):
            # Shapes.
            batch_size = tf.shape(z)[0]
            # Initial variables.
            state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
            x_t = -tf.ones((batch_size, 1), dtype=tf.float32)
            x_list = []
            for t in range(d):
                inp = tf.concat([x_t, conditioning], axis=1)
                m_t, state = rnn_cell(inp, state)
                z_t = tf.expand_dims(z[:, t], -1)
                x_t = z_t - m_t
                x_list.append(x_t)
            x = tf.concat(x_list, 1)
        return x
    return z, logdet, invmap


def leaky_relu(x, alpha):
    return tf.maximum(x, alpha * x)  # Assumes alpha <= 1.0


def general_leaky_relu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def leaky_transformation(x, alpha=None):
    """Implement an element wise leaky relu transformation."""
    if alpha is None:
        alpha = tf.nn.sigmoid(
            tf.get_variable('log_alpha', initializer=5.0, dtype=tf.float32))
    z = leaky_relu(x, alpha)
    num_negative = tf.reduce_sum(tf.cast(tf.less(z, 0.0), tf.float32), 1)
    logdet = num_negative * tf.log(alpha)

    def invmap(z):
        return tf.minimum(z, z / alpha)

    return z, logdet, invmap


def cond_leaky_transformation(x, conditioning, alpha=None):
    d = tf.shape(x)[1]
    if alpha is None:
        alpha = tf.nn.sigmoid(
            tf.get_variable('log_alpha', initializer=5.0, dtype=tf.float32))
    z = leaky_relu(x, alpha)
    num_negative = tf.cast(tf.less(z, 0.0), tf.float32)
    bitmask = conditioning[:, d:]
    sorted_bitmask = tf.contrib.framework.sort(
        1. - bitmask, axis=-1, direction='DESCENDING')
    num_negative = tf.reduce_sum(num_negative * sorted_bitmask, axis=1)
    logdet = num_negative * tf.log(alpha)

    def invmap(z, conditioning):
        return tf.minimum(z, z / alpha)

    return z, logdet, invmap


class Simple1dCell(tf.contrib.rnn.RNNCell):
    """Implements a simple distribution based recurrent unit that keeps moving
    averages of the mean map embeddings of features of inputs.
    Assumes that alpha in (0, 1]
    """

    def __init__(self, state_size, alpha=None, max_alpha=1.0,
                 state_activation=tf.nn.relu):
        self._state_size = state_size
        self._output_dims = 1
        if alpha is None:
            assert max_alpha <= 1.0
            self._alpha = max_alpha * tf.nn.sigmoid(
                tf.get_variable('log_alpha', initializer=5.0, dtype=tf.float32))
            # self._alpha = tf.minimum(
            #     tf.exp(tf.get_variable('log_alpha', initializer=0.0,
            #                            dtype=tf.float32)),
            #     max_alpha)
            # TODO: get rid of debug code
            # self._alpha = tf.Print(self._alpha, [self._alpha], 'alpha')
        else:
            if isinstance(alpha, float):
                assert alpha <= 1.0
            self._alpha = alpha
        if state_activation is not None:
            self._state_activation = state_activation
        else:
            def lr(x):
                return leaky_relu(x, self._alpha)

            self._state_activation = lr

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_dims

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # Output parameters.
            self._w_z_y = tf.get_variable(
                'w_z_y',  # shape=(1,),
                dtype=tf.float32,
                initializer=tf.ones((1,), dtype=tf.float32))
            self._w_z_h = tf.get_variable(
                'w_z_h',  # shape=(self.state_size, 1),
                dtype=tf.float32,
                initializer=tf.zeros((self.state_size, 1), tf.float32))
            self._b_z = tf.get_variable(
                'b_z',  # shape=(1,),
                dtype=tf.float32,
                initializer=tf.zeros((1,), tf.float32))
            # State parameters.
            self._w_h_y = tf.get_variable(
                'w_h_y', shape=(1,), dtype=tf.float32)
            self._w_h_h = tf.get_variable(
                'w_h_h',  # shape=(self.state_size, self.state_size),
                dtype=tf.float32,
                initializer=tf.eye(self.state_size, dtype=tf.float32))
            self._b_h = tf.get_variable(
                'b_h',  # shape=(self._state_size,),
                dtype=tf.float32,
                initializer=tf.zeros((self._state_size,), dtype=tf.float32))
            output = leaky_relu(
                self._w_z_y * inputs + tf.matmul(state, self._w_z_h) +
                self._b_z,
                self._alpha)
            out_state = self._state_activation(
                self._w_h_y * inputs + tf.matmul(state, self._w_h_h) + self._b_h)
        return output, out_state

    def transform(self, inputs, scope='rnn_transform'):
        batch_size = tf.shape(inputs)[0]
        d = int(inputs.get_shape()[1])
        state = self.zero_state(batch_size, dtype=tf.float32)
        z_list = []
        with tf.variable_scope(scope or type(self).__name__):
            for t in range(d):
                z_t, state = self(tf.expand_dims(inputs[:, t], -1), state)
                z_list.append(z_t)
            z = tf.concat(z_list, 1)
        return z

    def inverse(self, output, scope=None):
        """ Computes the inverse mapping for this rnn. May not be numerically
        stable for small |w_z_y|.
        Args:
            output: N x d tensors
        Returns:
            inverse: N x d tensor of original values
        """
        batch_size = tf.shape(output)[0]
        d = int(output.get_shape()[1])
        state = self.zero_state(batch_size, dtype=tf.float32)
        y_list = []
        with tf.variable_scope(scope or type(self).__name__):
            for t in range(d):
                z_t = tf.expand_dims(output[:, t], -1)
                z_t_scaled = tf.minimum(z_t, z_t / self._alpha)
                y_t = (z_t_scaled - tf.matmul(state, self._w_z_h) - self._b_z)
                y_t /= self._w_z_y
                y_list.append(y_t)
                state = self._state_activation(
                    self._w_h_y * y_t + tf.matmul(state, self._w_h_h) +
                    self._b_h
                )
            y = tf.concat(y_list, 1)
        return y


# %% NICE/NVP transformation function.
#
# TODO: add scale like in conditional transformation.
def additive_coupling(x, hidden_sizes, irange=None, output_irange=None,
                      activation=tf.nn.relu, name='additive_coupling'):
    """ NICE additive coupling layer. """
    if irange is not None:
        initializer = tf.random_uniform_initializer(-irange, irange)
    else:
        initializer = None
    with tf.variable_scope(name, initializer=initializer) as scope:
        d = int(x.get_shape()[1])
        d_half = d / 2
        x_1 = tf.slice(x, [0, 0], [-1, d_half], 'x_1')
        x_2 = tf.slice(x, [0, d_half], [-1, -1], 'x_2')
        m = nn.fc_network(x_2, d_half, hidden_sizes=hidden_sizes,
                          output_init_range=output_irange,
                          activation=activation, name='m')
        y = tf.concat((x_1 + m, x_2), 1, 'y')
        print(y.get_shape())
        logdet = 0.0

    # inverse
    def invmap(y):
        with tf.variable_scope(scope, reuse=True):
            y_1 = tf.slice(y, [0, 0], [-1, d_half], 'y_1')
            y_2 = tf.slice(y, [0, d_half], [-1, -1], 'y_2')
            m = nn.fc_network(y_2, d_half, hidden_sizes=hidden_sizes,
                              output_init_range=output_irange, reuse=True,
                              activation=activation, name='m')
            x = tf.concat((y_1 - m, y_2), 1, 'y_inv')
            return x

    return y, logdet, invmap


# %% Conditional based transformation
#
def conditioning_transformation(x, conditioning, hidden_sizes,
                                irange=None, output_irange=None,
                                activation=tf.nn.relu,
                                name='cond_trans'):
    """
    Transform covariates x using a scaling and shift coming from a fully
    connected network on extranous conditioning information y.
    z = x*exp(s) + m; m,s = split(fc_net(y)).
    Args:
        x: N x d input covariates.
        conditioning: N x p of extraneous conditioning values.
        hidden_sizes: list of hidden layer sizes for use in fc_net for shift
            and scaling.
        irange: scalar, used to initialize the weights of the fc_net randomly
            in [-irange, irange]; a small value helps keep initial
            transformations close to identity.
        output_irange: scalar, seperate initializer to overide irange for the
            output of fc_net.
        activation: activation function to use in fc_net.
        name: variable scope
    Returns:
        z: N x d transformed covariates.
        logdet: scalar, the log determinant of the Jacobian for transformation.
        invmap: function that computes the inverse transformation.
    """
    if conditioning is None:
        # Identity transformation.
        return x, 0.0, (lambda y, c: y)

    # TODO: remove print.
    print('Using conditional transformation...')
    if irange is not None:
        initializer = tf.random_uniform_initializer(-irange, irange)
    else:
        initializer = None
    with tf.variable_scope(name, initializer=initializer) as scope:
        d = int(x.get_shape()[1])
        ms = nn.fc_network(conditioning, 2 * d, hidden_sizes=hidden_sizes,
                           output_init_range=output_irange,
                           activation=activation, name='ms')
        m, s = tf.split(ms, 2, 1)
        y = tf.multiply(x, tf.exp(s)) + m
        logdet = tf.reduce_sum(s, 1)

    # inverse
    def invmap(y, conditioning):
        with tf.variable_scope(scope, reuse=True):
            ms = nn.fc_network(conditioning, 2 * d, hidden_sizes=hidden_sizes,
                               output_init_range=output_irange,
                               activation=activation, name='ms')
            m, s = tf.split(ms, 2, 1)
            x = tf.div(y - m, tf.exp(s))
            return x

    return y, logdet, invmap


# %% Simple Transformations.
#
def rescale(x, init_constant=None, name='rescale'):
    """Rescale z = s*x."""
    with tf.variable_scope(name) as scope:
        d = int(x.get_shape()[1])
        if init_constant is not None:
            s = tf.get_variable(
                's', initializer=init_constant * tf.ones((1, d)),
                dtype=tf.float32)
        else:
            s = tf.get_variable('s', shape=(1, d), dtype=tf.float32)
        y = tf.multiply(x, s, name='y')
        logdet = tf.reduce_sum(tf.log(tf.abs(s)))

    # inverse
    def invmap(y):
        with tf.variable_scope(scope, reuse=True):
            # TODO: neccesarryryryry?
            s = tf.get_variable('s', shape=(1, d), dtype=tf.float32)
            x = tf.divide(y, s, name='y_inv')
            return x

    return y, logdet, invmap


def log_rescale(x, init_zeros=True, name='rescale'):
    """Rescale z = exp(s)*x"""
    with tf.variable_scope(name) as scope:
        d = int(x.get_shape()[1])
        if init_zeros:
            s = tf.get_variable(
                's', initializer=tf.zeros((1, d)), dtype=tf.float32)
        else:
            s = tf.get_variable('s', shape=(1, d), dtype=tf.float32)
        y = tf.multiply(x, tf.exp(s), name='y')
        logdet = tf.reduce_sum(s)

    # inverse
    def invmap(y):
        with tf.variable_scope(scope, reuse=True):
            # TODO: neccesaary?
            s = tf.get_variable('s', shape=(1, d), dtype=tf.float32)
            x = tf.divide(y, tf.exp(s), name='y_inv')
            return x

    return y, logdet, invmap


def cond_log_rescale(x, conditioning, init_zeros=True, name='cond_rescale'):
    with tf.variable_scope(name) as scope:
        N = tf.shape(x)[0]
        d = int(x.get_shape()[1])
        if init_zeros:
            s = tf.get_variable(
                's', initializer=tf.zeros((1, d)), dtype=tf.float32)
        else:
            s = tf.get_variable('s', shape=(1, d), dtype=tf.float32)

        bitmask = conditioning[:, d:]
        ind = tf.contrib.framework.argsort(
            1 - bitmask, axis=-1, direction='DESCENDING', stable=True)
        s_tiled = tf.tile(s, [N, 1])
        su = tf.batch_gather(s_tiled, ind)
        y = tf.multiply(x, tf.exp(su), name='y')
        logdet = tf.reduce_sum(s_tiled * (1 - bitmask), axis=-1)

    def invmap(y, conditioning):
        with tf.variable_scope(scope, reuse=True):
            s = tf.get_variable('s', shape=(1, d), dtype=tf.float32)
            bitmask = conditioning[:, d:]
            ind = tf.contrib.framework.argsort(
                1 - bitmask, axis=-1, direction='DESCENDING', stable=True)
            s_tiled = tf.tile(s, [N, 1])
            su = tf.batch_gather(s_tiled, ind)
            x = tf.divide(y, tf.exp(su), name='y_inv')
            return x

    return y, logdet, invmap


def shift(x, init_zeros=True, name='shift'):
    """Shift z = x + b."""
    with tf.variable_scope(name) as scope:
        d = int(x.get_shape()[1])
        if init_zeros:
            s = tf.get_variable(
                's', initializer=tf.zeros((1, d)), dtype=tf.float32)
        else:
            s = tf.get_variable('s', shape=(1, d), dtype=tf.float32)
        y = x + s
        logdet = 0.0

        # inverse
    def invmap(y):
        with tf.variable_scope(scope, reuse=True):
            # TODO: neccesaary?
            s = tf.get_variable('s', shape=(1, d), dtype=tf.float32)
            x = y - s
            return x

    return y, logdet, invmap


def negate(x, name='negate'):
    """Negate z = -x."""
    with tf.variable_scope(name) as scope:
        y = -x
        logdet = 0.0

    # inverse
    def invmap(y):
        with tf.variable_scope(scope, reuse=True):
            return -y

    return y, logdet, invmap


def logit_transform(x, alpha=0.05, max_val=256.0, name='logit_transform',
                    logdet_mult=None):
    """Logit transform for compact values."""
    print('Using logit transform')

    def logit(x):
        return tf.log(x) - tf.log(1.0 - x)

    with tf.variable_scope(name) as scope:
        sig = alpha + (1.0 - alpha) * x / max_val
        z = logit(sig)
        logdet = tf.reduce_sum(
            tf.log(1 - alpha) - tf.log(sig) - tf.log(1.0 - sig) - tf.log(max_val), 1)
        if logdet_mult is not None:
            logdet = logdet_mult * logdet

    # inverse
    def invmap(z):
        with tf.variable_scope(scope, reuse=True):
            arg = 1.0 / (1.0 + tf.exp(-z))
            return (arg - alpha) * max_val / (1.0 - alpha)
    return z, logdet, invmap


# %% Transformation composition
#
# TODO: test to see if gives back identity with inverse.
def transformer(inputs, transformations, conditioning=None):
    """Makes transormation on the r.v. X
    Args:
        inputs: N x d tensor of inputs
        transformations: list of functions that take input (and conditioning)
            variables to transform and return output, logdet of Jacobian,
            and inverse for transformation.
        conditioning: N x p tensor of conditioning values
    Returns:
        y: N x d tensor of transformed values
        logdet: scalar tensor with the log determinant corresponding to
            the transformation.
        invmap: function that takes in N x d tensor of the transformed r.v.s
            and outputs the r.v. in originals space.
    """
    # Apply transformations.
    y = inputs
    invmaps = []
    logdet = 0.0
    for i, trans in enumerate(transformations):
        with tf.variable_scope('transformation_{}'.format(i)):
            try:
                y, ldet, imap = trans(y, conditioning)
            except TypeError:  # Does not take in conditioning values.
                y, ldet, imap = trans(y)
            logdet += ldet
            invmaps.append(imap)

    # Make inverse by stacking inverses in reverse order.
    ntrans = len(invmaps)
    # print(invmaps[::-1])

    def invmap(z, conditioning=None):
        for i in range(ntrans - 1, -1, -1):
            with tf.variable_scope('transformation_{}'.format(i)):
                try:
                    z = invmaps[i](z, conditioning)
                except TypeError:  # Does not take in conditioning values.
                    z = invmaps[i](z)
        return z
    return y, logdet, invmap
