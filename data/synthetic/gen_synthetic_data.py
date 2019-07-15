import numpy as np
import pickle as p
import os

np.random.seed(1234)


def unobserved_format(xs, bitmask):
    # move all observed values to the front of row and pad the back with zeros
    xs = np.array(xs)

    valid_mask = bitmask != 0
    flipped_mask = valid_mask.sum(
        1, keepdims=1) > np.arange(xs.shape[1] - 1, -1, -1)
    flipped_mask = flipped_mask[:, ::-1]
    xs[flipped_mask] = xs[valid_mask]
    xs[~flipped_mask] = 0
    return xs


def make_data(sample_size, d, dst_type='rand'):
    # Generate a random covariate matrix
    v = np.random.rand(d, d)
    sigma = np.matmul(v, np.transpose(v))
    mu = np.random.rand(d)
    data = np.random.multivariate_normal(mu, sigma, sample_size)

    if dst_type == 'rand':
        # Generate a random bitmask (0s and 1s uniformly distributed) 0 means unobserved
        bitmask = np.random.choice([0, 1], [sample_size, d])
        # Must always be at least 1 observed and 1 unobserved value
        for b in bitmask:
            if np.sum(b) == len(b):
                b[np.random.randint(len(b))] = 0
            elif np.sum(b) == 0:
                b[np.random.randint(len(b))] = 1
    elif dst_type == '2d':
        bitmask = np.ones([sample_size, d])
        for b in bitmask:
            ind = np.random.choice(d, 2, replace=False)
            b[ind] = 0
    elif dst_type == '2d_fix':
        bitmask = np.ones([sample_size, d])
        ind = np.random.choice(d, 2, replace=False)
        bitmask[:, ind] = 0
    else:
        raise NotImplementedError()

    xs, cond = input_format(data, bitmask)
    return data, bitmask, xs, cond, (mu, sigma)


def input_format(data, bitmask):
    # black out missing data
    xs = np.multiply(data, 1 - bitmask)
    obs = np.multiply(data, bitmask)
    # concatenate bitmask with missing data
    cond = np.concatenate((obs, bitmask), axis=1)
    xs = unobserved_format(xs, 1 - bitmask)

    return xs, cond


def main(path=None, dst_type='rand'):
    if path is None:
        home = os.path.expanduser('~')
        path = "{}/data/imputation/".format(home)
        misc.make_path(path)
        path += "imp.p"

    ys, bitmask, xs, cond, dist_data = make_data(102400, 15, dst_type)
    part = xs.shape[0] / 10

    training = xs[:int(5 * part)]
    training_l = cond[:int(5 * part)]

    validation = xs[int(5 * part):int(6 * part)]
    validation_l = cond[int(5 * part):int(6 * part)]

    test = xs[int(6 * part):]
    test_l = cond[int(6 * part):]

    dataset = {
        "train": training, "train_labels": training_l,
        "valid": validation, "valid_labels": validation_l,
        "test": test, "test_labels": test_l,
        'dist_data': dist_data
    }

    p.dump(dataset, open(path, 'wb'))


if __name__ == '__main__':
    main('./imp_rand.p', 'rand')
    main('./imp_2d.p', '2d')
    main('./imp_2d_fix.p', '2d_fix')
