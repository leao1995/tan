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


def input_format(data, bitmask):
    # black out missing data
    xs = np.multiply(data, 1 - bitmask)
    obs = np.multiply(data, bitmask)
    # concatenate bitmask with missing data
    cond = np.concatenate((obs, bitmask), axis=1)
    xs = unobserved_format(xs, 1 - bitmask)

    return xs, cond


def make_data(sample_size, d, num_modes):
    all_data = []
    dist_data = []
    for _ in range(num_modes):
        v = np.random.rand(d, d)
        sigma = np.matmul(v, np.transpose(v))
        mu = np.random.rand(d)
        data = np.random.multivariate_normal(mu, sigma, sample_size)
        all_data.append(data)
        dist_data.append((mu, sigma))
    all_data = np.stack(all_data, axis=-1)  # [N,d,m]
    inds = np.random.choice(num_modes, [sample_size])
    sel = np.zeros([sample_size, 1, num_modes])
    for i, ind in enumerate(inds):
        sel[i, :, ind] = 1.
    x = np.sum(all_data * sel, axis=-1)
    bitmask = np.ones([sample_size, d])
    for b in bitmask:
        ind = np.random.choice(d, 2, replace=False)
        b[ind] = 0
    xu, cond = input_format(x, bitmask)

    return x, bitmask, xu, cond, dist_data


def main(path, dimension, num_modes):
    x, bitmask, xs, cond, dist_data = make_data(102400, dimension, num_modes)
    part = x.shape[0] / 10

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
        'dist_data': dist_data,
        'x': x,
        'bitmask': bitmask
    }

    p.dump(dataset, open(path, 'wb'))


if __name__ == '__main__':
    # main('./dataset/synthetic/imp_m2_2d_4d.p', 4, 2)
    main('./dataset/synthetic/imp_m4_2d_4d.p', 4, 4)
