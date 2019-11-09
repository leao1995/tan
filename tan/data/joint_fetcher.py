import numpy as np
from PIL import Image
from ..utils import misc


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

def get_partitions(data):
    x = []
    c = []
    N, d = data.shape
    prt = np.random.randint(0, d, [N,d])
    for i in range(d):
        b1 = prt != i
        b2 = (prt > i) * -1
        xs = np.multiply(data, 1-b1)
        xs = unobserved_format(xs, 1-b1)
        xc = np.multiply(data, b1+b2)
        cs = np.concatenate([xc,b1,b2])
        x.append(xs)
        c.append(cs)
    x = np.stack(x)
    c = np.stack(c)
    x = np.reshape(x, [N*d, d])
    c = np.reshape(c, [N*d, d*3])

    return x, c    

class BatchFetcher:

    stats = (None, None)

    def __init__(self, *datasets, **kwargs):
        assert len(datasets) == 1
        self._datasets = datasets[0]
        self.ndatasets = 2
        self._N = datasets[0].shape[0]
        self._perm = np.random.permutation(self._N)
        self._curri = 0
        self._standardize = misc.get_default(kwargs, 'standardize', False)
        self._noise_std = misc.get_default(kwargs, 'noise_std', 0.0)
        self._loop_around = misc.get_default(kwargs, 'loop_around', True)
        self._global_step = 0
        N, D = datasets[0].shape
        self._shape = [(N, D), (N, D * 2)]
        self.dim = [s[-1] for s in self._shape]

    def reset_index(self):
        self._curri = 0

    def next_batch(self, batch_size):
        assert self._N > batch_size

        curri = self._curri
        if self._loop_around:
            endi = (curri + batch_size) % self._N
        else:
            if curri >= self._N:
                raise IndexError
            endi = np.minimum(curri + batch_size, self._N)
        if endi < curri:  # looped around
            inds = np.concatenate((np.arange(curri, self._N), np.arange(endi)))
        else:
            inds = np.arange(curri, endi)
        self._curri = endi

        if self._loop_around:
            batches = self._datasets[self._perm[inds]]
        else:
            batches = self._datasets[inds]

        # add noise
        if self._noise_std > 0.:
            batches += np.random.randn(*batches.shape) * self._noise_std

        # standardize
        if self._standardize:
            mean, std = BatchFetcher.stats
            batches = (batches - mean) / std

        # get partitions and condtioning
        B,d = batches.shape
        ## 2 partitions
        # batches = np.array(batches, dtype='float32')
        # bitmask = np.random.choice([0, 1], batches.shape, p=[0.5, 0.5])
        # xu, cu = input_format(batches, bitmask)
        # bu = np.zeros_like(cu[:,d:])
        # cu = np.concatenate([cu, bu], axis=-1)
        # xo, co = input_format(batches, 1-bitmask)
        # co[:,:d] = np.ones_like(co[:,:d])* -1
        # bo = co[:,d:]* -1
        # co = np.concatenate([co,bo],axis=-1)
        # x = np.concatenate([xu,xo],axis=1)
        # x = np.reshape(x, [B*2, d])
        # c = np.concatenate([cu,co],axis=-1)
        # c = np.reshape(c, [B*2, d*3])
        ## random partitions  
        batches = np.array(batches, dtype='float32')
        x, c = get_partitions(batches)        

        return x, c


class DatasetFetchers:

    def __init__(self, train, validation, test, **kwargs):
        self.train = BatchFetcher(*train, **kwargs)
        self.validation = BatchFetcher(*validation, **kwargs)
        self.test = BatchFetcher(*test, loop_around=False, **kwargs)

        self.set_stats()

    def set_stats(self):
        mean = self.train._datasets.mean(axis=0)
        std = self.train._datasets.std(axis=0)
        BatchFetcher.stats = (mean, std)

    @staticmethod
    def reverse(samples, standardize):
        if standardize:
            mean, std = BatchFetcher.stats
            return samples * std + mean
        else:
            return samples

    def reset_index(self):
        self.train.reset_index()
        self.validation.reset_index()
        self.test.reset_index()

    @property
    def dim(self):
        return self.train.dim


def generate_fetchers(noise_std, standardize):
    return lambda tr, va, ts: DatasetFetchers(
        tr, va, ts,
        noise_std=noise_std,
        standardize=standardize)

