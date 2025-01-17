import numpy as np
from PIL import Image
from ..utils import misc


def impute(x, c):
    '''
    x: [d]
    c: [2d]
    '''
    d = x.shape[0]
    xo = c[:d]
    bitmask = c[d:]
    sorted_mask = np.sort(1 - bitmask, axis=0)[::-1]
    sorted_mask = sorted_mask != 0
    valid_mask = bitmask != 0
    xu = x.copy()
    xu[~valid_mask] = x[sorted_mask]
    xu[valid_mask] = 0

    x = xu * (1 - bitmask) + xo * bitmask

    return x


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


def batch_resize(data, resize, channels):
    N, d = data.shape
    s = int(np.sqrt(d / channels))
    image = data.reshape([N, s, s, channels])
    image = np.squeeze(image, axis=-1)
    resized_image = []
    for i in range(N):
        img = Image.fromarray(image[i])
        img = img.resize((resize, resize), Image.BILINEAR)
        resized_image.append(np.array(img))
    resized_image = np.stack(resized_image)

    return resized_image.reshape([N, -1])


def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


class BatchFetcher:

    stats = (None, None)

    def __init__(self, *datasets, **kwargs):
        assert len(datasets) == 1
        self._datasets = datasets[0]
        self.ndatasets = 2
        self._N = datasets[0].shape[0]
        self._perm = np.random.permutation(self._N)
        self._curri = 0
        self._logit = misc.get_default(kwargs, 'logit', True)
        self._standardize = misc.get_default(kwargs, 'standardize', False)
        self._noise_std = misc.get_default(kwargs, 'noise_std', 0.0)
        self._missing_prob = misc.get_default(kwargs, 'missing_prob', 0.5)
        self._loop_around = misc.get_default(kwargs, 'loop_around', True)
        self._is_image = misc.get_default(kwargs, 'is_image', False)
        self._channels = misc.get_default(kwargs, 'channels', 1)
        self._resize = misc.get_default(kwargs, 'resize', -1)
        N, D = datasets[0].shape
        if self._is_image and self._resize > 0:
            D = self._resize ** 2
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

        # preprocess image data
        if self._is_image:
            if self._resize > 0:
                batches = batch_resize(batches, self._resize, self._channels)
            batches = np.array(batches, dtype='float32')
            batches += np.random.rand(*batches.shape)
            batches /= 256.
            if self._logit:
                data_constraint = 0.9
                pre_logit_scale = np.log(data_constraint)
                pre_logit_scale -= np.log(1. - data_constraint)
                pre_logit_scale = np.array(pre_logit_scale, dtype=np.float32)
                logit_x_in = 2. * batches  # [0, 2]
                logit_x_in -= 1.  # [-1, 1]
                logit_x_in *= data_constraint  # [-.9, .9]
                logit_x_in += 1.  # [.1, 1.9]
                logit_x_in /= 2.  # [.05, .95]
                # logit the data
                logit_x_in = np.log(logit_x_in) - np.log(1. - logit_x_in)
                batches = logit_x_in
        else:
            # add noise
            if self._noise_std > 0.:
                batches += np.random.randn(*batches.shape) * self._noise_std

            # standardize
            if self._standardize:
                mean, std = BatchFetcher.stats
                batches = (batches - mean) / std

        # get unobserved and condtioning
        batches = np.array(batches, dtype='float32')
        bitmask = np.random.choice(
            [0, 1], batches.shape, p=[self._missing_prob, 1 - self._missing_prob])

        # make sure all rows have at least 1 zero
        zerod = np.where(np.sum(1 - bitmask, axis=1) == 0)[0]
        bitmask[zerod, np.random.choice(batches.shape[1], zerod.shape)] = 0

        x, c = input_format(batches, bitmask)
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
    def reverse(samples, is_image, standardize, logit):
        if is_image:
            if logit:
                samples = sigmoid(samples)
            return samples * 255.
        else:
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


def generate_fetchers(is_image, channels, resize, noise_std, standardize, logit):
    return lambda tr, va, ts: DatasetFetchers(
        tr, va, ts,
        is_image=is_image, channels=channels,
        resize=resize, noise_std=noise_std,
        standardize=standardize, logit=logit)


def main():
    import pickle
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    fname = './data/mnist/mnist.p'
    dataset = pickle.load(open(fname, 'rb'))
    train = (dataset['train'],)
    valid = (dataset['valid'],)
    test = (dataset['test'],)
    fetcher = DatasetFetchers(train, valid, test, is_image=True, resize=-1)
    x, c = fetcher.train.next_batch(10)
    d = x.shape[1]
    xc = c[:, :d]
    im = (xc[0] * 255).astype('uint8')
    im = im.reshape((28, 28))
    plt.imsave('./test.png', im)
