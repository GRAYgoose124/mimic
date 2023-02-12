import numpy as np


xor_set = [([0.0, 1.0], [1.0]),
           ([1.0, 0.0], [1.0]),
           ([1.0, 1.0], [0.0]),
           ([0.0, 0.0], [0.0])]


def rand_dist(value, error=0.33):
    return value + np.random.uniform(-error, error, len(value))


def vary(ds, f=rand_dist, variance=0.1):
    """ Vary a dataset by up to 10%. """
    for inp, outp in ds:
        inp, outp = np.array(inp), np.array(outp)

        # if np.random.uniform(0, 1) > 0.5:
        inp = f(inp, error=variance)
        # else:
        #     outp = f(outp, error=variance)

        yield inp, outp