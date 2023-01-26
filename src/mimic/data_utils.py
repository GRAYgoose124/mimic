import numpy as np


xor_set = [([0.0, 1.0], [1.0]),
           ([1.0, 0.0], [1.0]),
           ([1.0, 1.0], [0.0]),
           ([0.0, 0.0], [0.0])]


def rand_dist(value, error=0.1):
    return value + np.random.uniform(-error, error, len(value))


def vary(ds, f=rand_dist):
    """ Vary a dataset by up to 10%. """
    for inp, outp in ds:
        inp, outp = np.array(inp), np.array(outp)

        if np.random.uniform(0, 1) > 0.5:
            inp = f(inp)
        else:
            outp = f(outp)

        yield inp, outp