import numpy as np
from numpy.random import RandomState

state = RandomState()


# N = 100000
# K = 2798

# NOTES:

# 1. I'm reasonably confident that these both produce uniform random unique
#    samples, but some amount of statistical verification of that fact is
#    probably warranted.
#
# 2. On my machine, for N and K noted above, fast_sample is about 4.5x faster
#    than using random.sample with a pre-computed index list. fastest_sample is
#    about 9x faster.

# $ ipython -i fast_rand.py

# In [1]: indices = list(range(N))

# In [2]: import random

# In [3]: %timeit random.sample(indices, K)
# 1000 loops, best of 3: 1.15 ms per loop

# In [4]: %timeit fast_sample(N, K)
# 1000 loops, best of 3: 279 µs per loop

# In [5]: %timeit fastest_sample(N, K)
# 10000 loops, best of 3: 140 µs per loop


def fastest_sample(n, k, state=state, extra=1.02):
    """
    Equivalent to random.sample(range(n), k).

    Works by generating k + buffer (currently 2%) random samples in one shot,
    then taking the first k uniques. This will fail if k is large enough that
    we have more than 2% collisions.

    You could probably do some interesting statistics to decide a good buffer
    value (and/or a good point at which this is no longer a reasonable
    algorithm).
    """
    # Uniquify in numpy directly. This is using an O(n * log (n)) method, but
    # it's all in numpy so it has much lower constants than the set-based
    # method below.

    def _fastest_sample(n, k, state=state, extra=1.02):
        # Generate and sort random ints.
        ints = state.randint(0, n, int(extra * k))
        ints.sort()

        # Take uniques by grabbing locations N where array[N] != array[N - 1].
        # We prepend 'True' so as to not throw away the first value and skew the
        # random distribution
        uniques = ints[
            np.concatenate(
                (np.array([True]), ints[1:] != ints[:-1])
            )
        ]

        return uniques


    uniques = _fastest_sample(n, k , state)

    while len(uniques) < k:
        uniques = np.concatenate(
            (uniques, _fastest_sample(n, k-len(uniques), state, extra=1.0))
        )

    # because our key-variable is ordered, we cannot have ordered indices as that
    # would bias the score statistic calculation and result in undersampling
    # at the extreme tail

    np.random.shuffle(uniques)

    return uniques[:k]

def fast_sample(n, k, state=state, extra=1.02):
    """
    Equivalent to random.sample(range(n), k).

    Works by generating k + buffer (currently 2%) random samples in one shot,
    then taking the first k uniques. This will fail if k is large enough that
    we have more than 2% collisions.

    You could probably do some interesting statistics to decide a good buffer
    value (and/or a good point at which this is no longer a reasonable
    algorithm).
    """
    # Uniquify using Python's built-in set. I'd expect this to scale better
    # than the sorting method below as N and K get larger.
    s = None

    while s is None:
        try:
            s = np.fromiter(
                set(state.randint(0, n, int(extra * k))),
                dtype='int64',
                count=k,
            )
        except Exception:
            pass

    return s

def sample(n, k, state=state, extra=1.02):
    s = fastest_sample(n, k, state, extra)

    while(len(s) != k):
        s = fastest_sample(n, k, state, extra)

    return(s)
