import numpy as np
import pandas as pd
import statsmodels.api as sm
import time
import random
from multiprocessing import Pool, TimeoutError
import os
from math import isclose

def import_data(path, response, key):
    # path, response, key: string

    # numpy import is real slow, even compared to pandas import with conversion
    data = pd.read_csv(path).as_matrix()

    # Manual indexing in numpy is super fast compared to pandas selection
    r = data[:, 0].copy()
    k = data[:, 6].copy()
    data = data[:, 1:6]

    # add constant
    data = np.column_stack((data, np.ones((data.shape[0], 1))))

    return data, r, k

def score_stat(residuals, variance, key_var):
    # residuals, variance, key_var : 1d np arrays
    key_var_sq = key_var**2

    score_numerator = (residuals.transpose().dot(key_var))**2
    score_denominator = key_var_sq.transpose().dot(variance)
    score = score_numerator/score_denominator

    return score

def null_score_stat(X, y, key_var):
    # X : 2d np array
    # y : 1d np array 0-1-2
    # key_var : 1d np array

    # key_var = X[key].copy()

    logreg = sm.Logit(y, X)
    fit = logreg.fit()

    fitted = fit.predict(X)

    residuals = y - fitted

    variance = fitted * (1 - fitted)

    score = score_stat(residuals, variance, key_var)

    return score, residuals, variance

def g(_):

    # sample random set of indices.
    # Faster than numpy or random

    # Generate and sort random ints.
    ints = state.randint(0, NROW, int(1.02 * NUM_NONZERO))
    ints.sort()

    # Take uniques by grabbing locations array[N] != array[N - 1].
    # We prepend 'True' so as to not throw away the first value and skew the
    # sampling distribution
    uniques = ints[
        np.concatenate(
            (np.array([True]), ints[1:] != ints[:-1])
        )
    ]

    # If the first pass does not return a list of length >= NUM_NONZERO
    while len(uniques) < NUM_NONZERO:

        ints = state.randint(0, NROW, NUM_NONZERO-len(uniques))
        ints.sort()

        more_uniques = ints[
            np.concatenate(
                (np.array([True]), ints[1:] != ints[:-1])
            )
        ]

        uniques = np.concatenate(
            (uniques, more_uniques)
        )

    np.random.shuffle(uniques)

    # In case we generated more than we need.
    uniques = uniques[:NUM_NONZERO]

    # Score Stat calculation
    key_var = perm_key_var
    key_var_sq = key_var**2
    residuals = null_residuals[uniques]
    variance  = null_variance[uniques]

    score_numerator = (residuals.dot(key_var))**2
    score_denominator = key_var_sq.dot(variance)
    score = score_numerator/score_denominator

    return score

if __name__ =='__main__':
    state = np.random.RandomState()

    KEY = 'X_1'
    RESPONSE = 'Y'
    NUM_PERMUTATIONS = 10**7

    data, response, key_var = import_data('data/data.csv', RESPONSE, KEY)
    NROW = len(response)

    NUM_ONES = sum(key_var == 1)
    NUM_TWOS = sum(key_var == 2)
    NUM_NONZERO = NUM_ONES + NUM_TWOS

    # Logistic Regression Score Test
    time1 = time.time()

    null_score, null_residuals, null_variance = null_score_stat(data, response, key_var)

    if not isclose(null_score, 2.2678647166344645):
        raise AssertionError('Observed Value of Score is Wrong!')

    # Instead of permuting the entire column, which is mostly 0s,
    # we permute the indices for 1s and 2s, and fill in the column
    perm_key_var = np.concatenate((np.ones(NUM_ONES), np.ones(NUM_TWOS) + 1))

    with Pool(processes=os.cpu_count()) as pool:
        perm_scores = pool.imap_unordered(g, range(NUM_PERMUTATIONS))
        # perm_scores = map(g, range(NUM_PERMUTATIONS))
        p_value = sum([(s > null_score) for s in perm_scores])/(NUM_PERMUTATIONS + 1)

    print(p_value)
    time2 = time.time()
    print(time2 - time1)
