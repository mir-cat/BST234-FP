import numpy as np
import pandas as pd
import statsmodels.api as sm
import time
import random
from multiprocessing import Pool, TimeoutError
import os
from math import isclose
from custom_sample import sample

def import_data(path, response, key):
    # path, response, key: string

    # numpy import is real slow, even compared to pandas import with conversion
    data = pd.read_csv(path).as_matrix()

    # Manual indexing in numpy is super fast compared to pandas
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

def permute_indices(N, n):
    # N, n : integers
    # return random.sample(range(N), n)
    return sample(N, n)

def g(_):
    pi = permute_indices(NROW, NUM_NONZERO)
    s = score_stat(null_residuals[pi], null_variance[pi], perm_key_var)
    return s

if __name__ =='__main__':
    state = np.random.RandomState()

    KEY = 'X_1'
    RESPONSE = 'Y'
    NUM_PERMUTATIONS = 10**4

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

    with Pool(processes=os.cpu_count()-1) as pool:
        perm_scores = pool.imap_unordered(g, range(NUM_PERMUTATIONS))
        # perm_scores = map(g, range(NUM_PERMUTATIONS))
        p_value = sum([(s > null_score) or (s < -1*null_score) for s in perm_scores])/NUM_PERMUTATIONS

    print(p_value)
    time2 = time.time()
    print(time2 - time1)
