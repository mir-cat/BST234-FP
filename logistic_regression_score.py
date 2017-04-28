import numpy as np
import pandas as pd
import statsmodels.api as sm
import time
import random
from multiprocessing import Pool, TimeoutError
import os

def import_data(path, response, key):
    # path : string

    # numpy import is real slow, even compared to pandas import with conversion
    data = pd.read_csv(path).as_matrix()

    # Manual indexing in numpy is super fast compared to pandas
    r = data[:, 0]
    k = data[:, 6]
    data = data[:, 1:5]
    data = np.column_stack((data, np.ones((data.shape[0], 1)))) #sm.add_constant(data, has_constant='add')
    return data, r, k

def score_stat(residuals, variance, key_var):
    # residuals, variance, key_var : pandas Series
    key_var_sq = key_var**2

    score_numerator = (residuals.transpose().dot(key_var))**2
    score_denominator = key_var_sq.transpose().dot(variance)
    score = score_numerator/score_denominator

    return score

def null_score_stat(X, y, key_var):
    # X : pandas dataframe
    # y : pandas series 0-1-2
    # key : string

    # key_var = X[key].copy()

    logreg = sm.Logit(y, X)
    fit = logreg.fit()

    fitted = fit.predict(X)

    residuals = y - fitted
    # convert to numpy array for matrix calculation speed
    # residuals = residuals.as_matrix()

    variance = fitted * (1 - fitted)

    score = score_stat(residuals, variance, key_var)

    return score, residuals, variance

def permute_indices(N, n):
    # N, n : integers
    return random.sample(range(N), n)

def g(_):
    pi = permute_indices(NROW, NUM_NONZERO)
    s = score_stat(null_residuals[pi], null_variance[pi], perm_key_var)
    return s

if __name__ =='__main__':
    KEY = 'X_1'
    RESPONSE = 'Y'
    NUM_PERMUTATIONS = 10000

    data, response, key_var = import_data('data/data.csv', RESPONSE, KEY)
    NROW = len(response)

    NUM_ONES = sum(key_var == 1)
    NUM_TWOS = sum(key_var == 2)
    NUM_NONZERO = NUM_ONES + NUM_TWOS

    # Logistic Regression Score Test
    time1 = time.time()

    null_score, null_residuals, null_variance = null_score_stat(data, response, key_var)

    # Instead of permuting the entire column, which is mostly 0s,
    # we permute the indices for 1s and 2s, and fill in the column
    perm_key_var = np.concatenate((np.ones(NUM_ONES), np.ones(NUM_TWOS) + 1))

    perm_scores = [False for _ in range(NUM_PERMUTATIONS)]

    # speed this up
    # for i in range(NUM_PERMUTATIONS):
    #     perm_scores[i] = g(True)
    #
    with Pool(processes=os.cpu_count()//2) as pool:
        perm_scores = pool.imap_unordered(g, range(NUM_PERMUTATIONS))
        # perm_scores = map(g, range(NUM_PERMUTATIONS))
        p_value = sum([(s > null_score) or (s < -1*null_score) for s in perm_scores])/NUM_PERMUTATIONS


    # perm_scores = map(g, range(NUM_PERMUTATIONS))

    # p_value = sum([(s > null_score) or (s < -1*null_score) for s in perm_scores])/NUM_PERMUTATIONS

    print(p_value)

    time2 = time.time()
    print(time2 - time1)
