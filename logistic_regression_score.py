import numpy as np
import pandas as pd
import statsmodels.api as sm
import time
import random


def import_data(path):
    # path : string
    data = pd.read_csv(path)
    data = sm.add_constant(data)
    return data

def score_stat(residuals, variance, key_var):
    # residuals, variance, key_var : pandas Series
    key_var_sq = key_var**2

    score_numerator = residuals.transpose().dot(key_var)**2
    score_denominator = key_var_sq.transpose().dot(variance)
    score = score_numerator/score_denominator

    return score

def null_score_stat(X, y, key):
    # X : pandas dataframe
    # y : pandas series 0-1-2
    # key : string

    key_var = X[key].copy()

    logreg = sm.Logit(y, X)
    fit = logreg.fit()

    fitted = fit.predict(X)

    residuals = fitted - y
    # convert to numpy array for matrix calculation speed
    residuals = residuals.as_matrix()

    variance = fitted * (1 - fitted)

    score = score_stat(residuals, variance, key_var)

    return score, residuals, variance

def permute_indices(N, n):
    # N, n : integers
    return [random.randint(0, N-1) for _ in range(n)]
if __name__ =='__main__':
    KEY = 'X_1'
    RESPONSE = 'Y'
    NUM_PERMUTATIONS = 10000

    perm_scores = [False for _ in range(NUM_PERMUTATIONS)]

    data = import_data('data/data.csv')
    NROW = len(data['Y'])
    key_var = data[KEY].copy()

    NUM_ONES = sum(key_var == 1)
    NUM_TWOS = sum(key_var == 2)
    NUM_NONZERO = NUM_ONES + NUM_TWOS

    response = data[RESPONSE]
    data.drop(RESPONSE, 1, inplace=True)

    # Logistic Regression Score Test
    time1 = time.process_time()

    key_var_sq = key_var**2 # 0-1-2 key
    null_score, null_residuals, null_variance = null_score_stat(data, response, KEY)

    # Instead of permuting the entire column, which is mostly 0s,
    # we permute the indices for 1s and 2s, and fill in the column
    perm_key_var = pd.Series(np.concatenate((np.ones(NUM_ONES), np.ones(NUM_TWOS) + 1)))

    # speed this up
    for i in range(NUM_PERMUTATIONS):

        pi = permute_indices(NROW, NUM_NONZERO)

        perm_null_residuals = null_residuals[pi]
        perm_null_variance = null_variance[pi]
        perm_scores[i] = score_stat(perm_null_residuals, perm_null_variance, perm_key_var)

    p_value = sum([(s > null_score) or (s < -1*null_score) for s in perm_scores])/NUM_PERMUTATIONS

    print(p_value)

    time2 = time.process_time()
    print(time2 - time1)
