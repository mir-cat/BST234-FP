import numpy as np
import pandas as pd
import statsmodels.api as sm
import time
import random
import threading

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

    # need to remove X_1 from the logistic regression model
    X_null = X.drop(key, axis=1)
    logreg = sm.Logit(y, X_null)
    fit = logreg.fit()

    fitted = fit.predict(X_null)

    residuals = fitted - y
    # convert to numpy array for matrix calculation speed
    residuals = residuals.as_matrix()

    variance = fitted * (1 - fitted)

    score = score_stat(residuals, variance, key_var)

    return score, residuals, variance

def permute_indices(N, n):
    # N, n : integers
    return [random.randint(0, N-1) for _ in range(n)]

def permute(numperm,index):
        for j in range(numperm):
            pi = permute_indices(NROW, NUM_NONZERO)
            perm_null_residuals = null_residuals[pi]
            perm_null_variance = null_variance[pi]
            perm_scores[j,index-1] = score_stat(perm_null_residuals, perm_null_variance, perm_key_var)
 
if __name__ =='__main__':
    KEY = 'X_1'
    RESPONSE = 'Y'
    NUM_PERMUTATIONS = 1000
    # num threads must go evenly into num_permutations
    NUM_THREADS = 4

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

    # perform permutations in parallel
    t = [None]*NUM_THREADS
    perm_scores = np.ones((NUM_PERMUTATIONS//NUM_THREADS, NUM_THREADS))*-1 
    for i in range(NUM_THREADS):
        #print(i)
        t[i] = threading.Thread(target=permute
                        , args = (NUM_PERMUTATIONS//NUM_THREADS,i))
        t[i].start()
    # let sleep so all results in before printing and returning
    time.sleep(1)
    #print(perm_scores)

    p_value = ((perm_scores > null_score).sum())/(NUM_PERMUTATIONS+1)
    print(p_value)

    time2 = time.process_time()
    print(time2 - time1)
