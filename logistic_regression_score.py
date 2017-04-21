import numpy as np
import pandas as pd
import statsmodels.api as sm
import time


def import_data(path):
    # path : string
    data = pd.read_csv(path)
    data = sm.add_constant(data)
    return data

def score_stat(residuals, variance, key_var):
    # residuals, variance, key_var : pandas Series

    resids = np.array(residuals)
    vars = np.array(variance)
    keyv = np.array(key_var)

    time1 = time.process_time()

    keyv_sq = keyv**2

    score_numerator = resids.transpose().dot(keyv)**2
    score_denominator = keyv_sq.transpose().dot(vars)
    score = score_numerator/score_denominator

    time2 = time.process_time()
    print(time2 - time1)

    return score


def score_statold(residuals, variance, key_var):
    # residuals, variance, key_var : pandas Series

    time1 = time.process_time()

    key_var_sq = key_var ** 2

    score_numerator = residuals.transpose().dot(key_var) ** 2
    score_denominator = key_var_sq.transpose().dot(variance)
    score = score_numerator / score_denominator

    time2 = time.process_time()
    print(time2 - time1)

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
    variance = fitted * (1 - fitted)

    score = score_stat(residuals, variance, key_var)

    return score, residuals, variance


def permute_column(data, col):
    # data : pandas df
    # col : string

    data[col] = data[col].sample(frac=1, replace=True).reset_index(drop=True)

if __name__ =='__main__':
    KEY = 'X_1'
    RESPONSE = 'Y'
    NUM_PERMUTATIONS = 100

    perm_scores = [False for _ in range(NUM_PERMUTATIONS)]


    data = import_data('data/data.csv')
    nrow = len(data['Y'])
    key_var = data[KEY].copy()


    response = data[RESPONSE]
    data.drop(RESPONSE, 1, inplace=True)

    # Logistic Regression Score Test
    time1 = time.process_time()

    key_var_sq = key_var**2 # 0-1-2 key
    null_score, null_residuals, null_variance = null_score_stat(data, response, KEY)


    # speed this up
    for i in range(NUM_PERMUTATIONS):
        permute_column(data, KEY)
        perm_key_var = data[KEY].copy()
        #print(len(perm_key_var),len(null_residuals))
        #print(perm_key_var[0:50])
        perm_scores[i] = score_stat(null_residuals, null_variance, perm_key_var)

    p_value = sum([(s > null_score) or (s < -1*null_score) for s in perm_scores])/NUM_PERMUTATIONS

    print(p_value)

    time2 = time.process_time()
    print(time2 - time1)
