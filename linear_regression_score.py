import numpy as np
import pandas as pd
import statsmodels.api as sm
import time
from logistic_regression_score import import_data, permute_column

# this is just for git
def solve_linear_regression(X, y):

    # X : pandas dataframe
    # y : pandas Series
    inv = np.linalg.inv(X.transpose().dot(X))
    beta = inv.dot(X.transpose()).dot(y)

    return beta


if __name__ =='__main__':
    KEY = 'X_1'
    RESPONSE = 'Y'
    NUM_PERMUTATIONS = 1000

    perm_betas = [False for _ in range(NUM_PERMUTATIONS)]


    data = import_data('data/data.csv')
    nrow = len(data['Y'])
    key_var = data[KEY].copy()

    response = data[RESPONSE]
    data.drop(RESPONSE, 1, inplace=True)

    key_index = [i for i in range(len(data.columns)) if data.columns[i] == KEY]
    print(data.columns[key_index])

    # Linear Regression Score Test
    time1 = time.process_time()

    obs_beta = solve_linear_regression(data, response)
    key_beta = obs_beta[key_index]

    for i in range(NUM_PERMUTATIONS):
        permute_column(data, KEY)
        perm_beta = solve_linear_regression(data, response)
        perm_betas[i] = perm_beta[key_index]


    time2 = time.process_time()
    print(time2 - time1)

    if key_beta < 0:
        p_value = sum([(b < obs_beta) or (b > -1*obs_beta) for b in perm_betas])/NUM_PERMUTATIONS
    else :
        p_value = sum([(b > obs_beta) or (b < -1*obs_beta) for b in perm_betas])/NUM_PERMUTATIONS

    print(p_value)
