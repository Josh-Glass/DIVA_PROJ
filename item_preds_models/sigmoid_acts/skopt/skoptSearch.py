import time
start = time.time()
import revised_ItemPred_sigmoid
import skopt_simutils
import matplotlib.pyplot as plt 
import numpy as np


tested_model = 'Reviseditempreds'

space = {
    "learn_rate": {
        "range": [0.5, 1.5],
        "type": "Real"
    },
    "num_hidden_nodes": {
        "range": [10, 15],
        "type": "Integer"
    },
    "weight_range": {
        "range": [2.0, 3.05],
        "type": "Real"
    },
    "beta": {
        "range": [400.0, 500.0],
        "type": "Integer"
    },
}


print('Search Started')

objective_func = revised_ItemPred_sigmoid.get_fit
best_params = skopt_simutils.skopt_search(objective_func=objective_func, space=space, iters= 100, inits=100, plot_results=False)


print('Search Complete!')
print('Best Parameters: ', best_params, '\n----------------------------------')
print('Hyperparameter search took {0:0.1f} seconds'.format(time.time() - start))