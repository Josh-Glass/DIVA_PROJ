import time
start = time.time()
import revised_cparam_sigmoid
import skopt_simutils
import matplotlib.pyplot as plt 
import numpy as np


tested_model = 'Revisedcparam'

space = {
    "learn_rate": {
        "range": [0.5, 1.5],
        "type": "Real"
    },
    "num_hidden_nodes": {
        "range": [5, 12],
        "type": "Integer"
    },
    "weight_range": {
        "range": [2.0, 3.0],
        "type": "Real"
    },
    "beta": {
        "range": [250.0, 450.0],
        "type": "Integer"
    },
    "c": {
        "range": [350.0, 450.0],
        "type": "Integer"
    },
    
    
}


print('Search Started')

objective_func = revised_cparam_sigmoid.get_fit
best_params = skopt_simutils.skopt_search(objective_func=objective_func, space=space, iters= 100, inits=100, plot_results=False)


print('Search Complete!')
print('Best Parameters: ', best_params, '\n----------------------------------')
print('Hyperparameter search took {0:0.1f} seconds'.format(time.time() - start))