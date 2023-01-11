import time
start = time.time()
import divaWrap_RespProbMethod_nosofskySuggestion
import simutils
import matplotlib.pyplot as plt 
import numpy as np


tested_model = 'diva_shj'

space = {
    "learn_rate": {
        "range": [1.9, 2.9],
        "type": "Real"
    },
    "num_hidden_nodes": {
        "range": [8, 12],
        "type": "Integer"
    },
    "weight_range": {
        "range": [1.5, 2.5],
        "type": "Real"
    },
    "beta": {
        "range": [400.0, 500.0],
        "type": "Integer"
    },
    "c": {
        "range": [400.0, 500.0],
        "type": "Integer"
    },
    
    
}


print('Search Started')

objective_func = divaWrap_RespProbMethod_nosofskySuggestion.get_fit
best_params = simutils.skopt_search(objective_func=objective_func, space=space, iters= 1500, inits=1500, plot_results=False)


print('Search Complete!')
print('Best Parameters: ', best_params, '\n----------------------------------')
print('Hyperparameter search took {0:0.1f} seconds'.format(time.time() - start))