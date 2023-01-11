import time
start = time.time()
import divaWrap_RespProbMethod_nosofskySuggestion
import simutils
import matplotlib.pyplot as plt 
import numpy as np


tested_model = 'diva_shj'

space = {
    "learn_rate": {
        "range": [0.1, 5],
        "type": "Real"
    },
    "num_hidden_nodes": {
        "range": [1, 25],
        "type": "Integer"
    },
    "weight_range": {
        "range": [.5, 5.5],
        "type": "Real"
    },
    "beta": {
        "range": [0.0, 500.0],
        "type": "Integer"
    },
    "c": {
        "range": [0.0, 500.0],
        "type": "Integer"
    },
    
    
}


print('Search Started')

objective_func = divaWrap_RespProbMethod_nosofskySuggestion.get_fit
best_params = simutils.skopt_search(objective_func=objective_func, space=space, iters= 1000, inits=1000, plot_results=False)


print('Search Complete!')
print('Best Parameters: ', best_params, '\n----------------------------------')
print('Hyperparameter search took {0:0.1f} seconds'.format(time.time() - start))