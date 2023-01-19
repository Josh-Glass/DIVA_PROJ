import csv 
import numpy as np
import itertools
import sys
sys.path.append('../models')

from models import divaWrap_RespProbMethod_nosofskySuggestion  

print(sys.path)


exit()
def grid_search(space, population_size = 10):



    granularity = population_size // len(space)

    param_space = []
    for param in space:
        if space[param]['type'] == 'Real':
            param_space.append(
                np.linspace(space[param]['range'][0], space[param]['range'][1], granularity)
            )
        elif space[param]['type'] == 'Integer':
            param_space.append(
                np.arange(
                    space[param]['range'][0], space[param]['range'][1], 
                    (space[param]['range'][1]  - space[param]['range'][0]) // granularity) # <-- step size
            )
    param_space = list(itertools.product(*param_space))

    return param_space





def run_parallel_grid_search(core_id, hyperparameter_batch):

    results_column_names = ['learn_rate', 'num_hidden_nodes', 'weight_range', 'beta', 'c', 'error']

    ## make results file
    with open('logs/' + str(core_id) + '.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(results_column_names)


    ## iterate through HP combos
    for hps in hyperparameter_batch:
        model_settings = {
            'learn_rate': hps[0],
            'num_hidden_nodes': hps[1],
            'weight_range': hps[2],
            'beta': hps[3],
            'c': hps[4],
        }

        errors = []
        error = divaWrap_RespProbMethod_nosofskySuggestion.get_fit(hps[0],hps[1], hps[2], hps[3], hps[4],)
        errors.append(error)

        ## save results for each dataset averaged across inits
        with open('logs/' + str(core_id) + '.csv', 'a') as file:
            writer = csv.writer(file)
            for val in errors:
                writer.writerow([ 
                    model_settings['learn_rate'], 
                    model_settings['num_hidden_nodes'], 
                    model_settings['weight_range'], 
                    model_settings['beta'], 
                    model_settings['c'],
                    val,
                ])