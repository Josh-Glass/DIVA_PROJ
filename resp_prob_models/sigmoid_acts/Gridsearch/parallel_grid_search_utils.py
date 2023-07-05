import csv 
import numpy as np
import itertools
import sys, os
import pandas as pd
import time
#sys.path.append(os.path.join(sys.path[0],'..'))

import divaWrap_RespProb_sigmoid  




def grid_search(space, population_size):



    granularity = population_size // len(space)
    

    param_space = []
    for param in space:
        
        if space[param]['type'] == 'Real':
            param_space.append(
                np.linspace(space[param]['range'][0], space[param]['range'][1], granularity)
            )
        elif space[param]['type'] == 'Integer':
            if ((space[param]['range'][1]  - space[param]['range'][0])//granularity) == 0:#<-- making sure step size is never zero
                param_space.append(np.arange(space[param]['range'][0], space[param]['range'][1], 1))
            else:
                param_space.append(np.arange(space[param]['range'][0], space[param]['range'][1], (space[param]['range'][1]  - space[param]['range'][0]) // granularity)) # <-- step size
    
    param_space = list(itertools.product(*param_space))
    #print(len(param_space))
    #print(param_space)

    return param_space




def run_parallel_grid_search(core_id, hyperparameter_batch):

    timestr = time.strftime("%m%d-%H%M")
    results_column_names = ['learn_rate', 'num_hidden_nodes', 'weight_range', 'beta', 'error']
    

    ## make results file
    with open('logs/' + str(core_id) +'GS' + 'respProb.csv', 'w') as file:
        writer = csv.writer(file, lineterminator = '\n')
        writer.writerow(results_column_names)
    

    ## iterate through HP combos

    for hps in hyperparameter_batch:
        

        model_settings = {
            'learn_rate': hps[0],
            'num_hidden_nodes': hps[1],
            'weight_range': hps[2],
            'beta': hps[3],
        }
        
        errors = []
        error = divaWrap_RespProb_sigmoid.get_fit(hps[0],hps[1], hps[2], hps[3],)
        errors.append(error)
        print(errors,'made it here too')

         
       
        ## save results for each dataset averaged across inits
        with open('logs/' + str(core_id) +'GS' + 'respProb.csv', 'a') as file:
            writer = csv.writer(file,lineterminator = '\n')
            for val in errors:
                writer.writerow([ 
                    model_settings['learn_rate'], 
                    model_settings['num_hidden_nodes'], 
                    model_settings['weight_range'], 
                    model_settings['beta'], 
                    val,
                ])