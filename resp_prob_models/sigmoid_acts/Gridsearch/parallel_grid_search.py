import sys, os
import parallel_grid_search_utils as utils
import multiprocessing as mp
import time

tested_model = 'diva_shj'



space = {
    "learn_rate": {
        "range": [2.0, 2.5],
        "type": "Real"
    },
    "num_hidden_nodes": {
        "range": [8, 12],
        "type": "Integer"
    },
    "weight_range": {
        "range": [2.0, 2.5],
        "type": "Real"
    },
    "beta": {
        "range": [400.0, 500.0],
        "type": "Integer"
    },  
}

num_cores= 3
hp_grid= utils.grid_search(space=space, population_size=8)

##__Break Grid into batches; send each batch to it's own 'process'
batch_size = len(hp_grid) // num_cores


## make a list of "processes"; each one gets their own core
processes = []
for core in range(num_cores):
    processes.append( # <-- map batches to model run function
        mp.Process(
            target = utils.run_parallel_grid_search,
            args = [
                core, # <-- gives each process it's own convenient ID num 
                hp_grid[(core * batch_size):(core * batch_size) + batch_size]
            ],
        )
    )

if __name__ == '__main__':
    #set working directory to to root directory
    print('search Started!')
    
 
    start = time.time()


    mp.freeze_support()

    ##__Start Processes
    for process in processes:
        process.start()

    ##__Wait until they finish
    for process in processes:
        process.join()


    

    print('search ended!')
    print('Hyperparameter search took {0:0.1f} seconds'.format(time.time() - start))




