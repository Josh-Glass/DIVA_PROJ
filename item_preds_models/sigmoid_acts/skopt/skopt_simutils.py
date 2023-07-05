import random
import numpy as np 
import pandas as pd 
import time

def split_data(dataset_id):

    # np.random.seed(0)
    data = pd.read_csv(dataset_id, delimiter = ',').sample(frac=1)

    train = data.values[:round(data.shape[0] * .7), :]
    test = data.values[round(data.shape[0] * .7):round(data.shape[0] * .9), :]
    validation = data.values[round(data.shape[0] * .9):, :]

    class_labels = np.unique(data.values[:,-1])

    return [train, test, validation, class_labels]


def random_search(objective_func, space, iters = 1, population_size = 1, print_iter_score = False):
    import itertools
    import random
    import multiprocessing as mp

    best_score = 1
    best_params = None
    for i in range(iters): 
        param_space = []
        for param in space:
            if space[param]['type'] == 'Real':
                param_space.append(
                    np.random.uniform(space[param]['range'][0], space[param]['range'][1], population_size)
                )
            elif space[param]['type'] == 'Integer':
                param_space.append(
                    np.random.randint(space[param]['range'][0], space[param]['range'][1], population_size)
                )

        for m in range(population_size):
            member_params = {
                param: param_space[p][m]
                for p, param in enumerate(space)
            }

            model_error = objective_func(**member_params)

            if model_error < best_score:
                best_score = model_error
                best_params = member_params   

        if print_iter_score == True:
            print(best_score)

    return best_params



#def grid_search(objective_func, space, population_size = 10):
def grid_search(space, population_size = 10):

    import itertools

    best_score = 1
    best_params = None

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



    '''for m in range(population_size):
        member_params = {
            param: param_space[m][p]
            for p, param in enumerate(space)
        }

        model_error = objective_func(**member_params)

        if model_error < best_score:
            best_score = model_error
            best_params = member_params'''

               

    return param_space





def skopt_search(objective_func, space, run = 1, iters = 1, inits = 1, plot_results = False):
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    from skopt import gp_minimize
    import sys, os
    sys.path.append(os.path.join(sys.path[0],'skopt'))
    import skoptSearch as sks



    timestr = time.strftime("%m%d-%H%M")

    


    #with open(f'logs/skopt-{tested_model}-{timestr}.txt', 'a') as file:
       # pass

    param_space = []
    for parameter in space:
        if space[parameter]['type'] == 'Real':
            param_space.append(
                Real(space[parameter]['range'][0], space[parameter]['range'][1], name = parameter)
            )
        elif space[parameter]['type'] == 'Integer':
            param_space.append(
                Integer(space[parameter]['range'][0], space[parameter]['range'][1], name = parameter)
            )
        else: 'parameter type not found'



    @use_named_args(param_space)
    def objective(**params):
        return objective_func(**params)



    res_gp = gp_minimize(objective, param_space, n_calls = iters, n_random_starts = inits)

    # for error_val, hps in res_gp.x_iters:
    #     print(error_val, '|', {hp_id: hp_val for hp_id, hp_val in zip(space, hps)})
        
    #with open(f'logs/skopt-{tested_model}-{timestr}.txt', 'a') as file:
        #for error_val, hps in zip(res_gp.func_vals, res_gp.x_iters):
           # print(error_val, '|', {hp_id: hp_val for hp_id, hp_val in zip(space, hps)}, file = file)

    

    if plot_results == True:
        import matplotlib.pyplot as plt
        from skopt.plots import plot_convergence
        plot_convergence(res_gp)
        plt.savefig(f'results_skopt_fig--{sks.tested_model}--{timestr}.png')

    
    df = pd.DataFrame({'error': res_gp.func_vals,
    '[learn_rate, num_hidden_nodes, weight_range, beta]': res_gp.x_iters,
    })
    df.to_csv(f'logs/skopt-{sks.tested_model}-{timestr}.csv')
   
    return (res_gp.fun, '|', {hp: val for hp, val in zip(space, res_gp.x)})