#%%

import numpy as np
from noisylabeltk.experiment import Experiment
from noisylabeltk.datasets import DatasetLoader
import pygad
import time
from multiprocessing import Pool

import os

host = 'localhost'
port = 27017
database_name = 'fairness'
tags = ['verify inversion']
collection_name = 'genetic_optimization'
n_jobs = 4
device = 'cpu'

if device == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_p_list(privileged_demotion, privileged_promotion, protected_demotion,
    protected_promotion):
    P_list = [np.array([[1 - privileged_demotion, privileged_demotion],
                        [privileged_promotion, 1 - privileged_promotion]]),
              np.array([[1 - protected_demotion, protected_demotion],
                        [protected_promotion, 1 - protected_promotion]])]
    return P_list

def fitness_func(solution, solution_idx):
    dataset_name = 'income'
    robust_method = 'fair-forward'
    auc_prune = 0.8
    target_metric = 'Positives'

    hyperparameters = {
        'num_layers': 3,
        'dropout': 0.2
    }

    for i, size in enumerate([32, 64, 32]):
        hyperparameters['hidden_size_%d' % i] = size

    protected_promotion = solution[0]
    protected_demotion = solution[1]
    privileged_promotion = solution[2]
    privileged_demotion = solution[3]
    p_list = get_p_list(privileged_demotion, privileged_promotion, protected_demotion,protected_promotion)

    parameters = {
        'batch-size': 32,
        'epochs': 10,
        'dataset': dataset_name,
        'model': 'simple-mlp',
        'noise': None,
        'noise-args': None,
        'robust-method': robust_method,
        'loss-args': [p_list],
        'protected-promotion': protected_promotion,
        'protected-demotion': protected_demotion,
        'privileged-promotion': privileged_promotion,
        'privileged-demotion': privileged_demotion,
        'loss-kwargs': None,
    }

    dataset = DatasetLoader(parameters['dataset'], parameters['batch-size'], sensitive_labels=True).load()

    exp = Experiment(dataset['num_features'],
                     dataset['num_classes'],
                     parameters,
                     database_name,
                     collection_name,
                     tags=tags)

    exp.build_model(hyperparameters)
    exp.fit_model(dataset['train'], dataset['validation'], parameters['batch-size'])
    exp.evaluate(dataset['test'], parameters['batch-size'])
    exp.evaluate_discrimination(dataset['test'], parameters['batch-size'])


    auc = float(exp.run_entry['metrics']['AUC_overall'])
    metric = abs(float(exp.run_entry['metrics']['%s_balance' % target_metric]))

    if (auc >= auc_prune and not np.isnan(metric) and metric > 0):
        fitness = fitness = 1/metric
    else:
        fitness = 0

    exp.run_entry['fitness'] = fitness
    exp.persist_metadata()

    return fitness


def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=1/ga_instance.best_solution()[1]))


def fitness_wrapper(solution):
    return fitness_func(solution, 0)


class PooledGA(pygad.GA):

    def cal_pop_fitness(self):
        global pool

        pop_fitness = pool.map(fitness_wrapper, self.population)
        print(pop_fitness)
        pop_fitness = np.array(pop_fitness)
        return pop_fitness

fitness_function = fitness_func

# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
num_generations = 50  # Number of generations.
num_parents_mating = 4  # Number of solutions to be selected as parents in the mating pool.
parent_selection_type = "sss"  # Type of parent selection.
crossover_type = "single_point"  # Type of the crossover operator.
mutation_type = "random"  # Type of the mutation operator.
mutation_percent_genes = 10  # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
keep_parents = -1  # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.
sol_per_pop = 8
num_genes = 4
init_range_low = 0.0
init_range_high = 1.0
initial_population = [ [0.1, 0.01, 0.01, 0.1],
                       [0.2, 0.02, 0.02, 0.2],
                       [0.3, 0.03, 0.03, 0.3],
                       [0.4, 0.04, 0.04, 0.4]]

start_time = time.time()



ga_instance = PooledGA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       num_genes=num_genes,
                       #initial_population=initial_population,
                       sol_per_pop=sol_per_pop,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       keep_parents=keep_parents,
                       on_generation=callback_generation)


with Pool(processes=4) as pool:
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=1/solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

    print("--- %s seconds ---" % (time.time() - start_time))
    # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
    ga_instance.plot_result(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)
