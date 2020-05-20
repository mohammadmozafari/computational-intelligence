"""
Notes:

- Main function is the start of the program

- There are 2 functions that can be called one 
        for solving tsp and the other for knapsack

- Functions that have prefix 'ks' are related 
        to knapsack and the ones with prefix 'tsp'
        are related to tsp problem
"""


import random as rnd
import numpy as np

def main():
    solve_tsp()
    # solve_knapsack()

def solve_knapsack():
    """
    Defines parameters for knapsack problem and
    solves knapsack using an evolutionary algorithm.
    """
    pop_size = 300
    mu = 100
    lam = 1000
    mutation_rate = 0.4
    iterations = 1000
    
    env = ks_extract_env('knapsack_3.txt')
    population = ks_init_pop(pop_size, int(env[0, 0]))
    for i in range(iterations):
        parents = ks_select_parents(population, env, mu, ks_fit)
        children = ks_create_children(parents, pop_size, ks_crossover, ks_mutate, mutation_rate)
        population = children
        check_population(population, i, ks_fit, env, 1)

def solve_tsp():
    """
    Defines parameters for tsp problem and
    solves tsp using an evolutionary algorithm.
    """
    pop_size = 1
    mu = 1
    lam = 1
    iterations = 1000000

    env = tsp_extract_env('tsp_data.txt')
    population = tsp_init_pop(pop_size, env.shape[0])
    for i in range(iterations):
        parents = population
        children = tsp_create_children(parents, tsp_mutate)
        population = tsp_select_children(parents, children, tsp_fit, env, lam)
        if (i + 1) % 1000 == 0:
            check_population(population, i, tsp_fit, env, 0)

def check_population(population, i, fitness_fn, env, mode):
    lens = []
    for member in population:
        lens.append(fitness_fn(member, env))
    highest_fitness = max(lens)
    best = population[lens.index(highest_fitness)]
    
    if mode == 0:
        print('generation ', i + 1)
        # print('   best answer:', best)
        print('   fitness:', highest_fitness)
        print('   length:', 1 / highest_fitness)
    else:
        values = env[1:, 0]
        weights = env[1:, 1]
        print('generation ', i + 1)
        print('   best answer:', best)
        print('   fitness:', highest_fitness)
        print('   total value:', best @ values)
        print('   total wight:', best @ weights)
        print('   capacity:', env[0, 1])

# functions related to knapsack problem
# =====================================

def ks_extract_env(file):
    """
    Reads the given file and builds a 2d list
    containing the weight and value of each item.
    """
    env = []
    with open(file, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            split = line.split(' ')
            env.append([float(split[0]), float(split[1])])
    return np.array(env)

def ks_init_pop(size, num_items):
    """
    Creates random initial population for knapsack problem.
    Each member is binary string that indicates
    whether each item is included or not.
    """
    initial_population = []
    for i in range(size):
        choice = np.random.uniform(0, 1, num_items)
        choice = (choice < 0.01) * 1
        initial_population.append(choice)
    return np.array(initial_population)

def ks_fit(choice, env):
    """
    Computes the the total value of items.
    Computes the difference between bag space and total weight.
    Uses these two to compute fitness
    """
    values = env[1:, 0]
    weights = env[1:, 1]
    cap = env[0, 1]
    total_value = choice @ values
    total_weight = choice @ weights
    if total_weight > cap:
        fitness = 0
    else:
        fitness = total_value
    return fitness

def ks_select_parents(choices, env, mu, fit_fn):
    """
    Selects parents according to their fitness.
    """
    fitness = []
    for choice in choices:
        fitness.append(fit_fn(choice, env))
    total = sum(fitness)
    weights = [x / (total + 1e-10) for x in fitness]
    parents = rnd.choices(choices, weights, k=mu)
    return parents

def ks_crossover(mum, dad):
    """
    Creates children 2 children from prents  
    """
    length = len(mum)
    rand = rnd.randint(1, length - 1)
    child1 = mum.copy()
    child2 = dad.copy()
    child1[rand:] = dad[rand:].copy()
    child2[rand:] = mum[rand:].copy()
    return child1, child2

def ks_mutate(choice):
    """
    Mutates a choice by randomly change 1 gene from 0 to 1 or vice versa.
    """
    length = len(choice)
    rand = rnd.randint(0, length - 1)
    new_choice = choice.copy()
    new_choice[rand] = 1 - new_choice[rand]
    return new_choice

def ks_create_children(parents, pop_size, crossover_fn, mutate_fn, mutation_rate):
    """
    Each 2 parents create 2 children.
    """
    children = []
    n = len(parents)
    for i in range(pop_size):
        mum = parents[rnd.randint(0, n - 1)]
        dad = parents[rnd.randint(0, n - 1)]
        child1, child2 = crossover_fn(mum, dad)
        if rnd.uniform(0, 1) < mutation_rate:
            child1 = mutate_fn(child1)
        if rnd.uniform(0, 1) < mutation_rate:
            child2 = mutate_fn(child2)
        children.append(child1)
        children.append(child2)
    return np.array(children)

# =====================================
# =====================================

def tsp_extract_env(file):
    """
    Reads the given file and builds a 2d list
    containing the position of all the cities.
    """
    env = []
    with open(file, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            split = line.split(' ')
            env.append([float(split[1]), float(split[2])])
    return np.array(env)

def tsp_init_pop(size, num_cities):
    """
    Creates random initial population for TSP problem.
    Each member is a random permutation of the cities.
    """
    initial_population = []
    for i in range(size):
        order = list(range(num_cities))
        rnd.shuffle(order)
        initial_population.append(order)
    return np.array(initial_population)

def tsp_fit(route, env):
    """
    Computes the length of given route.
    The fitness equals inverse of length.
    """
    order = np.zeros((route.shape[0]), dtype=int)
    order[0:-1] = np.arange(route.shape[0] - 1) + 1
    next_cities = route[order]
    a = env[route]
    b = env[next_cities]

    lens = np.sum((a - b) ** 2, axis=1)
    length = np.sum(lens ** 0.5)
    return 1 / length

def tsp_mutate(route):
    """
    Mutates a route by swapping two random genes
    """
    new_route = route.copy()
    n = route.shape[0]
    p1 = rnd.randint(0, n - 2)
    p2 = rnd.randint(p1 + 1, n - 1)
    segment = route[p1:p2+1].copy()
    segment = segment[::-1]
    new_route[p1:p2+1] = segment

    # print(new_route.shape)
    return new_route

def tsp_create_children(parents, mutate_fn):
    """
    Each parent creates one child.
    """
    children = []
    for parent in parents:
        child1 = mutate_fn(parent)
        children.append(child1)
    return children

def tsp_select_children(parents, children, fit_fn, env, lam):
    whole = np.concatenate((parents, children))
    fitness = []
    for route in whole:
        fitness.append(fit_fn(route, env))
    fitness = np.array(fitness)
    order = np.argsort(fitness)
    sorted_pop = whole[order]
    return sorted_pop[-lam:]

if __name__ == "__main__":
    main()
