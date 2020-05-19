import random as rnd
import numpy as np

def main():

    # settings = extract_tsp_env('tsp_data.txt')
    # init_size = 1000
    # mu = 10
    # lam = 10
    # iters = 100000
    # Evo(tsp_init_pop, init_size, len(settings), tsp_fit,
    #     settings, tsp_select_parents, mu, tsp_mutate, None, None,
    #     tsp_create_children, lam, tsp_select_children, iters, 0)

    solve_knapsack()

def solve_knapsack():
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

def check_population(population, i, fitness_fn, env, mode):
    lens = []
    for member in population:
        lens.append(fitness_fn(member, env))
    highest_fitness = max(lens)
    best = population[lens.index(highest_fitness)]
    
    if mode == 0:
        print('generation ', i + 1)
        print('   best answer:', best)
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

def extract_tsp_env(file):
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
    return env

def tsp_init_pop(size, num_cities):
    """
    Creates random initial population for TSP problem.
    Each member is a random permutation of the cities.
    """
    initial_population = []
    for i in range(size):
        order = list(range(num_cities - 1))
        rnd.shuffle(order)
        initial_population.append(order)
    return initial_population

def tsp_fit(route, env):
    """
    Computes the length of given route.
    The fitness equals inverse of length.
    """
    length = 0.0
    curr_x, curr_y = env[-1]
    final_x, final_y = curr_x, curr_y
    for city in route:
        next_x, next_y = env[city]
        length += ((curr_x - next_x) ** 2 + (curr_y - next_y) ** 2) ** 0.5
        curr_x, curr_y = next_x, next_y
    length += ((curr_x - final_x) ** 2 + (curr_y - final_y) ** 2) ** 0.5
    return 1 / length

def tsp_select_parents(routes, env, mu, fit_fn):
    """
    Selects parents according to their fitness.
    """
    fitness = []
    for route in routes:
        fitness.append(fit_fn(route, env))
    total = sum(fitness)
    weights = [x / total for x in fitness]
    parents = rnd.choices(routes, fitness, k=mu)
    return parents

def tsp_mutate(route):
    """
    Mutates a route by swapping two random genes
    """
    new_route = route.copy()
    n = len(route)
    p1 = rnd.randint(1, n - 1)
    p2 = rnd.randint(0, p1)
    new_route[p1], new_route[p2] = new_route[p2], new_route[p1]
    return new_route

def tsp_create_children(parents, mutate_fn, crossover_fn, mutation_rate):
    """
    Each parent creates one child.
    """
    children = []
    for parent in parents:
        children.append(mutate_fn(parent))
    return children

def tsp_select_children(parents, children, fit_fn, env, lam):
    whole = parents + children
    fitness = []
    for route in whole:
        fitness.append(fit_fn(route, env))
    sorted_pop = [x for _, x in sorted(zip(fitness, whole))]
    print(len(sorted_pop))
    print(len(sorted_pop[-1]))
    return sorted_pop[-lam:]

if __name__ == "__main__":
    main()
