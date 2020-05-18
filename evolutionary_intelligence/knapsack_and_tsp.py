import random as rnd

def main():
    env = extract_tsp_env('tsp_test.txt')
    pop = init_tsp_pop(10, 4)
    pars = tsp_select_parents(pop, env, 5, tsp_fit)

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

def init_tsp_pop(size, num_cities):
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
        length += (curr_x - next_x) ** 2 + (curr_y - next_y) ** 2
        curr_x, curr_y = next_x, next_y
    length += (curr_x - final_x) ** 2 + (curr_y - final_y) ** 2
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

def tsp_create_children(parents, mutate_fn):
    """
    Each parent creates one child.
    """
    children = []
    for parent in parents:
        children.append(mutate_fn(parent))
    return children

if __name__ == "__main__":
    main()
