import random as rnd

def main():
    x = init_tsp_pop(5, 10)
    print(x)

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
    The fitness equals minus length.
    """
    length = 0.0
    curr_x, curr_y = env[-1]
    for city in route:
        next_x, next_y = env[city]
        length += (curr_x - next_x) ** 2 + (curr_y - next_y) ** 2
        curr_x, curr_y = next_x, next_y
    return -length

if __name__ == "__main__":
    main()
