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
        order = list(range(num_cities))
        rnd.shuffle(order)
        initial_population.append(order)
    return initial_population

if __name__ == "__main__":
    main()
