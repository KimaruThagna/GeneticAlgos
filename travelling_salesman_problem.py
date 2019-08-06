'''
The classical problem.
Given a list of cities and the distances between each pair of cities,
what is the shortest possible route that visits each city and returns to the origin city?
'''

'''
General GA algorithm
Create initial Population --> Determine fitness --> Select mating pool --> Breed --> Mutate --> Repeat
'''
import numpy as np, random, operator
import pandas as pd
import matplotlib.pyplot as plt

# city is the gene in this specific case
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        x_dis = abs(self.x - city.x)
        y_dis = abs(self.y - city.y)
        distance = np.sqrt((x_dis ** 2) + (y_dis ** 2)) # pythagoras theorem
        return distance

    def __repr__(self):
        return f'{self.x},{self.y}'


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0
# objective function
    def route_distance(self):
        if self.distance == 0:
            path_distance = 0
            for i in range(0, len(self.route)):
                from_city = self.route[i]
                to_city = None
                if i + 1 < len(self.route):
                    to_city = self.route[i + 1]
                else:
                    to_city = self.route[0]
                path_distance += from_city.distance(to_city)

            self.distance = path_distance
        return self.distance

    def route_fitness(self): # inverse of objective function since its a minimization case
        if self.fitness == 0.0:
            self.fitness = 1 / float(self.route_distance())
        return self.fitness

def create_route(city_list):
    route = random.sample(city_list, len(city_list))
    return route

def initial_population(pop_size, city_list):
    population = []

    for i in range(pop_size):
        population.append(create_route(city_list))
    return population

# survival of the fittest

def rank_routes(population):
    fitness_results = {}
    for i in range(0,len(population)):
        fitness_results[i] = Fitness(population[i]).route_fitness()
    return sorted(fitness_results.items(), key = operator.itemgetter(1), reverse = True) # descending order of values


def selection(pop_ranked, elite_size):
    selection_results = []
    df = pd.DataFrame(np.array(pop_ranked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, elite_size): # we want the x best to make it to the breeding pool where x is the elite_size
        selection_results.append(pop_ranked[i][0])
    for i in range(0, len(pop_ranked) - elite_size):
        pick = 100 * random.random()
        for i in range(0, len(pop_ranked)):
            if pick <= df.iat[i, 3]:
                selection_results.append(pop_ranked[i][0])
                break
    return selection_results

def mating_pool(population, selection_results):
    matingpool = []
    for i in range(0, len(selection_results)):
        matingpool.append(population[selection_results[i]])# pick the selected individuals from the total population for breeding
    return matingpool


def breed(parent1, parent2):
    child = []
    child_p1 = []
    child_p2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        child_p1.append(parent1[i])

    child_p2 = [item for item in parent2 if item not in child_p1] # ensure no repeating items

    child = child_p1 + child_p2 # combine children
    return child


def breed_population(matingpool, elite_size):
    children = []
    length = len(matingpool) - elite_size
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, elite_size):
        children.append(matingpool[i]) # not necessarily children but since theyre the best of the best, they make it to the next generation


    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1]) # take any 2 "parents" that are  opposite in terms of their indices.
                                                            # Can be any, even from the elite
        children.append(child)


    return children


def mutate(individual, mutation_rate):
    for swapped in range(len(individual)):
        if (random.random() < mutation_rate): # gauging probability. If true, perform the mutation which is a swap
            swapWith = int(random.random() * len(individual))# random index of city
            # individual is a chromosome hence route
            city1 = individual[swapped]  # current city within the individual route
            city2 = individual[swapWith] # random city within the individual

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


def mutate_population(population, mutation_rate):
    mutated_pop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutation_rate)
        mutated_pop.append(mutatedInd)
    return mutated_pop



def next_generation(current_gen, elite_size, mutation_rate):
    popRanked = rank_routes(current_gen)
    selectionResults = selection(popRanked, elite_size)
    matingpool = mating_pool(current_gen, selectionResults)
    children = breed_population(matingpool, elite_size)
    next_generation = mutate_population(children, mutation_rate)
    return next_generation


def genetic_algorithm(population, pop_size, elite_size, mutation_rate, generations,visualize=True):
    pop = initial_population(pop_size, population)
    print("Initial distance: " + str(1 / rank_routes(pop)[0][1])) # inverse since what was returned was route fitness which is 1/route_distance()

    for i in range(0, generations):
        pop = next_generation(pop, elite_size, mutation_rate)

    print("Final distance: " + str(1 / rank_routes(pop)[0][1]))
    bestRouteIndex = rank_routes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    if visualize == True:
        progress = []
        progress.append(1 / rank_routes(pop)[0][1])

        for i in range(0, generations):
            pop = next_generation(pop, elite_size, mutation_rate)
            progress.append(1 / rank_routes(pop)[0][1])

        plt.plot(progress)
        plt.ylabel('Distance')
        plt.xlabel('Generation')
        plt.show()
    return bestRoute


city_list = []

for i in range(0,25):
    city_list.append(City(x=int(random.random() * 10), y=int(random.random() * 10)))

genetic_algorithm(population=city_list, pop_size=100, elite_size=10, mutation_rate=0.01, generations=500)


'''
SAME PROBLEM USING MLROSE LIBRARY
'''

import mlrose
# create a list of triplets (u,v,d) giving distance d between nodes u and v.
'''
assume
city-0,city-1,city-2,,city-3,city-4,city-5,city-6,city-7,city-8,city-9  
'''
dist_list = [(0, 1, 3.1623), (0, 2, 4.1231), (0, 3, 5.8310), (0, 4, 4.2426),
             (0, 5, 5.3852), (0, 6, 4.0000), (0, 7, 2.2361), (1, 2, 1.0000),
             (2, 6, 5.0000), (2, 7, 3.1623), (3, 4, 2.0000), (3, 5, 3.6056), 
             (3, 6, 5.0990), (3, 7, 4.1231), (4, 5, 2.2361), (4, 6, 3.1623),
             (4, 7, 2.2361), (5, 6, 2.2361), (5, 7, 3.1623), (6, 7, 2.2361)]

coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]

fitness_func = mlrose.TravellingSales(distances=dist_list)
optimization_problem = mlrose.TSPOpt(length=8, fitness_fn=fitness_func, maximize=False, distances=dist_list)
init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # Define initial state
best_state, best_fitness = mlrose.genetic_alg(optimization_problem,pop_size=1000,mutation_prob=0.07,random_state=2)

print('The best state : ', best_state)
print('Fitness: ', best_fitness)