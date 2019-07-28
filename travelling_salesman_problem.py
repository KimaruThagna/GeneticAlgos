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
class Gene:
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

    for i in range(0, elite_size):
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
        index = selection_results[i]
        matingpool.append(population[index])
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

    child_p2 = [item for item in parent2 if item not in child_p1]

    child = child_p1 + child_p2
    return child


def breed_population(matingpool, elite_size):
    children = []
    length = len(matingpool) - elite_size
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, elite_size):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children


def mutate(individual, mutation_rate):
    for swapped in range(len(individual)):
        if (random.random() < mutation_rate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

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
    nextGeneration = mutate_population(children, mutation_rate)
    return nextGeneration


def genetic_algorithm(population, pop_size, elite_size, mutation_rate, generations):
    pop = initial_population(pop_size, population)
    print("Initial distance: " + str(1 / rank_routes(pop)[0][1]))

    for i in range(0, generations):
        pop = next_generation(pop, elite_size, mutation_rate)

    print("Final distance: " + str(1 / rank_routes(pop)[0][1]))
    bestRouteIndex = rank_routes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute


city_list = []

for i in range(0,25):
    city_list.append(Gene(x=int(random.random() * 200), y=int(random.random() * 200)))

genetic_algorithm(population=city_list, pop_size=100, elite_size=20, mutation_rate=0.01, generations=500)


def visualize_evolution(population, pop_size, elite_size, mutation_rate, generations):
    pop = initial_population(pop_size, population)
    progress = []
    progress.append(1 / rank_routes(pop)[0][1])

    for i in range(0, generations):
        pop = next_generation(pop, elite_size, mutation_rate)
        progress.append(1 / rank_routes(pop)[0][1])

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

visualize_evolution(population=city_list, pop_size=100, elite_size=20, mutation_rate=0.01, generations=500)