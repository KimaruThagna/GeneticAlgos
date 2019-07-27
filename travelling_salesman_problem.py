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

    def route_fitness(self):
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
    return sorted(fitness_results.items(), key = operator.itemgetter(1), reverse = True)
