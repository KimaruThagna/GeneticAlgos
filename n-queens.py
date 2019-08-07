import mlrose # a library of common optimization problems such as n-queens ans TSP
import numpy as np

'''
THE 8-QUEENS PROBLEM
Having an 8x8 chess board and 8 queen pieces, what is the optimal way to place each piece such
that no one piece can attack the other?
Pre-requisite- Knowledge of chess that the queen piece can attack in any direction
'''
#define our fitness function
fitness_func = mlrose.Queens()
# the n-queens is a discrete state optimization problem since the values are integers
# these integers represent the column position and their indices, the row position
optimization_problem = mlrose.DiscreteOpt(max_val=8, maximize=False, length=8, fitness_fn=fitness_func)
# maximize=False since we are trying to reduce number of attacking pairs(queen pieces)


init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7]) # Define initial state


def visualize_state(state):
    vizualiation_board =[]
    for i in range(len(state)):
        row = []
        for j in range(len(state)):
            if state[i] == j:
                row.append(' Q ')
            else:
                row.append(' * ')

        vizualiation_board.append(row)
    for obj in vizualiation_board:
        print(obj)

# Solve problem using simulated annealing
best_state, best_fitness = mlrose.simulated_annealing(optimization_problem, schedule = mlrose.ExpDecay(),
                                                      max_attempts = 100, max_iters = 1000,
                                                      init_state = init_state, random_state = 1)

print('The best state : ', best_state)
print('Fitness: ', best_fitness)
visualize_state(best_state)

# solving using hill-climb algorithm
hc_state, hc_fitness, hc_curve = mlrose.hill_climb(optimization_problem,init_state=init_state,
                                                   curve=True,restarts=50,random_state=3, max_iters=10)

print('The best state : ', hc_state)
print('Fitness Hill Climb: ', hc_fitness)
print('Curve: ', hc_curve)
visualize_state(hc_state)

# Randomized hill climbing
rhc_state, rhc_fitness, rhc_curve = mlrose.random_hill_climb(optimization_problem,init_state=init_state,
                                                   curve=True,restarts=50,random_state=3,max_attempts=50, max_iters=1000)

print('The best state : ', rhc_state)
print('Fitness Randomized Hill Climb: ', rhc_fitness)
print('Curve: ', rhc_curve)
visualize_state(rhc_state)

# Genetic algorithm
ga_state, ga_fitness, ga_curve = mlrose.genetic_alg(optimization_problem, curve=True, pop_size=500,
                                                    max_attempts=100, max_iters=1000)

print('The best state : ', rhc_state)
print('Genetic Algorithm Fitness: ', rhc_fitness)
print('Curve: ', ga_curve)
visualize_state(ga_state)

# MIMIC algorithm---only applicable to DiscreteOpt() problems
mimic_state, mimic_fitness, mimic_curve = mlrose.mimic(optimization_problem, curve=True,pop_size=500,keep_pct=0.4,
                                                       random_state=3,max_attempts=10, max_iters=100)

print('The best state : ', mimic_state)
print('Fitness MIMIC: ', mimic_fitness)
print('Curve: ', mimic_curve)
visualize_state(mimic_state)

