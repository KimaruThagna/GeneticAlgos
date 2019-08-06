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

# Solve problem using simulated annealing
best_state, best_fitness = mlrose.simulated_annealing(optimization_problem, schedule = mlrose.ExpDecay(),
                                                      max_attempts = 100, max_iters = 1000,
                                                      init_state = init_state, random_state = 1)

print('The best state : ', best_state)
print('Fitness: ', best_fitness)

# visualize the best state

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

visualize_state(best_state)