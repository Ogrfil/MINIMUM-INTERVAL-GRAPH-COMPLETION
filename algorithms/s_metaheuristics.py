import itertools
import random
import time
import numpy as np
import math
from copy import deepcopy
from time import perf_counter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sage.all
from sage.graphs.graph import Graph
import interval_graph_check as igc


def initialize(num_edges):
    return []


def calc_solution_value(G, solution):
    G_copy = deepcopy(G)
    G_copy.add_edges(solution)
    check, _ = igc.check_interval_graph(G_copy)
    if check:
        return len(solution)
    else:
        return float('inf')


def make_small_change(solution, G, iteration, num_iters):
    all_possible_edges = set((u, v) for u in G.vertices() for v in G.vertices() if u < v)
    existing_edges = set(G.edges(labels=False))
    solution_set = set(solution)
    
    missing_edges = list(all_possible_edges - existing_edges - solution_set)
    new_solution = deepcopy(solution)
    
    add_prob = max(0.1, 1 - iteration / num_iters)  # Adaptive probability
    
    if random.random() < add_prob and missing_edges:
        edge = random.choice(missing_edges)
        new_solution.append(edge)
    elif new_solution:
        edge = random.choice(new_solution)
        new_solution.remove(edge)
                
    return new_solution


def safe_exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return float('inf')


def simulated_annealing(G, num_iters=1000):
    values = [None for _ in range(num_iters)]
    check, _ = igc.check_interval_graph(G)
    if check:
        return None, G, values
    
    all_possible_edges = set((u, v) for u in G.vertices() for v in G.vertices() if u < v)
    existing_edges = set(G.edges(labels=False))
    missing_edges = list(all_possible_edges - existing_edges)
    
    solution = initialize(len(missing_edges))
    value = calc_solution_value(G, solution)
    best_solution = deepcopy(solution)
    best_value = value


    for i in range(1, num_iters + 1):
        #print(f"Iteration {i}")        
        #print(f"solution: {solution}" )
        new_solution = make_small_change(solution, G, i, num_iters)
        #print(f"new_solution: {new_solution}")
        new_value = calc_solution_value(G, new_solution)
        #print(f"new_value: {new_value}")
        #print("----------------------------------")
        
        if new_value < value:
            value = new_value
            solution = deepcopy(new_solution)
            if new_value < best_value:
                best_value = new_value
                best_solution = deepcopy(new_solution)
        else:
            if random.random() < 1 / i:
                value = new_value
                solution = deepcopy(new_solution)

        values[i - 1] = value

        # Debugging outputs
        #print(f"Iteration {i}")
        #print(f"Current Value: {value}")
        #print(f"Best Value: {best_value}")

    G_minimal = deepcopy(G)
    G_minimal.add_edges(best_solution)
    
    return best_solution, G_minimal, values


#multiplicative cooling schemes

def linear_multiplicative_cooling(t, k, t_max, alpha=0.99):
    return t_max - alpha * k

def natural_log_exponential_multiplicative_cooling(t, k, t_max, alpha=0.99):
    return t_max * alpha**k

def logarithmic_multiplicative_cooling(t, k, t_max, alpha=0.99):
    return t_max / (1 + alpha * np.log(k + 1))

def quadratic_multiplicative_cooling(t, k, t_max, alpha=0.99):
    return t_max / (1 + alpha * k**2)


#additive cooling schemes

def linear_additive_cooling(t, k, t_max, step_max, t_min=1):
    return t_min + (t_max - t_min) * (step_max - k) / step_max

def exponential_additive_cooling(t, k, t_max, step_max, t_min=1):
    return t_min + (t_max - t_min) * (1 / (1 + np.exp(2 * np.log(t_max - t_min) * (k - 0.5 * step_max) / step_max)))

def quadratic_additive_cooling(t, k, t_max, step_max, t_min=1):
    return t_min + (t_max - t_min) * ((step_max - k) / step_max)**2


def simulated_annealing_cooling(G, cooling_function, t_max=1000, t_min=1, step_max=1000, alpha=0.99, initial_acceptance_threshold=0.5):
    values = [None] * step_max
    step = 0
    accept = 0
    
    check, _ = igc.check_interval_graph(G)
    if check:
        return None, G, values, 0, 1.0
    
    all_possible_edges = set((u, v) for u in G.vertices() for v in G.vertices() if u < v)
    existing_edges = set(G.edges())
    missing_edges = list(all_possible_edges - existing_edges)

    # Initialize with a small number of edges
    current_state = initialize(len(missing_edges))
    current_energy = calc_solution_value(G, current_state)
    best_state = deepcopy(current_state)
    best_energy = current_energy
    
    t = t_max   
    
    while step < step_max and t >= t_min:
        proposed_neighbor = make_small_change(current_state, G, step, step_max)
        E_n = calc_solution_value(G, proposed_neighbor)
        dE = E_n - current_energy
        
        # Acceptance logic
        if step < step_max * initial_acceptance_threshold:
            if random.random() < 0.8:  # Accept 80% of worse solutions in the beginning
                current_energy = E_n
                current_state = deepcopy(proposed_neighbor)
                accept += 1
                
        else:
            if dE < 0 or random.random() < safe_exp(-dE / t):
                current_energy = E_n
                current_state = deepcopy(proposed_neighbor)
                accept += 1
                
        if E_n < best_energy:
            best_energy = E_n
            best_state = deepcopy(proposed_neighbor)
            

        # Update temperature using the provided cooling function
        if 'alpha' in cooling_function.__code__.co_varnames:
            t = cooling_function(t, step, t_max, alpha=alpha)
        else:
            t = cooling_function(t, step, t_max, step_max, t_min=t_min)
        

        values[step] = current_energy
        step += 1
        
    acceptance_rate = float(accept) / step if step > 0 else 0

    G_minimal = deepcopy(G)
    G_minimal.add_edges(best_state)
    
    return best_state, G_minimal, values, best_energy, acceptance_rate


def shaking(solution, k, G, iteration, num_iters):
    all_possible_edges = set((u, v) for u in G.vertices() for v in G.vertices() if u < v)
    existing_edges = set(G.edges(labels=False))
    solution_set = set(solution)
    
    missing_edges = list(all_possible_edges - existing_edges - solution_set)
    new_solution = deepcopy(solution)
    
    add_prob = max(0.1, 1 - iteration / num_iters)  # Adaptive probability based on iteration
    
    for _ in range(k):
        if random.random() < add_prob and missing_edges:
            edge = random.choice(missing_edges)
            if edge not in new_solution:
                new_solution.append(edge)
        elif new_solution:
            edge = random.choice(new_solution)
            new_solution.remove(edge)
                
    return new_solution


def local_search_best_improvement_unbiased(solution, value, G):
    best_solution = deepcopy(solution)
    best_value = value
    improved = True
    
    all_possible_edges = set((u, v) for u in G.vertices() for v in G.vertices() if u < v)
    
    while improved:
        improved = False
        
        # Update missing edges for the current best_solution
        existing_edges = set(G.edges())
        current_edges = set(best_solution)
        missing_edges = list(all_possible_edges - existing_edges - current_edges)
        
        unbiased_order = list(missing_edges) + best_solution
        
        random.shuffle(unbiased_order)  # Shuffle the order of edges

        for edge in unbiased_order:
            if edge in missing_edges:  # Trying to add edge
                new_solution = best_solution + [edge]
            else:  # Trying to remove edge
                new_solution = [e for e in best_solution if e != edge]
            
            new_value = calc_solution_value(G, new_solution)
            if new_value < best_value:
                best_value = new_value
                best_solution = deepcopy(new_solution)
                improved = True
        
        # If an improvement was found, update the current solution
        if improved:
            solution = deepcopy(best_solution)
            value = best_value

    return best_solution, best_value


def vns_min_interval_completion(G, vns_params):
    check, _ = igc.check_interval_graph(G)
    if check:
        return None, G, 0, [], [], [], [], []

    start_time = perf_counter()
    solution = initialize(len(G.edges()))
    value = calc_solution_value(G, solution)

    # Initialize tracking variables
    all_values = []
    all_new_values = []

    iteration = 0

    while perf_counter() - start_time < vns_params['time_limit']:
        for k in range(vns_params['k_min'], vns_params['k_max'] + 1):
            new_solution = shaking(solution, k, G, iteration, len(G.edges()))
            new_value = calc_solution_value(G, new_solution)
            new_solution, new_value = local_search_best_improvement_unbiased(new_solution, new_value, G)

            if new_value < value or (new_value == value and random.random() < vns_params['move_prob']):
                value = new_value
                solution = deepcopy(new_solution)
            
            #print(f"Iteration: {iteration}, K: {k}, Value: {value}, New value: {new_value}")
            
            # Track values for plotting
            all_values.append(value)
            all_new_values.append(new_value)
        
        iteration += 1

    G_minimal = G.copy()
    G_minimal.add_edges(solution)
    
    return solution, G_minimal, value, all_values, all_new_values

