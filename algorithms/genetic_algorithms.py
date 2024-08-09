import random
import numpy as np
import interval_graph_check as igc
import sage.all
from sage.graphs.graph import Graph

def fitness(G, added_edges):
    G_copy = G.copy()
    G_copy.add_edges(added_edges)
    check, peo = igc.check_interval_graph(G_copy)
    
    #if new graph is interval, it's fitness is number of newly added edges
    if check:
        return len(added_edges)
    else:
        return float('inf')  # Penalize non-interval graphs

def initialize_population(G, population_size, missing_edges):
    population = []
    
    for _ in range(population_size):
        #randomize adding aditional edges
        individual = random.sample(missing_edges, random.randint(1, len(missing_edges)))
        population.append(individual)
    
    return population

def tournament_selection(population, fitnesses, tournament_size, num_parents):
    selected_parents = []
    for _ in range(num_parents):
        # Randomly select a subset of individuals for the tournament
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_individuals = [population[i] for i in tournament_indices]
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        
        # Ensure two distinct parents
        parent1_index = tournament_fitnesses.index(min(tournament_fitnesses))
        parent1 = tournament_individuals[parent1_index]
        
        # Remove the selected parent from the tournament pool to ensure distinct parent selection
        tournament_individuals.pop(parent1_index)
        tournament_fitnesses.pop(parent1_index)
        
        # Select the second best individual
        parent2_index = tournament_fitnesses.index(min(tournament_fitnesses))
        parent2 = tournament_individuals[parent2_index]
        
        selected_parents.append((parent1, parent2))
    
    return selected_parents

def roulette_selection(population, fitnesses, num_parents):
    fitness_sum = sum(1.0 / f if f != float('inf') else 0 for f in fitnesses)
    probabilities = [(1.0 / f if f != float('inf') else 0) / fitness_sum for f in fitnesses]
    
    selected_parents = []
    for _ in range(num_parents):
        parent_indices = np.random.choice(len(population), size=2, p=probabilities, replace=False)
        parent1, parent2 = population[parent_indices[0]], population[parent_indices[1]]
        selected_parents.append((parent1, parent2))
    
    return selected_parents

def crossover(parent1, parent2):
    len1 = len(parent1)
    len2 = len(parent2)
    
    # Ensure parents are large enough to perform crossover
    if len1 < 2 or len2 < 2:
        return parent1, parent2  # No crossover if not enough edges
    
    crossover_point = random.randint(1, min(len1, len2) - 1)
    
    parent1_edges = list(parent1)
    parent2_edges = list(parent2)
    
    child1_edges = list(set(parent1_edges[:crossover_point] + parent2_edges[crossover_point:]))
    child2_edges = list(set(parent2_edges[:crossover_point] + parent1_edges[crossover_point:]))
    
    return child1_edges, child2_edges

def mutate(individual, missing_edges, mutation_rate):
    new_individual = individual.copy()
    
    for i in range(len(new_individual)):
        if random.random() < mutation_rate:
            new_individual[i] = random.choice(missing_edges)
            
    return list(set(new_individual))

def genetic_algorithm(G, population_size=50, num_generations=100, mutation_rate=0.01, elitism_rate=0.1, tournament_size=5):
    
    check, _ = igc.check_interval_graph(G)
    if check:
        return None, G
    
    all_possible_edges = set((u, v) for u in G.vertices() for v in G.vertices() if u < v)
    existing_edges = set(G.edges())
    missing_edges = list(all_possible_edges - existing_edges)
    
    population = initialize_population(G, population_size, missing_edges)
    best_individual = None
    best_fitness = float('inf')
    
    for generation in range(num_generations):
        fitnesses = [fitness(G, individual) for individual in population]
        
        elite_size = int(elitism_rate * population_size)
        elite_indices = np.argsort(fitnesses)[:elite_size]
        elite = [population[i] for i in elite_indices]
        
        for i in range(len(population)):
            if fitnesses[i] < best_fitness:
                best_fitness = fitnesses[i]
                best_individual = population[i]
        
        if best_fitness == float('inf'):
            continue
        
        new_population = elite.copy()
        for _ in range((population_size - elite_size) // 2):
            parents = roulette_selection(population, fitnesses, 2)
            #parents = tournament_selection(population, fitnesses, tournament_size, 2)
            for parent1, parent2 in parents:
                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1, missing_edges, mutation_rate)
                child2 = mutate(child2, missing_edges, mutation_rate)
                new_population.extend([child1, child2])
        
        population = new_population.copy()
    
    G_minimal = G.copy()
    G_minimal.add_edges(best_individual)
    return best_individual, G_minimal
