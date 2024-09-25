import random
import math
import numpy as np
import interval_graph_check as igc
import sage.all
from sage.graphs.graph import Graph

def fitness(G, added_edges):
    G_copy = G.copy()
    G_copy.add_edges(added_edges)
    check, peo = igc.check_interval_graph(G_copy)
    
    if check:
        return len(added_edges)
    else:
        return float('inf')

def initialize_population(G, population_size, missing_edges):
    population = []
    
    for _ in range(population_size):
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


def roulette_selection(population, fitnesses):
    # Handle infinite fitness and calculate inverse fitnesses
    inv_fitnesses = [
        0 if math.isinf(f) else 1 / (f + 1e-6) for f in fitnesses
    ]
    
    total_inv_fitness = sum(inv_fitnesses)

    # If all fitnesses are inf or effectively zero, select two random parents
    if total_inv_fitness == 0:
        return random.sample(population, 2)
    
    # Create cumulative probabilities
    cumulative_probabilities = []
    cumulative_sum = 0
    for inv_fit in inv_fitnesses:
        cumulative_sum += inv_fit / total_inv_fitness
        cumulative_probabilities.append(cumulative_sum)

    # Select one parent based on cumulative probabilities
    def select_parent():
        rand_value = random.random()
        for i, cumulative_prob in enumerate(cumulative_probabilities):
            if rand_value <= cumulative_prob:
                return population[i]
        return population[-1]  # Return last population member if no parent is found

    # Select two distinct parents
    parent1 = select_parent()
    parent2 = select_parent()

    # Ensure distinct parents
    if parent1 == parent2:
        # Re-select parent2 only if it's the same as parent1
        parent2 = random.choice([p for p in population if p != parent1])

    return [parent1, parent2]

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

def genetic_algorithm(G, population_size=50, num_generations=100, mutation_rate=0.01, elitism_rate=0.1, s='r', tournament_size=5):
    
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
            if s == 't':
                parents = tournament_selection(population, fitnesses, tournament_size, 2)
                for parent1, parent2 in parents:
                    child1, child2 = crossover(parent1, parent2)
                    child1 = mutate(child1, missing_edges, mutation_rate)
                    child2 = mutate(child2, missing_edges, mutation_rate)
                    new_population.extend([child1, child2])
            else:
                parent1, parent2 = roulette_selection(population, fitnesses)
                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1, missing_edges, mutation_rate)
                child2 = mutate(child2, missing_edges, mutation_rate)
                new_population.extend([child1, child2])
        
        population = new_population.copy()
    
    if best_individual is not None:
        G_minimal = G.copy()
        G_minimal.add_edges(best_individual)
        return best_individual, G_minimal
    else:
        return None, None
