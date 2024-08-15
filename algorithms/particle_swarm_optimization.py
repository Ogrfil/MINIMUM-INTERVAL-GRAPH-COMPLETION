import random
import sage.all
from sage.graphs.graph import Graph
import interval_graph_check as igc

class Particle:
    global_best_position = None
    global_best_value = float('inf')

    def __init__(self, G, missing_edges, c_i, c_p, c_g):
        self.G = G
        self.missing_edges = missing_edges
        self.c_i = c_i
        self.c_p = c_p
        self.c_g = c_g
        
        self.position = [random.randint(0, 1) for _ in range(len(missing_edges))]  # Random binary vector
        self.velocity = [random.uniform(-1, 1) for _ in range(len(missing_edges))]  # Random initial velocities

        self.value = self.evaluate()
        self.personal_best_position = self.position.copy()
        self.personal_best_value = self.value

        # Update global best based on initial particles
        
        if Particle.global_best_position is None or self.value < Particle.global_best_value:
            Particle.global_best_value = self.value
            Particle.global_best_position = self.position.copy()

       
    def evaluate(self):
        G_copy = self.G.copy()
        edges_to_add = [edge for edge, pos in zip(self.missing_edges, self.position) if pos == 1 and edge not in self.G.edges()]
        
        edges_to_add = [edge for edge in edges_to_add if edge not in self.G.edges()]

        G_copy.add_edges(edges_to_add)
        check, _ = igc.check_interval_graph(G_copy)
        if check:
            return len(edges_to_add)
        else:
            return float('inf')

    def update_velocity(self):
        if Particle.global_best_position is None:
            raise ValueError("Global best position not initialized")
        
        r_p = [random.random() for _ in range(len(self.velocity))]
        r_s = [random.random() for _ in range(len(self.velocity))]
        cognitive_velocity = [self.c_p * rp * (p - pos) for rp, p, pos in zip(r_p, self.personal_best_position, self.position)]
        social_velocity = [self.c_g * rs * (gb - pos) for rs, gb, pos in zip(r_s, Particle.global_best_position, self.position)]
        inertia = [self.c_i * v for v in self.velocity]
        self.velocity = [i + c + s for i, c, s in zip(inertia, cognitive_velocity, social_velocity)]

    def update_position(self):
        new_position = [pos + vel for pos, vel in zip(self.position, self.velocity)]
        self.position = [1 if pos > 1 else 0 if pos < 0 else pos for pos in new_position]
        self.value = self.evaluate()

        if self.value < self.personal_best_value:
            self.personal_best_value = self.value
            self.personal_best_position = self.position.copy()
            if self.value < Particle.global_best_value:
                Particle.global_best_value = self.value
                Particle.global_best_position = self.position.copy()

def pso(G, swarm_size, c_i, c_p, c_g, num_iters):
    all_possible_edges = {(u, v) for u in G.vertices() for v in G.vertices() if u < v}
    existing_edges = set(G.edges())
    missing_edges = list(all_possible_edges - existing_edges)
    
    swarm = [Particle(G, missing_edges, c_i, c_p, c_g) for _ in range(swarm_size)]

    for _ in range(num_iters):
        for particle in swarm:
            particle.update_velocity()
            particle.update_position()

    if Particle.global_best_position is None:
        raise ValueError("Global best position was not found after iterations")

    best_edges_to_add = [edge for edge, pos in zip(missing_edges, Particle.global_best_position) if pos == 1]
    G_minimal = G.copy()
    G_minimal.add_edges(best_edges_to_add)
    
    Particle.global_best_position = None
    Particle.global_best_value = float('inf')

    return best_edges_to_add, G_minimal