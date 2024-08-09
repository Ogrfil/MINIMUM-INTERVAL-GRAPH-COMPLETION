import itertools
import interval_graph_check as igc
import sage.all
from sage.graphs.graph import Graph

def brute_force_min_interval_completion(G):
    
    '''
    Brute force approach to find the minimum interval graph completion.
    param G: Graph
    '''
    
    all_possible_edges = set((u, v) for u in G.vertices() for v in G.vertices() if u < v)
    
    # Find existing edges
    existing_edges = set(G.edges())
    
    # Determine missing edges
    missing_edges = all_possible_edges - existing_edges
    
    min_edges_to_add = None
    G_min = G.copy()
    
    check, peo = igc.check_interval_graph(G)
    if check:
        return min_edges_to_add, G_min
    
    # Try adding different combinations of missing edges
    for r in range(1, len(missing_edges) + 1):
        for edge_combination in itertools.combinations(missing_edges, r):
            G_copy = G.copy()
            G_copy.add_edges(edge_combination)
            
            check, peo = igc.check_interval_graph(G_copy)
            if check:
                if min_edges_to_add is None or len(edge_combination) < len(min_edges_to_add):
                    min_edges_to_add = edge_combination
                    G_min = G_copy.copy()
    
    return min_edges_to_add, G_min