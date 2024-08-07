#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sage.all
from sage.graphs.graph import Graph
from sage.graphs.traversals import lex_BFS
import networkx as nx
import matplotlib.pyplot as plt


# In[2]:


def find_PEO(graph):
    '''
    Finding Perfect Elimination Order of graph using lexBFS algorithm. 
    '''
    peo = list(lex_BFS(graph, reverse=True))
    return peo


# In[3]:


def plot_PEO(edges, peo):
    G_peo = nx.Graph()

    # Add nodes in the order of PEO
    G_peo.add_nodes_from(peo)

    # Add edges to the new graph
    for edge in edges:
        if edge[0] in peo and edge[1] in peo:
            G_peo.add_edge(edge[0], edge[1])

    # Plot the new graph based on PEO
    plt.figure(figsize=(8, 6))
    pos_peo = {node: (i, 0) for i, node in enumerate(peo)}  # Linear layout for simplicity
    nx.draw(G_peo, pos_peo, with_labels=True, node_color='lightgreen', node_size=500, edge_color='gray')
    plt.title('Graph with PEO')
    plt.show()


# In[4]:


def plot_graph(edges):
    G = nx.Graph()
    
    G.add_edges_from(edges)

    # Plot the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
    plt.title('Graph from SageMath')
    plt.show()


# In[5]:


def check_chordal(graph):   
    '''
    Every interval graph is chordal. Maximum Cardinality Search algorithm is used for checking wheter graph
    is chordal. F is empty array in case our graph is chordal.
    '''
    alpha, F, X = graph.maximum_cardinality_search_M()
    if not F:
        return True
    else:
        return False    


# In[6]:


def check_comparability(graph):
    '''
    Graph is comparability graph if and only if it is a chordal graph
    '''
    graph_complement = graph.complement()
    if check_chordal(graph_complement):
        return True
    else:
        return False


# In[7]:


def check_interval_graph(graph):
    '''
    Checking wheter graph is interval or not
    Graph is interval if and only if it is chordal and 
    its complement is comparability graph.
    
    1. step: finding PEO - Perfect Elimination Order of graph
    2. step: checking wheter graph is chordal or not.
    3. step: checking wheter complement graph is comparability graph
    '''
    peo = find_PEO(graph)
    if check_chordal(graph):
        result = check_comparability(graph)
        return result, peo
    return False, peo

