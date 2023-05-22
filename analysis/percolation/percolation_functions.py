import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import random
from numba import jit, prange
import cProfile
from multiprocessing import Pool


@jit(nopython=True)
def generate_microtubule_lengths(shape, scale, total_length):
    lengths = []
    current_total_length = 0.0
    while current_total_length < total_length:
        new_length = round(np.random.gamma(shape, scale), 1)
        if new_length >= 1 and new_length <= 25:
            if current_total_length + new_length <= total_length:
                lengths.append(new_length)
                current_total_length += new_length
            else:
                new_length = total_length - current_total_length
                if new_length >= 1:
                    lengths.append(new_length)
                    break
                else:
                    break
    return np.array(lengths)


def generate_graph(shape, scale, total_length):
    edge_lengths = generate_microtubule_lengths(shape, scale, total_length)
    n = len(edge_lengths)

    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i, edge_length in enumerate(edge_lengths):
        if edge_length > 9:  # If the edge length is greater than 9, add an edge
            G.add_edge(i, (i+1) % n)  # Use modulus to loop back to the start

    return G



def analyze_graph(G):
    giant_component_ratio = 0
    for component in nx.connected_components(G):
        if len(component) > giant_component_ratio:
            giant_component_ratio = len(component)
    return giant_component_ratio / G.number_of_nodes()

def compute_ratio(params):
    shape, scale = params
    ratios = []
    for _ in range(20):
        G = generate_graph(shape, scale, 1000) # predefined total length
        giant_component_ratio = analyze_graph(G)
        ratios.append(giant_component_ratio)
    return np.mean(ratios)
