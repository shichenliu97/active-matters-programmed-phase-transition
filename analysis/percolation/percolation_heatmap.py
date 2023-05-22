# heatmap_generator.py

import numpy as np
import networkx as nx
from numba import jit
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
        if edge_length > 9:
            for j in range(i+1, n):  # Add an edge between node i and all subsequent nodes
                G.add_edge(i, j)

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
    for _ in range(10):
        G = generate_graph(shape, scale, 1000)
        giant_component_ratio = analyze_graph(G)
        ratios.append(giant_component_ratio)
    return np.mean(ratios)

def generate_heatmap_data(shape_values, scale_values):
    params = [(shape, scale) for shape in shape_values for scale in scale_values]

    with Pool(16) as p:
        results = p.map(compute_ratio, params)

    return np.array(results).reshape(len(shape_values), len(scale_values))

if __name__ == '__main__':
    shape_values = np.linspace(0.1, 7, 50)
    scale_values = np.linspace(0.1, 7, 50)

    heatmap_data = generate_heatmap_data(shape_values, scale_values)
    np.save('heatmap_data.npy', heatmap_data)
