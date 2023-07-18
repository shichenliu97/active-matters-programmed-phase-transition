# heatmap_generator.py

import numpy as np
import networkx as nx
from numba import jit, njit
from multiprocessing import Pool
from scipy.stats import gamma

@jit
def generate_microtubule_lengths(shape, scale, total_length):
    lengths = []
    current_total_length = 0.0

    # Calculate the CDF at the truncation points
    cdf_low = gamma.cdf(1, shape, scale=scale)
    cdf_high = gamma.cdf(25, shape, scale=scale)

    while current_total_length < total_length:
        # Generate uniform random number between cdf_low and cdf_high
        u = np.random.uniform(cdf_low, cdf_high)

        # Convert back into gamma distribution using the ppf
        new_length = round(gamma.ppf(u, shape, scale=scale), 1)

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

    # Generate a permutation of the nodes
    nodes = np.random.permutation(n)

    for node in nodes:
        # Only consider edge lengths greater than or equal to 9
        if edge_lengths[node] >= 9:
            # Create a normalized edge probability between 0 and 1
            edge_prob = edge_lengths[node] / 25  # As maximum possible edge length is 25

            # Generate a random number between 0 and 1
            rand_num = np.random.uniform(0, 1)
                
            if rand_num <= edge_prob:
                # Add an edge to the first node in the remaining list that is not the current node
                for other_node in nodes:
                    if node != other_node:
                        G.add_edge(node, other_node)
                        break

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
    np.save('heatmap_data_v2.npy', heatmap_data)
