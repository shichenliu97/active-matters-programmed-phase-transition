# heatmap_generator.py

import numpy as np
import networkx as nx
import pickle
from numba import njit
from multiprocessing import Pool

@njit
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

    for i in range(n):
        for j in range(i+1, n):
            # Only consider edge lengths greater than or equal to 9
            if edge_lengths[i] >= 9:
                # Create a normalized edge probability between 0 and 1
                edge_prob = edge_lengths[i] / 25  # As maximum possible edge length is 25

                # Generate a random number between 0 and 1
                rand_num = np.random.uniform(0, 1)
                
                if rand_num <= edge_prob:
                    G.add_edge(i, j)

    return G


def analyze_graph(G):
    giant_component_ratio = 0
    for component in nx.connected_components(G):
        if len(component) > giant_component_ratio:
            giant_component_ratio = len(component)
    return giant_component_ratio / G.number_of_nodes()

def compute_plot_data(params):
    shape, scale = params
    ratios = []
    graph_data = []
    for i in range(5):
        G = generate_graph(shape, scale, 1000)  # total_length is set to 1000
        giant_component_ratio = analyze_graph(G)
        ratios.append(giant_component_ratio)
        if i == 0:  # Save the graph and its layout for the first trial
            pos = nx.spring_layout(G)
            graph_data.append((G, pos))
    avg_ratio = np.mean(ratios)
    std_dev_ratio = np.std(ratios)
    return shape, scale, avg_ratio, std_dev_ratio, graph_data

def generate_plot_data(shape_values, scale_values):
    params = [(shape, scale) for shape in shape_values for scale in scale_values]

    with Pool(16) as p:
        plot_data = p.map(compute_plot_data, params)

    return plot_data

if __name__ == '__main__':
    shape_values = [3]  # fixed shape for gamma
    scale_values = np.linspace(0.8, 1.8, 50)  # for scale in gamma

    plot_data = generate_plot_data(shape_values, scale_values)

    with open('network_plot_data.pkl', 'wb') as f:
        pickle.dump(plot_data, f)
