# heatmap_generator.py

import numpy as np
import networkx as nx
from numba import jit, njit
from multiprocessing import Pool
from scipy.stats import gamma
import pickle
import scipy.stats as stats



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


def generate_graph(shape, scale, total_length, max_edges_per_node = 10):
    edge_lengths = generate_microtubule_lengths(shape, scale, total_length)
    n = len(edge_lengths)

    G = nx.Graph()
    G.add_nodes_from(range(n))

    mean = 9
    std_dev = 2  # adjust as needed
    max_len = 25  # you used this as your max length

    for i in range(n):
        # The counter for the number of edges from the node
        edges_count = 0
        for j in range(i+1, n):
            edge_prob = stats.norm(mean, std_dev).pdf(edge_lengths[i]) / stats.norm(mean, std_dev).pdf(mean)

            # Generate a random number between 0 and 1
            rand_num = np.random.uniform(0, 1)
                
            if rand_num <= edge_prob and edges_count < max_edges_per_node:
                G.add_edge(i, j)
                edges_count += 1
    return G



def analyze_graph(G):
    giant_component_ratio = 0
    for component in nx.connected_components(G):
        if len(component) > giant_component_ratio:
            giant_component_ratio = len(component)
    return giant_component_ratio / G.number_of_nodes()


def compute_plot_data(params):
    shape, scale, max_edges_per_node = params  # assuming you pass max_edges_per_node as a parameter
    ratios = []
    graph_data = []
    for i in range(5):
        G = generate_graph(shape, scale, 1000, max_edges_per_node)  # total_length is set to 1000
        giant_component_ratio = analyze_graph(G)
        ratios.append(giant_component_ratio)
        if i == 0:  # Save the graph and its layout for the first trial
            pos = nx.kamada_kawai_layout(G)  # Using circular_layout instead of spring_layout
            graph_data.append((G, pos))
    avg_ratio = np.mean(ratios)
    std_dev_ratio = np.std(ratios)
    return shape, scale, avg_ratio, std_dev_ratio, graph_data

def generate_plot_data(shape_values, scale_values, max_edges_per_node=25):
    params = [(shape, scale, max_edges_per_node) for shape in shape_values for scale in scale_values]

    with Pool(16) as p:
        plot_data = p.map(compute_plot_data, params)

    return plot_data

if __name__ == '__main__':
    shape_values = [2]  # fixed shape for gamma
    scale_values = np.linspace(0.5, 5.5, 100)  # for scale in gamma

    plot_data = generate_plot_data(shape_values, scale_values)

    with open('network_plot_data_kk.pkl', 'wb') as f:
        pickle.dump(plot_data, f)
