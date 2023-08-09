import numpy as np
from scipy.sparse import csr_matrix


def add_noise_edges(adj, node_labels, noise_level, random_seed=0):
    np.random.seed(random_seed)
    num_nodes = adj.shape[0]
    num_edges = adj.sum() / 2
    if noise_level == 1:
        num_added_edges = int(num_edges * 0.3)
    elif noise_level == 2:
        num_added_edges = int(num_edges * 0.6)
    elif noise_level == 3:
        num_added_edges = int(num_edges * 0.9)
    else:
        return None
    node_labels = np.argmax(node_labels, axis=-1)

    rows = []
    cols = []
    edge_dict = dict()
    for src, dst in zip(adj.tocoo().row, adj.tocoo().col):
        if src <= dst: continue
        if src not in edge_dict: edge_dict[src] = set()
        edge_dict[src].add(dst)
        rows.append(src)
        cols.append(dst)
        rows.append(dst)
        cols.append(src)

    num_sampled = 0
    noise_edge_list = []
    while num_sampled < num_added_edges:
        sampled_src, sampled_dst = np.random.choice(num_nodes, 2, replace=False)
        if sampled_src <= sampled_dst:
            tmp = sampled_src
            sampled_src = sampled_dst
            sampled_dst = tmp
        if sampled_src in edge_dict:
            if sampled_dst in edge_dict[sampled_src]:
                continue
            if node_labels[sampled_dst] == node_labels[sampled_src]:
                continue
        if [sampled_src, sampled_dst] not in noise_edge_list:
            noise_edge_list.append([sampled_src, sampled_dst])
            num_sampled += 1
    
    for src, dst in noise_edge_list:
        rows.append(src)
        cols.append(dst)
        rows.append(dst)
        cols.append(src)
    noisy_adj = csr_matrix(([1] * len(rows), (rows, cols)), shape=(num_nodes, num_nodes))    
    return noisy_adj


def adarmic_adar(adj):
    num_nodes = adj.shape[0]
    node_set = set(np.arange(num_nodes))
    graph_dict = dict()
    for row, col in zip(adj.tocoo().row, adj.tocoo().col):
        if row == col:
            print(row, col)
            print("ERROR")
            exit()
        node_set.add(row)
        node_set.add(col)
        if row not in graph_dict: graph_dict[row] = set()
        graph_dict[row].add(col)
    num_nodes = max(node_set) + 1
    adamic_adar = np.zeros([num_nodes, num_nodes])
    for src in node_set:
        for dst in node_set:
            if src >= dst: continue
            aa_score = 0
            if dst in graph_dict and src in graph_dict:
                neigh_set = graph_dict[src].intersection(graph_dict[dst])
                for neigh in neigh_set:
                    aa_score += 1 / len(graph_dict[neigh])
            adamic_adar[src][dst] = aa_score
            adamic_adar[dst][src] = aa_score
    return adamic_adar