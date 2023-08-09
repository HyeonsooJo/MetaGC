import os
import sys
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

from scipy.sparse import csr_matrix


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalized_adj(adj):
    normalized_D = degree_power(adj, -0.5)
    norm_adj = normalized_D.dot(adj).dot(normalized_D)
    return norm_adj

def normalized_attributes(node_attributes):
    rowsum = np.array(node_attributes.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    node_attributes = r_mat_inv.dot(node_attributes)
    return node_attributes

def load_graph(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join("graph", "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join("graph", "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    node_attributes = sp.vstack((allx, tx)).tolil()
    node_attributes[test_idx_reorder, :] = node_attributes[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    num_nodes = adj.shape[0]
    rows = []
    cols = []
    for src, dst in zip(adj.tocoo().row, adj.tocoo().col):
        if src <= dst: continue
        rows.append(src)
        cols.append(dst)
        rows.append(dst)
        cols.append(src)
    adj = csr_matrix(([1] * len(rows), (rows, cols)), shape=(num_nodes, num_nodes))    

    node_labels = np.vstack((ally, ty))
    node_labels[test_idx_reorder, :] = node_labels[test_idx_range, :]

    node_attributes = normalized_attributes(node_attributes)
    return adj, node_attributes, node_labels
   
def degree_power(adj, pow):
    degrees = np.power(np.array(adj.sum(1)), pow).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(adj): 
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D
    