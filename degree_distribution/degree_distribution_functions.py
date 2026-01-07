"""
Code accompanying the papers:

Diego Febbe, Duccio Fanelli, Timoteo Carletti, "Exploring Simplicial Complexes by wandering across the dimensions", arXiv, 2026.
Diego Febbe, Duccio Fanelli, Timoteo Carletti, "Simplicial Complexes with dimension-wise preferential attachment", arXiv, 2026.

If you use this code, please cite the above works (bibtex in README file).
"""
from numba import njit
import numpy as np
import matplotlib.pyplot as plt

"""
To analyze the degree distribution arising from a preferential attachment mechanism, simplex growth can be modeled via
an abstract "game" regarding an array growth procedure, avoiding explicit construction of the full simplicial complex.
This abstraction is important for efficiency, as the simulation time required for the generation and the study of a 
simplex of the proper dimension for the correct estimate of the degree distribution can be substantial.
For example, consider a Barabási–Albert network initialized as a triangle with degree array d = [2,2,2].
At each step, an index of d is selected with probability proportional to its value, and a new entry 1 is appended to 
represent the newly added node.
E.G.:
[2, 2, 2] -> [2, 3, 2, 1].
"""

@njit
def weighted_choice(p):
    r = np.random.rand()
    cumsum = np.cumsum(p)
    return np.searchsorted(cumsum, r)

# @njit
def game_nodes(n, l = None, labels = None):
    p = n / np.sum(n)
    idx = weighted_choice(p)
    n[idx] += 1
    new_n = np.empty(len(n) + 1, dtype=np.uint32)
    new_n[:len(n)] = n
    new_n[-1:] = 1
    if l is not None:
        if labels is None:
            print("Labels must be provided if links are provided.")
            return new_n
        new_l = np.empty(len(l) + 1, dtype=np.uint32)
        new_l[:len(l)] = l
        new_l[-1:] = 0
        new_labels = np.array([idx, len(n)], dtype=labels.dtype)
        labels = np.vstack((labels, new_labels))
        return new_n, new_l, labels
    return new_n

@njit
def game_links(x):
    p = x / np.sum(x)
    idx = weighted_choice(p)
    x[idx] += 1
    new_x = np.empty(len(x) + 2, dtype=np.uint32)
    new_x[:len(x)] = x
    new_x[-2:] = 1
    return new_x

@njit
def game_triangles(x):
    p = x / np.sum(x)
    idx = weighted_choice(p)
    x[idx] += 1
    new_x = np.empty(len(x) + 3, dtype=np.uint32)
    new_x[:len(x)] = x
    new_x[-3:] = 1
    return new_x

@njit
def game_tetrahedra(x):
    p = x / np.sum(x)
    idx = weighted_choice(p)
    x[idx] += 1
    new_x = np.empty(len(x) + 4, dtype=np.uint32)
    new_x[:len(x)] = x
    new_x[-4:] = 1
    return new_x

def game_nodes_and_triangles(n, l, labels):
    p = l / np.sum(l)
    idx = np.random.choice(len(l), p=p)
    l[idx] += 1
    label = labels[idx]
    n[label[0]] += 1
    n[label[1]] += 1
    l = np.concatenate((l, np.array([1, 1], dtype=l.dtype)))
    new_labels = np.array([[label[0], len(n)], [label[1], len(n)]], dtype=labels.dtype)
    labels = np.vstack((labels, new_labels))
    n = np.concatenate((n, np.array([2], dtype=n.dtype)))
    return n, l, labels

def game_nodes_and_tetrahedra(n, t, labels):
    p = t / np.sum(t) if np.sum(t) > 0 else np.array([1.0])
    idx = np.random.choice(len(t), p=p)
    t[idx] += 1
    label = labels[idx]
    n[label[0]] += 1
    n[label[1]] += 1
    n[label[2]] += 1
    t = np.concatenate((t, np.array([1, 1, 1], dtype=t.dtype)))
    new_labels = np.array([[label[0], label[1], len(n)], [label[0], label[2], len(n)],
                           [label[1], label[2], len(n)]], dtype=labels.dtype)
    labels = np.vstack((labels, new_labels))
    n = np.concatenate((n, np.array([3], dtype=n.dtype)))
    return n, t, labels

def game_links_and_tetrahedra(l, t, labels_l, labels_t):
    # Compute already present nodes
    n = labels_l.max() + 1
    # Triangle selection
    p = t / np.sum(t) if np.sum(t) > 0 else np.array([1.0])
    idx = np.random.choice(len(t), p=p)
    triangle = labels_t[idx]
    selected_links = {
        (min(triangle[0], triangle[1]), max(triangle[0], triangle[1])),
        (min(triangle[0], triangle[2]), max(triangle[0], triangle[2])),
        (min(triangle[1], triangle[2]), max(triangle[1], triangle[2])),
    }
    # link mapping for quick index retrieval
    link_index_map = {tuple(link): i for i, link in enumerate(labels_l)}
    # link degree increment
    for link in selected_links:
        idx_l = link_index_map.get(link)
        if idx_l is not None:
            l[idx_l] += 1
    # triangle degree increment
    t[idx] += 1
    # New triangle and links creation
    new_node = n
    new_links = []
    new_triangles = []

    for node in triangle:
        # New link (node, new_node)
        new_links.append([min(node, new_node), max(node, new_node)])
        l = np.append(l, 1)

    for i in range(3):
        a, b = triangle[i], triangle[(i + 1) % 3]
        new_triangles.append([a, b, new_node])
        t = np.append(t, 1)

    # Adding batch to labels_l e labels_t
    labels_l = np.vstack((labels_l, np.array(new_links, dtype=np.uint32)))
    labels_t = np.vstack((labels_t, np.array(new_triangles, dtype=np.uint32)))
    return l, t, labels_l, labels_t

def evaluate_growth(k_tempo, threshold=0.1, sampling=100, just_max = True,  title = None, plot_flag = True):
    if just_max:
        max_k_list = [np.array([np.max(k) for k in k_tempo])]
    else:
        max_k0 = np.array([k[0] for k in k_tempo])
        max_k1 = np.array([k[1] for k in k_tempo])
        max_k2 = np.array([k[2] for k in k_tempo])
        max_k_list = [max_k0, max_k1, max_k2]
    max_k_final_vec = []
    # return max_k_list
    for max_k in max_k_list:
        start_idx = int(len(max_k) * threshold)
        t = (np.arange(len(max_k))[start_idx:] + 1) * sampling
        max_k_final = max_k[start_idx:]
        max_k_final_vec.append(max_k_final)
        # log-log fit
        log_t = np.log(t)
        log_k = np.log(max_k_final)
        slope, intercept = np.polyfit(log_t, log_k, deg=1)
        # exponential fit line for plotting
        fit_y = np.exp(intercept + slope * log_t)
        # Plot
        if plot_flag:
            plt.loglog(t, max_k_final, 'o', label='Data')
            plt.loglog(t, fit_y, '--r', label=f'Fit: slope = {slope:.2f}')
            # plt.title(f'growth')
            plt.xlabel('$t$', fontsize=15)
            plt.ylabel('k', fontsize=15)
            if title:
                plt.title(title, fontsize=18)
            plt.legend()
            plt.grid(True, which="both", ls="--", lw=0.5)
            plt.tight_layout()
            # plt.savefig('Plots/d=1_p_3=1_degree_growth_links.png')
            plt.show()
    if just_max:
        max_k = max_k_final_vec[0]
    else:
        max_values = [max(max_k) for max_k in max_k_final_vec]
        max_index = max_values.index(max(max_values))
        max_k = max_k_list[max_index]
    return [slope,intercept], [t, max_k]