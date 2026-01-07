"""
Code accompanying the papers:

Diego Febbe, Duccio Fanelli, Timoteo Carletti, "Exploring Simplicial Complexes by wandering across the dimensions", arXiv, 2026.
Diego Febbe, Duccio Fanelli, Timoteo Carletti, "Simplicial Complexes with dimension-wise preferential attachment", arXiv, 2026.

If you use this code, please cite the above works (bibtex in README file).
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import numba as nb

def preprocess_neighbors(G):
    """
    This function preprocesses the neighbors of each node in the graph G
    to create flat arrays suitable for Numba JIT compilation.
    """
    all_neighbors = []
    offsets = np.zeros(G.number_of_nodes(), dtype=np.int64)
    lengths = np.zeros(G.number_of_nodes(), dtype=np.int64)

    pos = 0
    for node in range(G.number_of_nodes()):
        neigh = list(G.neighbors(node))
        offsets[node] = pos
        lengths[node] = len(neigh)
        all_neighbors.extend(neigh)
        pos += len(neigh)

    return np.array(all_neighbors, dtype=np.int64), offsets, lengths

@nb.njit
def simulate_single_walk(start, end, neighbors, offsets, lengths, n_simul, max_steps):
    times = np.empty(n_simul, dtype=np.int64)
    for sim in range(n_simul):
        current_state = start
        t = 0
        while t <= max_steps:
            if current_state == end:
                break
            off = offsets[current_state]
            L = lengths[current_state]
            # uniform random choice among neighbors
            current_state = neighbors[off + np.random.randint(L)]
            t += 1
        times[sim] = t
    return np.mean(times)

def time_i_j_simulated(G, start, end, n_simul=10000, max_steps=1000):
    neighbors, offsets, lengths = preprocess_neighbors(G)
    return simulate_single_walk(start, end, neighbors, offsets, lengths, n_simul, max_steps)

def time_i_j_theoretical(G, i, j):
    A = nx.to_numpy_array(G)
    n = A.shape[0]
    d = A.sum(axis=1)
    m = int(d.sum() / 2) # edges

    D_sqrt_inv = np.diag(1.0 / np.sqrt(d))
    S = D_sqrt_inv @ A @ D_sqrt_inv
    eigvals, eigvecs = np.linalg.eigh(S)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    total = 0.0
    for k in range(1, n):  # from k=2 to N (indice 1-based)
        lam = eigvals[k]
        psi = eigvecs[:, k]
        term = (psi[j] ** 2) / d[j] - (psi[i] * psi[j]) / np.sqrt(d[i] * d[j])
        total += term / (1 - lam)

    H_ij = 2 * m * total
    return H_ij

def matrix_FPT_zhang(G, simulated_flag=True):
    """
    Compute the matrix of first passage times for all pairs of nodes in graph G.
    The theoretical time is computed according to:
    Zhang, Zhongzhi, et al. "Mean first-passage time for random walks on undirected networks."
    The European Physical Journal B 84.4 (2011): 691-697.
    """
    N = G.number_of_nodes()
    matrix_FPT = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                if simulated_flag:
                    matrix_FPT[i, j] = time_i_j_simulated(G, i, j)
                else:
                    matrix_FPT[i, j] = time_i_j_theoretical(G, i, j)
            else:
                matrix_FPT[i, j] = 0
    return matrix_FPT

@nb.njit
def simulate_uniform_mean_FPT_j_numba(start_indices, end_indices, neighbors, offsets, lengths, n_simul, max_steps):
    max_j = np.max(end_indices)
    T_j = np.zeros(max_j + 1)

    for j in end_indices:
        times = np.empty(n_simul, dtype=np.int64)
        for s in range(n_simul):
            # start != j
            while True:
                current = start_indices[np.random.randint(len(start_indices))]
                if current != j:
                    break
            t = 0
            while t < max_steps:
                off = offsets[current]
                L = lengths[current]
                current = neighbors[off + np.random.randint(L)]
                t += 1
                if current == j:
                    break
            times[s] = t
        T_j[j] = np.mean(times)
    return T_j[end_indices]

def simulate_uniform_mean_FPT_j(G, set_start=None, set_end=None, n_simul=10000, max_steps=1000):
    neighbors, offsets, lengths = preprocess_neighbors(G)
    start_indices = np.array(set_start if set_start is not None else np.arange(len(G.nodes)), dtype=np.int64)
    end_indices = np.array(set_end if set_end is not None else np.arange(len(G.nodes)), dtype=np.int64)
    return simulate_uniform_mean_FPT_j_numba(start_indices, end_indices, neighbors, offsets, lengths, n_simul, max_steps)

@nb.njit
def sample_weighted(start_indices, cdf):
    r = np.random.random()
    lo, hi = 0, len(cdf) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if r <= cdf[mid]:
            hi = mid
        else:
            lo = mid + 1
    return start_indices[lo]

@nb.njit
def simulate_weighted_mean_FPT_j_numba(start_indices, end_indices, neighbors, offsets, lengths, n_simul, max_steps, prob_init):
    cdf = np.cumsum(prob_init)
    max_j = np.max(end_indices)
    T_j = np.zeros(max_j + 1)

    for j in end_indices:
        times = np.empty(n_simul, dtype=np.int64)
        for s in range(n_simul):
            # start != j
            while True:
                current = sample_weighted(start_indices, cdf)
                if current != j:
                    break
            t = 0
            while t < max_steps:
                off = offsets[current]
                L = lengths[current]
                current = neighbors[off + np.random.randint(L)]
                t += 1
                if current == j:
                    break
            times[s] = t
        T_j[j] = np.mean(times)
    return T_j[end_indices]

def simulate_weighted_mean_FPT_j(G, set_start=None, set_end=None, n_simul=10000, max_steps=1000):
    neighbors, offsets, lengths = preprocess_neighbors(G)
    start_indices = np.array(set_start if set_start is not None else np.arange(len(G.nodes)), dtype=np.int64)
    end_indices = np.array(set_end if set_end is not None else np.arange(len(G.nodes)), dtype=np.int64)
    degrees = np.array(list(dict(G.degree(start_indices)).values()))
    prob_init = degrees / np.sum(degrees)
    return simulate_weighted_mean_FPT_j_numba(start_indices, end_indices, neighbors, offsets, lengths, n_simul, max_steps, prob_init)


def uniform_mean_FPT_j(matrix_FPT, set_start=None, set_end=None):
    """
    Compute the mean of first passage times starting from a set_start and ending in each of the set_end nodes.
    Args:
        matrix_FPT: matrix of first passage times among all pairs of nodes.
        set_start: the walker can start from each node of this set, with uniform probability.
        set_end: the walker can end in each node, considered individually, of this set.

    Returns: vector of trapping time for each ending node j.
    """
    start_indices = set_start if set_start is not None else np.arange(0, matrix_FPT.shape[0])
    end_indices = set_end if set_end is not None else np.arange(0, matrix_FPT.shape[0])
    T_j_matrix = matrix_FPT[np.ix_(start_indices, end_indices)]
    T_j = np.sum(T_j_matrix, axis=0)
    norm = np.array([1 / (len(start_indices) - 1) if j in start_indices else 1 / len(start_indices) for j in end_indices])
    T_j *= norm
    return T_j

def weighted_mean_FPT_j(matrix_FPT, G, set_start=None, set_end=None):
    """
    Compute the mean of first passage times starting from a set_start and ending in each of the set_end nodes.
    Args:
        matrix_FPT: matrix of first passage times among all pairs of nodes.
        set_start: the walker can start from each node of this set,
        with weighted probability given by the asymptotic state of the random walk.
        set_end: the walker can end in each node, considered individually, of this set.

    Returns: vector of trapping time for each ending node j.
    """
    start_indices = set_start if set_start is not None else np.arange(0, matrix_FPT.shape[0])
    end_indices = set_end if set_end is not None else np.arange(0, matrix_FPT.shape[0])
    degrees = np.array(list(dict(G.degree(start_indices)).values()))
    prob_init = degrees/np.sum(degrees)
    T_j_matrix = matrix_FPT[np.ix_(start_indices, end_indices)]
    weighted_T_j_matrix = np.diag(prob_init) @ T_j_matrix # weighting the rows for the degree
    T_j = np.sum(weighted_T_j_matrix, axis=0)
    norm = [1-prob_init[i] for i in end_indices]
    T_j = T_j / norm
    return T_j

def weighted_mean_FPT_j_with_nodes_to_nodes(matrix_FPT, G, n_n, n_l, n_tr, n_te):
    start_indices = np.arange(0, n_n)
    end_indices = np.arange(0, n_n)
    K = 4 * n_l + 6 * n_tr + 8 * n_te
    degrees = np.array(list(dict(G.degree()).values()))
    prob_init = degrees / np.sum(degrees)
    T_j_matrix = matrix_FPT[np.ix_(start_indices, end_indices)]
    weighted_T_j_matrix = np.diag(prob_init[np.ix_(start_indices)]) @ T_j_matrix  # weighting the rows for the degree
    T_j = np.sum(weighted_T_j_matrix, axis=0)
    norm = [1 - prob_init[i] * K / (2 * n_l) for i in end_indices]
    T_j = K / (2 * n_l) * T_j / norm
    return T_j

#%%
A = np.loadtxt('Data/adjacency_simplex_10.txt', delimiter=',')
# A = np.array([[0, 0, 1],
#               [0, 0, 1],
#               [1, 1, 0]])
G = nx.from_numpy_array(A)

N_n, N_l, N_tr, N_te = np.loadtxt('Data/simplex_structure_10.txt', dtype = np.int32) #[2,1,0,0]

## Check the First Passage Time for each pair of nodes
T_matrix_theo = matrix_FPT_zhang(G, simulated_flag=False)
T_matrix_simul = matrix_FPT_zhang(G, simulated_flag=True)

#%%
plt.plot(T_matrix_theo.flatten(), T_matrix_simul.flatten(), 'o')
plt.plot(T_matrix_theo.flatten(), T_matrix_theo.flatten(), '-')
plt.grid()
plt.title('First Passage Times', fontsize=15)
plt.xlabel('Theoretical', fontsize=15)
plt.ylabel('Simulated', fontsize=15)
plt.show()

#%%
## Check the First Trapping Time to a certain node (each of a given set) starting from a given set
mean_uniform_T_j_theo = uniform_mean_FPT_j(T_matrix_theo, set_start=None, set_end=None)
mean_uniform_T_j_simul = simulate_uniform_mean_FPT_j(G, set_start=None, set_end=None)

#%%
plt.plot(mean_uniform_T_j_theo, mean_uniform_T_j_simul, 'o')
plt.plot(mean_uniform_T_j_theo, mean_uniform_T_j_theo, '-')
plt.grid()
plt.title('Uniform First Trapping Times', fontsize=15)
plt.xlabel('Theoretical', fontsize=15)
plt.ylabel('Simulated', fontsize=15)
plt.show()

#%%
mean_weighted_T_j_simul = simulate_weighted_mean_FPT_j(G, set_start=None, set_end=None)
mean_weighted_T_j_theo = weighted_mean_FPT_j(T_matrix_theo, G, set_start=None, set_end=None)
#%%
plt.plot(mean_weighted_T_j_theo, mean_weighted_T_j_simul, 'o')
plt.plot(mean_weighted_T_j_theo, mean_weighted_T_j_theo, '-')
plt.grid()
plt.title('Weighted Firs Trapping Times', fontsize=15)
plt.xlabel('Theoretical', fontsize=15)
plt.ylabel('Simulated', fontsize=15)
plt.show()
#%%

# mean_weighted_T_j_simul_nodes = simulate_weighted_mean_FPT_j(G, set_start=np.arange(0,N_n), set_end=np.arange(0,N_n))
mean_weighted_T_j_theo_nodes = weighted_mean_FPT_j(T_matrix_theo, G, set_start=np.arange(0,N_n), set_end=np.arange(0,N_n))
mean_weighted_T_j_simplex = weighted_mean_FPT_j_with_nodes_to_nodes(T_matrix_theo, G, N_n, N_l, N_tr, N_te)
plt.plot(mean_weighted_T_j_theo_nodes, mean_weighted_T_j_simplex, 'o')
plt.plot(mean_weighted_T_j_theo_nodes, mean_weighted_T_j_theo_nodes, '-')
plt.grid()
plt.title('Weighted formula check', fontsize=15)
plt.xlabel('Theoretical', fontsize=15)
plt.ylabel('New Formula', fontsize=15)
plt.show()

#%%
##Nodes
## Check the First Trapping Time to a certain node (each of a given set) starting from a given set
mean_uniform_T_j_theo = uniform_mean_FPT_j(T_matrix_theo, set_start=np.arange(0, N_n), set_end=np.arange(0, N_n))
mean_uniform_T_j_simul = simulate_uniform_mean_FPT_j(G, set_start=np.arange(0, N_n), set_end=np.arange(0, N_n), n_simul=50000)

#%%
plt.plot(mean_uniform_T_j_theo, mean_uniform_T_j_simul, 'o')
plt.plot(mean_uniform_T_j_theo, mean_uniform_T_j_theo, '-')
plt.grid()
plt.title('Uniform First Trapping Times', fontsize=15)
plt.xlabel('Theoretical', fontsize=15)
plt.ylabel('Simulated', fontsize=15)
plt.show()

#%%
mean_weighted_T_j_simul = simulate_weighted_mean_FPT_j(G, set_start=np.arange(0, N_n), set_end=np.arange(0, N_n), n_simul=50000)
mean_weighted_T_j_theo = weighted_mean_FPT_j(T_matrix_theo, G, set_start=np.arange(0, N_n), set_end=np.arange(0, N_n), )
#%%
plt.plot(mean_weighted_T_j_theo, mean_weighted_T_j_simul, 'o')
plt.plot(mean_weighted_T_j_theo, mean_weighted_T_j_theo, '-')
plt.grid()
plt.title('Weighted First Trapping Times', fontsize=15)
plt.xlabel('Theoretical', fontsize=15)
plt.ylabel('Simulated', fontsize=15)
plt.show()

#%%
plt.plot(mean_uniform_T_j_theo, mean_weighted_T_j_simul, 'o')
plt.plot(mean_uniform_T_j_theo, mean_uniform_T_j_theo, '-')
plt.grid()
plt.title('Difference uniform weighted', fontsize=15)
plt.xlabel('Uniform', fontsize=15)
plt.ylabel('Weighted', fontsize=15)
plt.show()