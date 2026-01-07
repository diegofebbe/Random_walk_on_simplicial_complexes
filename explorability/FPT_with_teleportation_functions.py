"""
Code accompanying the papers:

Diego Febbe, Duccio Fanelli, Timoteo Carletti, "Exploring Simplicial Complexes by wandering across the dimensions", arXiv, 2026.
Diego Febbe, Duccio Fanelli, Timoteo Carletti, "Simplicial Complexes with dimension-wise preferential attachment", arXiv, 2026.

If you use this code, please cite the above works (bibtex in README file).
"""
import numpy as np
import networkx as nx

def noise_matrix_fun(N, delta, safe_flag=False, connected_flag=False, self_loops_flag=True):
    A = np.zeros((N, N), int)
    if safe_flag:
        if connected_flag:
            # Create a noise matrix ensuring one connected component
            shuffled_list = np.random.permutation(N)
            for i in range(N - 1):
                A[shuffled_list[i], shuffled_list[i + 1]] = 1
                A[shuffled_list[i + 1], shuffled_list[i]] = 1
        else:
            # Create a noise matrix ensuring no row is all zeros
            for i in range (N):
                if np.sum(A[i, :]) == 0:
                    loc = np.random.randint(N - 1)
                    loc += (loc >= i) # to avoid self-loop
                    A[i, loc] = 1
                    A[loc, i] = 1
        # Compute delta
        safe_delta = np.sum(A) / (N * N)
        if safe_delta > delta:
            print(f"Warning: safe density of noise matrix cannot be lower than {safe_delta:.3f} for Network of size {N}.")
            return A, safe_delta
        else:
            delta_reduced = (delta - safe_delta)
    else:
        delta_reduced = delta
    idx = np.triu_indices(N, 1)
    chosen = np.zeros(len(idx[0]), bool)
    chosen[:int(round(delta_reduced * len(idx[0])))] = True
    np.random.shuffle(chosen)
    A[idx] = chosen
    A += A.T
    if self_loops_flag:
        diag_idx = np.arange(N)
        diag_chosen = np.zeros(N, bool)
        diag_chosen[:int(round(delta_reduced * N))] = True
        np.random.shuffle(diag_chosen)
        A[diag_idx, diag_idx] = diag_chosen
        # Check if any row is all zeros, if so add a self-loop
        for i in range(N):
            if np.sum(A[i, :]) == 0:
                A[i, i] = 1
    return A, delta

def transition_matrix(graph, alpha, delta=1., noise_matrix=None):
    A = nx.adjacency_matrix(graph).todense()
    N = A.shape[0]
    deg = np.array(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    P = A * (1 / deg[:, None])
    if noise_matrix is None:
        noise_matrix, _ = noise_matrix_fun(N, delta)
        # noise_matrix = (lambda M: (np.triu(M) + np.triu(M).T).astype(int))(np.random.rand(N, N) < delta)
    deg_noise = np.array(noise_matrix.sum(axis=1)).ravel()
    deg_noise[deg_noise == 0] = 1.0
    noise_matrix = noise_matrix * (1 / deg_noise[:, None])
    T = alpha * P + (1 - alpha) * noise_matrix
    return T

def time_to_j(transition_matrix, j):
    Z = np.eye(transition_matrix.shape[0]-1) - np.delete(np.delete(transition_matrix, j, axis=0), j, axis=1)
    try:
        Z_inv = np.linalg.inv(Z)
    except:
        print(transition_matrix.shape)
        print(j)
        raise Exception("Matrix inversion failed")
    t_to_j = np.sum(Z_inv, axis=1)
    return t_to_j

def matrix_FPT_fanelli(transition_matrix, dim_target=None):
    ## Based on Di Patti, F., Fanelli, D., & Piazza, F. (2015).
    ## Optimal search strategies on complex multi-linked networks. Scientific reports, 5(1), 9869.
    FPT_matrix = []
    L = transition_matrix.shape[1] if dim_target is None else dim_target
    for j in range(L):
        fpt = time_to_j(transition_matrix, j)
        complete_fpt = np.insert(fpt, j, 0)
        FPT_matrix.append(complete_fpt)
    FPT_matrix = np.column_stack(FPT_matrix)
    FPT_matrix = FPT_matrix[:L, :L]
    return FPT_matrix


def coefficient_for_minimum(graph, noise_matrix, nodes_subset=None, order=2):
    A = nx.adjacency_matrix(graph).todense()
    N = A.shape[0]
    limit = N if nodes_subset is None else nodes_subset
    c_0 = 0
    c_1 = 0
    c_2 = 0
    c_3 = 0
    c_4 = 0
    for j in range(limit):
        k_A = np.diag(np.array(np.sum(A, axis=1)).flatten())
        k_a = np.delete(np.delete(k_A, j, axis=0), j, axis=1)
        A_j = np.delete(np.delete(A, j, axis=0), j, axis=1)
        S_j = np.delete(np.delete(noise_matrix, j, axis=0), j, axis=1)
        k_noise = np.diag(np.array(np.sum(noise_matrix, axis=1)).flatten())
        k_s = np.delete(np.delete(k_noise, j, axis=0), j, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_k_s = np.diag(1.0 / np.diag(k_s))
            inv_k_s[~np.isfinite(inv_k_s)] = 0
            inv_k_a = np.diag(1.0 / np.diag(k_a))
            inv_k_a[~np.isfinite(inv_k_a)] = 0

        B = inv_k_s @ S_j - inv_k_a @ A_j
        C = np.eye(N - 1) - inv_k_s @ S_j
        try:
            C_inv = np.linalg.inv(C)
        except:
            continue
        Step = B @ C_inv
        T0 = C_inv
        T1 = T0 @ Step  # The same as C_inv @ B @ C_inv
        T2 = T1 @ Step  # The same as C_inv @ B @ C_inv @ B @ C_inv
        vec_0 = np.sum(T0, axis=1)
        vec_1 = np.sum(T1, axis=1)
        vec_2 = np.sum(T2, axis=1)
        c_0 += np.sum(vec_0[:limit - 1])
        c_1 += np.sum(vec_1[:limit - 1])
        c_2 += np.sum(vec_2[:limit - 1])
        if order >= 3:
            T3 = T2 @ Step
            vec_3 = np.sum(T3, axis=1)
            c_3 += np.sum(vec_3[:limit - 1])
        if order >= 4:
            T4 = T3 @ Step
            vec_4 = np.sum(T4, axis=1)
            c_4 += np.sum(vec_4[:limit - 1])
    norm = limit * (limit - 1)
    c_0 /= norm
    c_1 /= norm
    c_2 /= norm
    if order == 2:
        return c_0, c_1, c_2
    elif order == 3:
        c_3 /= norm
        return c_0, c_1, c_2, c_3
    elif order >= 4:
        c_3 /= norm
        c_4 /= norm
        return c_0, c_1, c_2, c_3, c_4

    return c_0, c_1, c_2  # Fallback


def coefficients_generic_order(graph, noise_matrix, max_order, nodes_subset=None):
    A = nx.adjacency_matrix(graph).todense()
    N = A.shape[0]
    limit = N if nodes_subset is None else nodes_subset
    # Initialize coefficients [c0, c1, ..., c_max]
    coeffs = np.zeros(max_order + 1)

    for j in range(limit):
        k_A = np.diag(np.array(np.sum(A, axis=1)).flatten())
        k_a = np.delete(np.delete(k_A, j, axis=0), j, axis=1)
        A_j = np.delete(np.delete(A, j, axis=0), j, axis=1)
        S_j = np.delete(np.delete(noise_matrix, j, axis=0), j, axis=1)
        k_noise = np.diag(np.array(np.sum(noise_matrix, axis=1)).flatten())
        k_s = np.delete(np.delete(k_noise, j, axis=0), j, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_k_s = np.diag(1.0 / np.diag(k_s))
            inv_k_s[~np.isfinite(inv_k_s)] = 0
            inv_k_a = np.diag(1.0 / np.diag(k_a))
            inv_k_a[~np.isfinite(inv_k_a)] = 0
        B = inv_k_s @ S_j - inv_k_a @ A_j
        C = np.eye(N - 1) - inv_k_s @ S_j
        try:
            C_inv = np.linalg.inv(C)
        except:
            continue
        Current_T = C_inv
        Step_Matrix = B @ C_inv

        for k in range(max_order + 1):
            ## 1. Sum over k columns -> time vec for all starting node
            vec_k = np.sum(Current_T, axis=1)
            ## 2.Sum over i (rows) -> just the nodes in the subset considered
            coeffs[k] += np.sum(vec_k[:limit - 1])
            ## T^(k+1) = T^(k) @ (B C^-1)
            if k < max_order:
                Current_T = Current_T @ Step_Matrix
    # Final normalization
    norm = limit * (limit - 1)
    coeffs /= norm
    return coeffs