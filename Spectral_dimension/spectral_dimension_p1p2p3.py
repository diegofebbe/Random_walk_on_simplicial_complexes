from build_simplex_graph import define_simplex
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from Spectral_dimension.spectral_dimension_functions import *
from triangle_probabilities_plot import plot_triangle_probabilites

N_experiments = 5
N = 1000
p_1_vec = np.arange(0, 1.05, 0.05)
p_2_vec = np.arange(0, 1.05, 0.05)
# p_1_vec = [0.1]
# p_2_vec = [0.0]
results = np.empty((len(p_1_vec)*len(p_2_vec), 3 + 3 + 3)) # d_0, d_1, d_2, errors
eps_tol = 1e-12
n_row = 0
for p_1 in p_1_vec:
    for p_2 in p_2_vec:
        if p_1 + p_2 > 1 + eps_tol:
            results[n_row] = (p_1, p_2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            n_row += 1
        else:
            p_3 = 1 - p_1 - p_2
            if p_3 < 0:
                p_3 = 0.0
            d_0_vec = []
            d_1_vec = []
            d_2_vec = []
            for _ in range(N_experiments):
                print(f"p_1 = {p_1}", end="\t")
                print(f"p_2 = {p_2}", end="\t")
                print(f"p_3 = {p_3}")
                kwargs = {'initialization': 'triangle', 'prob_new_link': p_1,
                          'prob_new_triangle': p_2, 'prob_new_tetrahedron': p_3, }
                N_0 = 4 if kwargs['initialization'] == 'tetrahedron' else 3 if kwargs['initialization'] == 'triangle' \
                    else 2 if kwargs['initialization'] == 'link' else 1 if kwargs['initialization'] == 'node' else 0
                N_increase = max(N - N_0, 0)
                simplex = define_simplex(N=N_increase, **kwargs)
                A = simplex.adjacency_matrix()
                general_degrees = np.array(A.sum(axis=1)).flatten()
                D = sp.diags(general_degrees)
                L0 = D - A
                eigenvalues_0 = np.sort(np.linalg.eigvalsh(L0))
                count = (np.arange(len(eigenvalues_0))) / L0.shape[0]
                index_fit = np.where((eigenvalues_0 < 2.5e-0) & (eigenvalues_0 > 1e-2))[0]
                lambda_fit = eigenvalues_0[index_fit]
                count_fit = count[index_fit]
                d_s_half, const = np.polyfit(np.log(np.where(lambda_fit > 0, lambda_fit, np.nan)),
                                             np.log(np.where(count_fit > 0, count_fit, np.nan)), 1)
                d_0_vec.append(d_s_half * 2)
                if len(simplex.triangles) == 0:
                    d_1_vec.append(np.nan)
                else:
                    B2 = simplex.boundary_matrix_triangles()
                    L1_up = B2 @ B2.T
                    B1 = simplex.boundary_matrix_links()
                    L1_down = B1.T @ B1
                    L1 = L1_up + L1_down
                    eigenvalues_1_up = np.sort(np.linalg.eigvalsh(L1_up))
                    eigenvalues_1_up_ = eigenvalues_1_up[np.where(eigenvalues_1_up > 1e-5)[0]]
                    count = (np.arange(len(eigenvalues_1_up_)) + 1) / len(eigenvalues_1_up_)
                    index_fit = np.where((eigenvalues_1_up_ < 3e-0))[0]
                    lambda_fit = eigenvalues_1_up_[index_fit]
                    count_fit = count[index_fit]
                    d_s_half, const = np.polyfit(np.log(lambda_fit), np.log(count_fit), 1)
                    d_1_vec.append(d_s_half * 2)
                if (len(simplex.tetrahedra) == 0):
                    d_2_vec.append(np.nan)
                else:
                    B3 = simplex.boundary_matrix_tetrahedra()
                    L2_up = B3 @ B3.T
                    B2 = simplex.boundary_matrix_triangles()
                    eigenvalues_2_up = np.sort(np.linalg.eigvalsh(L2_up))
                    eigenvalues_2_up_ = eigenvalues_2_up[np.where(eigenvalues_2_up > 1e-5)[0]]
                    count = (np.arange(len(eigenvalues_2_up_)) + 1) / len(eigenvalues_2_up_)
                    index_fit = np.where((eigenvalues_2_up_ < 3e-0))[0]
                    lambda_fit = eigenvalues_2_up_[index_fit]
                    count_fit = count[index_fit]
                    if len(lambda_fit):
                        d_s_half, const = np.polyfit(np.log(lambda_fit), np.log(count_fit), 1)
                    else:
                        d_s_half, const = np.nan, np.nan
                    d_2_vec.append(d_s_half * 2)
            d_0_vec = np.array(d_0_vec)
            d_1_vec = np.array(d_1_vec)
            d_2_vec = np.array(d_2_vec)
            d_0_mean = nanmean(d_0_vec, axis=0)
            d_0_std = nanstd(d_0_vec, axis=0) / np.sqrt(N_experiments)
            d_1_mean = nanmean(d_1_vec, axis=0)
            d_1_std = nanstd(d_1_vec, axis=0) / np.sqrt(N_experiments)
            d_2_mean = nanmean(d_2_vec, axis=0)
            d_2_std = nanstd(d_2_vec, axis=0) / np.sqrt(N_experiments)
            results[n_row] = (p_1, p_2, p_3, d_0_mean, d_0_std, d_1_mean, d_1_std, d_2_mean, d_2_std)
            n_row += 1
nan_filter = np.isnan(results).any(axis=1)
clean_results = results[~nan_filter]

np.savetxt(f"Data/spectral_dimension_p1p2p3__N={N}.txt", clean_results)
