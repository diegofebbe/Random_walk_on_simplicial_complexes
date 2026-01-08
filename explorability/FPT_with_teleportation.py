"""
Code accompanying the papers:

Diego Febbe, Duccio Fanelli, Timoteo Carletti, "Exploring Simplicial Complexes by wandering across the dimensions", arXiv, 2026.
Diego Febbe, Duccio Fanelli, Timoteo Carletti, "Simplicial Complexes with dimension-wise preferential attachment", arXiv, 2026.

If you use this code, please cite the above works (bibtex in README file).
"""
from bootstrap import ensure_root_on_path; ensure_root_on_path() # if you don't use PyCharm
from build_simplex_graph import define_simplex_graph, define_graph
from explorability.FPT_with_teleportation_functions import (transition_matrix, matrix_FPT_fanelli,
                                                            noise_matrix_fun, coefficient_for_minimum,
                                                            coefficients_generic_order)
import matplotlib.pyplot as plt
import numpy as np

# %%
n_exp = 5
p_1 = 0.  # 1 / 3
p_2 = 0.  # 1 / 3
p_3 = 1 - p_1 - p_2
N = 50
kwargs = {'initialization': 'triangle', 'prob_new_link': p_1,
          'prob_new_triangle': p_2, 'prob_new_tetrahedron': p_3, }
N_0 = 4 if kwargs['initialization'] == 'tetrahedron' else 3 if kwargs['initialization'] == 'triangle' \
    else 2 if kwargs['initialization'] == 'link' else 1 if kwargs['initialization'] == 'node' else 0
if p_3>0 and kwargs['initialization'] not in ['triangle', 'tetrahedron']:
    raise ValueError("To have p_3>0 the initialization must be 'triangle' or 'tetrahedron'")
N_increase = max(N - N_0, 0)
alpha_vec = np.linspace(0., 1., 101)
delta = 0.1  # noise parameter
AST_graph_list = np.zeros((n_exp, len(alpha_vec)))
AST_simplex_list = np.zeros((n_exp, len(alpha_vec)))
for i in range(n_exp):
    simplex_graph, simplex = define_simplex_graph(N=N_increase, **kwargs)
    graph = define_graph(simplex)
    N_simplices = sum(simplex.total_number_of_simplices())
    dim_input_target = N # The walkers start and end just on the nodes
    noise_mat_graph, delta = noise_matrix_fun(N, delta, safe_flag=True, connected_flag=True)
    noise_mat_simplex, delta = noise_matrix_fun(N_simplices, delta, safe_flag=True, connected_flag=True)
    for j, alpha in enumerate(alpha_vec):
        print(f'alpha: {alpha:.3f}')
        transition_matrix_simplex = transition_matrix(simplex_graph, alpha, delta, noise_matrix=noise_mat_simplex)
        transition_matrix_graph = transition_matrix(graph, alpha, delta, noise_matrix=noise_mat_graph)
        FPT_simplex = matrix_FPT_fanelli(transition_matrix_simplex, dim_input_target)
        FPT_graph = matrix_FPT_fanelli(transition_matrix_graph)
        average_search_time_simplex = np.sum(FPT_simplex[np.nonzero(FPT_simplex)])/(dim_input_target*(dim_input_target-1))
        average_search_time_graph = np.sum(FPT_graph[np.nonzero(FPT_graph)])/(N*(N-1))
        AST_simplex_list[i][j] = average_search_time_simplex
        AST_graph_list[i][j] = average_search_time_graph
mean_AST_simplex = np.mean(AST_simplex_list, axis=0)
std_AST_simplex = np.std(AST_simplex_list, axis=0)/np.sqrt(n_exp)
mean_AST_graph = np.mean(AST_graph_list, axis=0)
std_AST_graph = np.std(AST_graph_list, axis=0)/np.sqrt(n_exp)
#%%
c_0, c_1, c_2, c_3, c_4 = coefficient_for_minimum(simplex_graph, noise_mat_simplex, dim_input_target, order=4)
def compute_approximation(graph, noise_matrix, max_order, alpha_values, nodes_subset=None):
    # General case
    coefficients = coefficients_generic_order(graph, noise_matrix, max_order, nodes_subset=nodes_subset)
    approx_values = np.zeros_like(alpha_values)
    for n, coef in enumerate(coefficients):
        sign = (-1)**n
        approx_values += sign * coef * (alpha_values**n)
    return approx_values

# %%
limit_plot = 20 #None
parabola = c_0 - c_1*alpha_vec + c_2 * alpha_vec**2 # - c_3 * alpha_vec**3 #+ c_4 * alpha_vec**4
plt.figure(figsize=(8,6))
plt.errorbar(alpha_vec[:limit_plot], mean_AST_simplex[:limit_plot], std_AST_simplex[:limit_plot], label='Mean FRT')
plt.plot(alpha_vec[:limit_plot], mean_AST_simplex[:limit_plot], label='Mean FPT')
plt.plot(alpha_vec[np.argmin(mean_AST_simplex)], np.min(mean_AST_simplex), marker='o', markersize=8, label='OSS', color='red')
plt.plot(alpha_vec[:limit_plot], parabola[:limit_plot], linestyle='--', marker='o', markersize = '3', label='Order 2')
for max_order in [3, 4, 7, 10, 20]:
    approx = compute_approximation(simplex_graph, noise_mat_simplex, max_order, alpha_vec, dim_input_target)
    plt.plot(alpha_vec[:limit_plot], approx[:limit_plot], linestyle='--', marker='o', label=f'Order {max_order}')
plt.legend()
plt.xlabel(r'$\alpha$', fontsize=16)
plt.ylabel(r'$\langle T\rangle$', fontsize=16)
plt.title(rf'Simplicial Complex', fontsize=18)
plt.grid()
plt.tight_layout()
plt.show()
# %%
limit_plot = 18 #None
plt.figure(figsize=(8,6))
plt.errorbar(alpha_vec[:limit_plot], mean_AST_graph[:limit_plot], std_AST_graph[:limit_plot], label='Analytic values')
for max_order in [3, 4, 7, 10]:
    approx = compute_approximation(graph, noise_mat_graph, max_order, alpha_vec, dim_input_target)
    plt.plot(alpha_vec[:limit_plot], approx[:limit_plot], linestyle='--', marker='o', label=f'Order {max_order}')
plt.legend()
plt.xlabel(r'$\alpha$', fontsize=16)
plt.ylabel(r'$\langle T\rangle$', fontsize=16)
plt.title(rf'Graph, $\delta$ = {delta:.2f}', fontsize=18)
plt.grid()
plt.tight_layout()
plt.show()
#%%
alpha_min_simplex = alpha_vec[np.argmin(mean_AST_simplex)]
min_simplex = np.min(mean_AST_simplex)
alpha_min_graph = alpha_vec[np.argmin(mean_AST_graph)]
min_graph = np.min(mean_AST_graph)
left_lim = 1
right_lim = None
plt.errorbar(alpha_vec[left_lim: right_lim], mean_AST_simplex[left_lim:right_lim]/N_simplices,
             std_AST_simplex[left_lim:right_lim]/N_simplices, fmt='-o', label='Simplicial Complex')
plt.errorbar(alpha_vec[left_lim:right_lim], mean_AST_graph[left_lim:right_lim]/N,
             std_AST_graph[left_lim:right_lim]/N, fmt='-o', label='Graph')
plt.plot(alpha_min_simplex, min_simplex/N_simplices, marker='o', markersize=10, color='r', label='OSS')
plt.plot(alpha_min_graph, min_graph/N, marker='o', markersize=10, color='r')
plt.xlabel(r'$\alpha$', fontsize=16)
plt.ylabel(r'$\langle T \rangle$/Size', fontsize=16)
# plt.title(rf'$\delta$ = {delta}', fontsize=18)
plt.legend()
plt.grid()
plt.show()

