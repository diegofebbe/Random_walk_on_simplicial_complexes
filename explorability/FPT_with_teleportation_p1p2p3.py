"""
Code accompanying the papers:

Diego Febbe, Duccio Fanelli, Timoteo Carletti, "Exploring Simplicial Complexes by wandering across the dimensions", arXiv, 2026.
Diego Febbe, Duccio Fanelli, Timoteo Carletti, "Simplicial Complexes with dimension-wise preferential attachment", arXiv, 2026.

If you use this code, please cite the above works (bibtex in README file).
"""
from build_simplex_graph import define_simplex_graph, define_graph
from explorability.FPT_with_teleportation_functions import (transition_matrix, matrix_FPT_fanelli,
                                                            noise_matrix_fun)
from triangle_probabilities_plot import plot_triangle_probabilites
import matplotlib.pyplot as plt
import numpy as np
#%%
delta = 0.9
N_experiments = 3
N = 50
alpha_vec = np.linspace(0, 1., 11)
p_1_vec = np.arange(0, 1.1, 0.1)
p_2_vec = np.arange(0, 1.1, 0.1)
results = np.empty((len(p_1_vec)*len(p_2_vec), 3 + 4 + 4)) # x_min, y_min/y(alpha=1), *2 (graph and simplex), errors
eps_tol = 1e-12
n_row = 0
for p_1 in p_1_vec:
    print(f"p_1 = {p_1}")
    for p_2 in p_2_vec:
        if p_1 + p_2 > 1 + eps_tol:
            results[n_row] = (p_1, p_2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            n_row += 1
        else:
            p_3 = 1 - p_1 - p_2
            if p_3 < 0:
                p_3 = 0.0
            x_min_graph_list = []
            y_min_graph_list = []
            y_rw_graph_list = []
            x_min_simplex_list = []
            y_min_simplex_list = []
            y_rw_simplex_list = []
            for _ in range(N_experiments):
                kwargs = {'initialization': 'tetrahedron', 'prob_new_link': p_1,
                          'prob_new_triangle': p_2, 'prob_new_tetrahedron': p_3, }
                N_0 = 4 if kwargs['initialization'] == 'tetrahedron' else 3 if kwargs['initialization'] == 'triangle' \
                    else 2 if kwargs['initialization'] == 'link' else 1 if kwargs['initialization'] == 'node' else 0
                N_increase = max(N - N_0, 0)
                simplex_graph, simplex = define_simplex_graph(N=N_increase, **kwargs)
                graph = define_graph(simplex)
                N_simplices = sum(simplex.total_number_of_simplices())
                dim_input_target = N  # The walkers starts and ends just on the nodes
                noise_mat_graph, delta = noise_matrix_fun(N, delta, safe_flag=True, connected_flag=True)
                noise_mat_simplex, delta = noise_matrix_fun(N_simplices, delta, safe_flag=True, connected_flag=True)
                AST_graph = []
                AST_simplex = []
                for alpha in alpha_vec:
                    transition_matrix_simplex = transition_matrix(simplex_graph, alpha, delta,
                                                                  noise_matrix=noise_mat_simplex)
                    transition_matrix_graph = transition_matrix(graph, alpha, delta, noise_matrix=noise_mat_graph)
                    FPT_simplex = matrix_FPT_fanelli(transition_matrix_simplex, dim_input_target)
                    FPT_graph = matrix_FPT_fanelli(transition_matrix_graph)
                    average_search_time_simplex = np.sum(FPT_simplex[np.nonzero(FPT_simplex)]) / (
                                dim_input_target * (dim_input_target - 1))
                    average_search_time_graph = np.sum(FPT_graph[np.nonzero(FPT_graph)]) / (N * (N - 1))
                    AST_simplex.append(average_search_time_simplex)
                    AST_graph.append(average_search_time_graph)
                x_min_graph_list.append(alpha_vec[np.argmin(AST_graph)])
                y_min_graph_list.append(np.min(AST_graph))
                y_rw_graph_list.append(AST_graph[-1])
                x_min_simplex_list.append(alpha_vec[np.argmin(AST_simplex)])
                y_min_simplex_list.append(np.min(AST_simplex))
                y_rw_simplex_list.append(AST_simplex[-1])
            x_min_graph = np.mean(x_min_graph_list)
            x_min_graph_error = np.std(x_min_graph_list)/np.sqrt(len(x_min_graph_list))
            y_ratio_graph_vec = np.array(y_min_graph_list)/np.array(y_rw_graph_list)
            y_ratio_graph = np.mean(y_ratio_graph_vec)
            y_ratio_graph_error = np.std(y_ratio_graph_vec)/np.sqrt(len(y_ratio_graph_vec))
            x_min_simplex = np.mean(x_min_simplex_list)
            x_min_simplex_error = np.std(x_min_simplex_list)/np.sqrt(len(x_min_simplex_list))
            y_ratio_simplex_vec = np.array(y_min_simplex_list)/np.array(y_rw_simplex_list)
            y_ratio_simplex = np.mean(y_ratio_simplex_vec)
            y_ratio_simplex_error = np.std(y_ratio_simplex_vec)/np.sqrt(len(y_ratio_simplex_vec))
            results[n_row] = (p_1, p_2, p_3,
                              x_min_graph, x_min_graph_error, y_ratio_graph, y_ratio_graph_error,
                              x_min_simplex, x_min_simplex_error, y_ratio_simplex, y_ratio_simplex_error)
            n_row += 1
nan_filter = np.isnan(results).any(axis=1)
clean_results = results[~nan_filter]

#%%
fig1, ax1 = plot_triangle_probabilites(data=clean_results[:, np.r_[0:3,3]],
                                       interpolate=True, title=r"$\alpha \Rightarrow AST_{min}$ Graph")
plt.show()

fig2, ax2 = plot_triangle_probabilites(data=clean_results[:, np.r_[0:3,4]],
                                       interpolate=True, title=r"$Error \alpha \Rightarrow AST_{min}$ Graph")
plt.show()

fig3, ax3 = plot_triangle_probabilites(data=clean_results[:, np.r_[0:3,5]],
                                       interpolate=True, title=r"$AST_{min}/AST(\alpha=1)$ Graph")
plt.show()

fig4, ax4 = plot_triangle_probabilites(data=clean_results[:, np.r_[0:3,6]],
                                       interpolate=True, title=r"Error $AST_{min}/AST(\alpha=1)$ Graph")
plt.show()

fig5, ax5 = plot_triangle_probabilites(data=clean_results[:, np.r_[0:3,7]],
                                       interpolate=True, title=r"$\alpha \Rightarrow AST_{min}$ Simplex")
plt.show()

fig6, ax6 = plot_triangle_probabilites(data=clean_results[:, np.r_[0:3,8]],
                                       interpolate=True, title=r"$Error \alpha \Rightarrow AST_{min}$ Simplex")
plt.show()

fig7, ax7 = plot_triangle_probabilites(data=clean_results[:, np.r_[0:3,9]],
                                       interpolate=True, title=r"$AST_{min}/AST(\alpha=1)$ Simplex")
plt.show()

fig8, ax8 = plot_triangle_probabilites(data=clean_results[:, np.r_[0:3,10]],
                                       interpolate=True, title=r"Error $AST_{min}/AST(\alpha=1)$ Simplex")
plt.show()

fig9, ax9 = plot_triangle_probabilites(data=np.column_stack((clean_results[:, 0:3], clean_results[:, 5] / clean_results[:, 9])),
                                       interpolate=True, title=f"Gain Simplex Graph N = {N}, delta = {delta}")
plt.show()
# %%
import matplotlib.image as mpimg
from pathlib import Path

outdir = Path("Plots/FPT/AST")
outdir.mkdir(parents=True, exist_ok=True)  # crea le cartelle se non ci sono

fig1.savefig(outdir / "alpha_AST_min_graph.png", bbox_inches="tight")
fig2.savefig(outdir / "alpha_AST_min_graph_error.png", bbox_inches="tight")
fig3.savefig(outdir / "AST_gain_graph.png", bbox_inches="tight")
fig4.savefig(outdir / "AST_gain_graph_error.png", bbox_inches="tight")
fig5.savefig(outdir / "alpha_AST_min_simplex.png", bbox_inches="tight")
fig6.savefig(outdir / "alpha_AST_min_simplex_error.png", bbox_inches="tight")
fig7.savefig(outdir / "AST_gain_simplex.png", bbox_inches="tight")
fig8.savefig(outdir / "AST_gain_simplex_error.png", bbox_inches="tight")
# fig9.savefig(outdir / "AST_gain_simplex_graph.png", bbox_inches="tight")

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle(rf'N = {N}, $\delta$ = {delta}', fontsize=20)
for ax, fname, title in zip(
    axes.flatten(),
    ["alpha_AST_min_graph.png", "alpha_AST_min_simplex.png", "AST_gain_graph.png", "AST_gain_simplex.png"],
    [r"alpha_AST_min_graph", r"alpha_AST_min_simplex.png", r"AST_gain_graph.png", r"AST_gain_simplex.png"]
):
    img = mpimg.imread(outdir / fname)
    ax.imshow(img)
    # ax.set_title(title)
    ax.axis("off")
    ax.set_frame_on(False)  # no edge

plt.tight_layout()
plt.show()
#
# #%%
## Save results to a txt file with np.savetxt
np.savetxt(f"Data/AST_p1p2p3_delta={delta}_N={N}.txt", clean_results)



