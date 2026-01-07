"""
Code accompanying the papers:

Diego Febbe, Duccio Fanelli, Timoteo Carletti, "Exploring Simplicial Complexes by wandering across the dimensions", arXiv, 2026.
Diego Febbe, Duccio Fanelli, Timoteo Carletti, "Simplicial Complexes with dimension-wise preferential attachment", arXiv, 2026.

If you use this code, please cite the above works (bibtex in README file).
"""

from build_simplex_graph import define_simplex_graph, define_graph
from triangle_probabilities_plot import plot_triangle_probabilites
from explorability.FPT_functions import matrix_FPT_zhang

import matplotlib.pyplot as plt
import numpy as np

#%%
N_experiments = 5
p_1_vec = np.arange(0, 1.1, 0.1)
p_2_vec = np.arange(0, 1.1, 0.1)
results = np.empty((len(p_1_vec)*len(p_2_vec), 3 + 6 + 2)) # Average of FPT simplex, Average of FPT graph, errors, N_structures with error
eps_tol = 1e-12
n_row = 0
for p_1 in p_1_vec:
    for p_2 in p_2_vec:
        if p_1 + p_2 > 1 + eps_tol:
            results[n_row] = (p_1, p_2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            n_row += 1
        else:
            p_3 = 1 - p_1 - p_2
            if p_3 < 0:
                p_3 = 0.0
            average_FPT_simplex_list = []
            average_FPT_graph_list = []
            average_FPT_simplex_normalized_list = []
            average_FPT_graph_normalized_list = []
            N_structures = []
            for _ in range(N_experiments):
                kwargs = {'initialization': 'tetrahedron', 'prob_new_link': p_1,
                          'prob_new_triangle': p_2, 'prob_new_tetrahedron': p_3, }
                N_0 = 4 if kwargs['initialization'] == 'tetrahedron' else 3 if kwargs['initialization'] == 'triangle' \
                    else 2 if kwargs['initialization'] == 'link' else 1 if kwargs['initialization'] == 'node' else 0
                N = 20 - N_0
                simplex_graph, simplex = define_simplex_graph(N=N, **kwargs)
                graph = define_graph(simplex)
                N_simplices = sum(simplex.total_number_of_simplices())
                T_matrix_theo_simplex = matrix_FPT_zhang(simplex_graph, simulated_flag=False, len_subset_nodes=len(graph.nodes))
                T_matrix_theo_graph = matrix_FPT_zhang(graph, simulated_flag=False)
                mean_FPT_simplex = np.mean(T_matrix_theo_simplex[T_matrix_theo_simplex > 0])
                average_FPT_simplex_list.append(mean_FPT_simplex)
                average_FPT_simplex_normalized_list.append(mean_FPT_simplex/N_simplices)
                mean_FPT_graph = np.mean(T_matrix_theo_graph[T_matrix_theo_graph > 0])
                average_FPT_graph_list.append(mean_FPT_graph)
                average_FPT_graph_normalized_list.append(mean_FPT_graph/(N+N_0))
                N_structures.append(N_simplices)
            average_FPT_simplex = np.mean(average_FPT_simplex_list)
            average_FPT_simplex_error = np.std(average_FPT_simplex_list)/np.sqrt(len(average_FPT_simplex_list))
            average_FPT_simplex_normalized = np.mean(average_FPT_simplex_normalized_list)
            average_FPT_graph = np.mean(average_FPT_graph_list)
            average_FPT_graph_error = np.std(average_FPT_graph_list)/np.sqrt(len(average_FPT_graph_list))
            average_FPT_graph_normalized = np.mean(average_FPT_graph_normalized_list)
            N_structures_mean = np.mean(N_structures)
            N_structures_error = np.std(N_structures)/np.sqrt(len(N_structures))
            results[n_row] = (p_1, p_2, p_3,
                              average_FPT_simplex, average_FPT_simplex_error, average_FPT_simplex_normalized,
                              average_FPT_graph, average_FPT_graph_error, average_FPT_graph_normalized,
                              N_structures_mean, N_structures_error)
            n_row += 1
nan_filter = np.isnan(results).any(axis=1)
clean_results = results[~nan_filter]

#%%
fig1, ax1 = plot_triangle_probabilites(data=clean_results[:, np.r_[0:3,3]],
                                       interpolate=True, title=r"$\langle T_{ij} \rangle$ Simplex")
plt.show()

fig2, ax2 = plot_triangle_probabilites(data=clean_results[:, np.r_[0:3,4]],
                                       interpolate=True, title=r"Error $ \langle T_{ij} \rangle$ Simplex")
plt.show()

fig3, ax3 = plot_triangle_probabilites(data=clean_results[:, np.r_[0:3,5]],
                                       interpolate=True, title=r"$\langle T_{ij} \rangle$ Simplex Normalized")
plt.show()

fig4, ax4 = plot_triangle_probabilites(data=clean_results[:, np.r_[0:3,6]],
                                       interpolate=True, title=r"$\langle T_{ij} \rangle$ Graph")
plt.show()

fig5, ax5 = plot_triangle_probabilites(data=clean_results[:, np.r_[0:3,7]],
                                       interpolate=True, title=r"Error $\langle T_{ij} \rangle$ Graph")
plt.show()

fig6, ax6 = plot_triangle_probabilites(data=clean_results[:, np.r_[0:3,8]],
                                       interpolate=True, title=r"$\langle T_{ij} \rangle$ Graph Normalized")
plt.show()

fig7, ax7 = plot_triangle_probabilites(data=clean_results[:, np.r_[0:3,9]],
                                       interpolate=True, title=r"Number of simplices")
plt.show()

fig8, ax8 = plot_triangle_probabilites(data=clean_results[:, np.r_[0:3,10]],
                                       interpolate=True, title=r"Error Number of simplices")
plt.show()

#%%
import matplotlib.image as mpimg
# save already created figures
from pathlib import Path

outdir = Path("Plots/FPT/Structures")
outdir.mkdir(parents=True, exist_ok=True)  # create directory if it doesn't exist

fig1.savefig(outdir / "Average_FPT_simplex.png", bbox_inches="tight")
fig2.savefig(outdir / "Average_FPT_simplex_error.png", bbox_inches="tight")
fig3.savefig(outdir / "Average_FPT_graph.png", bbox_inches="tight")
fig4.savefig(outdir / "Average_FPT_graph_error.png", bbox_inches="tight")
fig5.savefig(outdir / "Number_of_simplices.png", bbox_inches="tight")
fig6.savefig(outdir / "Number_of_simplices_error.png", bbox_inches="tight")

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for ax, fname, title in zip(
    axes.flatten(),
    ["Average_FPT_simplex.png", "Average_FPT_simplex_error.png", "Average_FPT_graph.png", "Average_FPT_graph_error.png"],
    [r"$\langle T_{ij} \rangle$ Simplex", r"Error $ \langle T_{ij} \rangle$ Simplex",
     r"$\langle T_{ij} \rangle$ Graph", r"Error $\langle T_{ij} \rangle$ Graph"]
):
    img = mpimg.imread(outdir / fname)
    ax.imshow(img)
    # ax.set_title(title)
    ax.axis("off")
    ax.set_frame_on(False)  # no edge

plt.tight_layout()
plt.show()
