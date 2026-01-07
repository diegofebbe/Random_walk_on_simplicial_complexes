"""
Code accompanying the papers:

Diego Febbe, Duccio Fanelli, Timoteo Carletti, "Exploring Simplicial Complexes by wandering across the dimensions", arXiv, 2026.
Diego Febbe, Duccio Fanelli, Timoteo Carletti, "Simplicial Complexes with dimension-wise preferential attachment", arXiv, 2026.

If you use this code, please cite the above works (bibtex in README file).
"""
from degree_distribution.degree_distribution_functions import *
import time

#%%
T = int(2*1e5)
sampling = int(1e2)
thresh_prob_vec = np.arange(0.0, 1.1, 0.1)
thresh_prob_vec = np.sort(np.concatenate([np.copy(thresh_prob_vec), np.arange(0.91, 1.0, 0.01)]))
final_degrees_collection = []
for thresh_prob in thresh_prob_vec:
    n = 2 * np.ones(3, dtype=np.uint32)
    l = np.ones(3, dtype=np.uint32)
    labels = np.array([[0, 1], [0, 2], [1, 2]])
    k_n_t = [n.copy()]
    k_l_t = [l.copy()]
    start_time = time.time()
    for t in range(T):
        r = np.random.rand()
        if r < thresh_prob:
            n, l, labels = game_nodes(n, l, labels)
        else:
            n, l, labels = game_nodes_and_triangles(n, l, labels)
        if t % sampling == 0:
            k_n_t.append(n.copy())
            # k_l_t.append(l.copy())
    growth = evaluate_growth(k_n_t, threshold=0.0, sampling = sampling, just_max = True,
                             title= rf"$p_1$ = {thresh_prob:.3f}", plot_flag=False)
    final_degrees_collection.append(growth[1])
    print(f"Time: {time.time() - start_time:.2f} seconds")
k = np.array(k_n_t[-1])