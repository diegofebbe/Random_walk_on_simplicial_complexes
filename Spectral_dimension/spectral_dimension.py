#%%
import numpy as np
import matplotlib.pyplot as plt
from simplicial_generator_class import SimplicialComplex
import scipy.sparse as sp
import time

N_exp = 5
p_vec = np.arange(0.0, 1.1, 0.1)
d_0_vec = np.zeros((N_exp, len(p_vec)))
d_1_vec = np.zeros((N_exp, len(p_vec)))
d_2_vec = np.zeros((N_exp, len(p_vec)))
for i in range(N_exp):
    for j, prob in enumerate(p_vec):
        p_link = 0
        p_triangle = prob
        p_tetrahedron = 1 - p_link - p_triangle
        kwargs = {
            'initialization': 'tetrahedron',
            'prob_new_link': p_link,
            'prob_new_triangle': p_triangle,
            'prob_new_tetrahedron': p_tetrahedron,
            'lower_degree': False
        }

        simplex = SimplicialComplex(**kwargs)
        N = 1996
        start = time.time()

        for _ in range(N):
            simplex.add_simplex()

        print(f"Tempo totale: {time.time() - start:.2f}s")

        # Mostra numero totale di semplici
        print("Numero totale di semplici:", simplex.total_number_of_simplices())
        # %%
        A = simplex.adjacency_matrix()
        general_degrees = np.array(A.sum(axis=1)).flatten()
        D = sp.diags(general_degrees)
        L0 = D - A
        eigenvalues_0 = np.sort(np.linalg.eigvalsh(L0))
        #%%
        count = (np.arange(len(eigenvalues_0))) / L0.shape[0]
        # plt.plot(eigenvalues_0[1:], count[1:], label='Cumulative density')
        # plt.xlabel('$\lambda$', fontsize=15)
        # plt.ylabel(r'$\rho_c(\lambda)$', fontsize=15)
        # plt.xscale('log')
        # plt.yscale('log')
        index_fit = np.where((eigenvalues_0 < 2.5e-0) & (eigenvalues_0>1e-2))[0]
        lambda_fit = eigenvalues_0[index_fit]
        count_fit = count[index_fit]
        d_s_half, const = np.polyfit(np.log(np.where(lambda_fit > 0, lambda_fit, np.nan)),
                                     np.log(np.where(count_fit > 0, count_fit, np.nan)), 1)
        # plt.loglog(lambda_fit, np.exp(const) * lambda_fit**d_s_half, 'r--', label=f'$d_s = {d_s_half *2:.2f}$')
        # plt.legend()
        # plt.title(f'$L_0$, $p_1=$ {prob:.1f}')
        # plt.grid()
        # plt.show()
        d_0_vec[i, j] = d_s_half * 2
        # %%
        if len(simplex.triangles) == 0:
            d_1_vec[i, j] = np.nan
        else:
            B2 = simplex.boundary_matrix_triangles()
            L1_up = B2 @ B2.T
            B1 = simplex.boundary_matrix_links()
            L1_down = B1.T @ B1
            L1 = L1_up + L1_down
            eigenvalues_1_up = np.sort(np.linalg.eigvalsh(L1_up))
            # %%
            eigenvalues_1_up_ = eigenvalues_1_up[np.where(eigenvalues_1_up > 1e-5)[0]]
            count = (np.arange(len(eigenvalues_1_up_)) + 1) / len(eigenvalues_1_up_)
            # plt.plot(eigenvalues_1_up_, count, label='Cumulative density')
            index_fit = np.where((eigenvalues_1_up_ < 3e-0))[0]
            lambda_fit = eigenvalues_1_up_[index_fit]
            count_fit = count[index_fit]
            # plt.xlabel('$\lambda$', fontsize=15)
            # plt.ylabel(r'$\rho_c(\lambda)$', fontsize=15)
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.title(f'$L_1^{{up}}$, $p_1=$ {prob:.1f}')
            d_s_half, const = np.polyfit(np.log(lambda_fit), np.log(count_fit), 1)
            # plt.loglog(lambda_fit, np.exp(const) * lambda_fit ** d_s_half, 'r--', label=f'$d_s = {d_s_half * 2:.2f}$')
            # plt.legend()
            # plt.tight_layout()
            # plt.grid()
            # plt.show()
            d_1_vec[i, j] = d_s_half * 2
        if(len(simplex.tetrahedra)==0):
            d_2_vec[i, j] = np.nan
        else:
            # %%
            B3 = simplex.boundary_matrix_tetrahedra()
            L2_up = B3 @ B3.T
            B2 = simplex.boundary_matrix_triangles()
            eigenvalues_2_up = np.sort(np.linalg.eigvalsh(L2_up))
            eigenvalues_2_up_ = eigenvalues_2_up[np.where(eigenvalues_2_up > 1e-5)[0]]
            count = (np.arange(len(eigenvalues_2_up_)) + 1) / len(eigenvalues_2_up_)
            # plt.plot(eigenvalues_2_up_, count, label='Cumulative density')
            index_fit = np.where((eigenvalues_2_up_ < 3e-0))[0]
            lambda_fit = eigenvalues_2_up_[index_fit]
            count_fit = count[index_fit]
            # plt.xlabel('$\lambda$', fontsize=15)
            # plt.ylabel(r'$\rho_c(\lambda)$', fontsize=15)
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.title(f'$L_2^{{up}}$, $p_2=$ {prob:.1f}')
            if(len(lambda_fit)):
                d_s_half, const = np.polyfit(np.log(lambda_fit), np.log(count_fit), 1)
            else:
                d_s_half, const = np.nan, np.nan
            # plt.loglog(lambda_fit, np.exp(const) * lambda_fit ** d_s_half, 'r--', label=f'$d_s = {d_s_half * 2:.2f}$')
            # plt.legend()
            # plt.tight_layout()
            # plt.grid()
            # plt.show()
            d_2_vec[i, j] = d_s_half * 2
#%%
def nanmean(x, axis=0):
    if x.size == 0:
        return np.nan
    # calcolo la media normale
    mean_vals = np.mean(x, axis=axis)
    # metto nan se in quella colonna/riga c’è almeno un nan
    mask = np.isnan(x).all(axis=axis)
    mean_vals = np.where(mask, np.nan, mean_vals)
    return mean_vals

def nanstd(x, axis=0, ddof=0):
    if x.size == 0:
        return np.nan
    # calcolo la std normale
    std_vals = np.std(x, axis=axis, ddof=ddof)
    # metto nan se in quella colonna/riga c’è almeno un nan
    mask = np.isnan(x).all(axis=axis)
    std_vals = np.where(mask, np.nan, std_vals)
    return std_vals

d_0_mean = nanmean(d_0_vec, axis=0)
d_0_std  = nanstd(d_0_vec, axis=0) / np.sqrt(N_exp)
d_1_mean = nanmean(d_1_vec, axis=0)
d_1_std  = nanstd(d_1_vec, axis=0) / np.sqrt(N_exp)
d_2_mean = nanmean(d_2_vec, axis=0)
d_2_std  = nanstd(d_2_vec, axis=0) / np.sqrt(N_exp)

#%%
plt.errorbar(p_vec, d_0_mean, yerr=d_0_std, label='$d_0$', marker='o')
plt.xlabel(r'$p_2$', fontsize=15)
plt.ylabel(r'$d_s^{[0]}$', fontsize=15)
plt.title(r'$p_2$+$p_3$=1', fontsize=15)
plt.errorbar(0, 4.82, 0.08, color='r', marker='o', alpha=0.5)
plt.grid()
plt.tight_layout()
plt.show()
#%%
plt.errorbar(p_vec, d_1_mean, yerr=d_1_std, label='$d_1$', marker='o')
plt.xlabel(r'$p_2$', fontsize=15)
plt.ylabel(r'$d_s^{[1]}$', fontsize=15)
plt.title(r'$p_2$+$p_3$=1', fontsize=15)
plt.grid()
plt.errorbar(0, 6.04, 0.05, color='r', marker='o', alpha=0.5)
plt.tight_layout()
plt.show()
#%%
plt.errorbar(p_vec, d_2_mean, yerr=d_2_std, label='$d_2$', marker='o')
plt.xlabel(r'$p_2$', fontsize=15)
plt.ylabel(r'$d_s^{[2]}$', fontsize=15)
plt.title(r'$p_2$+$p_3$=1', fontsize=15)
plt.errorbar(0, 7.8, 0.3, color='r', marker='o', alpha=0.5)
plt.grid()
plt.tight_layout()
plt.show()

#%%
import os
os.makedirs("Data", exist_ok=True)
np.savetxt("Data/d0_p2_p3.txt", np.column_stack((p_vec, d_0_mean, d_0_std)), delimiter="\t", fmt="%.6f")
np.savetxt("Data/d1_p2_p3.txt", np.column_stack((p_vec, d_1_mean, d_1_std)), delimiter="\t", fmt="%.6f")
np.savetxt("Data/d2_p2_p3.txt", np.column_stack((p_vec, d_2_mean, d_2_std)), delimiter="\t", fmt="%.6f")