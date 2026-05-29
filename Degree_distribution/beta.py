from Degree_distribution.degree_distribution_functions import *
import time

#%%
N_exp = 10
T = int(1e5)
sampling_time = int(1e2)
thresh_prob_vec_1 = np.arange(0.0, 1.0, 0.1)
thresh_prob_vec_2 = np.arange(0.9, 1.01, 0.01)
thresh_prob_vec = np.concatenate((thresh_prob_vec_1, thresh_prob_vec_2))
beta_vec = np.zeros((N_exp, len(thresh_prob_vec)))
for i in range(N_exp):
    final_degrees_collection = []
    for j, thresh_prob in enumerate(thresh_prob_vec):
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
            if t % sampling_time == 0:
                k_n_t.append(n.copy())
                # k_l_t.append(l.copy())
        growth = evaluate_growth(k_n_t, threshold=0.1, sampling = sampling_time, just_max = True,
                                 title= rf"$p_1$ = {thresh_prob:.3f}", plot_flag=False)
        final_degrees_collection.append(growth[1])
        print(f"Time: {time.time() - start_time:.2f} seconds")
        beta_vec[i, j] = growth[0][0]

#%%
beta_mean = np.mean(beta_vec, axis=0)
beta_std = np.std(beta_vec, axis=0)

# plt.errorbar(thresh_prob_vec, beta_mean, yerr=beta_std, fmt='o-', capsize=5)
# plt.xlabel(r'$p_1$', fontsize=15)
# plt.ylabel(r'$\beta$', fontsize=15)
# plt.grid()
# plt.tight_layout()
# plt.show()
#%%
# np.savetxt("beta_vs_p1.txt", np.column_stack((thresh_prob_vec, beta_mean, beta_std)), delimiter="\t")
plt.figure(figsize=(8, 6))
thresh_prob_vec, beta_mean, beta_std = np.loadtxt('Data/beta_vs_p1.txt',unpack=True)
plt.fill_between(thresh_prob_vec, beta_mean - beta_std, beta_mean + beta_std, color="blue", alpha=0.2)

# linea del valore medio
plt.plot(thresh_prob_vec, beta_mean, color="blue", linewidth=2, label="Valore medio")
# plt.errorbar(thresh_prob_vec, beta_mean, yerr=beta_std, fmt='o-', capsize=5)
plt.plot(thresh_prob_vec, beta_mean, '-o')
plt.xlabel(r'$p_1$', fontsize=25)
plt.ylabel(r'$\beta$', fontsize=25)
plt.title('$p_1 + p_2 = 1$', fontsize=25)
plt.grid()
plt.tight_layout()
plt.show()


