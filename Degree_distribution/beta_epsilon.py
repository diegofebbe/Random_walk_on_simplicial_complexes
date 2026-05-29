from degree_distribution_functions_epsilon import *
import os
import time
import numpy as np
import matplotlib.pyplot as plt


# Minimal scan of beta(p1), with p1+p2=1 and epsilon=1.
# Optimized version: stores only k_max(t), uses fixed-capacity Fenwick trees,
# and prints progress at run/p1 level.

N_exp = 10
T = int(1e5)
sampling_time = int(1e2)
m = 3
epsilon = 0.0
threshold = 0.2

# Coarse grid plus denser points close to p1=1.
p1_vec_1 = np.arange(0.0, 1.0, 0.1)
p1_vec_2 = np.arange(0.9, 1.01, 0.01)
p1_vec = np.unique(np.concatenate((p1_vec_1, p1_vec_2)))

beta_vec = np.zeros((N_exp, len(p1_vec)), dtype=float)

start_all = time.time()
print(
    f"Starting beta scan: epsilon={epsilon}, m={m}, T={T}, "
    f"N_exp={N_exp}, n_p1={len(p1_vec)}",
    flush=True,
)

for i in range(N_exp):
    print(f"\n=== Run {i + 1}/{N_exp} ===", flush=True)
    for j, p1 in enumerate(p1_vec):
        start = time.time()
        beta, intercept, _ = simulate_growth_epsilon_fast(
            p1=p1,
            T=T,
            m=m,
            epsilon=epsilon,
            sampling_time=sampling_time,
            threshold=threshold,
            plot_flag=False,
            seed=None,
        )
        beta_vec[i, j] = beta
        print(
            f"run={i + 1}/{N_exp}, p1_index={j + 1}/{len(p1_vec)}, "
            f"p1={p1:.3f}, beta={beta:.4f}, time={time.time() - start:.2f} s",
            flush=True,
        )

beta_mean = np.mean(beta_vec, axis=0)
beta_std = np.std(beta_vec, axis=0)
beta_theory = np.array([theoretical_beta_epsilon(p1, epsilon=epsilon) for p1 in p1_vec])

os.makedirs("Data", exist_ok=True)
output = np.column_stack((p1_vec, beta_mean, beta_std, beta_theory))
np.savetxt(
    "Data/beta_epsilon=1.txt",
    output,
    delimiter="\t",
    # header="p1\tbeta_mean\tbeta_std\tbeta_theory",
)

print(f"\nSaved Data/beta_epsilon={epsilon}.txt", flush=True)
print(f"Total time: {time.time() - start_all:.2f} s", flush=True)

#%%
plt.fill_between(p1_vec, beta_mean - beta_std, beta_mean + beta_std, alpha=0.2)
plt.plot(p1_vec, beta_mean, '-o', label='simulation')
# plt.plot(p1_vec, beta_theory, '--', linewidth = 3, label='theory')
plt.xlabel(r'$p_1$', fontsize=15)
plt.ylabel(r'$\beta$', fontsize=15)
# plt.title(rf'$\epsilon$ = {epsilon}', fontsize=15)
plt.grid()
plt.legend(fontsize = 15)
plt.tight_layout()
plt.show()
