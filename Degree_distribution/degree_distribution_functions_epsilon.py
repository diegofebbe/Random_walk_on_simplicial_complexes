import numpy as np
import matplotlib.pyplot as plt


class FenwickTreeFixed:
    """Fenwick tree with fixed maximum size and O(log n) updates/sampling.

    This avoids the expensive rebuild caused by appending to a Fenwick tree at
    every new node/edge.
    """

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.tree = np.zeros(self.capacity + 1, dtype=np.float64)

    def add(self, idx, delta):
        i = int(idx) + 1
        cap = self.capacity
        delta = float(delta)
        while i <= cap:
            self.tree[i] += delta
            i += i & -i

    def prefix_sum(self, idx):
        s = 0.0
        i = int(idx) + 1
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

    def total(self):
        return self.prefix_sum(self.capacity - 1)

    def sample(self):
        """Sample an index with probability proportional to the stored weights."""
        total_weight = self.total()
        if total_weight <= 0.0:
            raise ValueError("Cannot sample from an empty Fenwick tree.")

        r = np.random.random() * total_weight
        idx = 0
        bit = 1 << (self.capacity.bit_length() - 1)
        while bit:
            nxt = idx + bit
            if nxt <= self.capacity and self.tree[nxt] <= r:
                idx = nxt
                r -= self.tree[nxt]
            bit >>= 1
        return idx


def edge_key(u, v):
    u = int(u)
    v = int(v)
    return (u, v) if u < v else (v, u)


def triangle_key(a, b, c):
    x = sorted((int(a), int(b), int(c)))
    return (x[0], x[1], x[2])


def fit_growth_from_max_series(max_k_series, sampling_time=100, threshold=0.1, plot_flag=False, title=None):
    """Estimate beta by fitting max degree growth in log-log scale."""
    max_k = np.asarray(max_k_series, dtype=np.float64)
    start_idx = int(len(max_k) * threshold)

    # The first stored value corresponds to t=0. For the fit we remove it.
    sample_ids = np.arange(len(max_k), dtype=np.float64)
    t = sample_ids * float(sampling_time)

    t_fit = t[start_idx:]
    k_fit = max_k[start_idx:]
    mask = (t_fit > 0.0) & (k_fit > 0.0)

    log_t = np.log(t_fit[mask])
    log_k = np.log(k_fit[mask])
    beta, intercept = np.polyfit(log_t, log_k, deg=1)

    if plot_flag:
        fit_y = np.exp(intercept + beta * log_t)
        plt.loglog(t_fit[mask], k_fit[mask], 'o', label='data')
        plt.loglog(t_fit[mask], fit_y, '--', label=rf'fit: $\beta={beta:.3f}$')
        plt.xlabel(r'$t$', fontsize=15)
        plt.ylabel(r'$k_{\max}$', fontsize=15)
        if title is not None:
            plt.title(title)
        plt.grid(True, which='both', ls='--', lw=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return float(beta), float(intercept), (t_fit[mask], k_fit[mask])


def theoretical_beta_epsilon(p1, epsilon=1.0):
    """Mean-field exponent for p1+p2=1 with edge weight k_e + epsilon."""
    p1 = float(p1)
    p2 = 1.0 - p1
    if p2 <= 0.0:
        return 0.5

    a = p1 / (2.0 * (p1 + 2.0 * p2))
    b = 2.0 * p2 / (epsilon * p1 + (3.0 + 2.0 * epsilon) * p2)
    mat = np.array([[a + epsilon * b, 2.0 * b],
                    [epsilon * b,       2.0 * b]], dtype=np.float64)
    return float(np.max(np.linalg.eigvals(mat).real))


def simulate_growth_epsilon_fast(p1, T=int(1e5), m=3, epsilon=1.0,
                                 sampling_time=100, threshold=0.1,
                                 plot_flag=False, seed=None):
    """Fast list-based simulation for the p1-p2 model with epsilon attractiveness.

    Initial condition: filled triangle on nodes 0,1,2.

    At each macroscopic time step one new node enters and performs m elementary
    attachment attempts:
      - probability p1: node preferential attachment;
      - probability p2: edge preferential attachment with weight k_e + epsilon,
        forming a triangle with the incoming node.

    The function stores only k_max(t), not the full degree vector at every sample.
    """
    if seed is not None:
        np.random.seed(seed)

    p1 = float(p1)
    p2 = 1.0 - p1
    T = int(T)
    m = int(m)

    max_nodes = 3 + T
    # Each elementary triangle event can add at most two new edges involving the
    # incoming node; each node event can add at most one. This is a safe bound.
    max_edges = 3 + 2 * m * T + m * T

    node_degree = np.zeros(max_nodes, dtype=np.uint32)
    node_degree[:3] = 2
    n_nodes = 3
    max_degree = 2

    edge_u = np.zeros(max_edges, dtype=np.uint32)
    edge_v = np.zeros(max_edges, dtype=np.uint32)
    edge_upper = np.zeros(max_edges, dtype=np.uint32)

    edge_u[:3] = np.array([0, 0, 1], dtype=np.uint32)
    edge_v[:3] = np.array([1, 2, 2], dtype=np.uint32)
    edge_upper[:3] = 1
    n_edges = 3

    edge_to_idx = {(0, 1): 0, (0, 2): 1, (1, 2): 2}
    triangle_set = {(0, 1, 2)}

    node_tree = FenwickTreeFixed(max_nodes)
    edge_tree = FenwickTreeFixed(max_edges)

    for idx in range(3):
        node_tree.add(idx, float(node_degree[idx]))
        edge_tree.add(idx, float(edge_upper[idx]) + epsilon)

    max_k_series = [max_degree]

    for t in range(T):
        new_node = n_nodes
        n_nodes += 1

        for _ in range(m):
            if np.random.random() < p1:
                target = node_tree.sample()
                key = edge_key(target, new_node)
                if key in edge_to_idx:
                    continue

                edge_to_idx[key] = n_edges
                edge_u[n_edges] = key[0]
                edge_v[n_edges] = key[1]
                edge_upper[n_edges] = 0
                edge_tree.add(n_edges, epsilon)
                n_edges += 1

                node_degree[target] += 1
                node_degree[new_node] += 1
                node_tree.add(target, 1.0)
                node_tree.add(new_node, 1.0)
                if node_degree[target] > max_degree:
                    max_degree = int(node_degree[target])
                if node_degree[new_node] > max_degree:
                    max_degree = int(node_degree[new_node])

            else:
                idx = edge_tree.sample()
                u = int(edge_u[idx])
                v = int(edge_v[idx])
                tri = triangle_key(u, v, new_node)
                if tri in triangle_set:
                    continue

                triangle_set.add(tri)

                for a, b in ((u, v), (u, new_node), (v, new_node)):
                    key = edge_key(a, b)
                    old_idx = edge_to_idx.get(key)
                    if old_idx is None:
                        edge_to_idx[key] = n_edges
                        edge_u[n_edges] = key[0]
                        edge_v[n_edges] = key[1]
                        edge_upper[n_edges] = 1
                        edge_tree.add(n_edges, 1.0 + epsilon)
                        n_edges += 1

                        node_degree[a] += 1
                        node_degree[b] += 1
                        node_tree.add(a, 1.0)
                        node_tree.add(b, 1.0)
                        if node_degree[a] > max_degree:
                            max_degree = int(node_degree[a])
                        if node_degree[b] > max_degree:
                            max_degree = int(node_degree[b])
                    else:
                        edge_upper[old_idx] += 1
                        edge_tree.add(old_idx, 1.0)

        if (t + 1) % sampling_time == 0:
            max_k_series.append(max_degree)

    beta, intercept, trajectory = fit_growth_from_max_series(
        max_k_series,
        sampling_time=sampling_time,
        threshold=threshold,
        plot_flag=plot_flag,
        title=rf'$p_1={p1:.3f}$, $\epsilon={epsilon}$, $m={m}$'
    )

    return beta, intercept, trajectory
