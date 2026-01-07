"""
Code accompanying the papers:

Diego Febbe, Duccio Fanelli, Timoteo Carletti, "Exploring Simplicial Complexes by wandering across the dimensions", arXiv, 2026.
Diego Febbe, Duccio Fanelli, Timoteo Carletti, "Simplicial Complexes with dimension-wise preferential attachment", arXiv, 2026.

If you use this code, please cite the above works (bibtex in README file).
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from simplicial_generator_class import SimplicialComplex
from functools import cached_property

class RandomWalkSimplex:
    def __init__(self, simplex, **kwargs):
        simplices_lists = [sorted(s) for s in [simplex.nodes, simplex.links, simplex.triangles, simplex.tetrahedra]]
        self.simplices = [item for sublist in simplices_lists for item in sublist]
        random_index = np.random.randint(0, len(self.simplices) - 1)
        self.current_state = (random_index, self.simplices[random_index])
        self.simplex = simplex
        self.path = [self.current_state]
        self.kwargs = kwargs

    def degree(self, flag_separate=False):
        """ This function computes the general degree for the blocks normalization."""
        k_n, k_l, k_tr, k_te = [], [], [], []
        if (isinstance(self.simplex.nodes, int) or len(self.simplex.nodes)) != 0:
            k_n = np.sum(np.abs(self.simplex.boundary_matrix_links()), axis=1)
        if len(self.simplex.links) != 0:
            k_l = np.sum(np.abs(self.simplex.boundary_matrix_triangles()), axis=1) + 2
        if len(self.simplex.triangles) != 0:
            k_tr = np.sum(np.abs(self.simplex.boundary_matrix_tetrahedra()), axis=1) + 3
        if len(self.simplex.tetrahedra) != 0:
            k_te = np.ones(len(self.simplex.tetrahedra)) * 4  # There are not upper dimensions
        if flag_separate:
            return [k_n, k_l, k_tr, k_te]
        k = np.concatenate((k_n, k_l, k_tr, k_te))
        k = k / k.sum()
        return k

    def up_down_operators(self, normalization_flag, sign_flag):
        B_1 = self.simplex.boundary_matrix_links()
        B_2 = self.simplex.boundary_matrix_triangles()
        B_3 = self.simplex.boundary_matrix_tetrahedra()
        if normalization_flag or not sign_flag:
            B_1, B_2, B_3 = np.abs(B_1), np.abs(B_2), np.abs(B_3)  # if normalized must be taken positive
        up_matrices = [B_1.copy(), B_2.copy(), B_3.copy()] # product to the left
        down_matrices = [B_1.T.copy(), B_2.T.copy(), B_3.T.copy()]
        if normalization_flag:
            for i in range(len(up_matrices)):
                row_sums = up_matrices[i].sum(axis=1).flatten()
                lower_degree = i + 1 if i != 0 else 0
                denom = row_sums[:, None] + lower_degree
                valid_rows = denom[:, 0] != 0 # e.g. there are links not belonging to any triangle
                up_matrices[i][valid_rows, :] *= 1 / denom[valid_rows]
            for i in range(len(down_matrices) - 1): # The last dimension has nothing above it
                row_sums = down_matrices[i + 1].sum(axis=0).flatten() # Play attention, the matrices are transposed
                lower_degree = i + 2 # links -> i = 0, and so on
                denom = row_sums[:, None] + lower_degree # the same as row_sums.reshape(-1,1)
                valid_rows = denom[:, 0] != 0
                down_matrices[i][valid_rows, :] *= 1 / denom[valid_rows]
            down_matrices[-1] = down_matrices[-1] * (1 / 4) # # The last dimension
        return up_matrices, down_matrices

    @cached_property
    def rw_operator(self):
        return self.rw_operator_construction()

    def rw_operator_construction(self, normalization_flag=True, sign_flag=False):
        """ This function builds the random walk operator for the simplicial complex
        by building a sparse matrix made of different blocks."""
        matrices_upper, matrices_lower = self.up_down_operators(normalization_flag, sign_flag)
        n = len(matrices_upper) + 1  # +1 for the last row and column
        row_sizes = [M.shape[0] for M in matrices_upper] + [matrices_upper[-1].shape[1]]  # Adjust last dimension
        col_sizes = row_sizes  # The matrix is square

        blocks = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    zero_block = sp.csr_matrix((row_sizes[i], row_sizes[i]))  # Square null matrix (full of zeros)
                elif j == i + 1 and i < len(matrices_upper):
                    zero_block = matrices_upper[i]  # M_i
                elif j == i - 1 and j < len(matrices_lower):
                    zero_block = matrices_lower[j]
                else:
                    zero_block = sp.csr_matrix((row_sizes[i], col_sizes[j]))  # Square null matrix to complete M
                # print(f"Block ({i},{j}) shape: {zero_block.shape}")  # Debug
                row.append(zero_block)
            blocks.append(row)
        operator = sp.bmat(blocks).tocsr()
        return operator

    def walker(self):
        """Moves the walker to the next simplex based on the cached random walk operator."""
        current_index = self.current_state[0]
        start = self.rw_operator.indptr[current_index]
        end = self.rw_operator.indptr[current_index + 1]
        coords = self.rw_operator.indices[start:end]
        values = self.rw_operator.data[start:end]
        if np.abs(values.sum() - 1) > 1e-5:
            print(f"Prob = {values}\nSum = {values.sum()}")
            raise ValueError("Probability not normalized.")
        new_coord = coords[0] if len(coords) == 1 else np.random.choice(coords, p=values)
        self.current_state = (new_coord, self.simplices[new_coord])
        self.path.append(self.current_state)
        return self.current_state

    def eigenvalues_eigenvectors(self):
        operator = self.rw_operator
        if operator.shape[0] <= 3:
            if operator.shape[0] == 3:
                eigenvalues = [np.array([1, -1, 0])]
                eigenvectors = [0]
                print("The simplicial complex is too small.")
                return eigenvalues, eigenvectors
            raise ValueError("The simplicial complex is too small.")
        eigenvalues, eigenvectors = sp.linalg.eigs(operator.T, k=3,which='LM')  # first 3 largest eigenvalues, (1, -1, )
        return eigenvalues, eigenvectors

    def convergence_time(self, toll=1e-6):
        eigenvalues, _ = self.eigenvalues_eigenvectors()
        l_fiedler = np.min(np.abs(eigenvalues))
        return int(np.ceil(np.log(toll) / np.log(l_fiedler)))


class RandomWalkerGraph:
    def __init__(self, adjacency_matrix):
        self.graph = self._matrix_to_adj_list(adjacency_matrix)  # Convert matrix to adjacency list
        self.current_node = np.random.choice(list(self.graph.keys()))
        self.path = [self.current_node]

    def _matrix_to_adj_list(self, matrix):
        """Convert an adjacency matrix to an adjacency list (dict)."""
        adj_list = {}
        n = matrix.shape[0]
        for i in range(n):
            adj_list[i] = [j for j in range(n) if matrix[i, j] > 0]  # Only add neighbors where there's an edge
        return adj_list

    def walker(self):
        """Move to a random neighbor."""
        self.current_node = np.random.choice(self.graph[self.current_node])
        self.path.append(self.current_node)

if __name__ == "__main__":
    #%%
    # Usage
    # Create a simplicial complex
    p_1 = 1 / 3
    p_2 = 1 / 3
    p_3 = 1 - p_1 - p_2
    kwargs = {'initialization': 'tetrahedron', 'prob_new_link': p_1,
              'prob_new_triangle': p_2, 'prob_new_tetrahedron': p_3, }
    # initialization = 'tetrahedron', 'triangle', 'link', 'node'

    simplex = SimplicialComplex(**kwargs)
    for i in range(1000):
        simplex.add_simplex()
    simplex.plot_graph()
    #%%
    # Define the random_walk classes on the simplicial complex and on the graph
    rws = RandomWalkSimplex(simplex, **kwargs)
    rwg = RandomWalkerGraph(simplex.adjacency_matrix())

