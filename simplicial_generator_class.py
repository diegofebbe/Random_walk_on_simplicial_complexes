"""
Code accompanying the papers:

Diego Febbe, Duccio Fanelli, Timoteo Carletti, "Exploring Simplicial Complexes by wandering across the dimensions", arXiv, 2026.
Diego Febbe, Duccio Fanelli, Timoteo Carletti, "Simplicial Complexes with dimension-wise preferential attachment", arXiv, 2026.

If you use this code, please cite the above works (bibtex in README file).
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt

class SimplicialComplex:
    def __init__(self, **kwargs):
        self.kwargs = kwargs or {
            'initialization': 'tetrahedron',
            'prob_new_link': 1 / 3,
            'prob_new_triangle': 1 / 3,
            'prob_new_tetrahedron': 1 / 3,
            'lower_degree': False,
        }

        nodes = kwargs.get("nodes", None)
        links = kwargs.get("links", None)
        triangles = kwargs.get("triangles", None)
        tetrahedra = kwargs.get("tetrahedra", None)

        self.node_counter = 0
        self.links = []
        self.triangles = []
        self.tetrahedra = []

        self.link_dict = {}
        self.triangle_dict = {}
        self.tetrahedron_dict = {}

        self.node_link_inc = sp.lil_matrix((0, 0), dtype=np.int8)
        self.link_triangle_inc = sp.lil_matrix((0, 0), dtype=np.int8)
        self.triangle_tetra_inc = sp.lil_matrix((0, 0), dtype=np.int8)

        self.prob_growth = [self.kwargs.get('prob_new_link'),
                            self.kwargs.get('prob_new_triangle'), self.kwargs.get('prob_new_tetrahedron')]
        self.lower_degree = self.kwargs.get('lower_degree', False)

        if nodes is not None:
            self._load_structure(nodes, links, triangles, tetrahedra)
        else:
            self._initialize(self.kwargs.get('initialization'))

    @property
    def nodes(self):
        return np.arange(self.node_counter)

    def _initialize(self, kind):
        if kind == 'tetrahedron':
            self._add_tetrahedron(0, 1, 2, 3)
            self.node_counter = 4
        elif kind == 'triangle':
            self._add_triangle(0, 1, 2)
            self.node_counter = 3
        elif kind == 'link':
            self._add_link(0, 1)
            self.node_counter = 2
        elif kind == 'node':
            self._ensure_node_capacity(0)
            self.node_counter = 1
        elif kind is None:
            pass
        else:
            raise ValueError('Initialization not recognized')

    def _load_structure(self, nodes=None, links=None, triangles=None, tetrahedra=None):
        """Load an external structure for the simplicial complex building."""
        if nodes is None:
            nodes = []
        if links is None:
            links = []
        if triangles is None:
            triangles = []
        if tetrahedra is None:
            tetrahedra = []

        self.node_counter = max(nodes) + 1 if nodes else 0

        for u, v in links:
            self._add_link(u, v)
        for a, b, c in triangles:
            self._add_triangle(a, b, c)
        for a, b, c, d in tetrahedra:
            self._add_tetrahedron(a, b, c, d)

    def dimension(self):
        if self.tetrahedra:
            return 3
        elif self.triangles:
            return 2
        elif self.links:
            return 1
        elif self.node_counter > 0:
            return 0
        else:
            return -1

    def _ensure_node_capacity(self, n):
        if n >= self.node_link_inc.shape[0]:
            self.node_link_inc.resize((n + 1, self.node_link_inc.shape[1]))

    def _add_link(self, a, b):
        if a > b:
            a, b = b, a
            sign = -1
        else:
            sign = 1
        link = (a, b)
        if link in self.link_dict:
            return self.link_dict[link]

        idx = len(self.links)
        self.links.append(link)
        self.link_dict[link] = idx

        if self.node_link_inc.shape[1] <= idx:
            self.node_link_inc.resize((self.node_link_inc.shape[0], idx + 1))

        self._ensure_node_capacity(max(a, b))
        self.node_link_inc[a, idx] = -sign
        self.node_link_inc[b, idx] = +sign
        return idx

    def _add_triangle(self, a, b, c):
        tri = tuple(sorted((a, b, c)))
        if tri in self.triangle_dict:
            return self.triangle_dict[tri]
        idx = len(self.triangles)
        self.triangles.append(tri)
        self.triangle_dict[tri] = idx

        l1 = self._add_link(b, c)
        l2 = self._add_link(a, c)
        l3 = self._add_link(a, b)

        if self.link_triangle_inc.shape[1] <= idx:
            self.link_triangle_inc.resize((len(self.links), idx + 1))

        self.link_triangle_inc[l1, idx] = +1
        self.link_triangle_inc[l2, idx] = -1
        self.link_triangle_inc[l3, idx] = +1
        return idx

    def _add_tetrahedron(self, a, b, c, d):
        tet = tuple(sorted((a, b, c, d)))
        if tet in self.tetrahedron_dict:
            return self.tetrahedron_dict[tet]
        idx = len(self.tetrahedra)
        self.tetrahedra.append(tet)
        self.tetrahedron_dict[tet] = idx

        t1 = self._add_triangle(b, c, d)
        t2 = self._add_triangle(a, c, d)
        t3 = self._add_triangle(a, b, d)
        t4 = self._add_triangle(a, b, c)

        if self.triangle_tetra_inc.shape[1] <= idx:
            self.triangle_tetra_inc.resize((len(self.triangles), idx + 1))

        self.triangle_tetra_inc[t1, idx] = +1
        self.triangle_tetra_inc[t2, idx] = -1
        self.triangle_tetra_inc[t3, idx] = +1
        self.triangle_tetra_inc[t4, idx] = -1
        return idx

    def general_degree(self, dim=None, lower_deg_flag = False):
        lower_degree = self.lower_degree or lower_deg_flag
        if dim == 0:
            return np.array(np.abs(self.node_link_inc).sum(axis=1)).flatten()
        elif dim == 1:
            n_links = len(self.links)
            if self.link_triangle_inc.shape[0] < n_links:
                # Enlarge the matrix if needed
                self.link_triangle_inc.resize((n_links, self.link_triangle_inc.shape[1]))
            deg = np.array(np.abs(self.link_triangle_inc[:n_links]).sum(axis=1)).flatten()
            return deg + 2 if lower_degree else deg
        elif dim == 2:
            n_tri = len(self.triangles)
            if self.triangle_tetra_inc.shape[0] < n_tri:
                self.triangle_tetra_inc.resize((n_tri, self.triangle_tetra_inc.shape[1]))
            deg = np.array(np.abs(self.triangle_tetra_inc[:n_tri]).sum(axis=1)).flatten()
            return deg + 3 if lower_degree else deg
        elif dim == 3:
            return np.full(len(self.tetrahedra), 4 if lower_degree else 0, dtype=int)
        else:
            general_degree_nodes = self.general_degree(0, lower_degree)
            general_degree_links = self.general_degree(1, lower_degree)
            general_degree_triangles = self.general_degree(2, lower_degree)
            general_degree_tetrahedrons = self.general_degree(3, lower_degree)
            return np.concatenate((general_degree_nodes, general_degree_links,
                                   general_degree_triangles, general_degree_tetrahedrons))

    def add_simplex(self):
        dim = np.random.choice([1, 2, 3], p=self.prob_growth) # 1,2,3 correspond to link, triangle, tetrahedron
        new_node = self.node_counter

        if dim == 1 and self.node_counter >= 2:
            self._ensure_node_capacity(new_node)
            degrees = self.general_degree(0, self.lower_degree)[:new_node]
            if degrees.sum() == 0:
                base = np.random.randint(0, new_node)
            else:
                prob = degrees / degrees.sum()
                base = np.random.choice(new_node, p=prob)
            self._add_link(base, new_node)
            self.node_counter += 1

        elif dim == 2 and self.links:
            self._ensure_node_capacity(new_node)
            degrees = self.general_degree(1, self.lower_degree)
            if degrees.sum() == 0:
                base = self.links[np.random.randint(len(self.links))]
            else:
                prob = degrees / degrees.sum()
                idx = np.random.choice(len(self.links), p=prob)
                base = self.links[idx]
            self._add_triangle(base[0], base[1], new_node)
            self.node_counter += 1

        elif dim == 3 and self.triangles:
            self._ensure_node_capacity(new_node)
            degrees = self.general_degree(2, self.lower_degree)
            if degrees.sum() == 0:
                base = self.triangles[np.random.randint(len(self.triangles))]
            else:
                prob = degrees / degrees.sum()
                idx = np.random.choice(len(self.triangles), p=prob)
                base = self.triangles[idx]
            self._add_tetrahedron(base[0], base[1], base[2], new_node)
            self.node_counter += 1
        return

    def boundary_matrix_links(self, ordered=True):
        if ordered:
            link_list = sorted(self.links)
            node_list = list(range(self.node_counter))
        else:
            link_list = self.links
            node_list = list(range(self.node_counter))

        node_index = {n: i for i, n in enumerate(node_list)}
        B_1 = np.zeros((len(node_list), len(link_list)))

        for j, (a, b) in enumerate(link_list):
            if a > b:
                a, b = b, a
                sign = -1
            else:
                sign = 1
            B_1[node_index[a], j] = -sign
            B_1[node_index[b], j] = +sign
        return B_1

    def boundary_matrix_triangles(self, ordered=True):
        link_list = sorted(self.links) if ordered else self.links
        triangle_list = sorted(self.triangles) if ordered else self.triangles
        link_index = {l: i for i, l in enumerate(link_list)}
        B_2 = np.zeros((len(link_list), len(triangle_list)))

        for j, tri in enumerate(triangle_list):
            a, b, c = tri
            faces = [((b, c), 1), ((a, c), -1), ((a, b), 1)]
            for (u, v), s in faces:
                if u > v:
                    u, v = v, u
                    s *= -1
                i = link_index[(u, v)]
                B_2[i, j] = s

        return B_2

    def boundary_matrix_tetrahedra(self, ordered=True):
        triangle_list = sorted(self.triangles) if ordered else self.triangles
        tetra_list = sorted(self.tetrahedra) if ordered else self.tetrahedra
        triangle_index = {t: i for i, t in enumerate(triangle_list)}
        B_3 = np.zeros((len(triangle_list), len(tetra_list)))

        for j, tet in enumerate(tetra_list):
            a, b, c, d = tet
            faces = [
                ((b, c, d), 1),
                ((a, c, d), -1),
                ((a, b, d), 1),
                ((a, b, c), -1),
            ]
            for face, s in faces:
                face = tuple(sorted(face))
                i = triangle_index[face]
                B_3[i, j] = s

        return B_3

    def structure(self):
        return {
            'nodes': list(range(self.node_counter)),
            'links': self.links,
            'triangles': self.triangles,
            'tetrahedra': self.tetrahedra
        }

    def total_number_of_simplices(self):
        return [self.node_counter, len(self.links), len(self.triangles), len(self.tetrahedra)]

    def adjacency_matrix(self):
        B = self.node_link_inc.tocsr()
        BBt = B @ B.T
        BBt.setdiag(0)
        return np.abs(BBt.toarray())

    def plot_graph(self, node_color='lightblue', edge_color='gray', node_size=700, font_size=12, title=None):
        A = self.adjacency_matrix()
        G = nx.from_numpy_array(A)
        pos = nx.spring_layout(G)
        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw(G, pos, ax=ax, node_size=node_size, edge_color=edge_color, node_color=node_color, with_labels=False)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size, font_color="black", font_weight="bold")
        if title:
            ax.set_title(str(title), fontsize=20)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example usage
    p_1 = 1 / 3
    p_2 = 1 / 3
    p_3 = 1 - p_1 - p_2
    N = 50
    kwargs = {'initialization': 'tetrahedron', 'prob_new_link': p_1,
              'prob_new_triangle': p_2, 'prob_new_tetrahedron': p_3, }
    N_0 = 4 if kwargs['initialization'] == 'tetrahedron' else 3 if kwargs['initialization'] == 'triangle' else 2 \
        if kwargs['initialization'] == 'link' else 1 if kwargs['initialization'] == 'node' else 0
    N_increase = N - N_0
    simplex = SimplicialComplex(**kwargs)
    for i in range(N_increase):
        simplex.add_simplex()

    simplex.plot_graph()

    # Print the structure of the simplicial complex
    print(simplex.structure())

    # Save the adjacency matrix to a file
    from utilities import save_adjacency_matrix, simplicial_to_multipartite_graph
    multipartite_graph, _ = simplicial_to_multipartite_graph(simplex)
    save_adjacency_matrix(multipartite_graph, f'Data/adjacency_simplex_{N}.txt')
    A_graph = simplex.adjacency_matrix()
    np.savetxt(f'Data/adjacency_graph_{N}.txt', A_graph, delimiter=',', fmt='%d')

    np.savetxt(f'Data/simplex_structure_{N}.txt', simplex.total_number_of_simplices(), delimiter='\n', fmt='%d')