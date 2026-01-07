"""
Code accompanying the papers:

Diego Febbe, Duccio Fanelli, Timoteo Carletti, "Exploring Simplicial Complexes by wandering across the dimensions", arXiv, 2026.
Diego Febbe, Duccio Fanelli, Timoteo Carletti, "Simplicial Complexes with dimension-wise preferential attachment", arXiv, 2026.

If you use this code, please cite the above works (bibtex in README file).
"""
import networkx as nx
import numpy as np
import pickle as pkl

def simplicial_to_multipartite_graph(simplex, sort_flag = True, **kwargs):
    """
    Convert a simplicial complex to a multipartite graph.

    Parameters:
    - simplex: SimplicialComplex object
    - kwargs: Additional parameters for the conversion

    Returns:
    - A multipartite graph representation of the simplicial complex.

    """
    from simplicial_random_walk_class import RandomWalkSimplex
    A = RandomWalkSimplex(simplex, **kwargs).rw_operator_construction(normalization_flag=False, sign_flag=False).toarray()
    if sort_flag:
        simplicial_input = [sorted(simplex.nodes), sorted(simplex.links),
                            sorted(simplex.triangles), sorted(simplex.tetrahedra)]
    else:
        simplicial_input = [simplex.nodes, simplex.links, simplex.triangles, simplex.tetrahedra]
    G = nx.Graph()
    simplex_list = []

    # Flatten the simplices while keeping type info
    for dim, simplices in enumerate(simplicial_input):
        for s in simplices:
            simplex_list.append((s, dim))

    for i, (s, dim) in enumerate(simplex_list):
        label = tuple((s,) if isinstance(s, (int, np.integer)) else s) # int != np.int
        label_str = ",".join(map(str, label)) # Give the name to the new nodes according to the simplices they represent
        G.add_node(i, type=label_str)

    # Add edges according to A, the generalized adjacency matrix
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[0]):
            if A[i, j] != 0:
                G.add_edge(i, j, weight=A[i, j])

    return G, simplex_list

def multipartite_graph_to_simplicial(graph, n_simplices):
    """
    Convert a multipartite graph back to a simplicial complex.

    Parameters
    ----------
    graph : nx.Graph
        Multipartite graph where each node represents a simplex.
    n_simplices : list of int
        Number of simplices for each dimension [nodes, links, triangles, tetrahedra].

    Returns
    -------
    SimplicialComplex
    """
    from simplicial_generator_class import SimplicialComplex

    n_nodes, n_links, n_triangles, n_tetrahedra = n_simplices

    # Permutation indices
    idx_nodes = list(range(n_nodes))
    idx_links = list(range(n_nodes, n_nodes + n_links))
    idx_triangles = list(range(n_nodes + n_links, n_nodes + n_links + n_triangles))
    idx_tetra = list(range(n_nodes + n_links + n_triangles,
                           n_nodes + n_links + n_triangles + n_tetrahedra))

    # Lists construction
    nodes = idx_nodes[:]  # the vertices are already alone
    links = []
    triangles = []
    tetrahedra = []

    # Dictionary links to nodes
    link_to_nodes = {}

    # Link building from neighboring nodes
    for l in idx_links:
        neigh = [n for n in graph.neighbors(l) if n in idx_nodes]
        if len(neigh) == 2:
            link_nodes = tuple(sorted(neigh))
            links.append(link_nodes)
            link_to_nodes[l] = link_nodes

    # Triangle building from neighboring links
    tri_to_nodes = {}
    for t in idx_triangles:
        neigh_links = [l for l in graph.neighbors(t) if l in idx_links]
        if len(neigh_links) == 3:
            # Each triangle is made of 3 links, get the nodes from those links
            tri_nodes = sorted(set().union(*[link_to_nodes[l] for l in neigh_links]))
            if len(tri_nodes) == 3:
                triangles.append(tuple(tri_nodes))
                tri_to_nodes[t] = tuple(tri_nodes)

    # Tetrahedra building from neighboring triangles
    for tet in idx_tetra:
        neigh_tri = [tr for tr in graph.neighbors(tet) if tr in idx_triangles]
        if len(neigh_tri) == 4:
            tet_nodes = sorted(set().union(*[tri_to_nodes[tr] for tr in neigh_tri]))
            if len(tet_nodes) == 4:
                tetrahedra.append(tuple(tet_nodes))

    # Simplex creation
    kwargs = {'nodes': nodes, 'links': links,
              'triangles': triangles, 'tetrahedra': tetrahedra}

    simplex = SimplicialComplex(**kwargs)
    return simplex, [nodes, links, triangles, tetrahedra]


def write_edge_list(graph, filename, include_reverse=True):
    """
    Write edges of the graph to a file in the format:
    [node_i, node_j; node_j, node_i; ...]
    If include_reverse is True, both directions are written.
    """
    with open(filename, "w") as f:
        for u, v in graph.edges():
            f.write(f"{u} {v}\n")
            if include_reverse:
                f.write(f"{v} {u}\n")

def save_adjacency_matrix(graph, filename):
    """
    Save the adjacency matrix of the graph to a file.
    """
    A = nx.to_numpy_array(graph)
    np.savetxt(filename, A, delimiter=',', fmt='%d')

def save_graph(graph, filename):
    """
    Save the graph to a file in GML format.
    """
    nx.write_gml(graph, filename)

def load_graph(filename):
    """
    Load a graph from a file in GML format.
    """
    return nx.read_gml(filename)


def particular_initialization():
    n = {0, 1, 2, 3}
    l = {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
    tr = {(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)}
    te = {(0, 1, 2, 3)}
    n.update({4, 5, 6, 7, 8})
    l.update({(3, 4), (4, 5), (4, 6), (4, 7), (4, 8)})
    return n, l, tr, te

def triangles_with_link():
    n = {0, 1, 2, 3}
    l = {(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)}
    tr = {(0, 1, 2), (1, 2, 3)}
    te = set()
    return n, l, tr, te

def triangles_with_node():
    n = {0, 1, 2, 3, 4}
    l = {(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4)}
    tr = {(0, 1, 2), (0, 3, 4)}
    te = set()
    return n, l, tr, te

def structure_from_file(filename = 'simplex.pkl'):
    with open(filename, "rb") as f:
        data = pkl.load(f)
    n, l, tr, te = data["n"], data["l"], data["tr"], data["te"]
    return n, l, tr, te