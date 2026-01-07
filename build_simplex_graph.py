"""
Code accompanying the papers:

Diego Febbe, Duccio Fanelli, Timoteo Carletti, "Exploring Simplicial Complexes by wandering across the dimensions", arXiv, 2026.
Diego Febbe, Duccio Fanelli, Timoteo Carletti, "Simplicial Complexes with dimension-wise preferential attachment", arXiv, 2026.

If you use this code, please cite the above works (bibtex in README file).
"""

from utilities import simplicial_to_multipartite_graph
from simplicial_generator_class import SimplicialComplex
import networkx as nx

def define_simplex_graph(N, **kwargs):
    simplex = SimplicialComplex(**kwargs)
    for i in range(N):
        simplex.add_simplex()
    simplex_graph, _ = simplicial_to_multipartite_graph(simplex, **kwargs)
    return simplex_graph, simplex

def define_simplex(N, **kwargs):
    simplex = SimplicialComplex(**kwargs)
    for i in range(N):
        simplex.add_simplex()
    return simplex

def define_graph(simplex):
    A = simplex.adjacency_matrix()
    graph = nx.from_numpy_array(A)
    return graph