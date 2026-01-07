# Random walks on Simplicial Complexes

This repository contains the Python codes used in the numerical experiments
of the papers

> Diego Febbe, Duccio Fanelli, Timoteo Carletti,  
> *Exploring Simplicial Complexes by wandering across the dimensions*,  
> arXiv, 2026.
>
> Diego Febbe, Duccio Fanelli, Timoteo Carletti,  
> *Simplicial Complexes with dimension-wise preferential attachment*,  
> arXiv, 2026.

If you use this code, please cite these papers (bibtex are in section Reference).

## Description

Here we provide a Simplicial Complexes building algorithm based on a dimension-wise preferential attachment mechanism,
together with the definition of a new Random Walk process defined upon them.

The code allows to generate Simplicial Complexes of arbitrary dimension and size, and to simulate random walks.

Other dynamical processes are also considered by mixing local random walk moves and global jumps across the dimensions.
Topological properties of the generated Simplicial Complexes are also studied.


## License

The code is released under a permissive MIT License to promote reuse and further development within the open science community.

## Reference

```bibtex
@article{febbe2026exploring,
  author  = {Febbe, Diego and Fanelli, Duccio and Carletti, Timoteo},
  title   = {Exploring Simplicial Complexes by wandering across the dimensions},
  journal = {arXiv},
  year    = {2026},
}

@article{febbe2026simplicial,
  author  = {Febbe, Diego and Fanelli, Duccio and Carletti, Timoteo},
  title   = {Simplicial Complexes with dimension-wise preferential attachments},
  journal = {arXiv},
  year    = {2026},
}
```

## Organization into branches

The repository is organized into two main branches:

- `master/` : main source codes,
- `experiments/` : scripts to reproduce the numerical experiments'

## Requirements

The code has been tested with Python 3.10 and requires the following packages:

numpy
scipy
matplotlib
networkx
collections
time
pickle
functools
numba

