"""
Code accompanying the papers:

Diego Febbe, Duccio Fanelli, Timoteo Carletti, "Exploring Simplicial Complexes by wandering across the dimensions", arXiv, 2026.
Diego Febbe, Duccio Fanelli, Timoteo Carletti, "Simplicial Complexes with dimension-wise preferential attachment", arXiv, 2026.

If you use this code, please cite the above works (bibtex in README file).
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.interpolate import griddata
from matplotlib import colors

## --------------------------------------------------
## Vertices of an equilateral triangle
## --------------------------------------------------
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, np.sqrt(3)/2.0])
VERTICES = np.stack([V1, V2, V3])

# --------------------------------------------------
# Conversion p <-> (x,y)
# --------------------------------------------------
def probs_to_xy(p1, p2, p3):
    return p1*V1 + p2*V2 + p3*V3

def xy_to_probs(x, y):
    A = np.array([[V1[0], V2[0], V3[0]],
                  [V1[1], V2[1], V3[1]],
                  [1.0, 1.0, 1.0]])
    b = np.array([x, y, 1.0])
    return np.linalg.solve(A, b)

## --------------------------------------------------
## plot
## --------------------------------------------------
def plot_triangle_probabilites(f=None, data=None, levels=None, cmap="viridis",
                               ngrid=200, title=None, show_colorbar=True, colorbar_range=None,
                               interpolate=False):
    """
    Draw a ternary simplex plot.
    f: function f(p1,p2,p3) -> float
    data: Nx4 array with [p1,p2,p3,value]. If present, f is ignored
    interpolate: if True, interpolate discrete data on internal grid.
    """
    if data is None and f is None:
        f = lambda p1,p2,p3: p1  # default

    ## point grid
    ii, jj = np.meshgrid(np.arange(ngrid+1), np.arange(ngrid+1))
    mask = ii + jj <= ngrid
    p1 = ii[mask] / ngrid
    p2 = jj[mask] / ngrid
    p3 = 1 - p1 - p2

    ## 2D coordinates
    xy = np.array([probs_to_xy(a, b, c) for a, b, c in zip(p1, p2, p3)])
    xs, ys = xy[:, 0], xy[:, 1]

    ## values
    if data is not None:
        if interpolate:
            points = data[:, :2]  # (p1,p2)
            values = data[:, 3]
            vals = griddata(points, values, np.c_[p1, p2], method="cubic")
            mask_nan = np.isnan(vals)
            vals[mask_nan] = griddata(points, values, np.c_[p1[mask_nan], p2[mask_nan]], method="nearest")
        else:
            xs, ys, p1, p2, p3, vals = [], [], [], [], [], []
            for row in data:
                a, b, c, val = row
                x, y = probs_to_xy(a, b, c)
                xs.append(x); ys.append(y)
                p1.append(a); p2.append(b); p3.append(c)
                vals.append(val)
            xs, ys, p1, p2, p3, vals = map(np.array, (xs, ys, p1, p2, p3, vals))
    else:
        vals = np.array([float(f(a, b, c)) for a, b, c in zip(p1, p2, p3)])

    ## triangulation
    tri = mtri.Triangulation(xs, ys)

    ## plot
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

    ## triangle border
    ax.plot([V1[0], V2[0], V3[0], V1[0]],
            [V1[1], V2[1], V3[1], V1[1]], "k-", lw=1.5)

    ## colormap
    if colorbar_range is not None:
        vmin, vmax = colorbar_range
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        sc = ax.tricontourf(tri, vals, levels=100, cmap=cmap, norm=norm)
    else:
        sc = ax.tricontourf(tri, vals, levels=100, cmap=cmap)
    if show_colorbar:
        fig.colorbar(sc, ax=ax, pad=0.13, fraction=0.05, shrink=0.8)


    ## isolinee equiprobabilità
    if levels is None:
        levels = np.linspace(0,1,6)
    ax.tricontour(tri, p1, levels=levels, colors="r", linewidths=0.7, alpha=0.6)
    ax.tricontour(tri, p2, levels=levels, colors="g", linewidths=0.7, alpha=0.6)
    ax.tricontour(tri, p3, levels=levels, colors="b", linewidths=0.7, alpha=0.6)

    ## labels vertices
    ax.text(V1[0]-0.03, V1[1]-0.03, "$p_1$=1", ha="right", va="top", fontsize= 12)
    ax.text(V2[0]+0.03, V2[1]-0.03, "$p_2$=1", ha="left", va="top", fontsize= 12)
    ax.text(V3[0], V3[1]+0.07, "$p_3$=1", ha="center", va="bottom", fontsize= 12)

    ## title
    if title:
        ax.set_title(title, fontsize=15, pad=40)
    ax.axis("off")  # remove ticks and spines
    ax.set_frame_on(False)  # no edge
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    ## Example of usage
    f_example = lambda p1, p2, p3: (
        np.sqrt(max(p1, 0.0)) +
        p2**2 +
        (max(p3, 0.0))**(2/3)
    )
    fig, ax = plot_triangle_probabilites(f=f_example, title="Funzione continua")
    plt.show()

    ## Example with discrete data points
    data = np.array([
        [0,0,1, 1.0],
        [0,1,0, 0.5],
        [1,0,0, 0.2],
        [0.2,0.3,0.5, 0.8],
        [0.3,0.3,0.4, 0.6],
    ])
    fig, ax = plot_triangle_probabilites(data=data, interpolate=True, title="Dati interpolati")
    plt.show()

    ## test conversione p -> xy -> p
    p_test = (0.2, 0.3, 0.5)
    x, y = probs_to_xy(*p_test)
    print("Probabilità -> coordinate:", p_test, "->", (x,y))
    print("Coordinate -> probabilità:", (x,y), "->", xy_to_probs(x,y))
