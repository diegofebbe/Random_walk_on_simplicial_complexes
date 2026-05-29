import numpy as np

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