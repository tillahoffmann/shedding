import matplotlib as mpl
from matplotlib import pyplot as plt
import numbers
import numpy as np
import re
import sys


def skip_doctest(obj):
    """
    Decorator to skip doctests.
    """
    if 'doctest' in sys.argv:
        obj.__doc__ = "Skipping doctest."
    return obj


def dict_to_array(mapping, size=None, fill_value=0, dtype=None):
    """
    Convert a key-value mapping to a numpy array.

    Parameters
    ----------
    mapping : dict
        Mapping from indices to values.
    size : int or tuple
        Size of the resulting array or `max(mapping) + 1` if omitted.
    fill_value :
        Fill value for missing indices.
    dtype : type
        Data type of the resulting array.

    Returns
    -------
    x : ndarray[dtype]<size>
        Data after conversion to an array.
    """
    if size is None:
        size = max(mapping) + 1
    x = np.ones(size, dtype) * fill_value
    for i, value in mapping.items():
        x[i] = value
    return x


def extract_kvps(x, pattern='(.*?)_rep$'):
    """
    Extract key-value pairs matching a pattern from a dictionary or list of dictionaries.

    Parameters
    ----------
    x : dict or list[dict]
        Dictionary or list of dictionaries from which to extract key-value pairs.
    pattern : str
        Regular expression pattern to match keys against. The first capture group determines the
        key in the resulting dictionary.

    Returns
    -------
    x : dict or list[dict]
        Dictionary or list of dictionaries after extraction of key-value pairs.

    Examples
    --------
    >>> extract_kvps({'a_': 1, 'b': 2}, r'(\\w)_')
    {'a': 1}
    """
    pattern = re.compile(pattern)
    if isinstance(x, dict):
        result = {}
        for key, value in x.items():
            match = re.search(pattern, key)
            if not match:
                continue
            result[match.group(1)] = value
        return result
    elif isinstance(x, list):
        return [extract_kvps(y, pattern) for y in x]
    else:
        raise ValueError(type(x))


def qq_plot(y, dist, yerr=None, ax=None, **kwargs):
    """
    Generate a Q-Q plot for the empirical data and a theoretical distribution.

    Parameters
    ----------
    y : ndarray
        Empirical data.
    dist :
        Theoretical distribution.
    yerr : ndarray
        Errors on empirical data.
    ax :
        Axes to plot into.
    **kwargs : dict
        Additional arguments passed to `ax.errorbar`.
    """
    ax = ax or plt.gca()
    kwargs.setdefault('ls', 'none')
    kwargs.setdefault('marker', mpl.rcParams['scatter.marker'])
    y = np.sort(y)
    # Generate empirical quantiles (using 0.5 offset to avoid infs)
    empirical = (0.5 + np.arange(len(y))) / len(y)
    x = dist.ppf(empirical)
    # Plot the empirical values against the corresponding theoretical ones
    errorbar = ax.errorbar(x, y, yerr, **kwargs)
    ax.set_aspect('equal')
    mm = x[0], x[-1]
    ax.plot(mm, mm, color='k', ls=':')
    return errorbar


def replication_percentile_plot(data, replicates, key=None, percentiles=10, ax=None, **kwargs):
    """
    Generate a violin plot of replicated percentiles against empirical percentiles.

    Parameters
    ----------
    data :
        Empirical data.
    replicates : list
        Posterior predictive replicates of the data.
    key : callable
        Callable to extract summary statistics from the data and replicates.
    percentiles : ndarray or int
        Percentiles of the summary statistic to plot or an integer number of quantiles.
    ax :
        Axes to plot into.
    **kwargs : dict
        Additional arguments passed to `ax.violinplot`.
    """
    ax = ax or plt.gca()
    if isinstance(percentiles, numbers.Number):
        percentiles = np.linspace(0, 100, percentiles)

    if key:
        data = key(data)
        replicates = [key(x) for x in replicates]

    x = np.percentile(data, percentiles, axis=0)
    y = np.asarray([np.percentile(x, percentiles) for x in replicates])
    mm = x.min(), x.max()
    ax.plot(mm, mm, ls=':', color='k')
    ax.set_aspect('equal')

    label = kwargs.pop('label', None)
    violins = ax.violinplot(y, x, **kwargs)
    if label:
        violins['cbars'].set_label(label)
    return violins


def softmax(x, axis=None):
    """
    Evaluate the softmax of `x` in a stable fashion.
    """
    proba = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return proba / np.sum(proba, axis=axis, keepdims=True)
