import hashlib
import matplotlib as mpl
from matplotlib import pyplot as plt
import numbers
import numpy as np
import os
import pickle
import pystan
import re


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


def maybe_build_model(model_code, root='.pystan', **kwargs):
    """
    Build a pystan model or retrieve a cached version.

    Parameters
    ----------
    model_code : str
        Stan model code to build.
    root : str
        Root directory at which to cache models.
    **kwargs : dict
        Additional arguments passed to the `pystan.StanModel` constructor.

    Returns
    -------
    model : pystan.StanModel
        Compiled stan model.
    """
    # Construct a filename
    identifier = hashlib.sha1(model_code.encode()).hexdigest()
    filename = os.path.join(root, identifier + '.pkl')

    if os.path.isfile(filename):  # Try to load the model
        with open(filename, 'rb') as fp:
            return pickle.load(fp)
    else:  # Build and store the model otherwise
        model = pystan.StanModel(model_code=model_code, **kwargs)
        os.makedirs(root, exist_ok=True)
        with open(filename, 'wb') as fp:
            pickle.dump(model, fp)
        return model


def filter_pystan_data(data):
    """
    Filter a data dictionary to retain only values supported by `pystan`.

    Parameters
    ----------
    data : dict
        Data from which to filter out values not supported by `pystan`.

    Returns
    -------
    data : dict
        Data after removal of values not supported by `pystan`.
    """
    return {
        key: value for key, value in data.items()
        if isinstance(value, numbers.Number)
        or (isinstance(value, np.ndarray) and value.dtype in (float, int))
    }


def transpose_samples(fit, pars=None):
    """
    Transpose samples from a `pystan.StanFit4Model` to samples of dictionaries.

    Parameters
    ----------
    fit : pystan.StanFit4Model
        Fit obtained from a `pystan` model.
    pars : list[str]
        Parameter names to extract; defaults to all parameters.

    Returns
    -------
    samples : list[dict]
        Sequence of samples, each represented by a dictionary.
    """
    mapping = fit.extract(pars)
    samples = []
    for key, values in mapping.items():
        for i, value in enumerate(values):
            if not i < len(samples):
                samples.append({})
            samples[i][key] = value
    return samples


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
