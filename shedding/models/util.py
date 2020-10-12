import enum
import functools as ft
import hashlib
import inspect
from jinja2 import Template
import logging
import numbers
import numpy as np
import os
import pickle
import pystan
from ..util import skip_doctest, softmax, logmeanexp


LOGGER = logging.getLogger(__name__)
MODEL_BOILERPLATE = {
    'data': """
        // Information about samples and associations between samples and patients.
        int<lower=0> num_patients;
        int<lower=0> num_samples;
        // Lookup table for patients: i = idx[j] is the patient index for sample j.
        int idx[num_samples];

        // Information about number of positives and negatives for each patient.
        int<lower=0> num_samples_by_patient[num_patients];
        int<lower=0> num_positives_by_patient[num_patients];
        int<lower=0> num_negatives_by_patient[num_patients];

        // Concentration measurements and levels of quantification.
        vector[num_samples] load;
        vector[num_samples] loq;
    """,
    'patient_contrib': """
        // Iterate over patients and ...
        for (i in 1:num_patients) {
            // ... deal with ones that have zero-only observations
            if (num_positives_by_patient[i] == 0) {
                target += log_sum_exp(
                    // This patient truly doesn't shed virus
                    bernoulli_lpmf(0 | rho),
                    // This patient sheds virus, but not enough to have been detected
                    bernoulli_lpmf(1 | rho) + patient_contrib_[i]
                );
            }
            // ... deal with ones that have positive and negative observations
            else {
                target += bernoulli_lpmf(1 | rho) + patient_contrib_[i];
            }
        }
    """,
    'gengamma_lpdf_lcdf': """
        // Log pdf for the generalised gamma distribution according to Stacey (1962).
        real gengamma_lpdf(real x, real q, real mu, real sigma) {
            real q2 = q ^ 2;
            return - (sigma * x ^ (q / sigma) * exp(-mu * q / sigma) + (2 - q2) * sigma * log(q) +
                q * (mu - log(x) + q * sigma * (log(sigma * x) + lgamma(1 / q2)))) / (sigma * q2);
        }

        // Log cdf for the generalised gamma distribution according to Stacey (1962).
        real gengamma_lcdf(real x, real q, real mu, real sigma) {
            real log_arg = q / sigma * log(x) - mu * q / sigma - 2 * log(q);
            return log(gamma_p(1 / q ^ 2, exp(log_arg)));
        }
    """
}


def broadcast_samples(func):
    """
    Decorator to broadcast the function across samples.
    """
    @ft.wraps(func)
    def _wrapper(*args, **kwargs):
        # Use different behaviour for instance methods to account for `self`
        if 'self' in inspect.signature(func).parameters:
            self, x, *args = args
            partial = ft.partial(func, self)
        else:
            x, *args = args
            partial = func

        if x.__class__.__name__ == 'StanFit4Model':
            x = transpose_samples(x)
        if isinstance(x, list):
            return np.asarray([partial(y, *args, **kwargs) for y in x])
        return partial(x, *args, **kwargs)

    return _wrapper


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
            model = pickle.load(fp)
        LOGGER.info('loaded model from %s', filename)
    else:  # Build and store the model otherwise
        model = pystan.StanModel(model_code=model_code, **kwargs)
        os.makedirs(root, exist_ok=True)
        with open(filename, 'wb') as fp:
            pickle.dump(model, fp)
        # Also dump the stan code for reference
        with open(filename.replace('.pkl', '.stan'), 'w') as fp:
            fp.write(model_code)
        LOGGER.info('dumped model to %s', filename)
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
    fit : pystan.StanFit4Model or dict[str,list]
        Fit obtained from a `pystan` model or dictionary of parameter samples.
    pars : list[str]
        Parameter names to extract; defaults to all parameters.

    Returns
    -------
    samples : list[dict]
        Sequence of samples, each represented by a dictionary.
    """
    if fit.__class__.__name__ == 'StanFit4Model':
        fit = fit.extract(pars)
    samples = []
    for key, values in fit.items():
        for i, value in enumerate(values):
            if not i < len(samples):
                samples.append({})
            samples[i][key] = value
    return samples


def merge_data(data, *, load, **kwargs):
    """
    Merge keyword arguments with a data dictionary, including sanity checking.
    """
    forbidden_keywords = ['load10', 'positive', 'num_positives_by_patient',
                          'num_samples_by_patient', 'num_negatives_by_patient']
    for forbidden_keyword in forbidden_keywords:
        if forbidden_keyword in kwargs:
            raise KeyError(f'`{forbidden_keyword}` is not an allowed keyword')

    data = dict(data)
    data.update(kwargs)

    # Count the number of positives and negatives
    positive = load > data['loq']
    num_positives_by_patient = np.zeros(data['num_patients'], int)
    num_samples_by_patient = np.zeros(data['num_patients'], int)

    # Subtracting one here because of Stan's zero-based indices
    for i, p in zip(data['idx'] - 1, positive):
        num_samples_by_patient[i] += 1
        num_positives_by_patient[i] += p
    num_negatives_by_patient = num_samples_by_patient - num_positives_by_patient

    # If indicator variables are known, do a sanity check that non-shedders don't have positives
    z = data.get('z')
    if z is not None:
        np.testing.assert_equal(num_positives_by_patient[~z], 0)

    data.update({
        'load': load,
        'positive': positive,
        'num_positives_by_patient': num_positives_by_patient,
        'num_negatives_by_patient': num_negatives_by_patient,
        'num_samples_by_patient': num_samples_by_patient,
    })
    return data


class ReplicationMode(enum.Enum):
    """
    Enum to indicate whether posterior predictive replication should be for `EXISTING_GROUPS` to
    assess the fit of new samples being collected for known patients or for `NEW_GROUPS` to assess
    the fit of samples being collected from new, unknown patients.

    Notes
    -----
    The `NEW_GROUPS` replication mode can be used to simulate data given hyperparameters at the
    population level.
    """
    EXISTING_GROUPS = 1
    NEW_GROUPS = 2


class Model:
    """
    Pystan model abstraction with python-based replication.
    """
    def __init__(self, model_code=None, **kwargs):
        model_code = model_code or self.MODEL_CODE
        if not model_code:
            raise ValueError("missing model code")

        # Render the template, substituting boilerplate
        template = Template(model_code)
        self.model_code = template.render(**MODEL_BOILERPLATE)

        kwargs.setdefault('model_name', self.__class__.__name__)
        self._kwargs = kwargs
        self._pystan_model = None

    @property
    def pystan_model(self):
        if self._pystan_model is None:
            self._pystan_model = maybe_build_model(self.model_code, **self._kwargs)
        return self._pystan_model

    MODEL_CODE = None
    DEFAULT_DATA = {}

    @skip_doctest
    @ft.wraps(pystan.StanModel.sampling)
    def sampling(self, data, *args, **kwargs):
        # Filter the data and add defaults
        data = filter_pystan_data(data)
        for key, value in self.DEFAULT_DATA.items():
            data.setdefault(key, value)
        # Use only one chain by default because of a bug in pystan
        kwargs.setdefault('n_jobs', 1)
        fit = self.pystan_model.sampling(data, *args, **kwargs)
        return fit

    @broadcast_samples
    def replicate(self, sample, data, mode=ReplicationMode.EXISTING_GROUPS, **kwargs):
        """
        Generate replicated data for a posterior sample.

        Parameters
        ----------
        sample : dict or list[dict]
            Posterior sample or sequence of posterior samples.
        data : dict
            Data from which the posterior samples were inferred.
        mode : ReplicationMode
            Whether to replicate only the lowest level of the hierarchical model
            (e.g. generate new data from existing groups) or replicate the entire hierarchy
            (e.g. generate new data from new groups).

        Returns
        -------
        replicate : dict
            Replicate of the data.
        """
        raise NotImplementedError(f'{self.__class__} does not support replication')

    @broadcast_samples
    def evaluate_statistic(self, sample, statistic, n=1000, **kwargs):
        """
        Evaluate a statistic of the model (using simulation where necessary).

        Parameters
        ----------
        sample : dict or list[dict]
            Posterior sample or sequence of posterior samples.
        statistic : str or list[str]
            One or more statistics to evaluate.
        n : int
            Number of samples to use if simulation is required to evaluate the statistic.
        """
        if isinstance(statistic, list):
            return {key: self._evaluate_statistic(sample, key, n) for key in statistic}
        return self._evaluate_statistic(sample, statistic, n)

    @broadcast_samples
    def evaluate_observed_likelihood(self, x, data, n=1000, **kwargs):
        """
        Evaluate the likelihood of the observed data marginalised with respect to group-level
        parameters but conditional on hyperparameters (using simulation where necessary).

        Parameters
        ----------
        x : dict
            Posterior sample.
        data : dict
            Data from which the posterior samples were inferred.
        n : int
            Number of samples to use if simulation is required to evaluate the likelihood.

        Returns
        -------
        likelihood : np.ndarray[num_patients]
            Observed data likelihood for each patient.
        """
        lpdf, lcdf = self._evaluate_observed_likelihood_contributions(x, data, n, **kwargs)
        sample_likelihood = np.where(data['positive'], lpdf, lcdf)
        # Marginalise with respect to the patient-level attributes
        sample_likelihood = logmeanexp(sample_likelihood, axis=0)

        # Aggregate by patient
        patient_likelihood = np.zeros(data['num_patients'])
        for idx, likelihood in zip(data['idx'] - 1, sample_likelihood):
            patient_likelihood[idx] += likelihood
        return patient_likelihood

    @broadcast_samples
    def rvs(self, x, size=None):
        """
        Draw a sample from the posterior predictive distribution.

        Parameters
        ----------
        x : dict
            Posterior sample.
        size : int or tuple[int]
            Size of the sample to draw.

        Returns
        -------
        sample : ndarray[size]
            Sample drawn from the posterior predictive distribution.
        """
        raise NotImplementedError(f'{self.__class__} does not support posterior sampling')

    def _evaluate_statistic(self, sample, statistic, n, **kwargs):
        raise NotImplementedError(f'{self.__class__} does not support evaluation of statistics')

    def _evaluate_observed_likelihood_contributions(self, x, data, n=1000, **kwargs):
        """
        Evaluate contributions to the observed data likelihood for each sample.

        Parameters
        ----------
        x : dict
            Posterior sample.
        data : dict
            Data from which the posterior samples were inferred.
        n : int
            Number of samples to use if simulation is required to evaluate the likelihood.

        Returns
        -------
        lpdf : np.ndarray[n, num_samples]
            Likelihood contributions if loads were above the level of quantification.
        lcdf : np.ndarray[n, num_samples]
            Likelihood contributions if loads were below the level of quantification.
        """
        raise NotImplementedError(f'{self.__class__} does not support evaluation of the observed '
                                  'data likelihood')


class InflationMixin:
    @broadcast_samples
    def evaluate_observed_likelihood(self, x, data, n=1000, **kwargs):
        patient_likelihood = super(InflationMixin, self).evaluate_observed_likelihood(x, data, n,
                                                                                      **kwargs)
        # Patients that have all-negative samples may be non-shedders. So the data are either
        # generated by having some latent indicator z==0 or z==1 but the samples are too small to be
        # above the LOQ. So we need to evaluate the mixture distribution.
        all_negative = data['num_positives_by_patient'] == 0
        patient_likelihood = np.where(
            all_negative,
            np.logaddexp(patient_likelihood + np.log(x['rho']), np.log1p(-x['rho'])),
            patient_likelihood
        )
        return patient_likelihood

    @broadcast_samples
    def rvs(self, x, size=None):
        # Draw a sample from the non-inflated model
        sample = super(InflationMixin, self).rvs(x, size)
        # Account for zero-inflation
        return np.where(np.random.uniform(size=np.shape(sample)) < x['rho'], sample, np.nan)

    def _evaluate_statistic(self, x, statistic, n, **kwargs):
        if statistic == 'mean':
            return x['rho'] * super(InflationMixin, self)._evaluate_statistic(x, statistic, n)
        raise NotImplementedError(statistic)


def sample_indicators(x, data):
    """
    Sample shedding indicator variables conditional on the data and a sample from the posterior
    using a Gibbs sampling step.

    Parameters
    ----------
    x : dict
            Posterior sample which must include the `patient_contrib_` key.
    data : dict
        Data from which the posterior samples were inferred.

    Returns
    -------
    z : np.ndarray<bool>[num_patients]
        Binary indicators for patients being shedders or non-shedders.

    Notes
    -----
    Because `pystan` does not support discrete parameters, we need to marginalise with respect to
    the shedding indicator variables analytically for inference purposes. However, we need to sample
    the indicator variables to replicate data.
    """
    # Evaluate the probability of being a shedder or non-shedder in the log space using a Gibbs
    # sampling approach.
    logprobas = np.asarray([
        np.log1p(-x['rho']) * np.ones(data['num_patients']),
        np.log(x['rho']) + x['patient_contrib_']
    ])
    probas = softmax(logprobas, axis=0)
    # Ensure everyone who has a positive sample remains a shedder in the replication.
    probas[0, data['num_positives_by_patient'] > 0] *= 0
    # Renormalise and pick the probability to be a shedder.
    probas = (probas / np.sum(probas, axis=0))[1]
    return np.random.uniform(0, 1, data['num_patients']) < probas
