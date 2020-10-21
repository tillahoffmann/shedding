import enum
import functools as ft
import inspect
import numpy as np
from scipy import special
from .util import flush_traceback, softmax, logmeanexp


def broadcast_samples(func):
    """
    Decorator to broadcast the function across samples provided as the first argument.
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

        if isinstance(x, list):
            return np.asarray([partial(y, *args, **kwargs) for y in x])
        return partial(x, *args, **kwargs)

    return _wrapper


def evaluate_size(parameters):
    """
    Evaluate the size, i.e. number of elements, for a set of parameters.

    Parameters
    ----------
    parameters : dict[str, tuple]
        Mapping from parameter names to shapes.

    Returns
    -------
    int : size
        Total number of elements in the parameter set.
    """
    total = 0
    for shape in parameters.values():
        size = 1
        for dim in shape:
            size *= dim
        total += size
    return total


def values_to_vector(parameters, values, size=None):
    """
    Convert a dictionary of values to a vector.

    Parameters
    ----------
    values : dict[str, np.ndarray]
        Mapping from parameter names to values.
    parameters : dict[str, tuple]
        Mapping from parameter names to shapes.

    Returns
    -------
    vector : np.ndarray
        Vector of parameters corresponding to `values`.
    """
    size = size or evaluate_size(parameters)
    vector = np.empty(size)
    offset = 0
    for key, shape in parameters.items():
        value = values[key]
        if not shape:
            vector[offset] = value
            offset += 1
            continue
        if len(shape) > 1:
            value = np.ravel(value)
        vector[offset:offset + value.size] = value
        offset += value.size
    return vector


def vector_to_values(parameters, vector):
    """
    Convert a vector to a dictionary of values.

    Parameters
    ----------
    vector : np.ndarray
        Vector of parameters corresponding to `values`.
    parameters : dict[str, tuple]
        Mapping from parameter names to shapes.

    Returns
    -------
    values : dict[str, np.ndarray]
        Mapping from parameter names to values.
    """
    values = {}
    offset = 0
    for key, shape in parameters.items():
        if not shape:
            values[key] = vector[offset]
            offset += 1
            continue
        size = 1
        for dim in shape:
            size *= dim
        value = vector[offset:offset + size]
        if len(shape) > 1:
            value = np.reshape(value, shape)
        values[key] = value
        offset += size
    return values


def transpose_samples(samples, parameters=None):
    """
    Transpose samples from a list of dictionaries to a dictionary of lists and vice versa.

    Parameters
    ----------
    samples : list[dict[str,np.ndarray]] or dict[str,list]
        Samples to transpose.
    parameters : dict[str,tuple]
        Parameter shape definitions to use if `samples` is an array.

    Returns
    -------
    samples : list[dict]
        Sequence of samples, each represented by a dictionary.
    """
    if parameters:
        samples = [vector_to_values(parameters, x) for x in samples]

    if isinstance(samples, list):
        result = {}
        for x in samples:
            for key, value in x.items():
                result.setdefault(key, []).append(value)
        result = {key: np.asarray(value) for key, value in result.items()}
        return result
    elif isinstance(samples, dict):
        result = []
        for key, values in samples.items():
            for i, value in enumerate(values):
                if not i < len(result):
                    result.append({})
                result[i][key] = value
        return result
    else:  # pragma: no cover
        raise ValueError


def write_paramnames_file(parameters, filename, escape=True):
    """
    Write parameter names to a file.
    """
    with open(filename, 'w') as fp:
        for parameter, shape in parameters.items():
            ndims = len(shape)
            if escape:
                parameter = parameter.replace('_', '\\_')
            if ndims == 0:
                fp.write(f'{parameter}\n')
            else:
                indices = np.indices(shape).reshape((ndims, -1)).T
                for index in indices:
                    fp.write(f'{parameter}[{", ".join(map(str, index))}]\n')


def lognormal_lpdf(mu, sigma, x):
    """
    Evaluate the natural logarithm of the lognormal probability density function.
    """
    return - (np.log(2 * np.pi) / 2 + (mu - np.log(x)) ** 2 / (2 * sigma ** 2) + np.log(sigma * x))


def lognormal_lcdf(mu, sigma, x):
    """
    Evaluate the natural logarithm of the lognormal cumulative distribution function.
    """
    return np.log(special.erfc((mu - np.log(x)) / (sigma * np.sqrt(2))) / 2)


def lognormal_mean(mu, sigma):
    """
    Evaluate the mean of the lognormal distribution.
    """
    return np.exp(mu + sigma ** 2 / 2)


def lognormal_loc(sigma, mean):
    """
    Evaluate the lognormal location parameter given the mean.
    """
    return np.log(mean) - sigma ** 2 / 2


def _gengamma_lpdf(q, mu, sigma, x):
    """
    Evaluate the natural logarithm of the generalised gamma pdf.
    """
    a, b, c = to_abc(q, mu, sigma)
    return - b * x ** c + a * np.log(b) + np.log(c) + (a * c - 1) * np.log(x) - special.gammaln(a)


def _gengamma_lcdf(q, mu, sigma, x):
    """
    Evaluate the natural logarithm of the generalised gamma cdf.
    """
    a, b, c = to_abc(q, mu, sigma)
    arg = b * x ** c
    cdf = special.gammainc(a, arg)
    return np.log(cdf)


def _gengamma_mean(q, mu, sigma):
    """
    Evaluate the mean of the generalised gamma distribution.
    """
    a, b, c = to_abc(q, mu, sigma)
    log_mean = - np.log(b) / c + special.gammaln(a + 1 / c) - special.gammaln(a)
    return np.exp(log_mean)


def _gengamma_loc(q, sigma, mean):
    """
    Evaluate the scale of the generalised gamma distribution for given shape, exponent, and mean.
    """
    a, _, c = to_abc(q, None, sigma)
    b = np.exp(- c * (np.log(mean) + special.gammaln(a) - special.gammaln(a + 1 / c)))
    _, mu, _ = from_abc(a, b, c)
    return mu


def q_branch(gengamma, lognormal):
    """
    Generate a function that uses `lognormal` when the first argument is zero and `gengamma`
    otherwise.
    """
    @ft.wraps(gengamma)
    def _wrapper(q, *args, **kwargs):
        # If all qs are zero, just return the lognormal branch
        q0 = q == 0
        qs = np.isscalar(q)
        if (qs and q0) or np.all(q0):
            return lognormal(*args, **kwargs)

        value = gengamma(q, *args, **kwargs)
        # If none of the qs are zero, just return the generalised gamma lpdf
        if (qs and not q0) or not np.any(q0):
            return value
        # Return a composite of the two functions
        return np.where(q0, lognormal(*args, **kwargs), value)
    return _wrapper


gengamma_lpdf = q_branch(_gengamma_lpdf, lognormal_lpdf)
gengamma_lcdf = q_branch(_gengamma_lcdf, lognormal_lcdf)
gengamma_mean = q_branch(_gengamma_mean, lognormal_mean)
gengamma_loc = q_branch(_gengamma_loc, lognormal_loc)


class Prior:
    @property
    def lower(self):
        return None

    @property
    def upper(self):
        return None

    @property
    def bounds(self):
        return (self.lower, self.upper)

    def __call__(self, u):
        raise NotImplementedError


class NormalPrior(Prior):
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, u):
        return self.mu + self.sigma * np.sqrt(2) * special.erfinv(2 * u - 1)


class LognormalPrior(NormalPrior):
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    @property
    def lower(self):
        return 0

    def __call__(self, u):
        return np.exp(super(LognormalPrior, self).__call__(u))


class GengammaPrior(Prior):
    def __init__(self, q=1, mu=0, sigma=1):
        self.q = q
        self.mu = mu
        self.sigma = sigma
        if self.q != 0:
            self.a, self.b, self.c = to_abc(self.q, self.mu, self.sigma)

    @property
    def lower(self):
        return 0

    def __call__(self, u):
        if self.q == 0:
            return LognormalPrior(self.mu, self.sigma)(u)
        return (special.gammaincinv(self.a, u) / self.b) ** (1 / self.c)


class UniformPrior(Prior):
    def __init__(self, lower, upper):
        self._lower = lower
        self._upper = upper

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    def __call__(self, u):
        return self._lower + u * (self._upper - self._lower)


class LoguniformPrior(UniformPrior):
    def __init__(self, lower, upper, base=np.e):
        super(LoguniformPrior, self).__init__(lower, upper)
        self.base = base

    @property
    def lower(self):
        return self.base ** self._lower

    @property
    def upper(self):
        return self.base ** self._upper

    def __call__(self, u):
        return self.base ** super(LoguniformPrior, self).__call__(u)


def from_abc(a, b, c):
    """
    Transform from the parametrisation in terms of shape, scale, and exponent to Stacey's
    parametrisation.
    """
    q = 1 / np.sqrt(a)
    mu = np.log(a / b) / c
    sigma = q / c
    return q, mu, sigma


def to_abc(q, mu, sigma):
    """
    Transform from Stacey's parametrisation to the parametrisation in terms of shape, scale, and
    exponent.
    """
    a = 1 / q ** 2
    c = q / sigma
    b = None if mu is None else a * np.exp(- mu * c)
    return a, b, c


class Parametrisation(enum.Enum):
    GENERAL = 'general'
    GAMMA = 'gamma'
    WEIBULL = 'weibull'
    LOGNORMAL = 'lognormal'


class SimulationMode(enum.Enum):
    """
    Enum to indicate whether simulation should be for `EXISTING_PATIENTS` or `NEW_PATIENTS`. Given
    posterior samples, the former can be used to assess the fit of the model to samples collected
    from new, unknown patients, and the latter can be used to assess the fit to new samples
    collected from existing patients.

    Notes
    -----
    The `NEW_PATIENTS` replication mode can be used to simulate data given hyperparameters at the
    population level.
    """
    NEW_PATIENTS = 'new_patients'
    EXISTING_PATIENTS = 'existing_patients'


class Model:
    r"""
    Generalised gamma model for viral load in faecal samples.

    Parameters
    ----------
    num_patients : int
        Number of patients.
    parametrisation : Parametrisation
        Parametrisation used by the model (see notes for details).
    inflated : bool
        Whether there is a "zero-inflated" subpopulation of patients who do not shed RNA.
    priors : dict
        Mapping of parameter names to callable priors.

    Notes
    -----
    The generalised gamma distribution is a versatile distribution for positive continuous data. We
    use the parametrisation considered by Stacey with shape parameter :math:`q`, location parameter
    :math:`\mu`, and scale parameter :math:`\sigma`. Then if :math:`\gamma` follows a gamma
    distribution with shape parameter :math:`a=1/q^2`, the random variable

    .. math ::
       x = \exp\left(\mu + \frac{\sigma}{q}\log\left(q^2\gamma\right)\right)

    follows a generalised gamma distribution.

    The generalised gamma distribution includes the regular gamma distribution (:math:`\sigma=q`),
    Weibull distribution (:math:`q=1`), and lognormal distribution (:math:`q=0`) as special cases.
    The model can be restricted to a particular distribution using the :code:`parametrisation`
    parameter.
    """
    def __init__(self, num_patients, parametrisation='general', inflated=False, priors=None):
        self.num_patients = num_patients
        self.parametrisation = Parametrisation(parametrisation)
        self.inflated = inflated
        # Using Stacey's parametrisation
        self.parameters = {
            'population_shape': (),
            'population_loc': (),
            'population_scale': (),
            'patient_shape': (),
            'patient_scale': (),
            'patient_mean': (num_patients,)
        }
        if self.inflated:
            self.parameters['rho'] = ()
        self.size = evaluate_size(self.parameters)

        # Merge the supplied priors and default priors
        self.priors = priors or {}
        default_priors = {
                'population_scale': LoguniformPrior(-2, 3),
                'patient_scale': LoguniformPrior(-2, 3),
                'population_loc': UniformPrior(6, 20)
            }
        if self.parametrisation == Parametrisation.GENERAL:
            default_priors.update({
                'population_shape': UniformPrior(0, 20),
                'patient_shape': UniformPrior(0, 5),
            })
        if self.inflated:
            default_priors['rho'] = UniformPrior(0, 1)
        for key, prior in default_priors.items():
            self.priors.setdefault(key, prior)

    def sample_shared_params(self, values):
        """
        Sample parameters that are shared amongst individuals.
        """
        population_scale = self.priors['population_scale'](values['population_scale'])
        patient_scale = self.priors['patient_scale'](values['patient_scale'])

        if self.parametrisation == Parametrisation.GENERAL:
            population_shape = self.priors['population_shape'](values['population_shape'])
            patient_shape = self.priors['patient_shape'](values['patient_shape'])
        elif self.parametrisation == Parametrisation.GAMMA:
            # We get a regular gamma distribution when the shape and scale are equal (c = 1)
            population_shape = population_scale
            patient_shape = patient_scale
        elif self.parametrisation == Parametrisation.WEIBULL:
            # We get a Weibull distribution when the gamma random variable reduces to an exponential
            population_shape = 1
            patient_shape = 1
        elif self.parametrisation == Parametrisation.LOGNORMAL:
            # We get a lognormal when the gamma random variable becomes a sharp Gaussian
            population_shape = 0
            patient_shape = 0
        else:  # pragma: no cover
            raise ValueError(self.parametrisation)
        values.update({
            'population_shape': population_shape,
            'population_scale': population_scale,
            'population_loc': self.priors['population_loc'](values['population_loc']),
            'patient_shape': patient_shape,
            'patient_scale': patient_scale,
        })
        if self.inflated:
            values['rho'] = self.priors['rho'](values['rho'])
        return values

    def sample_individual_params(self, values):
        """
        Sample individual-level parameters.
        """
        prior = GengammaPrior(values['population_shape'], values['population_loc'],
                              values['population_scale'])
        values['patient_mean'] = prior(values['patient_mean'])
        return values

    def sample_params(self, values):
        """
        Sample shared and individual-level parameters from the hierarchical prior.
        """
        self.sample_shared_params(values)
        self.sample_individual_params(values)
        return values

    def _evaluate_sample_log_likelihood(self, values, data):
        """
        Evaluate the log likelihood for each sample conditional on all model parameters, assuming
        that all patients shed virus.

        Parameters
        ----------
        values : dict
            Parameter values.
        data : dict
            Data from which the posterior samples were inferred.

        Returns
        -------
        lxdf : np.ndarray[..., num_samples]
            Log likelihood contributions for each sample.
        """
        q = values['patient_shape']
        sigma = values['patient_scale']
        mu = gengamma_loc(q, sigma, values['patient_mean'])
        mu = np.repeat(mu, data['num_samples_by_patient'], axis=-1)
        lpdf = gengamma_lpdf(q, mu, sigma, data['load'])
        lcdf = gengamma_lcdf(q, mu, sigma, data['loq'])
        return np.where(data['positive'], lpdf, lcdf)

    def _evaluate_patient_log_likelihood(self, values, data):
        """
        Evaluate the log likelihood for each patient conditional on all model parameters, assuming
        that all patients shed virus.
        """
        result = self._evaluate_sample_log_likelihood(values, data)
        return np.bincount(data['idx'], result, minlength=data['num_patients'])

    def evaluate_log_likelihood(self, values, data):
        """
        Evaluate the log likelihood of the data conditional on all model parameters.
        """
        if not self.inflated:
            return self._evaluate_sample_log_likelihood(values, data).sum()

        patient_contrib = self._evaluate_patient_log_likelihood(values, data) + \
            np.log(values['rho'])
        np.logaddexp(patient_contrib, np.log1p(-values['rho']), out=patient_contrib,
                     where=data['num_positives_by_patient'] == 0)
        return patient_contrib.sum()

    @broadcast_samples
    def evaluate_marginal_log_likelihood(self, values, data, n=1000, **kwargs):
        """
        Evaluate the log likelihood of the observed data marginalised with respect to group-level
        parameters but conditional on hyperparameters.

        Parameters
        ----------
        values : dict
            Posterior sample.
        data : dict
            Data from which the posterior samples were inferred.
        n : int
            Number of samples to use if simulation is required to evaluate the likelihood.

        Returns
        -------
        likelihood : np.ndarray[num_patients]
            Marginal log likelihood for each patient.
        """
        values = dict(values)
        # Sample the patient means
        uniform = np.random.uniform(size=(n, data['num_patients']))
        values['patient_mean'] = GengammaPrior(values['population_shape'], values['population_loc'],
                                               values['population_scale'])(uniform)
        # Evaluate the sample log likelihood and marginalise with respect to the patient-level
        # attributes
        sample_likelihood = self._evaluate_sample_log_likelihood(values, data)
        sample_likelihood = logmeanexp(sample_likelihood, axis=0)

        # Aggregate by patient
        patient_likelihood = np.bincount(data['idx'], sample_likelihood,
                                         minlength=data['num_patients'])
        if not self.inflated:
            return patient_likelihood

        # Patients that have all-negative samples may be non-shedders. So the data are either
        # generated by having some latent indicator z==0 or z==1 but the samples are too small to be
        # above the LOQ. So we need to evaluate the mixture distribution.
        all_negative = data['num_positives_by_patient'] == 0
        patient_likelihood = np.where(
            all_negative,
            np.logaddexp(patient_likelihood + np.log(values['rho']), np.log1p(-values['rho'])),
            patient_likelihood
        )
        return patient_likelihood

    @broadcast_samples
    def rvs(self, values, size=None):
        """
        Draw a sample from the posterior predictive distribution.

        Parameters
        ----------
        values : dict
            Posterior sample.
        size : int or tuple[int]
            Size of the sample to draw.

        Returns
        -------
        sample : ndarray[size]
            Sample drawn from the posterior predictive distribution.
        """
        # Sample the patient means
        patient_mean = GengammaPrior(values['population_shape'], values['population_loc'],
                                     values['population_scale'])(np.random.uniform(size=size))
        # Evaluate the locations for the sample distribution
        loc = gengamma_loc(values['patient_shape'], values['patient_scale'], patient_mean)
        # Sample the RNA loads
        sample = GengammaPrior(values['patient_shape'], loc, values['patient_scale'])(
            np.random.uniform(size=size))
        if not self.inflated:
            return sample
        # Account for non-shedders
        z = np.random.uniform(size=np.shape(sample)) < values['rho']
        return np.where(z, sample, np.nan)

    @broadcast_samples
    def simulate(self, values, data, simulation_mode):
        """
        Generate simulated data for a posterior sample.

        Parameters
        ----------
        sample : dict or list[dict]
            Posterior sample or sequence of posterior samples.
        data : dict
            Data from which the posterior samples were inferred.
        simulation_mode : SimulationMode
            Whether to simulate only the lowest level of the hierarchical model (e.g. generate new
            data from existing groups) or simulate the entire hierarchy (e.g. generate new data from
            new groups).

        Returns
        -------
        simulation : dict
            Simulated data.
        """
        data = data.copy()
        simulation_mode = SimulationMode(simulation_mode)
        num_patients = data['num_patients']
        # Sample new individual-level attributes
        if simulation_mode == SimulationMode.NEW_PATIENTS:
            values['patient_mean'] = np.random.uniform(size=num_patients)
            self.sample_individual_params(values)
        # Sample loads
        loq = data['loq']
        q = values['patient_shape']
        sigma = values['patient_scale']
        mu = gengamma_loc(q, sigma, values['patient_mean'])
        mu = np.repeat(mu, data['num_samples_by_patient'])
        load = GengammaPrior(q, mu, sigma)(np.random.uniform(size=mu.size))

        # Account for the patients who do not shed any RNA
        if self.inflated:
            if simulation_mode == SimulationMode.NEW_PATIENTS:
                z = np.random.uniform(size=num_patients) < values['rho']
            else:
                patient_contrib = self._evaluate_patient_log_likelihood(values, data)
                # Evaluate the probability of being a shedder or non-shedder in the log space using
                # a Gibbs sampling approach.
                logprobas = np.asarray([
                    np.log1p(-values['rho']) * np.ones(num_patients),
                    np.log(values['rho']) + patient_contrib,
                ])
                probas = softmax(logprobas, axis=0)
                # Ensure everyone who has a positive sample remains a shedder in the replication.
                probas[0, data['num_positives_by_patient'] > 0] *= 0
                # Renormalise and pick the probability to be a shedder.
                probas = (probas / np.sum(probas, axis=0))[1]
                z = np.random.uniform(0, 1, num_patients) < probas
            values['z'] = z
            z = np.repeat(z, data['num_samples_by_patient'])
            load = np.where(z, load, loq / 2)
        data['load'] = load
        data['positive'] = positive = load >= loq

        # Update summary statistics
        idx = data['idx']
        data['num_positives_by_patient'] = np.bincount(idx, positive, minlength=num_patients)
        data['num_negatives_by_patient'] = np.bincount(idx, ~positive, minlength=num_patients)
        return data

    @broadcast_samples
    def evaluate_statistic(self, values, statistic):
        """
        Evaluate a statistic of the model.

        Parameters
        ----------
        sample : dict
            Parameter values.
        statistic : str
            Statistic to evaluate.

        Returns
        -------
        value : float
            Value of the desired statistic given parameter values.
        """
        if self.inflated != ('rho' in values):
            raise ValueError  # pragma: no cover
        if statistic == 'mean':
            mean = gengamma_mean(values['population_shape'], values['population_loc'],
                                 values['population_scale'])
            if self.inflated:
                mean *= values['rho']
            return mean
        else:  # pragma: no cover
            raise ValueError(statistic)

    @flush_traceback
    def sample_params_from_vector(self, vector):
        values = vector_to_values(self.parameters, vector)
        self.sample_params(values)
        return values_to_vector(self.parameters, values, self.size)

    @flush_traceback
    def evaluate_log_likelihood_from_vector(self, vector, data):
        values = vector_to_values(self.parameters, vector)
        result = self.evaluate_log_likelihood(values, data)
        # Use a large negative number if the likelihood cannot be evaluated
        if not np.isfinite(result):
            result = - np.finfo(float).max
        return result, []
