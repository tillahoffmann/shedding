import collections
import enum
import functools as ft
import inspect
import numpy as np
from scipy import special
from .util import flush_traceback, softmax, logmeanexp
from ._util import gengamma_lpdf, gengamma_lcdf, gengamma_loc


def broadcast_samples(func):
    """
    Decorator to broadcast the function across samples provided as the first argument.
    """
    @ft.wraps(func)
    def _broadcast_samples_wrapper(*args, **kwargs):
        # Use different behaviour for instance methods to account for `self`
        if 'self' in inspect.signature(func).parameters:
            self, x, *args = args
            partial = ft.partial(func, self)
        else:
            x, *args = args
            partial = func

        if not isinstance(x, collections.abc.Mapping):
            return np.asarray([partial(y, *args, **kwargs) for y in x])
        return partial(x, *args, **kwargs)

    return _broadcast_samples_wrapper


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
    vector = None
    offset = 0
    for key, shape in parameters.items():
        value = values[key]
        if vector is None:
            base_shape = np.shape(value)
            if shape:
                base_shape = base_shape[:-len(shape)]
            vector = np.empty(base_shape + (size,))
        if not shape:
            vector[..., offset] = value
            offset += 1
            continue
        if len(shape) > 1:
            value = np.reshape(value, base_shape + (-1,))
        vector[..., offset:offset + value.shape[-1]] = value
        offset += value.shape[-1]
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
    base_shape = np.shape(vector)[:-1]
    for key, shape in parameters.items():
        if not shape:
            values[key] = vector[..., offset]
            offset += 1
            continue
        size = 1
        for dim in shape:
            size *= dim
        value = vector[..., offset:offset + size]
        if len(shape) > 1:
            value = np.reshape(value, base_shape + shape)
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


_LN_PDF_CONSTANT = np.log(2 * np.pi) / 2
_SQRT2 = np.sqrt(2)


def _gengamma_lpdf(q, mu, sigma, logx, qmin=1e-6):
    r"""
    Evaluate the natural logarithm of the generalised gamma pdf.

    Parameters
    ----------
    q :
        Shape of the generalised gamma distribution.
    mu :
        Location of the generalised gamma distribution.
    sigma :
        Scale of the generalised gamma distribution.
    qmin :
        Threshold below which the generalised gamma is evaluated
        using a normal approximation.

    Notes
    -----
    A generalised gamma random variable :math:`x` can be generated as

    :: math..

       x = \exp\left(mu + \frac{\sigma}{q}\log\left[q^2\gamma\right]\right),

    where :math:`\gamma` follows a standard gamma distribution with shape
    parameter :math:`a = q^{-2}`. The gamma random variable has mean
    :math:`q^{-2}` and standard deviation :math:`1/q` such that :math:`q^2\gamma`
    has unit mean and standard deviation :math:`q`. For sufficiently small :math:`q`,
    we can approximate :math:`q^2\gamma\approx 1 + q s`, where :math:`s` is a
    standard normal distribution. Thus,

    :: math..

       x \approx \exp(\mu + \frac{\sigma}{q} \log\left[1 + q s\right]),

    which allows us to evaluate the pdf in a numerically stable fashion.
    """
    # Evaluate summary statistics
    a = 1 / (q * q)
    c = q / sigma
    logc = np.log(c)
    z = (logx - mu) / sigma
    qz = q * z
    # Evaluate the generalised gamma lpdf
    lpdf1 = logc - special.gammaln(a) + a * np.log(a) + (a * c - 1) * logx - \
        a * (mu * c + np.exp(qz))
    # Evaluate for small q close to the lognormal limit
    s = np.where(qz > qmin, np.expm1(qz) / q, z + qz * z / 2)
    lpdf2 = - _LN_PDF_CONSTANT - np.log(sigma) - s * s / 2 + qz - logx
    return np.where(q > qmin, lpdf1, lpdf2)


def _gengamma_lcdf(q, mu, sigma, logx, qmin=1e-6):
    """
    Evaluate the natural logarithm of the generalised gamma cdf.
    """
    a = 1 / (q * q)
    z = (logx - mu) / sigma
    qz = q * z
    arg = a * np.exp(qz)
    cdf1 = special.gammainc(a, arg)
    s = np.where(qz > qmin, np.expm1(qz) / q, z + qz * z / 2)
    cdf2 = (1 + special.erf(s / _SQRT2)) / 2
    return np.log(np.where(q > qmin, cdf1, cdf2))


def gengamma_mean(q, mu, sigma, qmin=1e-6):
    """
    Evaluate the mean of the generalised gamma distribution.
    """
    a = 1 / q ** 2
    cinv = sigma / q
    sigma2 = np.square(sigma)
    summand1 = special.gammaln(a + cinv) - special.gammaln(a) - np.log(a) * cinv
    # Obtained by expanding the line above to sixth order in q about q = 0
    summand2 = sigma2 / 2 - q * sigma / 6 * (sigma2 + 3)
    summand = np.where(q > qmin, summand1, summand2)
    log_mean = mu + summand
    return np.exp(log_mean)


def _gengamma_loc(q, sigma, mean, qmin=1e-6):
    """
    Evaluate the scale of the generalised gamma distribution for given shape, exponent, and mean.
    """
    a = 1 / q ** 2
    cinv = sigma / q
    sigma2 = np.square(sigma)
    summand1 = special.gammaln(a + cinv) - special.gammaln(a) - np.log(a) * cinv
    summand2 = sigma2 / 2 - q * sigma / 6 * (sigma2 + 3)
    summand = np.where(q > qmin, summand1, summand2)
    return np.log(mean) - summand


class Prior:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.lower = None
        self.upper = None

    @classmethod
    def from_uniform(cls, uniform, **kwargs):
        raise NotImplementedError

    def lpdf(self, x):
        raise NotImplementedError

    @property
    def bounds(self):
        return (self.lower, self.upper)

    def __call__(self, u):
        return self.from_uniform(u, **self.kwargs)


class PositivePrior(Prior):
    def __init__(self, **kwargs):
        super(PositivePrior, self).__init__(**kwargs)
        self.lower = 0


class HalfCauchyPrior(PositivePrior):
    @classmethod
    def from_uniform(cls, uniform, scale):
        return scale * np.tan(np.pi * uniform / 2)

    def lpdf(self, x):
        scale = self.kwargs['scale']
        return 2 * scale / (np.pi * (scale ** 2 + x ** 2))


class CauchyPrior(Prior):
    @classmethod
    def from_uniform(cls, uniform, loc, scale):
        return loc + scale * np.tan(np.pi * (uniform - 0.5))

    def lpdf(self, x):
        scale = self.kwargs['scale']
        loc = self.kwargs['loc']
        return scale / (np.pi * (scale ** 2 + (x - loc) ** 2))


class NormalPrior(Prior):
    @classmethod
    def from_uniform(cls, uniform, mu, sigma):
        return mu + sigma * _SQRT2 * special.erfinv(2 * uniform - 1)


class LognormalPrior(PositivePrior):
    @classmethod
    def from_uniform(cls, uniform, mu, sigma):
        return np.exp(NormalPrior.from_uniform(uniform, mu, sigma))


class GengammaPrior(PositivePrior):
    @classmethod
    def from_uniform(cls, uniform, q, mu, sigma):
        if q == 0:
            return LognormalPrior.from_uniform(uniform, mu, sigma)
        a = 1 / (q * q)
        cinv = sigma / q
        return (special.gammaincinv(a, uniform) / a) ** cinv * np.exp(mu)

    def lpdf(self, x, log=False):
        # Transform to log space if necessary
        if not log:
            x = np.log(x)
        return gengamma_lpdf(**self.kwargs, logx=x)


class UniformPrior(Prior):
    def __init__(self, lower, upper):
        super(UniformPrior, self).__init__(lower=lower, upper=upper)
        self.lower = lower
        self.upper = upper

    @classmethod
    def from_uniform(cls, uniform, lower, upper):
        return lower + uniform * (upper - lower)

    def lpdf(self, x):
        return -np.log(self.upper - self.lower)


class LoguniformPrior(UniformPrior):
    def __init__(self, lower, upper, base=None):
        base = base or np.e
        super(UniformPrior, self).__init__(lower=lower, upper=upper, base=base)
        self.lower = base ** lower
        self.upper = base ** upper

    @classmethod
    def from_uniform(cls, uniform, lower, upper, base):
        return base ** UniformPrior.from_uniform(uniform, lower, upper)


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


class Transformation:
    r"""
    A bijective transformation :math:`y = f(x)` for regular variables :math:`x` and transformed
    variables :math:`y`. For sampling purposes, the transformation induces a volume compression or
    expansion that needs to be accounted for, i.e.

    .. math::

       \log P(y) = \log P(x=f^{-1}(y)) + \log \frac{dx}{dy}.
    """
    def __call__(self, values):
        """
        Transform from the transformed space to the regular space.
        """
        raise NotImplementedError

    def inverse(self, values):
        """
        Transform from the regular space to the transformed space.
        """
        raise NotImplementedError

    def jacobianlogdet(self, values):
        """
        Evaluate the logarithm of the Jacobian determinant to account for sampling in the
        transformed space rather than the regular space.
        """
        raise NotImplementedError


class DefaultTransformation:
    POSITIVE_KEYS = ['patient_mean', 'patient_shape', 'patient_scale',
                     'population_shape', 'population_scale']

    def __call__(self, values):
        result = dict(values)
        result.update({key: np.exp(values[key]) for key in self.POSITIVE_KEYS if key in values})
        # Transform to the unit interval if required
        logit = values.get('rho')
        if logit is not None:
            result['rho'] = special.expit(logit)
        return result

    def inverse(self, values):
        result = dict(values)
        result.update({key: np.log(values[key]) for key in self.POSITIVE_KEYS if key in values})
        rho = values.get('rho')
        if rho is not None:
            result['rho'] = special.logit(rho)
        return result

    def jacobianlogdet(self, values):
        logdet = sum(values[key].sum() for key in self.POSITIVE_KEYS if key in values)
        # Account for logit transform if required
        logit = values.get('rho')
        if logit is not None:
            logdet += logit - 2 * np.log1p(np.exp(logit))
        return logdet


def _augment_values(func):
    """
    Decorator to augment the values by adding the population and patient shapes based on the
    parametrisation.
    """
    @ft.wraps(func)
    def _augment_wrapper(self, values, *args, **kwargs):
        if self.parametrisation == Parametrisation.GENERAL:
            pass
        elif self.parametrisation == Parametrisation.LOGNORMAL:
            values.update({
                'population_shape': np.float64(0),
                'patient_shape': np.float64(0),
            })
        elif self.parametrisation == Parametrisation.WEIBULL:
            values.update({
                'population_shape': 1,
                'patient_shape': 1,
            })
        elif self.parametrisation == Parametrisation.GAMMA:
            values.update({
                    'population_shape': values['population_scale'],
                    'patient_shape': values['patient_scale'],
                })
        else:
            raise ValueError
        return func(self, values, *args, **kwargs)
    return _augment_wrapper


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
    temporal : bool
        Whether to include an exponential decay component in the fit.
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
    def __init__(self, num_patients, parametrisation='general', inflated=False, temporal=False,
                 priors=None):
        self.num_patients = num_patients
        self.parametrisation = Parametrisation(parametrisation)
        self.inflated = inflated
        self.temporal = temporal
        # Using Stacey's parametrisation
        self.parameters = {
            'population_loc': (),
            'population_scale': (),
            'patient_scale': (),
            'patient_mean': (num_patients,)
        }
        if self.parametrisation == Parametrisation.GENERAL:
            self.parameters.update({
                'population_shape': (),
                'patient_shape': (),
            })
        if self.inflated:
            self.parameters['rho'] = ()
        if self.temporal:
            self.parameters['slope'] = ()
        self.size = evaluate_size(self.parameters)

        # Merge the supplied priors and default priors
        self.priors = priors or {}
        default_priors = {
                'population_scale': HalfCauchyPrior(scale=1),
                'patient_scale': HalfCauchyPrior(scale=1),
                'population_loc': UniformPrior(6, 20)
            }
        if self.parametrisation == Parametrisation.GENERAL:
            default_priors.update({
                'population_shape': HalfCauchyPrior(scale=1),
                'patient_shape': HalfCauchyPrior(scale=1),
            })
        if self.inflated:
            default_priors['rho'] = UniformPrior(0, 1)
        if self.temporal:
            default_priors['slope'] = CauchyPrior(loc=0, scale=1)
        for key, prior in default_priors.items():
            self.priors.setdefault(key, prior)

    def sample_shared_params(self, values):
        """
        Sample parameters that are shared amongst individuals.
        """
        values.update({key: self.priors[key](value) for key, value in values.items()
                       if key != 'patient_mean'})
        return values

    @_augment_values
    def sample_individual_params(self, values):
        """
        Sample individual-level parameters.
        """
        # Use from_uniform directly to avoid overhead of instance creation
        values['patient_mean'] = GengammaPrior.from_uniform(
            values['patient_mean'], values['population_shape'], values['population_loc'],
            values['population_scale'])
        return values

    def sample_params(self, values):
        """
        Sample shared and individual-level parameters from the hierarchical prior.
        """
        self.sample_shared_params(values)
        self.sample_individual_params(values)
        return values

    @_augment_values
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
        slope = values.get('slope')
        if slope is not None:
            mu += slope * data['day']
        lxdf = gengamma_lpdf(q, mu, sigma, data['loadln'], where=data['positive'])
        gengamma_lcdf(q, mu, sigma, data['loqln'], out=lxdf, where=~data['positive'])
        return lxdf

    @_augment_values
    def _evaluate_patient_log_likelihood(self, values, data):
        """
        Evaluate the log likelihood for each patient conditional on all model parameters, assuming
        that all patients shed virus.
        """
        result = self._evaluate_sample_log_likelihood(values, data)
        return np.bincount(data['idx'], result, minlength=data['num_patients'])

    @_augment_values
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
    @_augment_values
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
        values['patient_mean'] = GengammaPrior.from_uniform(
            uniform, values['population_shape'], values['population_loc'],
            values['population_scale'])
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
    @_augment_values
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
        uniform = np.random.uniform(size=size)
        patient_mean = GengammaPrior.from_uniform(
            uniform, values['population_shape'], values['population_loc'],
            values['population_scale'])
        # Evaluate the locations for the sample distribution
        loc = gengamma_loc(values['patient_shape'], values['patient_scale'], patient_mean)
        # Sample the RNA loads
        uniform = np.random.uniform(size=size)
        sample = GengammaPrior.from_uniform(uniform, values['patient_shape'], loc,
                                            values['patient_scale'])
        if not self.inflated:
            return sample
        # Account for non-shedders
        z = np.random.uniform(size=np.shape(sample)) < values['rho']
        return np.where(z, sample, np.nan)

    @broadcast_samples
    @_augment_values
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
        load = GengammaPrior.from_uniform(np.random.uniform(size=mu.size), q, mu, sigma)

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
        data['loadln'] = np.log(load)
        data['positive'] = positive = load >= loq

        # Update summary statistics
        data['num_positives_by_patient'] = np.bincount(data['idx'], positive,
                                                       minlength=num_patients).astype(int)
        data['num_negatives_by_patient'] = np.bincount(data['idx'], ~positive,
                                                       minlength=num_patients).astype(int)
        return data

    @broadcast_samples
    @_augment_values
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

    @_augment_values
    def evaluate_log_joint(self, values, data):
        # Population and patient shape and scale as well as population loc
        result = sum(prior.lpdf(values[key]) for key, prior in self.priors.items())
        # Patient means
        x = values.get('log_patient_mean')
        if x is None:
            x = np.log(values['patient_mean'])
        result += gengamma_lpdf(values['population_shape'], values['population_loc'],
                                values['population_scale'], x).sum()
        return result + self.evaluate_log_likelihood(values, data)

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
