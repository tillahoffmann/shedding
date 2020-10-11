import numpy as np
from scipy import special
from . import util
from ..util import broadcast_shapes


def qms2abc(q, mu, sigma):
    """
    Convert (shape, location, scale) to (shape, scale, exponent) parametrisation.
    """
    a = q ** -2
    b = a * np.exp(- q * mu / sigma)
    c = q / sigma
    return a, b, c


def abc2qms(a, b, c):
    """
    Convert (shape, scale, exponent) to (shape, location, scale) parametrisation.
    """
    mu = np.log(a / b) / c
    q = 1 / np.sqrt(a)
    sigma = q / c
    return q, mu, sigma


def gengamma_lpdf(x, q, mu, sigma):
    r"""
    Evaluate the natural logarithm of the generalised gamma probability density function.

    Notes
    -----
    We use the parametrisation proposed by Stacey (1962) because it tends to be more numerically
    stable (see https://www.rdocumentation.org/packages/flexsurv/versions/1.1.1/topics/GenGamma). In
    short, if :math:`\gamma\sim\mathrm{Gamma}(q^{-2},1)` and :math:`w=\log(q^2 \gamma)/q`, then
    :math:`x=\exp(\mu+\sigma w)` follows a generalised gamma distribution. This formulation also
    allows for a non-centred parametrisation of the hierarchical model.
    """
    q2 = q ** 2
    return - (sigma * x ** (q / sigma) * np.exp(-mu * q / sigma) + (2 - q2) * sigma * np.log(q) +
              q * (mu - np.log(x) + q * sigma * (np.log(sigma * x) + special.gammaln(1 / q2)))) / \
        (sigma * q ** 2)


def gengamma_lcdf(x, q, mu, sigma):
    """
    Evaluate the natural logarithm of the generalised gamma cumulative distribution function.
    """
    log_arg = q / sigma * np.log(x) - mu * q / sigma - 2 * np.log(q)
    cdf = special.gammainc(q ** -2, np.exp(log_arg))
    return np.log(cdf)


def gengamma_mean(q, mu, sigma):
    """
    Evaluate the mean of the generalised gamma distribution.
    """
    log_mean = mu + 2 * sigma * np.log(q) / q - special.gammaln(q ** -2) + \
        special.gammaln((1 + q * sigma) / q ** 2)
    return np.exp(log_mean)


def gengamma_rvs(q, mu, sigma, size=None):
    size = broadcast_shapes(np.shape(q), np.shape(mu), np.shape(sigma), size)
    gamma = np.random.standard_gamma(q ** -2, size=size)
    w = np.log(q ** 2 * gamma) / q
    return np.exp(mu + sigma * w)


class GeneralisedGammaModel(util.Model):
    """
    Hierarchical generalised gamma model with random effects for each patient.
    """
    MODEL_CODE = """
    functions {
        real gengamma_lpdf(real x, real q, real mu, real sigma) {
            real q2 = q ^ 2;
            return - (sigma * x ^ (q / sigma) * exp(-mu * q / sigma) + (2 - q2) * sigma * log(q) +
                q * (mu - log(x) + q * sigma * (log(sigma * x) + lgamma(1 / q2)))) / (sigma * q2);
        }

        real gengamma_lcdf(real x, real q, real mu, real sigma) {
            real log_arg = q / sigma * log(x) - mu * q / sigma - 2 * log(q);
            return log(gamma_p(1 / q ^ 2, exp(log_arg)));
        }
    }

    data {
        {{data}}

        // Whether to restrict to the regular gamma distribution
        int regular_gamma;
    }

    parameters {
        // Population level parameters
        real population_loc;
        real<lower=0> population_shape;
        real<lower=0> population_scale_;

        // Individual level parameters
        vector<lower=0>[num_patients] patient_gamma_;  // Random variable for non-centred setup
        real<lower=0> patient_shape;
        real<lower=0> patient_scale_;
    }

    transformed parameters {
        // Calculation to induce the right distribution using the non-centred setup
        vector<lower=0>[num_patients] patient_mean;
        // Contribution to the patient_loc
        real loc_contrib_;
        // Contribution to the target
        vector[num_samples] sample_contrib_;
        // Declare transformed parameters for the population_scale and patient_scale
        real<lower=0> population_scale;
        real<lower=0> patient_scale;
        if (regular_gamma == 1) {
            population_scale = population_shape;
            patient_scale = patient_shape;
        } else {
            population_scale = population_scale_;
            patient_scale = patient_scale_;
        }

        patient_mean = exp(population_loc + population_scale *
            log(population_shape ^ 2 * patient_gamma_) / population_shape);

        // Evaluate the contribution to the patient_loc
        loc_contrib_ = lgamma(1 / patient_shape ^ 2) -
            lgamma((1 + patient_shape * patient_scale) / patient_shape ^ 2) -
            2 * patient_scale * log(patient_shape) / patient_shape;
        // Evaluate contributions to the target
        for (j in 1:num_samples) {
            real loc = log(patient_mean[idx[j]]) + loc_contrib_;
            if (load[j] > loq[j]) {
                sample_contrib_[j] = gengamma_lpdf(load[j] | patient_shape, loc, patient_scale);
            } else {
                sample_contrib_[j] = gengamma_lcdf(loq[j] | patient_shape, loc, patient_scale);
            }
        }
    }

    model {
        // Sample individual parameters for each person (non-centred parametrisation)
        target += gamma_lpdf(patient_gamma_ | 1 / population_shape ^ 2, 1);
        target += sum(sample_contrib_);
        // Prior for unused parameters depending on parametrisation
        if (regular_gamma == 1) {
            patient_scale_ ~ gamma(1, 1);
            population_scale_ ~ gamma(1, 1);
        }
    }
    """
    DEFAULT_DATA = {
        'regular_gamma': 0,
    }

    def _evaluate_loc(self, q, mean, sigma):
        return special.gammaln(q ** -2) - special.gammaln((1 + q * sigma) / q ** 2) - \
            2 * sigma * np.log(q) / q + np.log(mean)

    @util.broadcast_samples
    def replicate(self, x, data, mode):
        if mode == util.ReplicationMode.NEW_GROUPS:
            patient_mean = gengamma_rvs(x['population_shape'], x['population_loc'],
                                        x['population_scale'], size=data['num_patients'])
        else:
            patient_mean = x['patient_mean']

        loc = self._evaluate_loc(x['patient_shape'], patient_mean, x['patient_scale'])
        loc = np.repeat(loc, data['num_samples_by_patient'])
        load = gengamma_rvs(x['patient_shape'], loc, x['patient_scale'])
        return util.merge_data(data, load=load, patient_mean=patient_mean)

    def _evaluate_statistic(self, x, statistic, n, **kwargs):
        if statistic == 'mean':
            return gengamma_mean(x['population_shape'], x['population_loc'], x['population_scale'])
        raise NotImplementedError(statistic)

    def _evaluate_observed_likelihood_contributions(self, x, data, n=1000, **kwargs):
        patient_mean = gengamma_rvs(x['population_shape'], x['population_loc'],
                                    x['population_scale'], (n, data['num_patients']))
        loc = self._evaluate_loc(x['patient_shape'], patient_mean, x['patient_scale'])
        loc = np.repeat(loc, data['num_samples_by_patient'], axis=-1)
        lpdf = gengamma_lpdf(data['load'], x['patient_shape'], loc, x['patient_scale'])
        lcdf = gengamma_lcdf(data['loq'], x['patient_shape'], loc, x['patient_scale'])
        return lpdf, lcdf

    @util.broadcast_samples
    def rvs(self, x, size=None):
        patient_mean = gengamma_rvs(x['population_shape'], x['population_loc'],
                                    x['population_scale'], size)
        loc = self._evaluate_loc(x['patient_shape'], patient_mean, x['patient_scale'])
        return gengamma_rvs(x['patient_shape'], loc, x['patient_scale'], size)
