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
        (sigma * q2)


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
    q2 = q ** 2
    log_mean = mu + 2 * sigma * np.log(q) / q - special.gammaln(1 / q2) + \
        special.gammaln((1 + q * sigma) / q2)
    return np.exp(log_mean)


def gengamma_rvs(q, mu, sigma, size=None):
    q2 = q ** 2
    size = broadcast_shapes(np.shape(q), np.shape(mu), np.shape(sigma), size)
    gamma = np.random.standard_gamma(1 / q2, size=size)
    w = np.log(q2 * gamma) / q
    return np.exp(mu + sigma * w)


class GeneralisedGammaModel(util.Model):
    """
    Hierarchical generalised gamma model with random effects for each patient.
    """
    MODEL_CODE = """
    functions {
        {{gengamma_lpdf_lcdf}}
    }

    data {
        {{data}}
    }

    parameters {
        // Population level parameters.
        real population_loc;
        real<lower=0> population_shape;
        real<lower=0> population_scale;

        // Individual level parameters
        real<lower=0> patient_scale;
        real<lower=0> patient_shape;
        // Random variable for non-centred setup.
        vector<lower=0>[num_patients] patient_gamma_;
    }

    transformed parameters {
        // Evaluate the patient mean in terms of the gamma random variable for an almost non-centred
        // parametrisatoin.
        vector<lower=0>[num_patients] patient_mean = exp(population_loc + population_scale *
            log(population_shape ^ 2 * patient_gamma_) / population_shape);
        // Contribution to the patient_loc, evaluated once for efficiency.
        real loc_contrib_ = lgamma(1 / patient_shape ^ 2) -
            lgamma((1 + patient_shape * patient_scale) / patient_shape ^ 2) -
            2 * patient_scale * log(patient_shape) / patient_shape;
        // Vector to hold the contributions to the target density for each sample.
        vector[num_samples] sample_contrib_;

        // Evaluate contributions to the target.
        for (j in 1:num_samples) {
            // Evaluation the location parameter for this sample.
            real loc = log(patient_mean[idx[j]]) + loc_contrib_;
            if (load[j] > loq[j]) {
                // Account for quantitative observations of the concentration in a sample.
                sample_contrib_[j] = gengamma_lpdf(load[j] | patient_shape, loc, patient_scale);
            } else {
                // Handle left-censoring if the concentration is below the level of quantification.
                sample_contrib_[j] = gengamma_lcdf(loq[j] | patient_shape, loc, patient_scale);
            }
        }
    }

    model {
        // Sample the latent gamma parameters that induce the generalised gamma prior for the
        // patient mean.
        target += gamma_lpdf(patient_gamma_ | 1 / population_shape ^ 2, 1);
        // Add the contributions from all the samples.
        target += sum(sample_contrib_);
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
