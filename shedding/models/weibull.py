import numpy as np
from scipy import special
from . import util


def weibull_lpdf(x, a, b):
    """
    Evaluate the natural logarithm of the Weibull probability density function.
    """
    return np.log(a / b) + (a - 1) * np.log(x / b) - (x / b) ** a


def weibull_lcdf(x, a, b):
    """
    Evaluate the natural logarithm of the Weibull cumulative distribution function.
    """
    return np.log1p(-np.exp(-(x / b) ** a))


def weibull_rng(a, b, size=None):
    """
    Draw samples from a scaled Weibull distribution.
    """
    return b * np.random.weibull(a, size)


class WeibullModel(util.Model):
    """
    Hierarchical Weibull model with random effects for each patient.
    """
    MODEL_CODE = """
    data {
        {{data}}

        // Prior hyperparameters
        real<upper=0> prior_population_scale_pow;
    }

    parameters {
        // Population level parameters
        real<lower=0> population_shape;
        real<lower=0> population_scale;

        // Individual level parameters (we want to keep a constant coefficient of variation)
        real<lower=0> patient_shape;
        vector<lower=0>[num_patients] patient_mean;
    }

    transformed parameters {
        vector[num_patients] patient_scale_ = patient_mean / tgamma(1 + 1 / patient_shape);
        vector[num_samples] sample_contrib_;
        for (i in 1:num_samples) {
            real scale = patient_scale_[idx[i]];
            if (load[i] > loq[i]) {
                sample_contrib_[i] = weibull_lpdf(load[i] | patient_shape, scale);
            } else {
                sample_contrib_[i] = weibull_lcdf(loq[i] | patient_shape, scale);
            }
        }
    }

    model {
        // Sample individual parameters for each person
        target += weibull_lpdf(patient_mean | population_shape, population_scale);
        target += sum(sample_contrib_);
        // Monomial priors for the scale parameter (defaults to log-uniform Jeffrey's prior)
        target += prior_population_scale_pow * log(population_scale);
    }
    """
    DEFAULT_DATA = {
        'prior_population_scale_pow': -1,
    }

    def _evaluate_scale(self, shape, mean):
        log_scale = np.log(mean) - special.gammaln(1 + 1 / shape)
        return np.exp(log_scale)

    @util.broadcast_samples
    def replicate(self, x, data, mode, **kwargs):
        if mode == util.ReplicationMode.NEW_GROUPS:
            patient_mean = weibull_rng(x['population_shape'], x['population_scale'],
                                       data['num_patients'])
        else:
            patient_mean = x['patient_mean']

        patient_scale = self._evaluate_scale(x['patient_shape'], patient_mean)
        patient_scale = np.repeat(patient_scale, data['num_samples_by_patient'])
        load = weibull_rng(x['patient_shape'], patient_scale, data['num_samples'])
        return util.merge_data(data, load=load, patient_mean=patient_mean)

    def _evaluate_statistic(self, x, statistic, n, **kwargs):
        if statistic == 'mean':
            return x['population_scale'] * special.gamma(1 / x['population_shape']) / \
                x['population_shape']
        raise NotImplementedError(statistic)

    def _evaluate_observed_likelihood_contributions(self, x, data, n=1000, **kwargs):
        patient_mean = weibull_rng(x['population_shape'], x['population_scale'],
                                   (n, data['num_patients']))
        patient_scale = self._evaluate_scale(x['patient_shape'], patient_mean)
        patient_scale = np.repeat(patient_scale, data['num_samples_by_patient'], axis=-1)
        lpdf = weibull_lpdf(data['load'], x['patient_shape'], patient_scale)
        lcdf = weibull_lcdf(data['loq'], x['patient_shape'], patient_scale)
        return lpdf, lcdf

    @util.broadcast_samples
    def rvs(self, x, size=None):
        patient_mean = weibull_rng(x['population_shape'], x['population_scale'], size)
        patient_scale = self._evaluate_scale(x['patient_shape'], patient_mean)
        return weibull_rng(x['patient_shape'], patient_scale, np.shape(patient_scale))


class WeibullInflatedModel(util.InflationMixin, WeibullModel):
    """
    Hierarchical Weibull model with random effects for each patient and a binary shedding indicator
    to support zero-inflated results.
    """
    MODEL_CODE = """
    data {
        {{data}}

        // Prior hyperparameters
        real<upper=0> prior_population_scale_pow;
    }

    parameters {
        // Proportion of individuals that are positive in stool
        real<lower=0, upper=1> rho;

        // Population level parameters
        real<lower=0> population_shape;
        real<lower=0> population_scale;

        // Individual level parameters (we want to keep a constant coefficient of variation)
        real<lower=0> patient_shape;
        vector<lower=0>[num_patients] patient_mean;
    }

    transformed parameters {
        // Contributions to the data likelihood
        vector[num_patients] patient_contrib_ = rep_vector(0, num_patients);
        vector[num_patients] patient_scale_ = patient_mean / tgamma(1 + 1 / patient_shape);

        // Iterate over samples
        for (j in 1:num_samples) {
            int i = idx[j];
            real scale = patient_scale_[i];
            if (load[j] > loq[j]) {
                patient_contrib_[i] += weibull_lpdf(load[j] | patient_shape, scale);
            } else {
                patient_contrib_[i] += weibull_lcdf(loq[j] | patient_shape, scale);
            }
        }
    }

    model {
        {{patient_contrib}}

        // Sample individual parameters for each person
        target += weibull_lpdf(patient_mean | population_shape, population_scale);
        // Monomial priors for the scale parameter (defaults to log-uniform Jeffrey's prior)
        target += prior_population_scale_pow * log(population_scale);
    }
    """

    @util.broadcast_samples
    def replicate(self, x, data, mode, **kwargs):
        if mode == util.ReplicationMode.NEW_GROUPS:
            patient_mean = weibull_rng(x['population_shape'], x['population_scale'],
                                       data['num_patients'])
            z = np.random.uniform(0, 1, data['num_patients']) < x['rho']
        else:
            patient_mean = x['patient_mean']
            z = util.sample_indicators(x, data)

        patient_scale = patient_mean / special.gamma(1 + 1 / x['patient_shape'])
        patient_scale = np.repeat(patient_scale, data['num_samples_by_patient'])
        load = weibull_rng(x['patient_shape'], patient_scale, data['num_samples'])
        load = np.where(np.repeat(z, data['num_samples_by_patient']), load, data['loq'] / 10)
        return util.merge_data(data, load=load, z=z, patient_mean=patient_mean)
