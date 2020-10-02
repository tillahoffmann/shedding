import numpy as np
from scipy import special
from . import util


def gamma_lpdf(x, a, b):
    """
    Evaluate the natural logarithm of the gamma probability density function.
    """
    return a * np.log(b) - special.gammaln(a) + (a - 1) * np.log(x) - b * x


def gamma_lcdf(x, a, b):
    """
    Evaluate the natural logarithm of the gamma cumulative distribution function.
    """
    return np.log(special.gammainc(a, b * x))


class GammaModel(util.Model):
    """
    Hierarchical gamma model with random effects for each patient.
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
        vector<lower=0>[num_patients] patient_mean;
        real<lower=0> patient_shape;
    }

    transformed parameters {
        // Evaluate contributions to the target
        vector[num_samples] sample_contrib_;
        for (i in 1:num_samples) {
            real scale = patient_shape / patient_mean[idx[i]];
            if (load[i] > loq[i]) {
                sample_contrib_[i] = gamma_lpdf(load[i] | patient_shape, scale);
            } else {
                sample_contrib_[i] = gamma_lcdf(loq[i] | patient_shape, scale);
            }
        }
    }

    model {
        // Sample individual parameters for each person
        target += gamma_lpdf(patient_mean | population_shape, population_scale);
        target += sum(sample_contrib_);

        // Monomial priors for the scale parameter (defaults to log-uniform Jeffrey's prior)
        target += prior_population_scale_pow * log(population_scale);
    }
    """
    DEFAULT_DATA = {
        'prior_population_scale_pow': -1,
    }

    @util.broadcast_samples
    def replicate(self, x, data, mode, **kwargs):
        if mode == util.ReplicationMode.NEW_GROUPS:
            patient_mean = np.random.gamma(x['population_shape'], scale=1 / x['population_scale'],
                                           size=data['num_patients'])
        else:
            patient_mean = x['patient_mean']

        patient_scale = np.repeat(x['patient_shape'] / patient_mean, data['num_samples_by_patient'])
        load = np.random.gamma(x['patient_shape'], scale=1 / patient_scale)

        return util.merge_data(data, load=load, patient_mean=patient_mean)

    def _evaluate_statistic(self, x, statistic, n, **kwargs):
        if statistic == 'mean':
            return x['population_shape'] / x['population_scale']
        elif statistic in ('var', 'std'):
            var = x['population_shape'] * (1 + x['population_shape'] + x['patient_shape']) / \
                (x['patient_shape'] * x['population_scale'] ** 2)
            if statistic == 'var':
                return var
            return np.sqrt(var)
        raise ValueError(statistic)

    def _evaluate_observed_likelihood_contributions(self, x, data, n=1000, **kwargs):
        patient_mean = np.random.gamma(x['population_shape'], 1 / x['population_scale'],
                                       (n, data['num_patients']))
        patient_scale = np.repeat(x['patient_shape'] / patient_mean, data['num_samples_by_patient'],
                                  axis=-1)
        lpdf = gamma_lpdf(data['load'], x['patient_shape'], patient_scale)
        lcdf = gamma_lcdf(data['loq'], x['patient_shape'], patient_scale)
        return lpdf, lcdf

    @util.broadcast_samples
    def rvs(self, x, size=None):
        patient_mean = np.random.gamma(x['population_shape'], 1 / x['population_scale'], size)
        patient_scale = x['patient_shape'] / patient_mean
        return np.random.gamma(x['patient_shape'], 1 / patient_scale, np.shape(patient_scale))


class GammaInflatedModel(util.InflationMixin, GammaModel):
    """
    Hierarchical gamma model with random effects for each patient and a binary shedding indicator to
    support zero-inflated results.
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
        vector<lower=0>[num_patients] patient_mean;
        real<lower=0> patient_shape;
    }

    transformed parameters {
        // Contributions to the data likelihood
        vector[num_patients] patient_contrib_ = rep_vector(0, num_patients);

        // Additional parameters to report summary statistics more easily
        real<lower=0> population_mean_ = population_shape / population_scale;

        // Parameters for the individual gamma distribution
        vector<lower=0>[num_patients] patient_scale_;
        for (i in 1:num_patients) {
            patient_scale_[i] = patient_shape / patient_mean[i];
        }

        // Iterate over samples
        for (j in 1:num_samples) {
            int i = idx[j];
            real scale = patient_scale_[i];
            if (load[j] > loq[j]) {
                patient_contrib_[i] += gamma_lpdf(load[j] | patient_shape, scale);
            } else {
                patient_contrib_[i] += gamma_lcdf(loq[j] | patient_shape, scale);
            }
        }
    }

    model {
        {{patient_contrib}}

        // Sample individual parameters for each person
        target += gamma_lpdf(patient_mean | population_shape, population_scale);
        // Monomial priors for the scale parameter (defaults to log-uniform Jeffrey's prior)
        target += prior_population_scale_pow * log(population_scale);
    }
    """

    @util.broadcast_samples
    def replicate(self, x, data, mode, **kwargs):
        if mode == util.ReplicationMode.NEW_GROUPS:
            # Generate the patient mean
            patient_mean = np.random.gamma(x['population_shape'], scale=1 / x['population_scale'],
                                           size=data['num_patients'])
            z = np.random.uniform(0, 1, data['num_patients']) < x['rho']
        else:
            z = util.sample_indicators(x, data)
            # Get the patient mean
            patient_mean = x['patient_mean']

        patient_scale = np.repeat(x['patient_shape'] / patient_mean, data['num_samples_by_patient'])
        load = np.random.gamma(x['patient_shape'], scale=1 / patient_scale)
        load = np.where(np.repeat(z, data['num_samples_by_patient']), load, data['loq'] / 10)
        return util.merge_data(data, load=load, z=z, patient_mean=patient_mean)
