import numpy as np
from scipy import special
from . import util


def lognormal_lpdf(x, mu, sigma):
    """
    Evaluate the natural logarithm of the lognormal probability density function.
    """
    return - (np.log(2 * np.pi) / 2 + (mu - np.log(x)) ** 2 / (2 * sigma ** 2) + np.log(sigma * x))


def lognormal_lcdf(x, mu, sigma):
    """
    Evaluate the natural logarithm of the lognormal cumulative distribution function.
    """
    return np.log(special.erfc((mu - np.log(x)) / (sigma * np.sqrt(2))) / 2)


def lognormal_mean(mu, sigma):
    """
    Evaluate the mean of the lognormal distribution.
    """
    return np.exp(mu + sigma ** 2 / 2)


class LognormalModel(util.Model):
    """
    Hierarchical lognormal model with random effects for each patient.
    """
    MODEL_CODE = """
    data {
        {{data}}

        // Prior hyperparameters
        real<upper=0> prior_population_scale_pow;
        real<upper=0> prior_patient_scale_pow;
    }

    parameters {
        // Population level parameters
        real population_loc;
        real<lower=0> population_scale;

        // Individual level parameters
        vector[num_patients] patient_loc;
        real<lower=0> patient_scale;
    }

    transformed parameters {
        vector[num_samples] sample_contrib_;
        // Iterate over the samples and distinguish between...
        for (i in 1:num_samples) {
            real mu = patient_loc[idx[i]] - patient_scale ^ 2 / 2;
            // ... a quantitative observation ...
            if (load[i] > loq[i]) {
                sample_contrib_[i] = lognormal_lpdf(load[i] | mu, patient_scale);
            }
            // ... or a censored observation.
            else {
                sample_contrib_[i] = lognormal_lcdf(loq[i] | mu, patient_scale);
            }
        }
    }

    model {
        // Sample individual parameters for each person ...
        target += normal_lpdf(patient_loc | population_loc, population_scale);
        // ... and the contributions from samples
        target += sum(sample_contrib_);
        // Monomial priors for the scale parameters (defaults to log-uniform Jeffrey's prior)
        target += prior_population_scale_pow * log(population_scale);
        target += prior_patient_scale_pow * log(patient_scale);
    }
    """
    DEFAULT_DATA = {
        'prior_population_scale_pow': -1,
        'prior_patient_scale_pow': -1,
    }

    @util.broadcast_samples
    def replicate(self, x, data, mode, **kwargs):
        # Sample new group-level parameters if required
        if mode == util.ReplicationMode.NEW_GROUPS:
            patient_loc = np.random.normal(x['population_loc'], x['population_scale'],
                                           data['num_patients'])
        else:
            patient_loc = x['patient_loc']

        # Sample viral loads
        mu = np.repeat(patient_loc, data['num_samples_by_patient']) - x['patient_scale'] ** 2 / 2
        load = np.random.lognormal(mu, x['patient_scale'], data['num_samples'])
        return util.merge_data(data, load=load, patient_mean=np.exp(patient_loc))

    def _evaluate_statistic(self, x, statistic, n, **kwargs):
        if statistic == 'mean':
            return lognormal_mean(x['population_loc'], x['population_scale'])
        raise NotImplementedError(statistic)

    def _evaluate_observed_likelihood_contributions(self, x, data, n, **kwargs):
        # Sample the patient-level variables
        patient_loc = np.random.normal(x['population_loc'], x['population_scale'],
                                       (n, data['num_patients']))
        # Evaluate the likelihood contributions for each sample given those means
        mu = np.repeat(patient_loc, data['num_samples_by_patient'], axis=-1) - \
            x['patient_scale'] ** 2 / 2
        lpdf = lognormal_lpdf(data['load'], mu, x['patient_scale'])
        lcdf = lognormal_lcdf(data['loq'], mu, x['patient_scale'])
        return lpdf, lcdf

    @util.broadcast_samples
    def rvs(self, x, size=None):
        patient_loc = np.random.normal(x['population_loc'], x['population_scale'], size) - \
            x['patient_scale'] ** 2 / 2
        return np.random.lognormal(patient_loc, x['patient_scale'], np.shape(patient_loc))


class LognormalInflatedModel(util.InflationMixin, LognormalModel):
    """
    Hierarchical lognormal model with random effects for each patient and a binary shedding
    indicator to support zero-inflated results.
    """
    MODEL_CODE = """
    data {
        {{data}}

        // Prior hyperparameters
        real<upper=0> prior_population_scale_pow;
        real<upper=0> prior_patient_scale_pow;
    }

    parameters {
        // Proportion of individuals that are positive in stool
        real<lower=0, upper=1> rho;

        // Population level parameters
        real population_loc;
        real<lower=0> population_scale;

        // Individual level parameters
        vector[num_patients] patient_loc;
        real<lower=0> patient_scale;
    }

    transformed parameters {
        vector[num_patients] patient_contrib_ = rep_vector(0, num_patients);

        // Iterate over samples
        for (j in 1:num_samples) {
            int i = idx[j];
            real mu = patient_loc[i] - patient_scale ^ 2 / 2;
            if (load[j] > loq[j]) {
                patient_contrib_[i] += lognormal_lpdf(load[j] | mu, patient_scale);
            } else {
                patient_contrib_[i] += lognormal_lcdf(loq[j] | mu, patient_scale);
            }
        }
    }

    model {
        {{patient_contrib}}

        // Sample individual parameters for each person.
        target += normal_lpdf(patient_loc | population_loc, population_scale);
        // Monomial priors for the scale parameters (defaults to log-uniform Jeffrey's prior)
        target += prior_population_scale_pow * log(population_scale);
        target += prior_patient_scale_pow * log(patient_scale);
    }
    """

    @util.broadcast_samples
    def replicate(self, x, data, mode, **kwargs):
        if mode == util.ReplicationMode.NEW_GROUPS:
            # Generate group-level means and indicators
            patient_loc = np.random.normal(x['population_loc'], x['population_scale'],
                                           data['num_patients'])
            z = np.random.uniform(0, 1, data['num_patients']) < x['rho']
        else:
            # Just sample the indicators we marginalised over analytically
            z = util.sample_indicators(x, data)
            patient_loc = x['patient_loc']

        mu = np.repeat(patient_loc, data['num_samples_by_patient']) - x['patient_scale'] ** 2 / 2
        load = np.random.lognormal(mu, x['patient_scale'])
        load = np.where(np.repeat(z, data['num_samples_by_patient']), load, data['loq'] / 10)

        return util.merge_data(data, load=load, z=z, patient_mean=np.exp(patient_loc))
