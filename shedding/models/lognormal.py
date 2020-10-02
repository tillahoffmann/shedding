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
        vector<lower=0>[num_patients] patient_mean;
        real<lower=0> patient_scale;
    }

    transformed parameters {
        vector[num_samples] sample_contrib_;
        // Iterate over the samples and distinguish between...
        for (i in 1:num_samples) {
            real mu = log(patient_mean[idx[i]]) - patient_scale ^ 2 / 2;
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
        target += lognormal_lpdf(patient_mean | population_loc, population_scale);
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
            patient_mean = np.random.lognormal(x['population_loc'], x['population_scale'],
                                               data['num_patients'])
        else:
            patient_mean = x['patient_mean']

        # Sample viral loads
        mu = np.log(np.repeat(patient_mean, data['num_samples_by_patient'])) - \
            x['patient_scale'] ** 2 / 2
        load = np.random.lognormal(mu, x['patient_scale'], data['num_samples'])
        return util.merge_data(data, load=load, patient_mean=patient_mean)

    def _evaluate_statistic(self, x, statistic, n, **kwargs):
        if statistic == 'mean':
            return lognormal_mean(x['population_loc'], x['population_scale'])
        raise ValueError(statistic)

    def _evaluate_observed_likelihood_contributions(self, x, data, n, analytic=True, **kwargs):
        if analytic:
            mu = x['population_loc'] - x['patient_scale'] ** 2 / 2
            lpdf = lognormal_lpdf(data['load'], mu, x['population_scale'])[None]
            lcdf = lognormal_lcdf(data['loq'], mu, x['population_scale'])[None]
        else:
            # Sample the patient-level variables
            patient_mean = np.random.lognormal(x['population_loc'], x['population_scale'],
                                               (n, data['num_patients']))
            # Evaluate the likelihood contributions for each sample given those means
            mu = np.repeat(np.log(patient_mean), data['num_samples_by_patient'], axis=-1) - \
                x['patient_scale'] ** 2 / 2
            lpdf = lognormal_lpdf(data['load'], mu, x['patient_scale'])
            lcdf = lognormal_lcdf(data['loq'], mu, x['patient_scale'])
        return lpdf, lcdf


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
        vector<lower=0>[num_patients] patient_mean;
        real<lower=0> patient_scale;
    }

    transformed parameters {
        vector[num_patients] patient_contrib_ = rep_vector(0, num_patients);

        // Iterate over samples
        for (j in 1:num_samples) {
            int i = idx[j];
            real mu = log(patient_mean[i]) - patient_scale ^ 2 / 2;
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
        target += lognormal_lpdf(patient_mean | population_loc, population_scale);
        // Monomial priors for the scale parameters (defaults to log-uniform Jeffrey's prior)
        target += prior_population_scale_pow * log(population_scale);
        target += prior_patient_scale_pow * log(patient_scale);
    }
    """

    @util.broadcast_samples
    def replicate(self, x, data, mode, **kwargs):
        if mode == util.ReplicationMode.NEW_GROUPS:
            # Generate group-level means and indicators
            patient_mean = np.random.lognormal(x['population_loc'], x['population_scale'],
                                               data['num_patients'])
            z = np.random.uniform(0, 1, data['num_patients']) < x['rho']
        else:
            # Just sample the indicators we marginalised over analytically
            z = util.sample_indicators(x, data)
            patient_mean = x['patient_mean']

        mu = np.log(np.repeat(patient_mean, data['num_samples_by_patient'])) - \
            x['patient_scale'] ** 2 / 2
        load = np.random.lognormal(mu, x['patient_scale'])
        load = np.where(np.repeat(z, data['num_samples_by_patient']), load, data['loq'] / 10)

        return util.merge_data(data, load=load, z=z, patient_mean=patient_mean)
