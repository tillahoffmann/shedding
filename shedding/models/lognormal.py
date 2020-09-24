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
    MODEL_CODE = util.MODEL_BOILERPLATE + """
    parameters {
        // Population level parameters
        real population_mean;
        real<lower=0> population_scale;

        // Individual level parameters
        vector<lower=0>[num_patients] patient_mean;
        real<lower=0> patient_scale;
    }

    transformed parameters {
        vector[num_samples] sample_contrib_;
        // Iterate over the samples and distinguish between...
        for (i in 1:num_samples) {
            real mu = log(patient_mean[idx[i]]);
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
        target += lognormal_lpdf(patient_mean | population_mean, population_scale);
        // ... and the contributions from samples
        target += sum(sample_contrib_);
    }
    """

    @util.broadcast_samples
    def replicate(self, x, data, mode, **kwargs):
        # Sample new group-level parameters if required
        if mode == util.ReplicationMode.NEW_GROUPS:
            patient_mean = np.random.lognormal(x['population_mean'], x['population_scale'],
                                               data['num_patients'])
        else:
            patient_mean = x['patient_mean']

        # Sample viral loads
        mu = np.log(np.repeat(patient_mean, data['num_samples_by_patient']))
        load = np.random.lognormal(mu, x['patient_scale'], data['num_samples'])
        return util.merge_data(data, load=load, patient_mean=patient_mean)

    def _evaluate_statistic(self, x, statistic, n):
        mu = x['population_mean']
        sigma = np.sqrt(x['population_scale'] ** 2 + x['patient_scale'] ** 2)
        if statistic == 'mean':
            return lognormal_mean(mu, sigma)
        raise ValueError(statistic)

    def _evaluate_observed_likelihood_contributions(self, x, data, n, analytic=True):
        if analytic:
            sigma = np.sqrt(x['population_scale'] ** 2 + x['patient_scale'] ** 2)
            lpdf = lognormal_lpdf(data['load'], x['population_mean'], sigma)[None]
            lcdf = lognormal_lcdf(data['loq'], x['population_mean'], sigma)[None]
        else:
            # Sample the patient-level variables
            patient_mean = np.random.lognormal(x['population_mean'], x['population_scale'],
                                               (n, data['num_patients']))
            # Evaluate the likelihood contributions for each sample given those means
            mu = np.repeat(np.log(patient_mean), data['num_samples_by_patient'], axis=-1)
            lpdf = lognormal_lpdf(data['load'], mu, x['patient_scale'])
            lcdf = lognormal_lcdf(data['loq'], mu, x['patient_scale'])
        return lpdf, lcdf


class LognormalInflatedModel(util.InflationMixin, LognormalModel):
    """
    Hierarchical lognormal model with random effects for each patient and a binary shedding
    indicator to support zero-inflated results.
    """
    MODEL_CODE = util.MODEL_BOILERPLATE + """
    parameters {
        // Proportion of individuals that are positive in stool
        real<lower=0, upper=1> rho;

        // Population level parameters
        real population_mean;
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
            real mu = log(patient_mean[i]);
            if (load[j] > loq[j]) {
                patient_contrib_[i] += lognormal_lpdf(load[j] | mu, patient_scale);
            } else {
                patient_contrib_[i] += lognormal_lcdf(loq[j] | mu, patient_scale);
            }
        }
    }

    model {
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

        // Sample individual parameters for each person.
        target += lognormal_lpdf(patient_mean | population_mean, population_scale);
    }
    """

    @util.broadcast_samples
    def replicate(self, x, data, mode, **kwargs):
        if mode == util.ReplicationMode.NEW_GROUPS:
            # Generate group-level means and indicators
            patient_mean = np.random.lognormal(x['population_mean'], x['population_scale'],
                                               data['num_patients'])
            z = np.random.uniform(0, 1, data['num_patients']) < x['rho']
        else:
            # Just sample the indicators we marginalised over analytically
            z = util.sample_indicators(x, data)
            patient_mean = x['patient_mean']

        mu = np.log(np.repeat(patient_mean, data['num_samples_by_patient']))
        load = np.random.lognormal(mu, x['patient_scale'])
        load = np.where(np.repeat(z, data['num_samples_by_patient']), load, data['loq'] / 10)

        return util.merge_data(data, load=load, z=z, patient_mean=patient_mean)
