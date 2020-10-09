import numpy as np
import pystan
import pytest
from scipy import stats
import shedding


@pytest.fixture(params=[shedding.GammaModel, shedding.GammaInflatedModel, shedding.LognormalModel,
                        shedding.LognormalInflatedModel, shedding.WeibullModel,
                        shedding.WeibullInflatedModel])
def model(request):
    return request.param()


def test_model_code(model):
    pystan.api.stanc(model_code=model.model_code)


@pytest.fixture
def hyperparameters(model):
    if isinstance(model, shedding.LognormalModel):
        params = {'patient_scale': 1, 'population_scale': 1, 'population_loc': 1}
    elif isinstance(model, shedding.GammaModel):
        params = {'patient_shape': 1, 'population_scale': 1e-5, 'population_shape': 1}
    elif isinstance(model, shedding.WeibullModel):
        params = {'patient_shape': 1, 'population_scale': 1e5, 'population_shape': 1}
    else:
        raise ValueError(model)
    if isinstance(model, shedding.InflationMixin):
        params['rho'] = 0.9
    return params


@pytest.fixture
def replicate1(model, hyperparameters):
    # Construct enough context to generate some data from scratch
    num_patients = 5
    num_samples_by_patient = 7
    data = {
        'num_samples': num_samples_by_patient * num_patients,
        'num_patients': num_patients,
        'num_samples_by_patient': num_samples_by_patient * np.ones(num_patients, int),
        'loq': 100,
        'idx': np.repeat(1 + np.arange(num_patients), num_samples_by_patient)
    }
    return model.replicate(hyperparameters, data, mode=shedding.ReplicationMode.NEW_GROUPS)


@pytest.fixture
def replicate2(model, hyperparameters, replicate1):
    # Transfer the patient mean to the hyperparameters to be able to replicate existing groups
    hyperparameters['patient_mean'] = replicate1['patient_mean']
    hyperparameters['patient_loc'] = np.log(replicate1['patient_mean'])
    # Generate some synthetic patient contributions we would've got from fitting
    if isinstance(model, shedding.InflationMixin):
        hyperparameters['patient_contrib_'] = -np.random.gamma(1, size=replicate1['num_patients'])
    # Then use those data to replicate within groups
    return model.replicate(hyperparameters, replicate1,
                           mode=shedding.ReplicationMode.EXISTING_GROUPS)


def test_observed_likelihood(model, hyperparameters, replicate2):
    model.evaluate_observed_likelihood(hyperparameters, replicate2)


def test_evaluate_mean(model, hyperparameters):
    assert model.evaluate_statistic(hyperparameters, 'mean') > 0


def _test_dist_pdf_cdf(dist, lpdf, lcdf, *args, **kwargs):
    x = dist.rvs(10)
    np.testing.assert_allclose(dist.logpdf(x), lpdf(x, *args, **kwargs))
    np.testing.assert_allclose(dist.logcdf(x), lcdf(x, *args, **kwargs))


def test_gamma_pdf_cdf():
    shape = 3
    scale = 4.5
    dist = stats.gamma(shape, scale=1 / scale)
    _test_dist_pdf_cdf(dist, shedding.gamma_lpdf, shedding.gamma_lcdf, shape, scale)


def test_lognormal_pdf_cdf():
    mu = 1
    scale = 2
    dist = stats.lognorm(scale, scale=np.exp(mu))
    _test_dist_pdf_cdf(dist, shedding.lognormal_lpdf, shedding.lognormal_lcdf, mu, scale)


def test_weibull_pdf_cdf():
    shape = 2
    scale = 5
    dist = stats.weibull_min(shape, scale=scale)
    _test_dist_pdf_cdf(dist, shedding.weibull_lpdf, shedding.weibull_lcdf, shape, scale)


def test_rvs(model, hyperparameters):
    sample = model.rvs(hyperparameters, 100)
    assert np.shape(sample) == (100,)
    if 'Inflated' in model.__class__.__name__:
        sample = sample[np.isfinite(sample)]
    np.testing.assert_array_less(0, sample)
