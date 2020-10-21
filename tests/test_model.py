import itertools as it
import numpy as np
import pytest
from scipy import stats
import shedding


@pytest.fixture(params=[
    (1, .2, 1),
    (0, .3, 2),
    (2, .4, 1),
    np.random.gamma(1, size=3),
])
def qms(request):
    return request.param


@pytest.fixture
def dist(qms):
    q, mu, sigma = qms
    if q == 0:
        return stats.lognorm(sigma, scale=np.exp(mu))
    a, b, c = shedding.to_abc(*qms)
    return stats.gengamma(a, c, scale=b ** (- 1 / c))


@pytest.fixture
def x(dist):
    return dist.rvs(size=100)


def test_gengamma_lpdf(qms, dist, x):
    np.testing.assert_allclose(shedding.gengamma_lpdf(*qms, x), dist.logpdf(x))


def test_gengamma_lcdf(qms, dist, x):
    np.testing.assert_allclose(shedding.gengamma_lcdf(*qms, x), dist.logcdf(x))


def test_gengamma_mean(qms, dist):
    np.testing.assert_allclose(shedding.gengamma_mean(*qms), dist.mean())


def test_gengamma_loc(qms):
    q, mu, sigma = qms
    mean = shedding.gengamma_mean(*qms)
    loc = shedding.gengamma_loc(q, sigma, mean)
    np.testing.assert_allclose(loc, mu)


def test_vector_values_roundtrip():
    parameters = {
        'x': (),
        'y': (10,),
        'z': (3, 4),
    }
    values = {key: np.random.normal(size=shape) for key, shape in parameters.items()}
    vector = shedding.values_to_vector(parameters, values)
    assert vector.size == 23
    reconstructed = shedding.vector_to_values(parameters, vector)
    [np.testing.assert_allclose(value, values[key]) for key, value in reconstructed.items()]


def test_transpose_samples_roundtrip():
    n = 100
    parameters = {
        'x': (n,),
        'y': (n, 10,),
        'z': (n, 3, 4),
    }
    values = {key: np.random.normal(size=shape) for key, shape in parameters.items()}
    reconstructed = shedding.transpose_samples(shedding.transpose_samples(values))
    [np.testing.assert_allclose(value, values[key]) for key, value in reconstructed.items()]


def test_gengamma_lpdf_composite():
    qms = (np.arange(2), .5, 1.2)
    x = np.random.gamma(1, size=(10, 1))
    lpdf = shedding.gengamma_lpdf(*qms, x)
    assert np.all(np.isfinite(lpdf))


@pytest.fixture(params=it.product(shedding.Parametrisation, [False, True]))
def model(request):
    return shedding.Model(10, *request.param)


@pytest.fixture
def data(model):
    num_samples_by_patient = 2
    n = model.num_patients * num_samples_by_patient
    loq = np.random.gamma(1, size=n)
    load = np.random.gamma(1, size=n)
    positive = load > loq
    idx = np.repeat(np.arange(model.num_patients), num_samples_by_patient)
    return {
        'num_patients': model.num_patients,
        'num_samples_by_patient': num_samples_by_patient,
        'loq': loq,
        'load': load,
        'idx': idx,
        'positive': positive,
        'num_positives_by_patient': np.bincount(idx, positive)
    }


def test_model_from_vector(model, data):
    x = np.random.uniform(0, 1, model.size)
    y = model.sample_params_from_vector(x)
    model.evaluate_log_likelihood_from_vector(y, data)


@pytest.fixture
def hyperparameters(model):
    if model.parametrisation == shedding.Parametrisation.LOGNORMAL:
        params = {'patient_scale': 1, 'population_scale': 1, 'population_loc': 1,
                  'patient_shape': 0, 'population_shape': 0}
    elif model.parametrisation == shedding.Parametrisation.GAMMA:
        params = {'patient_scale': 2, 'population_scale': 3, 'population_loc': 1,
                  'patient_shape': 2, 'population_shape': 3}
    elif model.parametrisation == shedding.Parametrisation.WEIBULL:
        params = {'patient_scale': 1, 'population_scale': 1, 'population_loc': 1,
                  'patient_shape': 1, 'population_shape': 1}
    elif model.parametrisation == shedding.Parametrisation.GENERAL:
        params = {'patient_scale': 1, 'population_scale': 1, 'population_loc': 1,
                  'patient_shape': 2, 'population_shape': 3}
    else:
        raise ValueError(model)
    if model.inflated:
        params['rho'] = 0.9
    return params


def test_simulate(model, hyperparameters, data):
    # Keep a copy of the hyperparameters
    values = dict(hyperparameters)
    data = model.simulate(values, data, 'new_patients')
    # Replicate data
    replicate_values = dict(values)
    replicate = model.simulate(replicate_values, data, 'existing_patients')
    np.testing.assert_allclose(values['patient_mean'], replicate_values['patient_mean'])

    if model.inflated:
        # Ensure that anyone with a negative indicator does not shed
        for x, d in zip([values, replicate_values], [data, replicate]):
            z = np.repeat(x['z'], d['num_samples_by_patient'])
            np.testing.assert_array_equal(d['positive'][~z], False)
        # Ensure that anyone who shed in the original data remains a shedder
        fltr = data['num_positives_by_patient'] > 0
        np.testing.assert_array_equal(replicate_values['z'][fltr], True)


def test_marginal_log_likelihood(model, hyperparameters, data):
    marginal = model.evaluate_marginal_log_likelihood(hyperparameters, data)
    assert marginal.shape == (data['num_patients'],)


def test_evaluate_mean(model, hyperparameters):
    assert model.evaluate_statistic(hyperparameters, 'mean') > 0


def test_rvs(model, hyperparameters):
    sample = model.rvs(hyperparameters, 100)
    assert np.shape(sample) == (100,)
    if model.inflated:
        sample = sample[np.isfinite(sample)]
    np.testing.assert_array_less(0, sample)
