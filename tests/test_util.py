import numpy as np
import shedding
import pytest


def test_extract_kvps_list():
    x = [
        {'a_': 1, 'b': 2},
        {'a': 7, 'c_': 4},
    ]
    y = shedding.extract_kvps(x, '(.*?)_')
    assert y == [
        {'a': 1},
        {'c': 4},
    ]


@pytest.mark.parametrize('shape, axis', [
    ((3, 4), None),
    ((3, 4), 0),
    ((5, 7), 1),
    ((3, 4, 5), (1, 2))
])
def test_logmeanexp(shape, axis):
    x = np.random.normal(0, 1, shape)
    actual = shedding.logmeanexp(x, axis)
    desired = np.log(np.mean(np.exp(x), axis))
    np.testing.assert_allclose(actual, desired)
