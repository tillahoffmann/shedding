import shedding


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
