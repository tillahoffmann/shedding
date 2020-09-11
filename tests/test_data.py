from glob import glob
import json
import jsonschema
import os
import pytest


@pytest.fixture(params=glob('publications/*/*.json'))
def filename(request):
    return request.param


@pytest.fixture(scope='module')
def schema():
    with open('schema.json') as fp:
        return json.load(fp)


def test_dataset(filename, schema):
    with open(filename) as fp:
        dataset = json.load(fp)

    # Validate against the schema
    try:
        jsonschema.validate(dataset, schema)
    except jsonschema.exceptions.ValidationError as ex:
        raise ValueError(ex.message)

    # Make sure everything adds up
    for key in ['patients', 'samples']:
        summary = dataset.get(key)
        if not summary:
            continue
        if any(x not in summary for x in ['n', 'negative', 'positive']):
            continue
        assert summary['n'] == summary['negative'] + summary['positive']

    # If there are viral loads and patient details, make sure that any patient reference is fine
    patient_details = dataset.get('patients', {}).get('details')
    if patient_details:
        loads = dataset.get('loads', [])
        for load in loads:
            patient = load.get('patient')
            assert patient < len(patient_details)


def test_docs(filename):
    basename, _ = os.path.splitext(filename)
    assert os.path.isfile(basename + '.rst'), f'missing documentation for {filename}'
