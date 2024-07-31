from glob import glob
import jsonschema
import os
import pytest
import shedding
import yaml


@pytest.fixture(params=glob("publications/*/*.yaml"))
def filename(request):
    return request.param


@pytest.fixture(scope="module")
def schema():
    with open("schema.yaml") as fp:
        return yaml.safe_load(fp)


def test_dataset(filename, schema):
    with open(filename) as fp:
        dataset = yaml.safe_load(fp)

    # Validate against the schema
    try:
        jsonschema.validate(dataset, schema)
    except jsonschema.exceptions.ValidationError as ex:
        raise ValueError(ex.message)

    # Make sure everything adds up
    for key in ["patients", "samples"]:
        summary = dataset.get(key)
        if not summary:
            continue
        if any(x not in summary for x in ["n", "negative", "positive"]):
            continue
        assert summary["n"] == summary["negative"] + summary["positive"]

    # If there are viral loads and patient details, make sure that any patient
    # reference is fine.
    patient_details = dataset.get("patients", {}).get("details")
    if patient_details:
        loads = dataset.get("loads", [])
        for load in loads:
            patient = load.get("patient")
            assert patient < len(patient_details)

    # Check if any samples report temporal information
    if any("day" in load for load in dataset.get("loads", [])):
        assert "temporal" in dataset


def test_docs(filename):
    basename, _ = os.path.splitext(filename)
    assert os.path.isfile(basename + ".rst"), f"missing documentation for {filename}"


def test_flatten_datasets():
    woelfel = shedding.load_dataset("Woelfel2020", "publications")
    data = shedding.flatten_datasets({"key": woelfel})
    assert data["num_patients"] == woelfel["patients"]["n"]
    assert data["num_samples"] == woelfel["samples"]["n"]

    # Make sure the number of positives and negatives of the second patient are right
    i = 1
    assert data["num_samples_by_patient"][i] == 11
    assert data["num_positives_by_patient"][i] == 10
    assert data["num_negatives_by_patient"][i] == 1

    # Consistency check (with one more negative because it's below the level of
    # quantification).
    assert data["num_samples_by_patient"].sum() == woelfel["samples"]["n"]
    assert data["num_positives_by_patient"].sum() == woelfel["samples"]["positive"] - 1
    assert data["num_negatives_by_patient"].sum() == woelfel["samples"]["negative"] + 1


def test_load_datasets():
    datasets = shedding.load_datasets(["Woelfel2020", "Han2020"], "publications")
    assert "Woelfel2020" in datasets
    assert "Han2020" in datasets
