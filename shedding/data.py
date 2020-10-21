import collections
import json
import numpy as np
import os
from .util import dict_to_array


def load_dataset(name, root=None):
    """
    Load a dataset given its name.

    Parameters
    ----------
    name : str
        Name of the dataset to load.
    root : str
        Root directory to load the dataset from.

    Returns
    -------
    dataset : dict
        Loaded dataset as a dictionary.
    """
    root = root or os.getcwd()
    with open(os.path.join(root, name, name + '.json')) as fp:
        return json.load(fp)


def load_datasets(names, root=None):
    """
    Load multiple datasets into a dictionary.

    Parameters
    ----------
    names : list[str]
        Sequence of datasets to load.
    root : str
        Root directory to load datasets from.

    Returns
    -------
    datasets : dict
        Mapping of datasets keyed by name.
    """
    return {name: load_dataset(name, root) for name in names}


def flatten_datasets(datasets, loq_fill_value=np.nan, day_fill_value=np.nan):
    """
    Flatten datasets into a uniform structure for inference.

    Parameters
    ----------
    datasets : dict
        Mapping of datasets.
    loq_fill_value : float
        Fill value to replace loads below the level of quantification (LOQ) as log10 gene copies per
        mL.
    day_fill_value : float
        Fill value to replace missing information about days since symptoms began as days.

    Returns
    -------
    data : dict
        Data obtained by flattening datasets, including

        * **num_patients** (`int`): Total number of patients across all datasets.
        * **num_samples** (`ndarray[int]<num_patients>`): Total number of samples across all
          datasets.
        * **num_samples_by_patient** (`ndarray[int]<num_patients>`): Number of samples for each
          patient.
        * **num_positives_by_patient** (`ndarray[int]<num_patients>`): Number of positive samples
          for each patient.
        * **num_negatives_by_patient** (`ndarray[int]<num_patients>`): Number of negative samples
          for each patient.
        * **idx** (`ndarray[int]<num_samples>`): Index of the patient from whom each sample was
          collected. The indices are one-based for compatibility with `pystan`.
        * **load** (`ndarray[float]<num_samples>`): Viral RNA load for each sample as gene copies
          per mL or `10 ** loq_fill_value` if the concentration is below the level of
          quantification.
        * **loq** (`ndarray[float]<num_samples>`): Level of quantification for each sample as gene
          copies per mL.
        * **positive** (`ndarray[bool]<num_samples>`): Indicator for each sample as to whether the
          RNA load is above the level of quantification.
        * **day** (`ndarray[int]<num_samples>`): Day after symptom onset on which the sample was
          collected or `day_fill_value` if unavailable.
        * **dataset** (`ndarray[str]<num_samples>`): Dataset from which each sample was obtained.
        * **patient** (`ndarray[int]<num_samples>`): Patient from which each sample was obtained as
          reported in the corresponding dataset.
    """
    # Lookup for integer identifiers for each patient
    patient_lookup = {}
    data = {}

    num_samples_by_patient = collections.Counter()
    num_positives_by_patient = collections.Counter()
    num_negatives_by_patient = collections.Counter()
    for key, dataset in datasets.items():
        loq = 10 ** dataset['loq']
        for i, x in enumerate(dataset['loads']):
            # Flatten dataset-level attributes
            data.setdefault('loq', []).append(loq)
            data.setdefault('dataset', []).append(key)

            # Assign a consistent patient index across different samples
            patient = x['patient']
            i = patient_lookup.setdefault((key, patient), len(patient_lookup))
            data.setdefault('idx', []).append(i)
            data.setdefault('patient', []).append(patient)

            # Add the values
            if x['value'] is None or x['value'] < dataset['loq']:
                value = loq_fill_value
                num_negatives_by_patient[i] += 1
            else:
                value = x['value']
                num_positives_by_patient[i] += 1
            data.setdefault('load', []).append(10 ** value)
            data.setdefault('day', []).append(x.get('day', day_fill_value))

            num_samples_by_patient[i] += 1

    # Convert to numpy arrays and add some additional contextual information
    num_patients = len(patient_lookup)
    data = {key: np.asarray(value) for key, value in data.items()}
    data.update({
        'num_samples': len(data['load']),
        'num_patients': num_patients,
        'num_samples_by_patient': dict_to_array(num_samples_by_patient,
                                                size=num_patients, dtype=int),
        'num_positives_by_patient': dict_to_array(num_positives_by_patient,
                                                  size=num_patients, dtype=int),
        'num_negatives_by_patient': dict_to_array(num_negatives_by_patient,
                                                  size=num_patients, dtype=int),
        'positive': data['load'] > data['loq'],
    })

    return data
