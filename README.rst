💩 Shedding of SARS-CoV-2 RNA in faeces
=======================================

.. image:: https://github.com/tillahoffmann/shedding/workflows/CI/badge.svg
  :target: https://github.com/tillahoffmann/shedding/actions?query=workflow%3A%22CI%22

.. image:: https://readthedocs.org/projects/shedding/badge/?version=latest
  :target: https://shedding.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/github/stars/tillahoffmann/shedding?style=social
  :target: https://github.com/tillahoffmann/shedding

Uncertainties surrounding the concentration of viral RNA fragments in the faeces of individuals infected with SARS-CoV-2 poses a major challenge for wastewater-based surveillance of Covid-19. This repository serves to collate data from quantitative studies on RNA load in faecal samples to constrain the shedding distribution.

📊 Datasets
-----------

Datasets can be found in the ``publications`` directory as JSON files following a common schema (see ``schema.json`` for details). All RNA loads are reported as log10 gene copies per mL. The results from individual samples are linked to patients wherever possible to provide longitudional information.

.. jinja::

   .. flat-table::
      :header-rows: 2
      :stub-columns: 1

      * - :rspan:`1` Key
        - :rspan:`1` Assay
        - :rspan:`1` LOQ
        - :cspan:`2` Patients
        - :cspan:`2` Samples
        - Detailed data
        - Temporal data
      * - n
        - \+
        - \-
        - n
        - \+
        - \-
      {% for key, pub in publications.items() -%}
      {% set patients = pub.get('patients', {}) %}
      {% set samples = pub.get('samples', {}) %}
      {% set assay = pub['assay'] %}
      * - :ref:`{{pub['key']}} <{{pub['key'].replace(' ', '_')}}>`
        - {{', '.join(assay) if assay is iterable and assay is not string else assay}}
        - {{('%.2f' | format(pub['loq'])) if pub['loq'] else '?'}}
        - {{patients.get('n', '?')}}
        - {{patients.get('positive', '?')}}{% if 'positive' in patients and 'n' in patients %} ({{'%.1f' | format(100 * patients['positive'] / patients['n'])}}%){% endif %}
        - {{patients.get('negative', '?')}}{% if 'negative' in patients and 'n' in patients %} ({{'%.1f' | format(100 * patients['negative'] / patients['n'])}}%){% endif %}
        - {{samples.get('n', '?')}}
        - {{samples.get('positive', '?')}}
        - {{samples.get('negative', '?')}}
        - {{ '\\+' if 'loads' in pub else '\\-' }}
        - {{ pub.get('temporal', '\\-') }}
      {% endfor -%}

.. plot:: plot_datasets.py

   *Overview of viral RNA load data available in different publications.* Black vertical dotted lines represent the level of quantification or threshold for a sample to be considered positive. Arrows represent limits of viral RNA load reported in some studies. Squares and triangles represent the mean and median, respectively.

🤝 Contributing
---------------

Contributions in the form of new datasets or corrections are most welcome in the form of pull requests from forks. See `here <https://guides.github.com/activities/forking/>`__ for more details on how to contribute.

🧪 Reproducing results
----------------------

Results can be reproduced by following these steps:

* Make sure you have python 3.8 or newer installed.
* Install the python dependencies by running :code:`pip install -r requirements.txt` (ideally in a dedicated virtual environment).
* Install the polychord sampler by running :code:`make pypolychord`.
* Reproduce the figures (in the :code:`figures` directory) and results (in the :code:`results.html` file) by running :code:`make results.html`.

.. note::

   Reproducing the results will take a considerable amount of time (several hours if you have a fast machine, days if you have a slow machine). You have two options to speed up the process (and they can be combined).

   1. Use :code:`make -j [number of processors you use] results.html` which will distribute the workload across the given number of processors.
   2. Use :code:`SEEDS=0 NLIVE=1 NREPEAT=1 make results.html` which will only run the inference for one random number generator seed and use fewer points for the nested sampling (giving rise to less reliably but faster results).

If you are not able to reproduce the results using the steps above, try running :code:`make tests` which may help identify the problem. Otherwise, please `raise a new issue <https://github.com/tillahoffmann/shedding/issues>`__.

📋 Contents
-----------

.. toctree::
   :glob:

   docs/shedding
   publications/*/*
