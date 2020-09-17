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
      {% endfor -%}

.. plot:: plot_datasets.py

   *Overview of viral RNA load data available in different publications.* Black vertical dotted lines represent the level of quantification or threshold for a sample to be considered positive. Arrows represent limits of viral RNA load reported in some studies. Squares and triangles represent the mean and median, respectively.

🤝 Contributing
---------------

Contributions in the form of new datasets or corrections are most welcome in the form of pull requests from forks. See `here <https://guides.github.com/activities/forking/>`__ for more details on how to contribute.

.. toctree::
   :glob:
   :hidden:

   publications/*/*
