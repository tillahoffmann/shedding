shedding python package
=======================

.. jinja::

   {% set modules = ['data', 'util'] %}
   {% for module in modules %}
   shedding.{{module}}
   ---------{{'-' * (module | length)}}

   .. automodule:: shedding.{{module}}
      :members:
   {% endfor %}
