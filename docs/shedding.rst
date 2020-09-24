shedding python package
=======================

.. jinja::

   {% for module in modules %}
   {% set depth = module.split('.') | length %}
   {{module}}
   {{('-' if depth == 2 else '^') * (module | length)}}

   .. automodule:: {{module}}
      :members:
   {% endfor %}
