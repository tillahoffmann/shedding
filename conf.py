# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Custom directives -------------------------------------------------------
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from docutils import nodes
from glob import glob
import inspect
from jinja2 import Template
import os
import shedding
import sphinx
import yaml


def _parse_key(argument):
    return argument.split("/")


class JinjaDirective(Directive):
    """
    Custom directive for using Jinja templates in reStructuredText.
    """

    has_content = True
    optional_arguments = 1
    option_spec = {
        "file": directives.path,
        "header_char": directives.unchanged,
        "debug": directives.flag,
        "key": _parse_key,
    }

    def run(self):
        # Load the template
        template_filename = self.options.get("file")
        if template_filename:
            with open(template_filename) as fp:
                template = fp.read()
        else:
            template = "\n".join(self.content)

        # Render the template
        context = self.app.config.jinja_context
        keys = self.options.get("key", [])
        for key in keys:
            context = context[key]
        context["_jinja_key"] = keys
        template = Template(template)
        rst = template.render(
            **context, header_char=self.options.get("header_char", "=")
        )

        if "debug" in self.options:
            print(rst)

        # Parse the generated rst
        node = nodes.Element()
        rst = StringList(rst.splitlines())
        sphinx.util.nested_parse_with_titles(self.state, rst, node)
        return node.children


def setup(app):
    JinjaDirective.app = app
    app.add_directive("jinja", JinjaDirective)
    app.add_config_value("jinja_context", {}, "env")
    return {"parallel_read_safe": True, "parallel_write_safe": True}


# -- Datasets for jinja templates --------------------------------------------
filenames = glob("publications/*/*.yaml")
jinja_context = {}
for filename in sorted(filenames):
    with open(filename) as fp:
        key, _ = os.path.splitext(os.path.basename(filename))
        jinja_context.setdefault("publications", {})[key] = yaml.safe_load(fp)

# Get all modules for the documentation
modules = set()
queue = [shedding]
while queue:
    module = queue.pop(0)
    for _, module in inspect.getmembers(module):
        if inspect.ismodule(module) and module.__name__.startswith("shedding"):
            queue.append(module)
            modules.add(module.__name__)
jinja_context["modules"] = list(sorted(modules))

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "shedding"
copyright = "2020, Till Hoffmann"
author = "Till Hoffmann"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "linuxdoc.rstFlatTable",  # For fancy tables
    "matplotlib.sphinxext.plot_directive",  # For plots
    "sphinx.ext.autodoc",  # To generate documentation
    "sphinx.ext.napoleon",  # For docstring parsing
    "sphinx.ext.doctest",  # For testing in docstrings
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    ".eggs",
    "build",
    "PolyChordLite",
    "README.rst",
    "venv",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "nature"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# -- Options for matplotlib --------------------------------------------------

plot_formats = [
    ("png", 144),
]

doctest_global_setup = "from shedding import *"
