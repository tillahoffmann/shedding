from setuptools import setup, find_packages, Extension


# Try to build from source if cython is installed, use preprocessed C file otherwise.
try:
    from Cython.Build import cythonize
    ext = cythonize('shedding/_util.pyx', language_level=3)
except ImportError:
    assert False
    ext = [Extension('_util', ['shedding/_util.c'])]


class numpy_include(object):
    """
    Lazily evaluate numpy include path because numpy may not be installed.
    """
    def __str__(self):
        import numpy
        return numpy.get_include()


setup(
    name='shedding',
    version='0.1',
    packages=find_packages(),
    author='Till Hoffmann',
    install_requires=[
        'numpy',
        'scipy',
    ],
    setup_requires=[
        'cython',
    ],
    extras_require={
        'tests': [
            'flake8',
            'jsonschema',
            'pytest',
            'pytest-cov',
        ],
        'docs': [
            'matplotlib',
            'sphinx',
            'linuxdoc @ git+http://github.com/return42/linuxdoc.git',
            'jinja2',
        ]
    },
    ext_modules=ext,
    include_dirs=[
        numpy_include(),
    ],
)
