from setuptools import setup, find_packages

setup(
    name='shedding',
    version='0.1',
    packages=find_packages(),
    author='Till Hoffmann',
    install_requires=[
        'numpy',
        'scipy',
        'tf-nightly',
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
    }
)
