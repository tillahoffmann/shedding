.PHONY : clean docs doctests tests

build : flake8 tests docs

flake8 : requirements.txt
	flake8

tests : requirements.txt
	pytest -v --cov=shedding --cov-report=html --cov-report=term-missing

doctests : requirements.txt
	sphinx-build -b doctest . docs/_build

docs : doctests requirements.txt
	sphinx-build . docs/_build

clean :
	rm -rf docs/_build

# Generate pinned dependencies
requirements.txt : requirements.in setup.py
	pip-compile -v --upgrade
	pip-sync

# Build the repository using a GitHub action for local debugging
# (cf. https://github.com/nektos/act)
build_action :
	act -P ubuntu-latest=nektos/act-environments-ubuntu:18.04
