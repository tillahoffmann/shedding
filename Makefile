.PHONY : docs tests clean

build : flake8 tests docs

flake8 :
	flake8

tests :
	pytest -v

docs :
	sphinx-build . docs/_build

clean :
	rm -rf docs/_build

requirements.txt : requirements.in
	pip-compile -v
	pip-sync

build_action :
    # Build the repository using a GitHub action for local debugging
	# (cf. https://github.com/nektos/act)
	act -P ubuntu-latest=nektos/act-environments-ubuntu:18.04
