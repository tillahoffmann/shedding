.PHONY : clean docs doctests tests pypolychord

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
	rm -rf docs/_build PolyChordLite

# Generate pinned dependencies
requirements.txt : requirements.in setup.py
	pip-compile -v --upgrade
	pip-sync

# Build the repository using a GitHub action for local debugging
# (cf. https://github.com/nektos/act)
build_action :
	act -P ubuntu-latest=nektos/act-environments-ubuntu:18.04

PolyChordLite :
	git clone --depth 1 --branch 1.18.1 git@github.com:PolyChord/PolyChordLite.git

pypolychord : PolyChordLite
	cd PolyChordLite \
		&& make MPI=0 libchord.so \
		&& python setup.py --no-mpi install

# Function to get parts of a string split by a dash
wordd = $(word $2,$(subst -, ,$1))

# Code to generate samples
PARAMETRISATIONS = general gamma weibull lognormal
INFLATED = 0 1
INFLATED_1 = --inflated
SEEDS = 0 1 2 3
TARGET_DIRS = $(addprefix workspace/,\
	$(foreach p,${PARAMETRISATIONS},\
	$(foreach i,${INFLATED}, \
	$(foreach s,${SEEDS},$p-$i-$s))))
TARGETS = $(addsuffix /chain.txt,${TARGET_DIRS})

samples: ${TARGETS}

$(TARGETS) : workspace/%/chain.txt : polychord-sampling.ipynb
	ARGS="--seed=$(call wordd,$*,3) ${INFLATED_$(call wordd,$*,2)} --nlive-factor=10 --nrepeat-factor=3 $(call wordd,$*,1) workspace/$*" \
		jupyter-nbconvert --execute --allow-errors --ExecuteProcessor.timeout=-1 \
		--output-dir=workspace/$* --to=html $<
