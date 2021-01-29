.PHONY : clean docs doctests tests pypolychord

build : flake8 tests docs

flake8 :
	flake8

tests :
	pytest -v --cov=shedding --cov-report=html --cov-report=term-missing

doctests :
	sphinx-build -b doctest . docs/_build

docs : doctests
	sphinx-build . docs/_build

clean :
	rm -rf docs/_build PolyChordLite

# Generate pinned dependencies
requirements.txt : requirements.in setup.py
	pip-compile -v --upgrade

sync : requirements.txt
	pip-sync
	$(MAKE) pypolychord

# Build the repository using a GitHub action for local debugging
# (cf. https://github.com/nektos/act)
gh-action :
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
PARAMETRISATIONS = general

INFLATED = standard inflated
INFLATED_inflated = --inflated

TEMPORAL = constant temporal
TEMPORAL_temporal = --temporal

SEEDS = 0 1 2

# See the polychord publication for reference
NLIVE = 25
NREPEAT = 5

TARGET_DIRS = $(addprefix workspace/,\
	$(foreach s,${SEEDS}, \
	$(foreach p,${PARAMETRISATIONS},\
	$(foreach i,${INFLATED}, \
	$(foreach t,${TEMPORAL}, \
	$p-$i-$t-$s)))))


# Evidences using polychord
EVIDENCE_TARGETS = $(addsuffix /polychord/result.pkl,${TARGET_DIRS})

evidences: ${EVIDENCE_TARGETS}

$(EVIDENCE_TARGETS) : workspace/%/polychord/result.pkl : polychord-sampling.ipynb
	ARGS="--evidence --seed=$(call wordd,$*,4) ${TEMPORAL_$(call wordd,$*,3)} ${INFLATED_$(call wordd,$*,2)} -f --nlive-factor=${NLIVE} --nrepeat-factor=${NREPEAT} $(call wordd,$*,1) workspace/$*/polychord" \
		jupyter-nbconvert --execute --allow-errors --ExecuteProcessor.timeout=-1 \
		--output-dir=workspace/$* --to=html $<

# Additional samples for the constant parameters including Wang's data
EXTRA_TARGET_DIRS = $(addprefix workspace/,\
	$(foreach s,${SEEDS}, \
	$(foreach p,${PARAMETRISATIONS},\
	$(foreach i,${INFLATED}, \
	$p-$i-constant-$s))))

EXTRA_SAMPLE_TARGETS = $(addsuffix -extra/polychord/result.pkl,${EXTRA_TARGET_DIRS})

extra_samples : ${EXTRA_SAMPLE_TARGETS}

$(EXTRA_SAMPLE_TARGETS) : workspace/%/polychord/result.pkl : polychord-sampling.ipynb
	ARGS="--seed=$(call wordd,$*,4) ${INFLATED_$(call wordd,$*,2)} -f --nlive-factor=${NLIVE} --nrepeat-factor=${NREPEAT} $(call wordd,$*,1) workspace/$*/polychord" \
		jupyter-nbconvert --execute --allow-errors --ExecuteProcessor.timeout=-1 \
		--output-dir=workspace/$* --to=html $<

all : evidences extra_samples
