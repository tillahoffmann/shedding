.PHONY : clean docs doctests tests pypolychord inference_test

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
TEMPORAL_temporal = --temporal=exponential

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

# Targets for assessing sensitvity to errors in days past symptom onset (don't need as many points
# because we're not trying to evaluate evidences)
SENSITIVITY_TARGET_DIRS = $(foreach s,${SEEDS},$(foreach d,1 2 3,workspace/sensitivity-$d-$s))
SENSITIVITY_TARGETS = $(addsuffix /result.pkl,${SENSITIVITY_TARGET_DIRS})

sensitivity : ${SENSITIVITY_TARGETS}

${SENSITIVITY_TARGETS} : workspace/sensitivity-%/result.pkl : polychord-sampling.ipynb
	ARGS="-f --nlive-factor=5 --nrepeat-factor=2 --day-noise=$(call wordd,$*,1) --seed=$(call wordd,$*,2) --temporal general workspace/sensitivity-$*" \
		jupyter-nbconvert --execute --allow-errors --ExecuteProcessor.timeout=-1 \
		--output-dir=workspace/sensitivity-$* --to=html $<


# Targets for investigating different shedding profiles
PROFILE_TARGET_DIRS = $(foreach s,${SEEDS},$(foreach t,teunis gamma,workspace/profile-$t-$s))
PROFILE_TARGETS = $(addsuffix /result.pkl,${PROFILE_TARGET_DIRS})

profiles : ${PROFILE_TARGETS}

${PROFILE_TARGETS} : workspace/profile-%/result.pkl : polychord-sampling.ipynb
	ARGS="-f --nlive-factor=25 --nrepeat-factor=5 --temporal=$(call wordd,$*,1) --seed=$(call wordd,$*,2) general workspace/profile-$*" \
		jupyter-nbconvert --execute --allow-errors --ExecuteProcessor.timeout=-1 \
		--output-dir=workspace/profile-$* --to=html $<

inference_test : polychord-sampling.ipynb
	mkdir -p $@
	ARGS="-f --nlive-factor=1 --nrepeat-factor=1 --temporal=exponential --seed=0 general $@" \
		jupyter-nbconvert --execute --ExecuteProcessor.timeout=-1 --output-dir $@ --to=html $<
