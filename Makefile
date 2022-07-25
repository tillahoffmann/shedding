.PHONY : clean docs doctests tests pypolychord inference_test clean-results

JUPYTER_CMD = MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 jupyter-nbconvert --execute --ExecuteProcessor.timeout=-1 --to=html

build : flake8 tests docs

flake8 :
	flake8

tests :
	pytest tests -v --cov=shedding --cov-report=html --cov-report=term-missing

doctests :
	sphinx-build -b doctest . docs/_build

docs : doctests
	sphinx-build . docs/_build

clean :
	rm -rf docs/_build PolyChordLite

clean-results:
	rm -rf workspace figures results.html

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
	git clone --depth 1 --branch 1.18.1 https://github.com/PolyChord/PolyChordLite.git

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

SEEDS ?= 0 1 2

# See the polychord publication for reference
NLIVE ?= 25
NREPEAT ?= 5

TARGET_DIRS = $(addprefix workspace/,\
	$(foreach s,${SEEDS}, \
	$(foreach p,${PARAMETRISATIONS},\
	$(foreach i,${INFLATED}, \
	$(foreach t,${TEMPORAL}, \
	$p-$i-$t-$s)))))


# Evidences using polychord
EVIDENCE_TARGETS = $(addsuffix /polychord/result.pkl,${TARGET_DIRS})

evidences: ${EVIDENCE_TARGETS}

$(EVIDENCE_TARGETS) : workspace/%/polychord/result.pkl :
	ARGS="--evidence --seed=$(call wordd,$*,4) ${TEMPORAL_$(call wordd,$*,3)} ${INFLATED_$(call wordd,$*,2)} -f --nlive-factor=${NLIVE} --nrepeat-factor=${NREPEAT} $(call wordd,$*,1) workspace/$*/polychord" \
		${JUPYTER_CMD} --output-dir=workspace/$* polychord-sampling.ipynb
	rm -rf workspace/$*/polychord/clusters

# Additional samples for the constant parameters including Wang's data
EXTRA_TARGET_DIRS = $(addprefix workspace/,\
	$(foreach s,${SEEDS}, \
	$(foreach p,${PARAMETRISATIONS},\
	$(foreach i,${INFLATED}, \
	$p-$i-constant-$s))))

EXTRA_SAMPLE_TARGETS = $(addsuffix -extra/polychord/result.pkl,${EXTRA_TARGET_DIRS})

extra_samples : ${EXTRA_SAMPLE_TARGETS}

$(EXTRA_SAMPLE_TARGETS) : workspace/%/polychord/result.pkl :
	ARGS="--seed=$(call wordd,$*,4) ${INFLATED_$(call wordd,$*,2)} -f --nlive-factor=${NLIVE} --nrepeat-factor=${NREPEAT} $(call wordd,$*,1) workspace/$*/polychord" \
		${JUPYTER_CMD} --output-dir=workspace/$* polychord-sampling.ipynb
	rm -rf workspace/$*/polychord/clusters

all : evidences extra_samples

# Targets for assessing sensitvity to errors in days past symptom onset (don't need as many points
# because we're not trying to evaluate evidences)
SENSITIVITY_TARGET_DIRS = $(foreach s,${SEEDS},$(foreach d,1 2 3,workspace/sensitivity-$d-$s))
SENSITIVITY_TARGETS = $(addsuffix /result.pkl,${SENSITIVITY_TARGET_DIRS})

sensitivity : ${SENSITIVITY_TARGETS}

${SENSITIVITY_TARGETS} : workspace/sensitivity-%/result.pkl :
	ARGS="-f --nlive-factor=${NLIVE} --nrepeat-factor=${NREPEAT} --day-noise=$(call wordd,$*,1) --seed=$(call wordd,$*,2) --temporal exponential general workspace/sensitivity-$*" \
		${JUPYTER_CMD} --output-dir=workspace/sensitivity-$* polychord-sampling.ipynb
	rm -rf workspace/sensitivity-$*/polychord/clusters


# Targets for investigating different shedding profiles
PROFILE_TARGET_DIRS = $(foreach s,${SEEDS},$(foreach t,teunis gamma,workspace/profile-$t-$s))
PROFILE_TARGETS = $(addsuffix /result.pkl,${PROFILE_TARGET_DIRS})

profiles : ${PROFILE_TARGETS}

${PROFILE_TARGETS} : workspace/profile-%/result.pkl :
	ARGS="-f --nlive-factor=${NLIVE} --nrepeat-factor=${NREPEAT} --temporal=$(call wordd,$*,1) --seed=$(call wordd,$*,2) general workspace/profile-$*" \
		${JUPYTER_CMD} --output-dir=workspace/profile-$* polychord-sampling.ipynb
	rm -rf workspace/profile-$*/polychord/clusters

inference_test : polychord-sampling.ipynb pypolychord
	mkdir -p $@
	ARGS="-f --nlive-factor=0.1 --nrepeat-factor=0.1 --temporal=exponential --seed=0 general $@" \
		jupyter-nbconvert --execute --ExecuteProcessor.timeout=-1 --output-dir $@ --to=html $<

FIGURES = model decay positivity-replicates prediction profiles replication shape-scale

workspace/results.html $(addprefix workspace/figures/,${FIGURES:=.pdf}) : results.ipynb evidences extra_samples sensitivity profiles
	mkdir -p workspace/figures
	${JUPYTER_CMD} $< --output-dir=workspace

PLATFORM =

image :
	docker build ${PLATFORM} -t shedding .

container :
	mkdir -p workspace
	docker run --rm -it ${PLATFORM} -v `pwd`/workspace:/workspace shedding bash
