FROM python:3.8.10

# Install build tools.
RUN apt-get update && apt-get install -y \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and cython source because it is needed at install time.
COPY requirements.txt setup.py ./
COPY shedding/_util.pyx shedding/__init__.py shedding/

# Install python requirements without cache directory (because we'll have an empty cache every time
# we build this layer anyway). We pre-install cython, cyfunc, and scipy to ensure everything is
# properly linked. We need to also rebuild cyfunc for this to work--not sure why.
RUN pip install --no-cache-dir `cat requirements.txt | grep -E '(cython|cyfunc|scipy)=='` \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --force `cat requirements.txt | grep cyfunc==`

# Install pypolychord (this is duplicated from the Makefile to avoid rebuilding the image every time
# we update Makefile targets). This needs to happen after python dependencies because building
# requires numpy.
RUN git clone --depth 1 --branch 1.18.1 https://github.com/PolyChord/PolyChordLite.git \
    && cd PolyChordLite \
    && make MPI=0 libchord.so \
    && python setup.py --no-mpi install

# Copy over everything else.
COPY . .
