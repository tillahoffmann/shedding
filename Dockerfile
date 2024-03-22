FROM python:3.10.13

# Install build tools.
RUN apt-get update && apt-get install -y \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and cython source because it is needed at install time.
COPY Makefile requirements.txt pyproject.toml setup.py ./
COPY shedding/_util.pyx shedding/__init__.py shedding/

# Install python requirements without cache directory (because we'll have an empty cache every time
# we build this layer anyway).
RUN pip install --no-cache-dir -r requirements.txt

# Install pypolychord.
RUN make pypolychord

# Copy over everything else.
COPY . .
