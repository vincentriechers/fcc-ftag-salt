# One image for the whole FCC-ee flavour-tagging chain.
#
# The upstream Salt image already carries salt, umami-preprocessing, puma,
# atlas-ftag-tools, torch+CUDA, h5py, hdf5plugin and PyYAML, so this Dockerfile
# only adds what it is missing (uproot / awkward / fsspec-xrootd) plus this
# package. That replaces the three separate images the scripts/ used to juggle
# (salt, upp-images/upp, puma-images/puma).
#
# The default base is the CERN registry, so a local `podman build` works with no
# credentials. CI overrides SALT_IMAGE with the GHCR mirror
# (.github/workflows/mirror-base.yml), which pulls faster from a runner and lets
# the push blob-mount the unchanged base layers instead of re-uploading them.
ARG SALT_IMAGE=gitlab-registry.cern.ch/aft/algorithms/salt:latest
FROM ${SALT_IMAGE}

ENV PYTHONNOUSERSITE=1
ENV MPLBACKEND=Agg
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PIP_ROOT_USER_ACTION=ignore
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_NO_CACHE=1

# hdf5plugin ships in the base image but HDF5_PLUGIN_PATH is not exported, so
# h5py there cannot read UPP's blosc-compressed output. Setting it here is what
# retires the hand-rolled external/hdf5plugin install in scripts/.
ENV HDF5_PLUGIN_PATH=/usr/local/lib/python3.12/dist-packages/hdf5plugin/plugins

WORKDIR /opt/fcc-ftag-salt-converter

# puma imports tkinter unconditionally (puma/plot_base.py), and the base image
# has no tkinter - which is why the plotting stage needed its own image.
RUN apt-get update && apt-get install -y --no-install-recommends python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer: cached independently of the source tree.
RUN uv pip install --system \
    "uproot>=5" \
    "awkward>=2" \
    "fsspec-xrootd>=0.5"

# Only what building the wheel needs, so the image changes when the package
# changes and not when a script, config or README does. --no-deps so this
# package's loose floors (numpy>=1.24, h5py>=3.10) can never resolve over the
# versions Salt pins.
COPY pyproject.toml README.md ./
COPY src/ src/
RUN uv pip install --system . --no-deps

# Bind targets used by scripts/env.sh: the checkout goes to /repo and the sample
# area to /data. UPP and Salt resolve neither relative nor $-expanded paths in
# their configs, so those configs reference /repo and /data - fixed inside the
# container, and therefore the same for every checkout and every site. The
# checkout is NOT baked into the image; every stage bind-mounts it.
RUN mkdir -p /repo /data /work

WORKDIR /work
CMD ["/bin/bash"]
