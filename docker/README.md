# Container image

One image with everything the pipeline needs: the converter, Salt, UPP, puma and
torch+CUDA. It is built on top of the upstream Salt image
(`gitlab-registry.cern.ch/aft/algorithms/salt`) and published to

```
ghcr.io/vincentriechers/fcc-ftag-salt-converter:latest
```

by `.github/workflows/docker.yml` on pushes to `main` (and on `v*` tags) that
touch `docker/Dockerfile`, `pyproject.toml`, `uv.lock` or `src/**` — nothing else
can change the image.

The base is mirrored from the CERN registry into
`ghcr.io/vincentriechers/salt-base` by `.github/workflows/mirror-base.yml`
(weekly, or on demand). Building from the same registry it pushes to makes the
pull faster and lets the unchanged base layers be blob-mounted instead of
re-uploaded.

## Run it

On a cluster with apptainer:

```bash
IMG=docker://ghcr.io/vincentriechers/fcc-ftag-salt-converter:latest

# conversion / merging / comparison — CPU
apptainer exec -B /home -B /tmp -B /srv $IMG \
    fcc-ftag-higgs-decay --help

# training — add --nv for the GPU
apptainer exec --nv -B /home -B /tmp -B /srv \
    -B "$FCC_FTAG_REPO":/repo -B "$FCC_FTAG_DATA":/data $IMG \
    salt fit --config /repo/configs/salt/gn3.yaml
```

### `/repo` and `/data`

**The image does not contain the repository.** It carries the installed package
and the environment only, so `configs/` and `scripts/` come from the bind mount
at runtime — which is what `scripts/env.sh` already sets up. Running the raw
image without `-B` gets you the environment, not the repo.

That is deliberate: the image is built from `pyproject.toml` and `src/` alone, so
editing a script or a config no longer produces a different image and no longer
triggers a 15-minute rebuild.

`scripts/env.sh` binds the checkout to `/repo` and the sample area to `/data`,
and the UPP and Salt configs are written against those paths. This is not
cosmetic: UPP resolves a relative `!include` against its *own* package config
directory, and neither UPP nor Salt expands environment variables in a config —
so the only way to have one set of configs work for every user is to make the
absolute paths identical inside the container.

The scripts under `scripts/` currently point at three separate CVMFS images. Each
of those can be pointed at this one instead:

| script variable | was | now |
|---|---|---|
| `IMAGE=` (`submit_upp.sh`) | `…/upp-images/upp:latest` | this image |
| `SALT=` / `SALT_IMG=` (`salt_submit.sh`, `salt_test_plot.sh`, `submit_clip_features.sbatch`, `submit_fix_tau_blocks.sbatch`) | `…/aft/algorithms/salt:latest` | this image |
| `PUMA_IMG=` (`salt_test_plot.sh`) | `…/puma-images/puma:latest` | this image |

`HDF5_PLUGIN_PATH` is set inside the image, so the
`export HDF5_PLUGIN_PATH=…/external/hdf5plugin/…` lines in the command scripts
are no longer needed.

## Build it locally

On Baobab, podman is already configured with a node-local graphroot, and rootless
netns creation does not work there, so use `--network=host`:

```bash
podman build --network=host -f docker/Dockerfile -t fcc-ftag-salt-converter .
```

To build against a pinned Salt release instead of the moving `latest`:

```bash
podman build --network=host -f docker/Dockerfile \
    --build-arg SALT_IMAGE=gitlab-registry.cern.ch/aft/algorithms/salt:0-13-0 \
    -t fcc-ftag-salt-converter .
```

The same `salt_image` value can be passed to the workflow via
`workflow_dispatch`.
