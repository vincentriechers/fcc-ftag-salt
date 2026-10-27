# FCC f-tag Salt Converter

End-to-end pipeline for training an FCC-ee jet flavour tagger in the ATLAS
[Salt](https://github.com/umami-hep/salt) framework: converts FCC-ee ROOT
ntuples into Salt-shaped HDF5 and drives preprocessing, training and
evaluation. Also ships a streaming EDM4hep→Salt converter and a generic
two-file H5 comparison tool.

Designed for million-event runs: events are read in chunks via
`uproot.iterate()` and appended to the output H5 incrementally, so memory
stays roughly O(chunk-size) regardless of the total number of events.

---

## The pipeline

Pre-computed *stage2* FCC-ee ntuples in, a trained Salt flavour tagger out:
convert, preprocess with UPP, train, evaluate. That is the path everything
below describes.

A second converter (`fcc-ftag-convert`, documented further down) is also
included: it builds Salt inputs from raw EDM4hep ROOT in the ATLAS
`tracks_ghost` schema, computing the track parameters itself. It is not part of
the pipeline above and ships without a UPP or Salt config.

> SLURM partitions and constraints in `scripts/` are **cluster-specific**
> (Baobab/DPNC). Paths are not: they come from three environment variables with
> Baobab defaults, set in [`scripts/env.sh`](scripts/env.sh).

| variable | default | what it is |
|---|---|---|
| `FCC_FTAG_REPO` | `$SLURM_SUBMIT_DIR`, else `$PWD` | this checkout — submit from the repo root and it is found automatically |
| `FCC_FTAG_DATA` | `/srv/beegfs/scratch/groups/rodem/FCC_FTAG` | samples, preprocessing output, training logs |
| `FCC_IMAGE` | `docker://ghcr.io/vincentriechers/fcc-ftag-salt-converter:latest` | the one container all stages run in |

Inside the container the checkout is bound to `/repo` and the data area to
`/data`, and the UPP and Salt configs are written against those. That indirection
is necessary rather than stylistic: UPP resolves a relative `!include` against
its *own* package config directory, and neither UPP nor Salt expands environment
variables in a config, so identical in-container paths are the only way a single
set of configs can work for every checkout.

## FCC-ee Higgs-decay flavour tagger

Trains a 7-class jet tagger (**u / d / s / c / b / g / tau**) on the
`pre_summer2026` Higgs-decay samples, in the ATLAS [Salt](https://github.com/umami-hep/salt)
framework. Inputs are flat `stage2_<decay>_*.root` ntuples from the FCCAnalyses
weaver (the 35 `pfcand_*` features are already computed); the converter
repackages them and adds three constituent kinematics for Salt's pairwise
edge features (`pfcand_e` copied, `pfcand_deta` / `pfcand_dphi` derived from
theta / phi) — no other physics recomputation. The tagger can be evaluated either as 7-class or as a merged
**5-class** view (light = u+d+g, then s/c/b/tau) by summing scores.

**Chain:** `weaver (stage1+stage2, truth-extended) → convert (ROOT→Salt H5) →
UPP preprocess → ±5σ clip → tau-block guard → Salt train → salt test + puma plots`.
Stage 1 is the expensive half and is often already on disk, in which case
`scripts/submit_stage2_higgs_decay.sh` does the flattening on its own.

Key pieces:

| stage | code / config | runs in |
|-------|---------------|---------|
| weaver | truth-extended FCCAnalyses (`scripts/submit_weaver_higgs_decay.sh`) | key4hep env |
| convert | `src/fcc_ftag_converter/fcc_higgs_decay.py` (`scripts/submit_convert_weaver_higgs_decay.sbatch`) | normal env |
| preprocess | `configs/upp/fcc-zh.yaml` (+ `variables-…`, `split-…`, `flavours-fcc.yaml`) | Salt container |
| clip / guard | `scripts/submit_clip_features.sbatch`, `scripts/submit_fix_tau_blocks.sbatch` | Salt container |
| train | `configs/salt/gn3.yaml` or `configs/salt/part.yaml` | Salt container, GPU |
| eval | `scripts/puma_plot_higgs_decay.py`, `scripts/puma_compare_configs.py`, `scripts/puma_compare_confmats.py` | Salt container |

Prerequisites (outside this repo):

- **Truth-labelling FCCAnalyses build** for step 1: a checkout carrying the
  ghost jet labels and the ATLAS-style constituent truth, i.e. the six stacked
  pull requests
  [#549](https://github.com/HEP-FCC/FCCAnalyses/pull/549),
  [#550](https://github.com/HEP-FCC/FCCAnalyses/pull/550),
  [#551](https://github.com/HEP-FCC/FCCAnalyses/pull/551),
  [#552](https://github.com/HEP-FCC/FCCAnalyses/pull/552),
  [#553](https://github.com/HEP-FCC/FCCAnalyses/pull/553) and
  [#554](https://github.com/HEP-FCC/FCCAnalyses/pull/554), built inside a
  key4hep environment. `scripts/weaver_higgs_decay.sh` picks it up via
  `FCCANALYSES_DIR` and sources the environment script given by
  `FCC_WEAVER_ENV` (a small wrapper that loads the key4hep stack and the local
  FCCAnalyses `setup.sh`) — set both to your paths, e.g.
  `FCCANALYSES_DIR=$HOME/FCCAnalyses FCC_WEAVER_ENV=$HOME/fccenv.sh`.
- **Apptainer** for steps 3–5: the preprocessing, training and plotting stages
  all run inside `FCC_IMAGE` (no pip installs needed).
- **Optional — Comet logging**: a shell file exporting
  `COMET_API_KEY` / `COMET_WORKSPACE` / `COMET_PROJECT_NAME`, pointed to by
  `COMET_KEY_FILE` (see `scripts/salt_commands.sh`). Without
  it, training logs offline.

Step by step (each step reads the previous one's output — wait for the jobs to
finish, or chain them with `sbatch --dependency=afterok:<jobid>`):

```bash
# 1. run the truth-extended weaver over the Delphes samples (SLURM array)
bash scripts/submit_weaver_higgs_decay.sh /path/to/higgs_decay/weaver_stage2 50
#    or, when stage1 already exists on disk, flatten it to stage2 only:
bash scripts/submit_stage2_higgs_decay.sh /path/to/stage1 /path/to/stage2
# 2. convert stage2 chunks -> fcc_zh_<decay>.h5 (one array task per decay)
sbatch scripts/submit_convert_weaver_higgs_decay.sbatch /path/to/stage2 /path/to/h5_out
# 3. UPP preprocessing, then the tau-block guard
sbatch scripts/submit_upp.sh
sbatch --array=1-35%8 scripts/submit_fix_tau_blocks.sbatch  # size = number of train splits
#    Input outliers are bounded at training time by `norm_config.clamp` in the
#    salt config, NOT by rewriting the H5 (see configs/salt/*.yaml).
#    scripts/submit_clip_features.sbatch still does the in-place variant; throttle
#    it (`--array=1-39%8`) if you use it, an interrupted run corrupts every file.
# 4. Salt training  5. eval + puma plots
sbatch scripts/salt_submit.sh
sbatch scripts/salt_test_plot.sh                    # auto-picks best ckpt
```

The eval writes to `logs/<run>/plots/`: per-class score histograms (`score_p*`,
`score5_*`), ROC curves (`roc_btag` / `roc_ctag` / `roc_stag`, each rejecting
light + the other heavy flavours incl. s), and differential rejection vs jet
pT/η (`rej_vs_pt_*`, `rej_vs_eta_*`); per-class accuracies print to the log.

Design notes and caveats are inline in the configs and in
`src/fcc_ftag_converter/fcc_higgs_decay.py` (label mapping, the 75-constituent
cap matching the reference training, `eventNumber` split key, etc.).

### Where the jet flavour comes from

`flavour_label` is built per jet from the ghost labels:

| condition | class |
|---|---|
| `jet_HadronGhostTruthLabelID == 5` | 4, b-jets |
| `== 4` | 3, c-jets |
| `== 15` | 6, tau-jets |
| `== 0` and `jet_PartonTruthLabelID == 2` | 0, u-jets |
| `== 0` and `== 1` | 1, d-jets |
| `== 0` and `== 3` | 2, s-jets |
| `== 0` and `== 21` | 5, gluon-jets |
| anything else | -1, excluded |

The hadron label settles b, c and tau but calls u, d, s and gluon all "light",
which is why the parton label is needed for those four. Note PDG order: 1 is d
and 2 is u. `atlas-ftag-tools` has those two the other way round in the
`partonic` category of its `ftag/flavours.yaml`, so do not copy it from there.

On 4.2 M jets of the 260918 production, 0.066% of jets resolve to neither and
are excluded (worst sample 0.31%).

Older ntuples have no ghost labels, and there the converter falls back to the
7-way one-hot `recojet_is*`. FCCAnalyses `stage2.py` sets those from the input
**file name**, so every jet in an `Hbb` file is a b-jet whatever it really is.
That is what the ghost labels replace. The one-hot columns are still written to
the H5, which is what makes the disagreement measurable — the converter prints
it at the end of every run:

| sample | disagrees with the file-name label |
|---|---|
| Hbb | 1.7% |
| Hcc | 2.0% |
| Hss | 22.0% |
| Hdd | 22.8% |
| Huu | 22.7% |
| Hgg | 25.4% |
| Htautau | 0.0% |

Around a fifth of the jets in the light samples are really gluon jets, and Hgg
contains real heavy flavour from gluon splitting (6.6% c, 3.2% b). Both are
physics, not a bug in the labelling.

### Truth labels (GN3 aux heads)

The central flat ntuples carry only the 35 features and a jet flavour taken
from the file name. The truth-labelling FCCAnalyses build adds real per-jet and
per-constituent truth, which is what the aux heads train on. Per constituent:

| branch | meaning |
|---|---|
| `pfcand_truthOriginLabel` | 2 Primary, 3 FromB, 4 FromBC, 5 FromC, 6 FromTau, 7 OtherSecondary, -1 no MC link |
| `pfcand_truthTypeLabel` | 0 NoTruth, 1 Other, ±2 pion, ±3 kaon, 4 lambda, ±5 electron, ±6 muon, 7 photon, signed by charge |
| `pfcand_truthSourceLabel` | 0 NoTruth, 1 NotSecondary, 3 StrangeMesonDecay, 4 StrangeBaryonDecay, 5 GammaConversion |
| `pfcand_truthVertexIndex` | 0 truth primary vertex, 1, 2, … secondaries, -2 no MC link |
| `pfcand_truthPdgId` | PDG of the matched MC particle, 0 if unmatched |

Per jet, the converter carries the full ghost-association and truth-jet block:
`jet_HadronGhostTruthLabelID` (0 light, 4 c, 5 b, 15 tau) and its extended and
initial-hadron variants, the labelling hadron's PDG and pT, the ghost b/c
hadron counts, `jet_PartonTruthLabelID` (-1 none, 1 d, 2 u, 3 s, 4 c, 5 b,
21 gluon) with its pT and ΔR, and the matched truth jet's pT, energy, ΔR,
purity and completeness.

Truth particles are added to the jet clustering at negligible momentum and
followed into the jet they end up in, so these are real per-jet quantities for
any jet algorithm and no jet axis moves.

The ATLAS-style names are not cosmetic: salt's `VertexingTask` finds its pair
weights by rewriting `VertexIndex` into `OriginLabel`, so the two constituent
labels have to be spelled exactly that way. Ntuples from the older weaver that
spell them `pfcand_truthOrigin` / `pfcand_truthVertex` / `pfcand_truthPID` (and
`jet_hadronPdgId` / `jet_truthPt`) still convert: the converter renames them on
write, and derives the type label from the matched PDG, so one set of configs
works either way. The renaming does not translate the values, though — the old
weaver used different conventions for "no MC link" — so do not mix old and new
ntuples inside a single training sample.

Everything is detected per input, so the salt config's aux heads work whenever
the labels are present and are simply skipped when they are not. When
re-converting over an existing UPP output directory, clear the stale caches
first (`rm -rf <base>/vds <base>/preprocessed`), otherwise UPP reuses the old
virtual datasets.

## Repository layout

```
fcc-ftag-salt/
├── src/fcc_ftag_converter/    # package source
│   ├── cli.py                 # argparse + streaming orchestrator
│   ├── streaming.py           # uproot chunk iterator
│   ├── reader.py              # per-chunk row builder
│   ├── physics.py             # kinematics, q/p, track uncertainties
│   ├── truth/                 # MC-truth labels, split by concern:
│   │   ├── pdg.py             #   heavy-flavour from PDG id
│   │   ├── origin.py          #   parent-ancestry b/c/tau (track_origin)
│   │   ├── vertex.py          #   production-vertex clustering
│   │   ├── track_type.py      #   ftagTruthTypeLabel from PDG
│   │   └── cone.py            #   HadronConeExcl jet labels
│   ├── fcc_higgs_decay.py     # flat stage2 ntuple -> Salt H5 (higgs_decay path)
│   ├── configs/               # salt_schema.json, shipped with the package
│   ├── schema.py              # SaltH5Writer + reference helpers
│   ├── merge.py               # concat sharded H5 outputs
│   ├── report.py              # streaming statistics + summary writer
│   └── compare.py             # generic H5 comparison
├── configs/                   # UPP/Salt yaml configs
├── scripts/                   # SLURM submit wrappers, weaver front-end, plots
├── docker/                    # all-in-one image (converter + UPP + Salt + puma)
├── tests/
└── pyproject.toml
```

## Setup

### Container (everything in one image)

The whole chain — converter, UPP, Salt, puma, torch+CUDA — is available as a
single image, built on top of the upstream Salt image and published to
`ghcr.io/vincentriechers/fcc-ftag-salt-converter`:

```bash
apptainer exec --nv -B /home -B /tmp -B /srv \
    docker://ghcr.io/vincentriechers/fcc-ftag-salt-converter:latest \
    fcc-ftag-higgs-decay --help
```

The image holds the environment, not the repository: `configs/` and `scripts/`
come from the `-B` bind mounts that `scripts/env.sh` sets up. See
[`docker/README.md`](docker/README.md) for building it locally and for which
`scripts/` variable each of the three old images maps to.

### Local install

Reproducible install with [uv](https://docs.astral.sh/uv/) (a `uv.lock` is
committed, so the environment is pinned):

```bash
uv sync --extra dev          # core + pytest/ruff/mypy/pre-commit/matplotlib
uv run fcc-ftag-convert ...   # run any console script in the locked env
```

Extras: `stream` (xrootd:// EOS inputs), `plots` (matplotlib + puma for
compare/plots), `cluster` (hdf5plugin + PyYAML for the post-UPP clip and
tau-block steps), `dev` (everything except puma, for development). Plain pip
works too:

```bash
python -m pip install -e ".[dev]"
```

Either way you get the `fcc-ftag-convert` / `fcc-ftag-compare` / `fcc-ftag-merge`
/ `fcc-ftag-higgs-decay` executables. Without installing, each is also reachable
as `python -m fcc_ftag_converter.{cli,compare,merge,fcc_higgs_decay}`.

## Convert FCC ROOT files

```bash
fcc-ftag-convert \
  root://eospublic.cern.ch//eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_tt_ecm365/events_002305720.root \
  --reference /srv/beegfs/scratch/groups/rodem/FCC_FTAG/structure_test_file.h5 \
  --output outputs/events.h5 \
  --max-events 100
```

By default the converter loads its Salt schema from
`src/fcc_ftag_converter/configs/salt_schema.json` (~80 kB, shipped inside the
package), so it needs no access to the original reference H5 at runtime. To
regenerate the JSON after a schema change:

```bash
python scripts/dump_schema.py /path/to/structure_test_file.h5
```

You can still pass a `.h5` to `--reference` directly if you prefer.

### Streaming for large runs (1M+ events)

```bash
fcc-ftag-convert \
  /path/to/sample/events_*.root \
  --output outputs/sample.h5 \
  --chunk-size 10000   # default; tune for memory vs. throughput
```

Each chunk is read, converted, and written to the H5 before the next chunk
starts. The summary report uses **reservoir sampling** (default 100k samples
per stats variable) so median/p95 stay accurate without buffering all values.

### Parallel conversion across many ROOT files

```bash
fcc-ftag-convert /path/to/sample/events_*.root \
  --output outputs/sample.h5 \
  --workers 8
```

`--workers N` shards the input file list round-robin across `N`
subprocesses, each writing a temporary H5; once all workers finish, the
shards are concatenated with `fcc-ftag-merge` and the merged summary
report is written. Only file-level parallelism — a single huge ROOT file
still goes through one worker. Diagnostics counters and `missing_collections`
are aggregated across all shards.

You can also merge manually:

```bash
fcc-ftag-merge shard_*.h5 -o combined.h5 --report
```

### Summary report

After every run two sidecar files appear next to the output:

- `<stem>.summary.md` — human-readable table
- `<stem>.summary.json` — machine-readable

They include: input files, number of jets, valid object counts
(`tracks_ghost`/`flows`/`electrons`/`muons`/`calo`), reservoir-sampled
median/p95 for key variables, missing EDM4hep collections, lists of
approximate/placeholder variables, and **conversion diagnostics**
(empty events, jets without tracks/flows, jets with non-finite kinematics).
All momentum/energy values are in **MeV**; `qOverP` in **MeV⁻¹**.
Skip with `--no-report`.

## Useful options

| Flag | Default | Description |
|------|---------|-------------|
| `--chunk-size` | 10000 | Events per uproot iterate step. Lower = less RAM, higher = less I/O overhead. |
| `--workers` | 1 | If >1 and there are ≥2 input files: shard files across N parallel subprocesses, then merge. |
| `--lepton-match-mode` | `constituent` | Or `nearest-jet`: match all event leptons to the nearest jet within `--lepton-match-dr`. |
| `--lepton-match-dr` | 0.4 | Max ΔR for nearest-jet lepton matching. |
| `--max-tracks`/`--max-flows`/`--max-electrons`/`--max-muons` | 50/50/10/5 | Slot caps. |
| `--magnetic-field-t` | 2.0 | Detector B field used for q/p conversion. |
| `--truth-cone-radius` | 0.4 | Cone for heavy-flavour ghost-label approximation. |
| `--no-report` | off | Skip writing `.summary.{md,json}`. |
| `--quiet` | off | Suppress per-chunk progress. |

### Missing optional collections

If `Electron#0` or `Muon#0` are missing from the input, the converter warns
once and writes zero/placeholder arrays for those datasets. Required
collections still raise a clear error.

## Compare two H5 files

```bash
fcc-ftag-compare ref.h5 converted.h5 \
  --label-a reference --label-b fcc \
  --output-dir plots/reference_vs_fcc
```

To compare only selected variables:

```bash
fcc-ftag-compare ref.h5 converted.h5 \
  --variable jets.pt_btagJes \
  --variable tracks_ghost.qOverP \
  --variable electrons.pt
```

Writes one PNG per variable plus `summary.csv` and `summary.md`. Each
histogram is normalized independently per sample.

## Checks

The repo uses Salt's lint, type and test settings, so the same rules apply in
both places. Install the hooks once and they run on every commit:

```bash
uv run --extra dev pre-commit install
```

To run everything by hand:

```bash
uv run --extra dev pre-commit run --all-files   # ruff, ruff-format, mypy, yamllint, pydoclint
uv run --extra dev pytest -q
```

CI runs the same checks: a single `lint` job (ruff, ruff-format, mypy) plus the
test matrix on Python 3.11-3.14.

`git blame` skips the bulk reformat via `.git-blame-ignore-revs`; teach your
local git about it with:

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```
