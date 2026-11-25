# A buoy-token attention model for real-time significant wave height reconstruction during outages

Code, data, and trained models for the paper of the same title (Oh, Seo, Woo, Jeon, and Lee).
The repository reproduces the preprocessing, the training and the inference of every method, and the
evaluation of Section 4.2 (accuracy by region and by the number of reporting neighbors, in complete outages), Section 4.3
(complete and partial outages), Section 4.4 (adding buoys to the network), and Appendix B (accuracy of the ensemble members).
It does not include the analyses of Sections 4.1 and 4.5, the tables of Section 2, or the figures.

## Contents

| Path | Contents |
|---|---|
| `data/raw/` | The NDBC standard meteorological files used in the study, as downloaded (22 buoys, 2016–2025, about 89 MB) |
| `models/` | The trained models (below), six members each |
| `configs/` | Every setting: buoy lists (`stations/`), the years of each split (`data.json`), the seeds (`pipeline.json`), data, sampling, models, training, fine-tuning, interpolators, and evaluation |
| `download/`, `preprocess/` | Downloading the raw files from NDBC; reading them onto the hourly grid |
| `src/oceanbuoy/` | The library (below) |
| `scripts/` | One script per pipeline step |
| `Snakefile` | The pipeline from the raw files to the evaluation |

| Module | Role |
|---|---|
| `config.py`, `io.py`, `rng.py` | Paths, settings, and seeds; writing files; the random-number streams |
| `data.py`, `sensor_groups.py`, `states.py` | Hourly records, removal of outliers, normalization, sensor groups, outage patterns |
| `samples.py`, `build_samples.py`, `validation.py` | Training and validation samples with synthetic outages, the validation RMSE weighted over the eight outage patterns |
| `views.py` | Inputs: tokens for the attention model and the token MLP, fixed rows for the ridge regression and the MLP |
| `models.py`, `train.py`, `ridge.py` | The networks, their training (`fit`) and fine-tuning, the ridge regression |
| `interp.py` | The interpolators: inverse-distance weighting (IDW), ordinary kriging (OK), and anisotropic kriging (AK) |
| `predict.py`, `expansion.py` | Estimates on the test samples; the network with the two added buoys |
| `evaluation.py` | RMSE, block bootstrap, and differences between methods |

## Names

The regions are `socal` (Southern California, ten buoys), `gom` (Gulf of Mexico, six), and `pnw` (Pacific Northwest, four).
`socal12` is the Southern California network with the two added buoys of Section 4.4, 46239 and 46047.
The training years are 2016–2022, the validation year 2023, and the test years 2024 and 2025. The six members of each
learned method and of the attention model are trained with the seeds 20260805 to 20260810, written `s05` to `s10` in file names.

| Paper | Model files | Estimates | Key in `results/` |
|---|---|---|---|
| Ridge regression | `models/ridge/<region>/sNN.nc` | `predictions/ridge/` | `ridge` |
| MLP | `models/mlp/<region>/sNN_tGG.pt`, one per target buoy `GG` | `predictions/mlp/` | `mlp` |
| Token MLP | `models/token_mlp/<region>/sNN.pt` | `predictions/token_mlp/` | `token_mlp` |
| Attention model | `models/attention/<region>/sNN.pt` | `predictions/attention/` | `attention` |
| IDW, OK, AK | `models/interp/<region>.json` | `predictions/idw|ok|ak/` | `idw`, `ok`, `ak` |
| Base (Section 4.4) | the attention model of `socal` | `predictions/attention/socal.nc` | `base` |
| None | the same model, with the added buoys as input | `predictions/expansion/none.nc` | `none` |
| FT-emb, FT-enc, FT | `models/expansion/ft_emb|ft_enc|ft/` | `predictions/expansion/ft_emb|ft_enc|ft.nc` | `ft_emb`, `ft_enc`, `ft` |
| Retrain | `models/expansion/retrain/` | `predictions/expansion/retrain.nc` | `retrain` |

The estimates at the two added buoys are in `predictions/added_buoys/<method>_<buoy>.nc`, with the same method names.

## Installation

```bash
conda env create -f environment.yml
conda activate oceanbuoy-reconstruction
```

The pipeline was run with Python 3.12, PyTorch 2.8.0 and CUDA 12.8 on Linux. Estimation runs on the CPU;
training uses one GPU.

## Data

All observations are public records of the National Data Buoy Center (NDBC) standard meteorological archive
(https://www.ndbc.noaa.gov). The files in `data/raw/<station>/<station>h<year>.txt.gz` are the archive files
without any change. Five of the buoys, 46218, 46219, 46239, 46251, and 46259, are Coastal Data Information Program
(CDIP) wave buoys operated by the Scripps Institution of Oceanography and distributed through the same archive.
CDIP data are cited as https://doi.org/10.18437/C7WC72.
`python download/download_ndbc.py` downloads the files again from NDBC; files already in `data/raw/` are skipped.

Before the samples are built, significant wave heights that stay far below those of the other buoys of the region are
removed as outliers (`scripts/find_wave_faults.py`, rule in `src/oceanbuoy/data.py`). An outlier is a run of at least
12 consecutive hours in which a buoy reports a significant wave height below 0.1 m and no dominant period, while the
median significant wave height of the other buoys of the region over the run is at least 0.5 m. The wave variables of
each run are set to missing. Every run found lies in the training years (2016–2022); the runs, with their buoy, start,
end and length, are written to `data/qc/suspect_wave_runs.json` by the pipeline.

## Reproducing the results with the trained models

```bash
snakemake -c4
```

This runs, in order: alignment of the raw files to the hourly grid (`data/aligned/`), the records before and after
the removal of outliers (`data/obs_unscreened/`, `data/qc/`, `data/obs/`), normalization constants
(`data/norm/`), the test samples (`data/samples/test/`), the outage patterns of `socal12` (`data/outage_patterns/`),
the fit and the estimates of the interpolators (`models/interp/`, `predictions/idw|ok|ak/`), the estimates of the
learned methods and the attention model from the trained models, the estimates of Section 4.4 (`predictions/expansion/`,
`predictions/added_buoys/`), and the evaluation (`results/`).
The trained models in `models/` are used as they are, so that no training is run, and the evaluation gives the
numbers of the paper. Other versions of the numerical libraries can change the last digits of the results (by about
10<sup>-14</sup> m), which changes no number reported in the paper. The whole run takes about one hour on a 12-core CPU.
`-c4` sets the number of steps run at once; each estimation step uses the number of threads set in
`configs/models.json` (`determinism.cpu_threads`).

## Retraining

```bash
snakemake -c4 --resources gpu=1 --config retrain=1
```

With `retrain=1` the pipeline also builds the training and validation samples (`data/samples/train/`,
`data/samples/val/`) and defines the training rules; a model is trained again when its file in `models/` is
missing or older than its inputs. Training on a GPU is not bit-for-bit deterministic, so retrained models give
numbers that differ slightly from the published ones.

A single model is trained with `scripts/train.py` after its samples are built. For the attention model of
Southern California with the first seed:

```bash
snakemake -c4 data/norm/socal.json
python scripts/make_samples.py --region socal --what val
python scripts/make_samples.py --region socal --what train --seed 20260805
python scripts/train.py --model attention --region socal --seed 20260805
```

`--model` is `attention`, `token_mlp`, `mlp` (with `--target`, the index of the target buoy), or `ridge`.
The models of Section 4.4 use the `socal12` samples (`make_samples.py --region socal12`, after
`snakemake -c4 data/norm/socal12.json data/outage_patterns/socal12.json`): Retrain is
`train.py --model attention --region socal12 --seed S`, and fine-tuning, which starts from the attention model
of `socal` with the same seed, is `train.py --model finetune --variant ft_emb|ft_enc|ft --seed S`.
A new model file replaces the published one at the same path.

## Outputs and the tables of the paper

| File | Key | Paper | Difference |
|---|---|---|---|
| `results/evaluation_regions.json` | `<region>.groups`, `<region>.all`, `<region>.two_or_more_neighbors` | Table 6 | `attention-<method>`, with the best baseline in `best_baseline` |
| | `<region>.nrmse` | Table 7 | |
| | `<region>.outage_type.complete`, `.partial`, `.partial-complete` | Table 8 | `attention-<method>`, `partial-complete` |
| | `member_rmse_m` of each method in `<region>.all` | Table B.1 | |
| `results/evaluation_added_buoys.json` | `added_buoys.<buoy>.complete` (`46239`, `46047`) and `added_buoys.46047.partial` | Table 9 | `<method>-none` |
| | `original_buoys.complete`, `.partial` | Table 10 | `<method>-base` |

Each entry holds `methods`, the RMSE of each method, and `differences`, the difference of two RMSEs on the same test
samples, written `A-B` for RMSE<sub>A</sub> − RMSE<sub>B</sub> as in the captions of the tables. Every RMSE is in meters, with its 95% confidence interval from the block bootstrap of Section 3.4 (blocks of 240 h).
The main results are in complete outages, in which no sensor of the target reports (outage pattern 0). Partial outages, in which
the target's anemometer, barometer, or sea surface temperature sensor still reports, weight the seven partial patterns by the
target's training-year occurrence rates, renormalized to sum to one, and are scored at the NDBC buoys, the targets that have them (Eq. 7 of
the paper). The neighbors enter as recorded in both. The models are selected by the validation RMSE weighted over all eight patterns.
The interpolators are scored on the test samples with two or more reporting neighbors (`two_or_more_neighbors`).
Table 6 leaves the difference blank for the two groups excluded from the comparison of the methods, the Southern
California group with six or fewer reporting neighbors (209 samples) and the Gulf of Mexico group with one (10 samples),
whose differences are still given in the JSON.

## License

The code is released under the MIT License (`LICENSE`). The observations in `data/raw/` are public NDBC records.
