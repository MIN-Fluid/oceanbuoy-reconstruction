# A buoy-token attention model for real-time significant wave height reconstruction during outages

Code, data, and trained models for the paper of the same title (Oh, Seo, Woo, Jeon, and Lee).

When a wave buoy stops reporting, the model estimates its significant wave height at that hour from the records of
the neighboring buoys of the same region. This repository holds the raw buoy files, the code of the attention model
and of six baseline methods, the trained models, and a pipeline that runs from the raw files to the results of the paper.

## Quick start

```bash
conda env create -f environment.yml
conda activate oceanbuoy-reconstruction
snakemake -c4
```

The last command reproduces the results with the trained models, without any training. It takes about one hour on
a 12-core CPU and writes the results to `results/`. The table [Results and the tables of the paper](#results-and-the-tables-of-the-paper)
shows where each table of the paper is found in these files.

## What is reproduced

| Paper | Content | Reproduced |
|---|---|---|
| Section 4.2 | Accuracy by region and by the number of reporting neighbors, in complete outages | Yes |
| Section 4.3 | Complete and partial outages | Yes |
| Section 4.4 | Adding buoys to the network | Yes |
| Appendix B | Accuracy of the ensemble members | Yes |
| Sections 4.1 and 4.5, the tables of Section 2, the figures | | No |

## Regions, years, and seeds

| Name | Region | Buoys |
|---|---|---|
| `socal` | Southern California | 10 |
| `gom` | Gulf of Mexico | 6 |
| `pnw` | Pacific Northwest | 4 |
| `socal12` | Southern California with the two added buoys of Section 4.4, 46239 and 46047 | 12 |

The training years are 2016–2022, the validation year is 2023, and the test years are 2024 and 2025.
Each learned method and the attention model is an ensemble of six members, trained with the seeds 20260805 to 20260810.

## Files in the repository

| Path | Contents |
|---|---|
| `data/raw/` | The NDBC standard meteorological files used in the study, as downloaded (22 buoys, the twenty of the three regions and the two added ones, 2016–2025, about 89 MB) |
| `models/` | The trained models, six members each |
| `configs/` | Every setting: buoy lists (`stations/`), the years of each split (`data.json`), the seeds (`pipeline.json`), data, sampling, models, training, fine-tuning, interpolators, and evaluation |
| `download/`, `preprocess/` | Downloading the raw files from NDBC; reading them onto the hourly grid |
| `src/oceanbuoy/` | The library |
| `scripts/` | One script per pipeline step |
| `Snakefile` | The pipeline from the raw files to the evaluation |

### Trained models

A file name holds the last two digits of the seed, from `s05` to `s10`.

| Method in the paper | Files | Number of files |
|---|---|---|
| Attention model | `models/attention/<region>/s05.pt` to `s10.pt` | 6 per region |
| Token MLP | `models/token_mlp/<region>/s05.pt` to `s10.pt` | 6 per region |
| Ridge regression | `models/ridge/<region>/s05.nc` to `s10.nc` | 6 per region |
| MLP | `models/mlp/<region>/s05_t00.pt`, `s05_t01.pt`, and so on | 6 per target buoy |
| FT-emb, FT-enc, FT (Section 4.4) | `models/expansion/ft_emb/`, `ft_enc/`, `ft/` | 6 each |
| Retrain (Section 4.4) | `models/expansion/retrain/` | 6 |

The attention model, the token MLP, and the ridge regression have one file per seed for a whole region.
The MLP is trained separately for each target buoy. Its file name adds the position of the target in the buoy list of
the region, counted from zero, so that `models/mlp/gom/s05_t00.pt` is the model of the first buoy in `configs/stations/gom.json`.
The interpolators (IDW, OK, AK) have no trained model in the repository; the pipeline fits them.

### Library

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

### Removal of outliers

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

The pipeline runs the following steps and writes each output to the folder shown.

| Step | Output |
|---|---|
| 1. Alignment of the raw files to the hourly grid | `data/aligned/` |
| 2. Records before and after the removal of outliers | `data/obs_unscreened/`, `data/qc/`, `data/obs/` |
| 3. Normalization constants | `data/norm/` |
| 4. Test samples | `data/samples/test/` |
| 5. Outage patterns of `socal12` | `data/outage_patterns/` |
| 6. Fit and estimates of the interpolators | `models/interp/<region>.json`, `predictions/idw/`, `ok/`, `ak/` |
| 7. Estimates of the learned methods and the attention model, from the trained models | `predictions/ridge/`, `mlp/`, `token_mlp/`, `attention/` |
| 8. Estimates of Section 4.4 | `predictions/expansion/`, `predictions/added_buoys/<method>_<buoy>.nc` |
| 9. Evaluation | `results/` |

The trained models in `models/` are used as they are, so that no training is run, and the evaluation gives the
numbers of the paper. Other versions of the numerical libraries can change the last digits of the results (by about
10<sup>-14</sup> m), which changes no number reported in the paper.

`-c4` sets the number of steps run at once; each estimation step uses the number of threads set in
`configs/models.json` (`determinism.cpu_threads`).

## Retraining

```bash
snakemake -c4 --resources gpu=1 --config retrain=1
```

With `retrain=1` the pipeline also builds the training and validation samples (`data/samples/train/`,
`data/samples/val/`) and defines the training rules; a model is trained again when its file in `models/` is
missing or older than its inputs. Training on a GPU is not bit-for-bit deterministic, so retrained models give
numbers that differ slightly from the published ones. A new model file replaces the published one at the same path.

A single model is trained with `scripts/train.py` after its samples are built. For the attention model of
Southern California with the first seed:

```bash
snakemake -c4 data/norm/socal.json
python scripts/make_samples.py --region socal --what val
python scripts/make_samples.py --region socal --what train --seed 20260805
python scripts/train.py --model attention --region socal --seed 20260805
```

| Model | Command |
|---|---|
| Attention model, token MLP, ridge regression | `train.py --model attention` (or `token_mlp`, `ridge`) `--region <region> --seed S` |
| MLP | `train.py --model mlp --target <index of the target buoy> --region <region> --seed S` |
| Retrain (Section 4.4) | `train.py --model attention --region socal12 --seed S` |
| FT-emb, FT-enc, FT (Section 4.4) | `train.py --model finetune --variant ft_emb` (or `ft_enc`, `ft`) `--seed S` |

The models of Section 4.4 use the `socal12` samples (`make_samples.py --region socal12`, after
`snakemake -c4 data/norm/socal12.json data/outage_patterns/socal12.json`). Fine-tuning starts from the attention
model of `socal` with the same seed.

## Results and the tables of the paper

| Paper | File in `results/` | Key |
|---|---|---|
| Table 6 | `evaluation_regions.json` | `<region>.groups`, `<region>.all`, `<region>.two_or_more_neighbors` |
| Table 7 | `evaluation_regions.json` | `<region>.nrmse` |
| Table 8 | `evaluation_regions.json` | `<region>.outage_type.complete`, `.partial`, `.partial-complete` |
| Table 9 | `evaluation_added_buoys.json` | `added_buoys.<buoy>.complete` (`46239`, `46047`) and `added_buoys.46047.partial` |
| Table 10 | `evaluation_added_buoys.json` | `original_buoys.complete`, `.partial` |
| Table B.1 | `evaluation_regions.json` | `member_rmse_m` of each method in `<region>.all` |

Each entry holds `methods`, the RMSE of each method, and `differences`, the difference of two RMSEs on the same test
samples. A difference is written `A-B` for RMSE<sub>A</sub> − RMSE<sub>B</sub>, as in the captions of the tables.
Every RMSE is in meters, with its 95% confidence interval from the block bootstrap of Section 3.4 (blocks of 240 h).

The methods have the following names in the files.

| Method in the paper | Name |
|---|---|
| Attention model, token MLP, MLP, ridge regression | `attention`, `token_mlp`, `mlp`, `ridge` |
| IDW, OK, AK | `idw`, `ok`, `ak` |
| Base, None (Section 4.4) | `base`, `none` |
| FT-emb, FT-enc, FT, Retrain (Section 4.4) | `ft_emb`, `ft_enc`, `ft`, `retrain` |

The differences are named `attention-<method>` in Tables 6 and 8, with the best baseline in `best_baseline`,
`<method>-none` in Table 9, and `<method>-base` in Table 10. Base is the attention model of `socal`, and None is the
same model with the added buoys as input.

### How the results are computed

The main results are in complete outages, in which no sensor of the target reports (outage pattern 0), and the
neighbors enter as recorded.

In partial outages, the target's anemometer, barometer, or sea surface temperature sensor still reports. The seven
partial patterns are weighted by the target's training-year occurrence rates, renormalized to sum to one (Eq. 7 of
the paper), and the results are computed at the NDBC buoys, the targets that have these sensors.

The models are selected by the validation RMSE weighted over all eight patterns.

The interpolators are evaluated on the test samples with two or more reporting neighbors (`two_or_more_neighbors`).

Table 6 leaves the difference blank for the two groups excluded from the comparison of the methods, the Southern
California group with six or fewer reporting neighbors (209 samples) and the Gulf of Mexico group with one (10 samples).
Their differences are still given in the JSON.

## Citation

`CITATION.cff` gives the citation of this repository.

## License

The code is released under the MIT License (`LICENSE`). The observations in `data/raw/` are public NDBC records.
