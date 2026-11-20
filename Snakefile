import json

PY = "python"
_DET = json.load(open("configs/models.json"))["determinism"]
ENV = (f"export OMP_NUM_THREADS={_DET['cpu_threads']} MKL_NUM_THREADS={_DET['cpu_threads']} "
       f"OPENBLAS_NUM_THREADS={_DET['cpu_threads']} CUDA_VISIBLE_DEVICES={_DET['cuda_device']}; ")
REGIONS = ["socal", "gom", "pnw"]
TAGS = [f"s{s - 20260800:02d}" for s in json.load(open("configs/pipeline.json"))["train_seeds"]]
N_STATIONS = {"socal": 10, "gom": 6, "pnw": 4}
RETRAIN = bool(config.get("retrain", False))
CFG = [f"configs/{c}.json" for c in ("data", "design", "eval", "interp", "models", "pipeline", "sampling", "sensor_groups", "socal12",
                                     "train", "views", "views_groups", "finetune")]
LEARNED = ["ridge", "mlp", "token_mlp", "attention"]
RUNGS = ["none", "ft_emb", "ft_enc", "ft", "retrain"]
ADDED = ["46239", "46047"]
ADDED_METHODS = RUNGS + ["idw", "ok", "ak"]


def seed_of(w):
    return 20260800 + int(w.tag[1:])


def obs_in(name):
    return [f"data/obs/{name}.nc"]


wildcard_constraints:
    r="socal|gom|pnw", name="socal|gom|pnw|socal12", tag="s\\d\\d", g="\\d\\d", model="attention|token_mlp",
    m="ridge|mlp|token_mlp|attention", variant="ft_emb|ft_enc|ft"


rule all:
    input:
        "results/evaluation_4_1.json", "results/evaluation_4_2.json",


rule aligned:
    input:
        "configs/stations/{name}.json",
    output:
        "data/aligned/{name}.npz"
    shell:
        "{PY} preprocess/build_aligned.py --regions {wildcards.name}"


rule obs_unscreened:
    input:
        "data/aligned/{name}.npz", CFG,
    output:
        "data/obs_unscreened/{name}.nc"
    shell:
        "{PY} scripts/make_obs.py --region {wildcards.name}"


rule wave_faults:
    input:
        expand("data/obs_unscreened/{r}.nc", r=REGIONS),
    output:
        "data/qc/suspect_wave_runs.json"
    shell:
        "{PY} scripts/find_wave_faults.py"


rule obs:
    input:
        lambda w: ["data/obs_unscreened/socal12.nc"] if w.name == "socal12" else [f"data/obs_unscreened/{w.name}.nc", "data/qc/suspect_wave_runs.json"],
    output:
        "data/obs/{name}.nc"
    shell:
        "{PY} scripts/screen_obs.py --region {wildcards.name}"


rule norm:
    input:
        lambda w: ["data/obs/socal12.nc", "data/norm/socal.json"] if w.name == "socal12" else [f"data/obs/{w.name}.nc"],
    output:
        "data/norm/{name}.json"
    shell:
        "{PY} scripts/make_norm.py --region {wildcards.name}"


rule test_set:
    input:
        "data/obs_unscreened/{r}.nc",
    output:
        "data/samples/test/{r}.nc"
    shell:
        "{PY} scripts/make_test_set.py --region {wildcards.r}"


rule outage_patterns:
    input:
        "data/obs/socal12.nc",
    output:
        "data/outage_patterns/socal12.json"
    shell:
        "{PY} scripts/outage_patterns.py"


rule interp_fit:
    input:
        lambda w: obs_in(w.name),
    output:
        "models/interp/{name}.json"
    shell:
        "{PY} scripts/fit_interp.py --region {wildcards.name}"


rule interp_predict:
    input:
        "models/interp/{r}.json", "data/samples/test/{r}.nc", "data/obs/{r}.nc",
    output:
        expand("predictions/{m}/{{r}}.nc", m=["idw", "ok", "ak"])
    shell:
        "{PY} scripts/predict_interp.py --region {wildcards.r}"


def model_files(m, r):
    if m in ("attention", "token_mlp"):
        return [f"models/{m}/{r}/{tag}.pt" for tag in TAGS]
    if m == "mlp":
        return [f"models/{m}/{r}/{tag}_t{g:02d}.pt" for tag in TAGS for g in range(N_STATIONS[r])]
    return [f"models/ridge/{r}/{tag}.nc" for tag in TAGS]


def rung_files(rung):
    if rung == "none":
        return model_files("attention", "socal")
    return [f"models/expansion/{rung}/{tag}.pt" for tag in TAGS]


rule predict:
    input:
        lambda w: model_files(w.m, w.r), "data/samples/test/{r}.nc", "data/obs/{r}.nc", "data/norm/{r}.json",
    output:
        "predictions/{m}/{r}.nc"
    shell:
        ENV + "{PY} scripts/predict.py --method {wildcards.m} --region {wildcards.r}"


rule predict_original_buoys:
    input:
        [f for rung in RUNGS for f in rung_files(rung)], "data/samples/test/socal.nc", "data/obs/socal.nc", "data/obs/socal12.nc",
        "data/norm/socal.json", "data/norm/socal12.json",
    output:
        expand("predictions/expansion/{rung}.nc", rung=RUNGS)
    shell:
        ENV + "{PY} scripts/predict_expansion.py --buoys original"


rule predict_added_buoys:
    input:
        [f for rung in RUNGS for f in rung_files(rung)], "models/interp/socal12.json", "data/obs/socal12.nc", "data/norm/socal12.json",
    output:
        expand("predictions/added_buoys/{m}_{b}.nc", m=ADDED_METHODS, b=ADDED)
    shell:
        ENV + "{PY} scripts/predict_expansion.py --buoys added"


rule evaluate:
    input:
        [f"predictions/{m}/{r}.nc" for m in LEARNED + ["idw", "ok", "ak"] for r in REGIONS],
        expand("predictions/expansion/{rung}.nc", rung=RUNGS), expand("predictions/added_buoys/{m}_{b}.nc", m=ADDED_METHODS, b=ADDED),
        expand("data/obs/{r}.nc", r=REGIONS), "data/outage_patterns/socal12.json",
    output:
        "results/evaluation_4_1.json", "results/evaluation_4_2.json",
    shell:
        "{PY} scripts/evaluate.py"


if RETRAIN:
    rule val_samples:
        input:
            lambda w: obs_in(w.name) + (["data/outage_patterns/socal12.json"] if w.name == "socal12" else []),
        output:
            "data/samples/val/{name}.nc"
        shell:
            "{PY} scripts/make_samples.py --region {wildcards.name} --what val"

    rule train_samples:
        input:
            lambda w: obs_in(w.name) + (["data/outage_patterns/socal12.json"] if w.name == "socal12" else []),
        output:
            "data/samples/train/{name}_{tag}.nc"
        params:
            seed=seed_of
        shell:
            "{PY} scripts/make_samples.py --region {wildcards.name} --what train --seed {params.seed}"

    def train_in(r, tag):
        return [f"data/samples/train/{r}_{tag}.nc", f"data/samples/val/{r}.nc", f"data/norm/{r}.json"] + obs_in(r)

    rule train_nn:
        input:
            lambda w: train_in(w.r, w.tag),
        output:
            "models/{model}/{r}/{tag}.pt"
        params:
            seed=seed_of
        resources:
            gpu=1
        shell:
            ENV + "{PY} scripts/train.py --model {wildcards.model} --region {wildcards.r} --seed {params.seed}"

    rule train_mlp:
        input:
            lambda w: train_in(w.r, w.tag),
        output:
            "models/mlp/{r}/{tag}_t{g}.pt"
        params:
            seed=seed_of
        resources:
            gpu=1
        shell:
            ENV + "{PY} scripts/train.py --model mlp --region {wildcards.r} --seed {params.seed} --target {wildcards.g}"

    rule train_ridge:
        input:
            lambda w: train_in(w.r, w.tag),
        output:
            "models/ridge/{r}/{tag}.nc"
        params:
            seed=seed_of
        shell:
            ENV + "{PY} scripts/train.py --model ridge --region {wildcards.r} --seed {params.seed}"

    rule train_retrain:
        input:
            lambda w: train_in("socal12", w.tag),
        output:
            "models/expansion/retrain/{tag}.pt"
        params:
            seed=seed_of
        resources:
            gpu=1
        shell:
            ENV + "{PY} scripts/train.py --model attention --region socal12 --seed {params.seed}"

    rule train_finetune:
        input:
            lambda w: train_in("socal12", w.tag) + [f"models/attention/socal/{w.tag}.pt"],
        output:
            "models/expansion/{variant}/{tag}.pt"
        params:
            seed=seed_of
        resources:
            gpu=1
        shell:
            ENV + "{PY} scripts/train.py --model finetune --variant {wildcards.variant} --seed {params.seed}"
