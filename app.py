"""FastAPI server for the load-classification random forests.

Run from the project root (next to model.py / score.py):
    uvicorn app:app --host 0.0.0.0 --port 8000
"""
import glob
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# `model` must be importable: the pickles reference model.RandomForest
from model import RandomForest  # noqa: F401

BASE = Path(__file__).parent
RESULTS = Path(os.getenv("RESULTS_DIR", BASE / "results"))
ARTIFACTS = Path(os.getenv("ARTIFACTS_DIR", BASE / "artifacts"))

CIR_CLASS = ["RC", "RL", "LC", "L", "C", "R", "RLC"]
BEHAV_CLASS = ["CZ", "CP", "CC"]

# name, group, unit
FEATURE_META = [
    ("Vrms", "Overall measurements", "V"),
    ("Irms", "Overall measurements", "A"),
    ("P", "Overall measurements", "W"),
    ("Q1", "Overall measurements", "var"),
    ("S", "Overall measurements", "VA"),
    ("PF", "Overall measurements", ""),
    ("phi_deg", "Overall measurements", "°"),
    ("Zmag", "Overall measurements", "Ω"),
    ("THD_V", "Distortion", "%"),
    ("THD_I", "Distortion", "%"),
] + [
    (f"{q}_L{i}", f"Stage L{i}", u)
    for i in range(4)
    for q, u in (("V", "V"), ("I", "A"), ("P", "W"))
]
FEATURES = [f[0] for f in FEATURE_META]

STATE: dict = {}


def _load_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _find_model(env_var: str, pattern: str) -> Path:
    if os.getenv(env_var):
        return Path(os.environ[env_var])
    hits = sorted(glob.glob(str(RESULTS / pattern)), key=os.path.getmtime)
    if not hits:
        raise FileNotFoundError(f"No model matching {pattern} in {RESULTS}")
    return Path(hits[-1])


def _load_classes(filename, fallback):
    """Class names in the exact order the model's integer outputs use."""
    path = ARTIFACTS / filename
    if path.exists():
        return [str(c) for c in joblib.load(path).classes_]
    return list(fallback)


def _from_training_pickles(scaler):
    """Placeholder defaults + example rows from the files train.py already uses."""
    try:
        X = pd.DataFrame(joblib.load(ARTIFACTS / "X_train.pkl"))
        if X.shape[1] != len(FEATURES):
            return {}, []
        X.columns = FEATURES
        if scaler is not None:  # show defaults in original units
            X = pd.DataFrame(scaler.inverse_transform(X), columns=FEATURES)
        stats = {
            f: {"median": float(X[f].median()), "min": float(X[f].min()), "max": float(X[f].max())}
            for f in FEATURES
        }
        yb = pd.Series(joblib.load(ARTIFACTS / "y_behav_train.pkl")).values
        yc = pd.Series(joblib.load(ARTIFACTS / "y_cir_train.pkl")).values
        df = X.assign(_b=yb, _c=yc)
        ex = df.groupby(["_b", "_c"], group_keys=False).sample(1, random_state=42)
        samples = [
            {
                "behaviour": decode(r["_b"], STATE["classes"]["behav"]),
                "circuit": decode(r["_c"], STATE["classes"]["cir"]),
                "values": {f: float(r[f]) for f in FEATURES},
            }
            for _, r in ex.iterrows()
        ]
        return stats, samples
    except Exception as e:  # app still works, placeholders fall back to 0
        print("Could not build defaults from training pickles:", e)
        return {}, []


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["behav"] = joblib.load(_find_model("BEHAV_MODEL", "rf_behaviour-*.pkl"))
    STATE["cir"] = joblib.load(_find_model("CIR_MODEL", "rf_circuit-*.pkl"))
    scaler = ARTIFACTS / "scaler.pkl"  # optional, only if you scaled in training
    STATE["scaler"] = joblib.load(scaler) if scaler.exists() else None
    STATE["classes"] = {
        "behav": _load_classes("le_behav.pkl", BEHAV_CLASS),
        "cir": _load_classes("le_cir.pkl", CIR_CLASS),
    }
    stats = _load_json(ARTIFACTS / "feature_stats.json", None)
    samples = _load_json(ARTIFACTS / "samples.json", None)
    if stats is None or samples is None:  # derive from the training pickles
        stats2, samples2 = _from_training_pickles(STATE["scaler"])
        stats = stats if stats is not None else stats2
        samples = samples if samples is not None else samples2
    STATE["stats"], STATE["samples"] = stats, samples
    yield


app = FastAPI(title="Load Classifier", lifespan=lifespan)


def default_for(name: str) -> float:
    return float(STATE["stats"].get(name, {}).get("median", 0.0))


def decode(v, classes):
    v = v.item() if hasattr(v, "item") else v
    return v if isinstance(v, str) else classes[int(v)]


def run_predict(model, df: pd.DataFrame):
    try:
        return np.asarray(model.predict(df))
    except Exception:  # model may expect a raw array
        return np.asarray(model.predict(df.values))


def run_proba(model, df: pd.DataFrame, classes):
    if not hasattr(model, "predict_proba"):
        return None
    try:
        p = np.asarray(model.predict_proba(df))[0]
        if len(p) != len(classes):
            return None
        return {c: float(v) for c, v in zip(classes, p)}
    except Exception:
        return None


class PredictIn(BaseModel):
    # Missing / null fields are filled with the training-set median.
    values: Dict[str, Optional[float]] = {}


@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": "behav" in STATE and "cir" in STATE}


@app.get("/api/features")
def features():
    stats = STATE["stats"]
    return [
        {
            "name": n,
            "group": g,
            "unit": u,
            "default": default_for(n),
            "min": stats.get(n, {}).get("min"),
            "max": stats.get(n, {}).get("max"),
        }
        for n, g, u in FEATURE_META
    ]


@app.get("/api/samples")
def samples():
    return STATE["samples"]


@app.get("/api/metrics")
def metrics():
    path = RESULTS / "performance.json"
    if not path.exists():
        raise HTTPException(404, "performance.json not found in results/")
    data = json.loads(path.read_text())
    legacy = {"behaviour": ("behav", BEHAV_CLASS), "circuit": ("cir", CIR_CLASS)}
    for key, (k, old_list) in legacy.items():
        t = data.get(key)
        names = STATE["classes"][k]
        if t and t.get("classes") == old_list and len(names) == len(old_list) == len(t["confusion_matrix"]):
            # Old train.py: score.confusion() named index i as old_list[i], then
            # pd.crosstab sorted rows/cols alphabetically by that name. Undo that:
            # the row for name n is encoder class index old_list.index(n).
            t["classes"] = [names[old_list.index(n)] for n in sorted(old_list)]
    return data


@app.post("/api/predict")
def predict(body: PredictIn):
    unknown = set(body.values) - set(FEATURES)
    if unknown:
        raise HTTPException(422, f"Unknown fields: {sorted(unknown)}")

    row, defaulted = {}, []
    for name in FEATURES:
        v = body.values.get(name)
        if v is None:
            v = default_for(name)
            defaulted.append(name)
        row[name] = v
    df = pd.DataFrame([row], columns=FEATURES)
    if STATE["scaler"] is not None:
        df = pd.DataFrame(STATE["scaler"].transform(df), columns=FEATURES)

    b = run_predict(STATE["behav"], df)[0]
    c = run_predict(STATE["cir"], df)[0]
    return {
        "behaviour": {
            "label": decode(b, STATE["classes"]["behav"]),
            "probabilities": run_proba(STATE["behav"], df, STATE["classes"]["behav"]),
        },
        "circuit": {
            "label": decode(c, STATE["classes"]["cir"]),
            "probabilities": run_proba(STATE["cir"], df, STATE["classes"]["cir"]),
        },
        "defaulted_fields": defaulted,
    }


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")