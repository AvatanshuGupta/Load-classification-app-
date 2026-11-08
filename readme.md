# Load Classifier

A FastAPI web app that serves two custom random-forest models and shows how well they perform.

- **Behaviour classifier** predicts one of three behaviour classes (CZ, CP, CC).
- **Circuit classifier** predicts one of seven circuit types (R, L, C, RL, RC, LC, RLC).

The page has a form for the 22 electrical measurements, a prediction panel, and a model performance section (accuracy, macro F1, training time, confusion matrix, per-class precision, recall and F1).

## Features

- **Placeholder inputs:** every field shows the training-set median as grey placeholder text. Leave a field empty and that median is used.
- **Load example:** fills the form with a real row from the training data.
- **Probability bars:** shown when the model implements `predict_proba`.
- **Metrics dashboard:** reads `results/performance.json`, with a tab for each task.
- **Dark and light mode:** follows the system setting.

## Project structure

```
load-classifier/
├── app.py                 FastAPI server and API
├── model.py               your RandomForest class (copy from training project)
├── score.py               your scoring helpers (not needed at runtime)
├── prepare_deploy.py      optional: builds the placeholder JSON files
├── requirements.txt
├── Dockerfile             optional
├── static/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── results/
│   ├── rf_behaviour-*.pkl
│   ├── rf_circuit-*.pkl
│   └── performance.json
└── artifacts/
    ├── le_behav.pkl       label encoders (required)
    ├── le_cir.pkl
    ├── X_train.pkl        used for placeholders and examples
    ├── y_behav_train.pkl
    ├── y_cir_train.pkl
    ├── feature_stats.json optional (replaces the three files above)
    └── samples.json       optional
```

`model.py` must be in the project root. The saved model files refer to `model.RandomForest`, so they cannot be loaded without it.

## Quick start

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://localhost:8000.

To make it reachable from other devices on your network:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Input features

| Group | Fields |
|---|---|
| Overall measurements | `Vrms`, `Irms`, `P`, `Q1`, `S`, `PF`, `phi_deg`, `Zmag` |
| Distortion | `THD_V`, `THD_I` |
| Stage L0 to L3 | `V_L0`, `I_L0`, `P_L0`, `V_L1`, `I_L1`, `P_L1`, `V_L2`, `I_L2`, `P_L2`, `V_L3`, `I_L3`, `P_L3` |

The units shown in the form are guesses from the column names. Edit `FEATURE_META` in `app.py` to change them.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/` | Web page |
| GET | `/api/health` | Health check |
| GET | `/api/features` | Feature names, groups, units, defaults and training ranges |
| GET | `/api/samples` | Example rows with their true labels |
| GET | `/api/metrics` | Contents of `performance.json` with corrected class labels |
| POST | `/api/predict` | Classify one measurement |

Example request. Fields you leave out are filled with the training median:

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"values": {"Vrms": 230, "Irms": 1.2, "PF": 0.95}}'
```

Example response:

```json
{
  "behaviour": {"label": "CZ", "probabilities": {"CZ": 0.84, "CP": 0.10, "CC": 0.06}},
  "circuit":   {"label": "RL", "probabilities": null},
  "defaulted_fields": ["P", "Q1", "..."]
}
```

`probabilities` is `null` if your `RandomForest` has no `predict_proba`. Interactive API docs are at `/docs`.

## Configuration

All optional.

| Environment variable | Default | Purpose |
|---|---|---|
| `RESULTS_DIR` | `./results` | Folder with the model files and `performance.json` |
| `ARTIFACTS_DIR` | `./artifacts` | Folder with encoders and training data |
| `BEHAV_MODEL` | newest `rf_behaviour-*.pkl` | Path to a specific behaviour model |
| `CIR_MODEL` | newest `rf_circuit-*.pkl` | Path to a specific circuit model |

## Faster, lighter startup

By default the app loads `X_train.pkl` at startup to work out the placeholder values. To skip that, run this once from the original, unscaled dataset:

```bash
python prepare_deploy.py --csv path/to/data.csv
```

It writes `artifacts/feature_stats.json` and `artifacts/samples.json`. When both exist, the app uses them and you can leave `X_train.pkl` and the `y_*_train.pkl` files out of the deployment. Keep the two label encoders.

## Label order

Class names come from `le_behav.pkl` and `le_cir.pkl`, so predictions always use the encoders' order.

For confusion matrices saved by the old `train.py`, the app undoes two effects: the matrix was labelled with the hardcoded `BEHAV_CLASS` and `CIR_CLASS` lists, and `pd.crosstab` then sorted rows and columns alphabetically. This only applies when the saved class list matches those old lists exactly.

If you rerun evaluation with `le.classes_` passed to `confusion()` and saved as `classes` in `performance.json`, the app leaves the matrix as it is.

## Scaling

If you scaled the features before training, save that scaler as `artifacts/scaler.pkl`. The app then applies it to every input and shows placeholders in the original units. Without a saved scaler, the app assumes your models were trained on unscaled features.

## Deployment

- **Any VM or VPS:** run `uvicorn app:app --host 0.0.0.0 --port 8000` behind a process manager such as systemd.
- **Docker:** `docker build -t load-classifier .` then `docker run -p 8000:8000 load-classifier`.
- **Render, Railway and similar:** build command `pip install -r requirements.txt`, start command `uvicorn app:app --host 0.0.0.0 --port $PORT`.

Free tiers that run a container usually sleep after a period of inactivity. Use the same Python, `numpy`, `pandas` and `scikit-learn` versions you trained with, and pin them in `requirements.txt`.

## Security

The model and encoder files are loaded with `joblib`, which can execute code when unpickling. Only deploy files you created yourself, and never accept uploaded model files.

## Troubleshooting

| Problem | Likely cause |
|---|---|
| `ModuleNotFoundError: model` on startup | `model.py` is not in the project root |
| `No model matching rf_behaviour-*.pkl` | Model files are missing from `results/`, or `RESULTS_DIR` is wrong |
| Placeholders all show `0` | `artifacts/` has neither the JSON files nor a readable `X_train.pkl`; check the terminal for the "Could not build defaults" message |
| "Metrics unavailable" on the page | `results/performance.json` is missing |
| Wrong predictions that look random | Training used scaled features but no `scaler.pkl` was saved |
| Pickle load errors | Library versions differ from those used in training |