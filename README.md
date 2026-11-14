---
title: FlameGuard SI
emoji: 🔥
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# FlameGuard SI

**Live demo:** <https://flameguard-si.onrender.com>

A calibrated decision layer for spacecraft cabin fire safety. Enter a scenario —
material, O2 concentration, pressure, opposed-flow velocity, confinement — and get
a flammability verdict (`sustained_spread` / `marginal` / `extinction`), a
suppression-strategy category, and the labelled experimental runs closest to that
scenario, in well under a second.

The point is not raw accuracy. It is that every call carries a **calibrated
confidence**, and the model **abstains** (`needs_human_review`) when it is not
confident enough to act — whether because the probability is low, because the top
two classes are nearly tied, or because an input lies outside the range the model
was trained on.

## How it works

```
data/raw/*.csv ──> flameguard.train ──> artifacts/{model.joblib,features.json,metrics.json,validation.md}
                                              │
  static/index.html ──POST /api/predict──> flameguard.api ──> core scoring
```

- **Model** — `HistGradientBoostingClassifier` over discovered features, wrapped in
  `CalibratedClassifierCV(method="sigmoid")` (Platt scaling).
- **Calibration is conditional** — kept only if it beats the uncalibrated model on
  macro Brier score (`metrics.json → calibration_kept`).
- **Abstention** — withheld when `max_prob < tau_global`, when `(p1 - p2) <
  tau_margin`, or when a numeric input is outside the observed training range.
  `tau` is chosen for highest accuracy on the auto-clear subset among thresholds
  whose review rate falls in the 10–30% band, ties broken toward fewer deferrals.
- **Schema-driven** — `features.json` is inferred from the CSV (numeric vs
  categorical, categories, min/max/mean/std) and the API request model is
  generated from it at startup, so feature names are never hard-coded. The four
  *physical roles* the suppression lookup needs (O2, pressure, confinement, flow)
  are detected by name heuristic; if one is missing the strategy degrades to an
  "unknown" band instead of guessing.
- **Similar-run comparison** — the k nearest labelled runs in standardised feature
  space, shown with their actual recorded outcomes (feature columns only).
- **Suppression mapping** — deterministic lookup from published NASA fire-safety
  guidance. Explicitly **not learned**, and labelled as such in the UI.

## Run locally

The system interpreter is often PEP 668-locked and can lack `ensurepip`, so build
the venv without pip and bootstrap it. Keep the venv on the Linux-native
filesystem: importing scikit-learn from a Windows mount (`/mnt/c`) takes ~40 s
versus ~3 s natively.

```bash
python3 -m venv --without-pip ~/.venvs/flameguard
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
~/.venvs/flameguard/bin/python /tmp/get-pip.py
~/.venvs/flameguard/bin/python -m pip install -r requirements-dev.txt
```

```bash
python -m flameguard.train                       # writes artifacts/
python -m pytest                                 # 11 tests
python -m uvicorn flameguard.api:app --port 7860 # http://127.0.0.1:7860
```

## Use the real dataset

Drop the organizer CSV into `data/raw/` and retrain:

```bash
cp /path/to/organizer.csv data/raw/
python -m flameguard.train
```

`python -m flameguard.train --data /path/to/file.csv --target <column>` also works.
The target column is auto-detected from a list of common names, falling back to
the last column; labels are normalised to the three canonical classes and include
negations ("did not spread" → extinction) and "extinguished". If too few rows map
to a class the run **fails loudly** and lists the unmapped values rather than
training on a silently shrunken set.

If `data/raw/` is empty, the pipeline generates a **clearly-labelled SYNTHETIC**
placeholder so the service stays exercisable; that fact is written into
`metrics.json`, `validation.md` and shown as a warning badge in the UI.

## Deployment

The image is digest-pinned, runs as UID 1000, honours `$PORT` (falling back to
7860), and retrains at build time so the live service always matches the code.

### Render — free tier (`render.yaml` included)

1. Push this directory to GitHub, GitLab or Bitbucket — Render deploys from a
   Git provider, not from a local folder.
2. In Render: **New → Blueprint**, point it at the repository and apply.
   `render.yaml` sets `runtime: docker`, `plan: free`, `region: frankfurt` and
   `healthCheckPath: /api/health`.
   If you prefer the manual path: **New → Web Service → Docker**, health check
   path `/api/health`, leave the start command empty (the Dockerfile `CMD` is
   used).
3. First build takes roughly 3–5 minutes.

Free-tier behaviour worth knowing: 512 MB RAM (measured footprint at runtime:
**67 MB**), less than one CPU, and the service spins down after 15 minutes
without traffic, so the **first** request after a quiet period takes about a
minute. Subsequent requests are normal. A free uptime monitor pinging
`/api/health` every 10 minutes keeps it warm and still fits inside Render's 750
free instance-hours per month.

### Hugging Face Spaces — requires a paid plan

Docker Spaces are not available on the free HF plan ("creating a Space that runs
on compute (Gradio or Docker) requires a paid plan"). The README frontmatter is
already correct for it (`sdk: docker`, `app_port: 7860`), so with PRO:

```bash
git init && git add -A && git commit -m "FlameGuard SI"
git remote add space https://huggingface.co/spaces/<user>/<space-name>
git push space main
```

## Limits

See `artifacts/validation.md` for the per-class table, confusion matrix, Brier
scores and the full abstention trade-off. Validation is against the provided
labelled dataset only; the NTRS paper-extraction track was cut for time, so no
claim is made beyond that dataset. With no organizer CSV present, the shipped
metrics are synthetic and marked as such in the UI, `metrics.json` and
`validation.md`.