# SQL and DBMS integration

This project now stores experiment metadata and evaluation results in a local
SQLite database. The integration is intentionally dependency-free: Python's
standard-library `sqlite3` module is enough.

## Files

- `schema.sql`: normalized tables, constraints, and indexes.
- `dbms.py`: database access layer with transactions and parameterized SQL.
- `dbms_demo.py`: reporting and `final_stats.csv` import example.
- `train.py`: creates an experiment row and stores ResNet/ViT test metrics.
- `app.py`: FastAPI website and prediction API that reads/writes the SQLite database.
- `keep_experiment.py`: one-time cleanup utility for selecting the best run.
- `templates/dashboard.html` and `static/styles.css`: dashboard UI.

## Run it

```powershell
python train.py
python dbms_demo.py
```

After training, `experiments.db` contains:

```text
experiments  1 ---- many  class_labels
experiments  1 ---- many  models  1 ---- many  model_metrics
experiments  1 ---- many  statistical_results
```

Import the existing statistical CSV into experiment 1 with:

```powershell
python dbms_demo.py --import-stats 1 final_stats.csv
```

## Open the website

Install Flask once:

```powershell
python -m pip install -r requirements_web.txt
```

Start the local website:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 5000 --reload
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. The
dashboard uses parameterized SQL queries to display experiment history, model
metrics, imported robustness results, and prediction history. The prediction
form accepts an image, model selection, and confidence threshold. Predictions
are written to the `predictions` table. The JSON view is available at
`http://127.0.0.1:5000/api/experiments/2`, and the interactive API docs are at
`http://127.0.0.1:5000/docs`.

The code demonstrates normalization, primary/foreign keys, `CHECK` and
`UNIQUE` constraints, indexes, transactions, parameterized queries, and a
`JOIN` for the model leaderboard.

To keep only the best run, experiment #2:

```powershell
python keep_experiment.py --keep 2
```
