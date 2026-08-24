-- SQLite schema for the EuroSAT experiment tracker.
-- The schema is normalized: experiment data, models, metrics, and statistical
-- summaries are stored in separate tables and connected with foreign keys.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_name TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    data_path TEXT,
    mode TEXT NOT NULL,
    seed INTEGER NOT NULL,
    device TEXT,
    status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'failed')),
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS class_labels (
    class_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    class_index INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,
    UNIQUE (experiment_id, class_index),
    UNIQUE (experiment_id, class_name)
);

CREATE TABLE IF NOT EXISTS models (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    architecture TEXT NOT NULL,
    checkpoint_path TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,
    UNIQUE (experiment_id, model_name)
);

CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('train', 'validation', 'test')),
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES models(model_id)
        ON DELETE CASCADE,
    UNIQUE (model_id, split, metric_name)
);

CREATE TABLE IF NOT EXISTS statistical_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    condition_name TEXT NOT NULL,
    resnet_mean_pct REAL NOT NULL,
    resnet_ci_low_pct REAL,
    resnet_ci_high_pct REAL,
    vit_mean_pct REAL NOT NULL,
    vit_ci_low_pct REAL,
    vit_ci_high_pct REAL,
    cohens_d REAL,
    effect_size TEXT,
    p_value REAL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,
    UNIQUE (experiment_id, condition_name)
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    filename TEXT,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    threshold REAL NOT NULL CHECK (threshold >= 0 AND threshold <= 1),
    accepted INTEGER NOT NULL CHECK (accepted IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_models_experiment
    ON models(experiment_id);

CREATE INDEX IF NOT EXISTS idx_metrics_model_split
    ON model_metrics(model_id, split);

CREATE INDEX IF NOT EXISTS idx_stats_experiment_pvalue
    ON statistical_results(experiment_id, p_value);

CREATE INDEX IF NOT EXISTS idx_predictions_experiment_created
    ON predictions(experiment_id, created_at DESC);
