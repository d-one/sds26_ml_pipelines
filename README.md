# End-to-End ML Workshop: "Do You Like Coffee?"

This repository hosts the Databricks notebooks and supporting utilities for an end-to-end machine learning workshop. Participants explore the Global Coffee Health dataset, engineer features, tune an XGBoost model, and finish by deploying the classifier behind a Databricks Model Serving endpoint.

Coffee Dataset: https://www.kaggle.com/datasets/uom190346a/global-coffee-health-dataset  
GNI Dataset: https://data.worldbank.org/indicator/NY.GNP.PCAP.PP.CD

## Repository Structure
- `quest_notebooks/`: gamified, fill-in-the-blanks versions of each module used during the live quest.
- `solution_notebooks/`: completed reference implementations for every module.
- `LICENSE` and supporting project metadata.

## Environment Setup
1. We used a Databricks cluster with:
   1. Runtime 16.4 LTS (includes Apache Spark 3.5.2, Scala 2.12)
   2. Machine learning enabled
   3. Photon acceleration enabled
   4. Node type Standard_D4ds_v4

## Notebook Flow
The notebooks are meant to be run sequentially inside a Unity Catalog–enabled Databricks workspace:
1. `00_setup.py` – Detects the current user, provisions their personal schema, and exposes helper utilities (scoreboard, feature-name mapping, etc.).
2. `01_data_ingest.py` – Cleans the Kaggle dataset, aligns it with the World Bank GNI table, builds the labeled training set, and creates a production holdout.
3. `02_eda.py` – Generates descriptive statistics, a ydata-profiling report, and seaborn/matplotlib visualizations for the labeled dataset.
4. `03_feature_store.py` – Builds a Feature Engineering table as the foundation for downstream training.
5. `04_automl_model.py` – Uses Databricks AutoML to baseline a classifier and log artifacts to Unity Catalog.
6. `05_optuna.py` – Runs Optuna-driven hyperparameter search for `SparkXGBClassifier`, logs results to MLflow, and registers the winning model.
7. `06_predictions_simulation.py` – Scores the production holdout with the champion model, logs evaluation metrics, and persists predictions.
8. `07_model_serving.py` – Provisions/updates a Databricks Model Serving endpoint and demonstrates querying it through the MLflow Deployments SDK.

Switch between `quest_notebooks/` and `solution_notebooks/` depending on whether you want the guided exercise or the final answer key.

## Running the Workshop
1. Import either the quest or solution notebooks into your Databricks workspace (drag-and-drop or `databricks workspace import_dir`).
2. Attach each notebook to a DBR ML cluster that has Unity Catalog access and the source tables mentioned above.
3. Execute the modules in order, starting from `01_data_ingest`, verifying the checkpoints and MLflow runs at each stage.


## Notes
- Spark ML and XGBoost-on-Spark are the primary modeling engines; Databricks AutoML is leveraged to jump-start the modeling baseline before custom tuning.
- MLflow is configured to log to Unity Catalog throughout the workflow so the model registry, feature store, and serving endpoint share the same governance boundary.
