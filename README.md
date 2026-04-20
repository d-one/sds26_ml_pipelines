# SDS 2026 / Mastering the ML Lifecycle on Databricks

This repository hosts the Databricks notebooks and supporting utilities for the SDS 2026 workshop "Mastering the ML Lifecycle on Databricks". The workshop focuses on the training of an XGBoost classifier using a synthetic dataset of 10,000 records, predicting whether a person drinks coffee or not.

The model's hyperparameters are tuned using Optuna, and the best performing model is registered in the Unity Catalog using MLFlow, and finally served as an endpoint to be used in real-time predictions.

## Repository Structure
- `notebooks/`: quest versions of each module used during the live quest, with fill-in-the-blanks and answer the question approach, using also optional hints. It also contains the AutoML and Data Exploration notebooks as additional information, should any participant want to experiment further with these parts.
- `LICENSE` and supporting project metadata.

## Environment Setup
1. We used a Databricks cluster with:
   1. Runtime 16.4 LTS (includes Apache Spark 3.5.2, Scala 2.12)
   2. Machine learning enabled
   3. Photon acceleration enabled
   4. Node type Standard_D4s_v3

## Notebook Flow
The solutions notebooks are meant to be run sequentially inside a Unity Catalog–enabled Databricks workspace:
1. `model_training.py` – Runs Optuna-driven hyperparameter search for `SparkXGBClassifier`, logs results to MLflow, and registers the champion model.
2. `predictions.py` – Scores the production holdout with the champion model, logs evaluation metrics, and saves a predictions table.

Participants are expected to complete the `_quest` notebooks, and if they get stuck or wish to proceed, the `_solution` notebooks is available to proceed to the next step.

## Notes
- Spark ML and XGBoost-on-Spark are the primary modeling engines; Databricks AutoML is leveraged to jump-start the modeling baseline before custom tuning.
- MLflow is configured to log to Unity Catalog (default setting in MLFlow Version 3 and above) throughout the workflow so the model registry, feature store, and serving endpoint share the same governance boundary.
