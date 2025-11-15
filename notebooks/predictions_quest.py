# Databricks notebook source
# MAGIC %md
# MAGIC # Coffee Module 06 · Prediction Simulation
# MAGIC This module now reads the production holdout created in Module 01 and scores it directly with the registered XGBoost model. Fill in the `...` placeholders in each quest before running the cell.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialization

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run ./___setup

# COMMAND ----------

# DBTITLE 1,Imports
import mlflow
from mlflow import MlflowClient

# COMMAND ----------

# DBTITLE 1,Constants
HOLDOUT_TABLE_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_prod_holdout"
PREDICTIONS_TABLE_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_prod_predictions"

PRIMARY_KEY_COL = "ID"
TIMESTAMP_COL = "Timestamp"
LABEL_COL = "Coffee_Intake_Binary"
PREDICTION_COLUMN = "prediction"
KEY_COLUMNS = [PRIMARY_KEY_COL, TIMESTAMP_COL]

MODEL_NAME = "coffee_xgb_model"
MODEL_PATH = f"{CATALOG}.{MY_SCHEMA}.{MODEL_NAME}"
CHAMPION_MODEL_URI = f"models:/{MODEL_PATH}@champion"
MLFLOW_EXPERIMENT_NAME = (
    f"/Workspace/Users/{USER_EMAIL}/coffee_prod_predictions"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 1 · Configure MLflow and Load the Holdout
# MAGIC **Goal:** point MLflow to Unity Catalog, ensure the experiment exists, and load the holdout DataFrame.
# MAGIC
# MAGIC **Hints**
# MAGIC - Replace the `...` inside `mlflow.set_registry_uri("...")` with `"databricks-uc"`.
# MAGIC - Fill the `spark.table(...)` placeholder with `HOLDOUT_TABLE_PATH` and print the resulting count.

# COMMAND ----------

mlflow.set_registry_uri("...")                                                 # replace placeholder
mlflow.autolog(disable=True)

exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
if exp is None:
    exp_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
else:
    exp_id = exp.experiment_id
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

holdout_df = spark.table(...)                                                  # replace placeholder
holdout_count = holdout_df.count()
print(f"Loaded holdout table with {holdout_count:,} rows.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 2 · Score the Holdout
# MAGIC **Goal:** load the champion model and generate predictions in Spark.
# MAGIC
# MAGIC **Hints**
# MAGIC - Swap the `...` in `mlflow.spark.load_model(...)` with `CHAMPION_MODEL_URI`.
# MAGIC - Predict the `holdout_df`

# COMMAND ----------

spark_model = mlflow.spark.load_model(...)                                      # replace placeholders
predictions_df = spark_model.transform(...)
display(predictions_df.select(*KEY_COLUMNS, PREDICTION_COLUMN).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 3 · Log the MLflow Evaluation
# MAGIC **Goal:** run `mlflow.models.evaluate` on the champion model using the holdout features and labels.
# MAGIC
# MAGIC **Hints**
# MAGIC - Fill the alias placeholder with `"champion"` when calling `get_model_version_by_alias`.
# MAGIC - Reuse `CHAMPION_MODEL_URI` and the holdout DataFrame inside `mlflow.models.evaluate`.

# COMMAND ----------

client = MlflowClient()
version = client.get_model_version_by_alias(name=MODEL_PATH, alias="...")        # replace placeholder
run_name = f"{MODEL_NAME}_v{version.version}_evaluation"

with mlflow.start_run(run_name=run_name) as run:
    results = mlflow.models.evaluate(                                            # replace placeholders
        ...,  # model uri
        ...,  # df to be evaluated
        targets=LABEL_COL,
        model_type="classifier",
        evaluators=["default"],
        evaluator_config={"default": {"pos_label": 0}},
    )

print(f"\nResults saved in {MLFLOW_EXPERIMENT_NAME}, to run {run_name} :")
for key, value in results.metrics.items():
    print(f"\t{key}: {round(value, 2)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 4 · Persist the Predictions
# MAGIC **Goal:** save `predictions_df` to Unity Catalog for downstream reporting.

# COMMAND ----------

# DBTITLE 1,Persist predictions
predictions_df.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(PREDICTIONS_TABLE_PATH)
print(
    f"DataFrame predictions_df has been written to table:\n\t- {PREDICTIONS_TABLE_PATH}"
)
