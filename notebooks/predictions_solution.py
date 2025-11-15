# Databricks notebook source
# MAGIC %md
# MAGIC # Simulation of predictions in production
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Loads the time-based scoring holdout prepared in `03_feature_store`.
# MAGIC 2. Simulates receiving fresh entity keys and performs Feature Store lookups.
# MAGIC 3. Scores the holdout rows with the registered XGBoost model and optionally evaluates performance.

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

# DBTITLE 1,Experiment setup
# Experiment name
MLFLOW_EXPERIMENT_NAME = f"/Workspace/Users/{USER_EMAIL}/coffee_prod_predictions"

# Disable autologging (because it can create extra experiments when fitting models in final run)
mlflow.autolog(disable=True)

exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
if exp is None:
    # If it doesn't exist, create it
    exp_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# COMMAND ----------

# DBTITLE 1,Load Holdout table
HOLDOUT_TABLE_PATH = f"{CATALOG}.{SCHEMA_WITH_SOURCE_DATA}.coffee_prod_holdout"
holdout_df = spark.table(HOLDOUT_TABLE_PATH)
print(f"Loaded holdout table with {holdout_df.count():,} rows.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predictions & Evaluation

# COMMAND ----------

# DBTITLE 1,Get predictions
# Load Spark pipeline **on the driver** and score with transform (no UDF)
MODEL_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_xgb_model"
CHAMPION_MODEL_URI = f"models:/{MODEL_PATH}@champion"

spark_model = mlflow.spark.load_model(CHAMPION_MODEL_URI)
predictions_df = spark_model.transform(holdout_df)

display(predictions_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## What columns did the prediction generate? Can you explain what they are?
# MAGIC
# MAGIC ### Code hint to show columns before, after, and explain results

# COMMAND ----------

# DBTITLE 1,MLFlow evaluation
client = MlflowClient()

version = client.get_model_version_by_alias(name=MODEL_PATH, alias="champion")

run_name = f"coffee_xgb_model_v{version.version}_evaluation"

# Log the baseline model to MLflow
with mlflow.start_run(run_name=run_name) as run:

    # Evaluate the logged model
    results = mlflow.models.evaluate(
        CHAMPION_MODEL_URI,
        holdout_df,
        targets="Coffee_Intake_Binary",
        model_type="classifier",
        evaluators=["default"],
        evaluator_config={"default": {"pos_label": 0}}
    )

print("\n\n\n")
print("-" * 100)
print(f"Results saved in {MLFLOW_EXPERIMENT_NAME}, in Run {run_name} :")
for key, value in results.metrics.items():
    print(f"\t{key}: {round(value, 2)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save predictions

# COMMAND ----------

# DBTITLE 1,Save predictions
PREDICTIONS_TABLE_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_predictions"
predictions_df.write.format("delta").mode("overwrite").saveAsTable(PREDICTIONS_TABLE_PATH)
print(f"DataFrame predictions_df has been written to table:\n\t- {PREDICTIONS_TABLE_PATH}")
