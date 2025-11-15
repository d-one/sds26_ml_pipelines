# Databricks notebook source
# MAGIC %md
# MAGIC # 06 – Coffee Intake Predictions Simulation
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
# MAGIC %run ./00_setup

# COMMAND ----------

# DBTITLE 1,Imports
import mlflow
# from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import functions as F
from mlflow import MlflowClient

# COMMAND ----------

# DBTITLE 1,Constants
# Table paths
HOLDOUT_TABLE_PATH = f"{CATALOG}.{SCHEMA_WITH_SOURCE_DATA}.coffee_prod_holdout"
PREDICTIONS_TABLE_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_predictions"

# Columns
PRIMARY_KEY_COL = "ID"
TIMESTAMP_COL = "Timestamp"
LABEL_COL = "Coffee_Intake_Binary"
PREDICTION_COLUMN = "prediction"
KEY_COLUMNS = [PRIMARY_KEY_COL, TIMESTAMP_COL]

# mlflow parameters
MODEL_NAME = "coffee_xgb_model"
MODEL_PATH = f"{CATALOG}.{MY_SCHEMA}.{MODEL_NAME}"
CHAMPION_MODEL_URI = f"models:/{MODEL_PATH}@champion"
MLFLOW_EXPERIMENT_NAME = f"/Workspace/Users/{USER_EMAIL}/coffee_prod_predictions"

# COMMAND ----------

# DBTITLE 1,Experiment setup
mlflow.set_registry_uri("databricks-uc")
# Disable autologging (because it can create extra experiments when fitting models in final run)
mlflow.autolog(disable=True)

exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
if exp is None:
    # If it doesn't exist, create it
    exp_id = mlflow.create_experiment(
        MLFLOW_EXPERIMENT_NAME
    )  # parent already exists

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# COMMAND ----------

# DBTITLE 1,Load Scoring Holdout and Features
holdout_df = spark.table(HOLDOUT_TABLE_PATH)
print(f"Loaded holdout table with {holdout_df.count():,} rows.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predictions & Evaluation

# COMMAND ----------

# DBTITLE 1,Get predictions
# Load Spark pipeline **on the driver** and score with transform (no UDF)
spark_model = mlflow.spark.load_model(CHAMPION_MODEL_URI)
predictions_df = spark_model.transform(holdout_df)

display(predictions_df.select(*KEY_COLUMNS, PREDICTION_COLUMN).limit(10))

# COMMAND ----------

# DBTITLE 1,Alternative way: works with native models logged with fe
# LINEAGE_MODEL_URI = f"models:/{MODEL_PATH}@lineage"
# fe = FeatureEngineeringClient()
# predictions_df = fe.score_batch(
#     model_uri=LINEAGE_MODEL_URI,
#     df=scoring_keys_df
# )

# print("Sample predictions enriched with features:")
# display(predictions_df.select(*KEY_COLUMNS, PREDICTION_COLUMN).limit(10))

# COMMAND ----------

# DBTITLE 1,MLFlow evaluation
client = MlflowClient()

version = client.get_model_version_by_alias(
    name=MODEL_PATH, alias="champion"
)

run_name = f"{MODEL_NAME}_v{version.version}_evaluation"

with mlflow.start_run(run_name=run_name) as run:
    # Log the baseline model to MLflow

    # Evaluate the logged model
    results = mlflow.models.evaluate(
        CHAMPION_MODEL_URI,
        holdout_df,
        targets=LABEL_COL,
        model_type="classifier",
        evaluators=["default"],
        evaluator_config={"default": {"pos_label": 0}}
    )

print(f"\nResults saved in {MLFLOW_EXPERIMENT_NAME}, to run {run_name} :")
for key, value in results.metrics.items():
    print(f"\t{key}: {round(value, 2)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save predictions

# COMMAND ----------

# DBTITLE 1,Persist predictions
predictions_df.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(PREDICTIONS_TABLE_PATH)
print(
    f"DataFrame predictions_df has been written to table:\n\t- {PREDICTIONS_TABLE_PATH}"
)

# COMMAND ----------

predictions_df.display()
