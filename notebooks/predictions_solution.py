# Databricks notebook source
# MAGIC %md
# MAGIC # Predictions in Production
# MAGIC
# MAGIC In this notebook, the goal is to:
# MAGIC 1. Load the registered model
# MAGIC 2. Use it to make predictions on unseen data
# MAGIC 3. Evaluate its performance

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

exp_id = init_experiment(MLFLOW_EXPERIMENT_NAME)

# COMMAND ----------

# DBTITLE 1,Load Holdout table
HOLDOUT_TABLE_PATH = f"{CATALOG}.{SCHEMA_WITH_SOURCE_DATA}.coffee_prod_holdout"
holdout_df = spark.table(HOLDOUT_TABLE_PATH)
print(f"Loaded holdout table with {holdout_df.count():,} rows.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 1 · Load the model
# MAGIC
# MAGIC It is time to load the model that we trained in the previous notebook and use it to make predictions on unseen data!
# MAGIC
# MAGIC Can you complete the missing alias to load the model?
# MAGIC
# MAGIC [Databricks Documentation for Model Aliases](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/#use-model-aliases)
# MAGIC

# COMMAND ----------

# DBTITLE 1,Load hint for Quest 1
load_hint("predictions", "quest_1")

# COMMAND ----------

# DBTITLE 1,Load the model
MODEL_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_xgb_model"
alias = "@champion"  # The 'champion' tag is assigned to the best model
CHAMPION_MODEL_URI = f"models:/{MODEL_PATH}{alias}"

model = mlflow.spark.load_model(CHAMPION_MODEL_URI)

# COMMAND ----------

# DBTITLE 1,Get predictions
predictions_df = model.transform(holdout_df)
display(predictions_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 2 · Understanding the result
# MAGIC
# MAGIC You used the model to make predictions on the holdout dataset.
# MAGIC
# MAGIC **1.** What columns where added by the model?
# MAGIC
# MAGIC **2.** Could you explain what they are?
# MAGIC

# COMMAND ----------

# DBTITLE 1,Load hint for Quest 2
load_hint("predictions", "quest_2")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 3 · Evaluate the model's performance with mlflow
# MAGIC
# MAGIC Run the following cells and answer the questions:
# MAGIC
# MAGIC 1. Where are the results logged?
# MAGIC
# MAGIC 2. Which metrics where logged?
# MAGIC

# COMMAND ----------

# DBTITLE 1,Load hint for Quest 3
load_hint("predictions", "quest_3")

# COMMAND ----------

# DBTITLE 1,mlflow evaluation
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


# COMMAND ----------

# MAGIC %md
# MAGIC ## Save predictions

# COMMAND ----------

# DBTITLE 1,Save predictions table
PREDICTIONS_TABLE_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_predictions"
predictions_df.write.format("delta").mode("overwrite").saveAsTable(PREDICTIONS_TABLE_PATH)
print(f"DataFrame predictions_df has been written to table:\n\t- {PREDICTIONS_TABLE_PATH}")
