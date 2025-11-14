# Databricks notebook source
# MAGIC %md
# MAGIC # Coffee Module 04 · AutoML Baseline
# MAGIC Configure and launch the updated Databricks AutoML run. Replace every `...` placeholder with real code before executing each quest.

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run ./00_setup

# COMMAND ----------

# DBTITLE 1,Imports
from datetime import date

import databricks.automl as automl

# COMMAND ----------

# DBTITLE 1,Constants
FEATURE_TABLE_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_features"
PRIMARY_KEY_COL = "ID"
TIMESTAMP_COL = "Timestamp"
LABEL_COL = "Coffee_Intake_Binary"
AUTOML_TIMEOUT_MINUTES = 10
AUTOML_EXPERIMENT_DIRECTORY = (
    f"/Workspace/Users/{USER_EMAIL}/automl_experiments/"
)
EXPERIMENT_NAME = f"coffee_automl_{date.today().strftime('%Y_%m_%d')}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 1 · Load the Feature Table
# MAGIC **Goal:** assign `training_data_df` from the Feature Store table and preview a sample.
# MAGIC
# MAGIC **Hints**
# MAGIC - Use `spark.table(FEATURE_TABLE_PATH)`.
# MAGIC - Call `display(training_data_df.limit(10))` to preview features and labels.

# COMMAND ----------

training_data_df = spark.table(...)                                                                 # replace placeholder
display(training_data_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 2 · Run the AutoML Classification Job
# MAGIC **Goal:** call `automl.classify` with the refreshed parameters, including the experiment directory and name.
# MAGIC
# MAGIC **Hints**
# MAGIC - Replace the ellipses for `dataset` and `target_col` with `training_data_df` and `LABEL_COL`.
# MAGIC - Keep `experiment_dir=AUTOML_EXPERIMENT_DIRECTORY` and `experiment_name=EXPERIMENT_NAME`.
# MAGIC - Exclude `[PRIMARY_KEY_COL, TIMESTAMP_COL]` and set `pos_label="0"` so metrics focus on non-drinkers.

# COMMAND ----------

summary = automl.classify(
    dataset=...,                                                                                    # replace placeholders
    target_col=...,
    primary_metric="...",
    experiment_dir=AUTOML_EXPERIMENT_DIRECTORY,
    experiment_name=EXPERIMENT_NAME,
    exclude_cols=[PRIMARY_KEY_COL, TIMESTAMP_COL],
    pos_label="...",
    timeout_minutes=AUTOML_TIMEOUT_MINUTES,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 3 · Review the Best Trial Notebook
# MAGIC **Goal:** print the notebook path of the best trial for follow-up inspection.
# MAGIC

# COMMAND ----------

print(f"Best trial notebook: {summary.best_trial.notebook_path}")
