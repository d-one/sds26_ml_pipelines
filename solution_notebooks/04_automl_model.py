# Databricks notebook source
# MAGIC %md
# MAGIC ### AutoML
# MAGIC
# MAGIC AutoML (Automated Machine Learning) simplifies the process of building machine learning models by automating tasks such as data preprocessing, feature engineering, model selection, and hyperparameter tuning. This enables users to quickly develop high-quality models with minimal manual intervention, making machine learning accessible to both experts and non-experts.
# MAGIC
# MAGIC Here, we employ Databricks AutoML to quickly get a baseline model and find a suited model type for the current setting.

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run ./00_setup

# COMMAND ----------

# DBTITLE 1,Imports
import databricks.automl as automl
from datetime import date

# COMMAND ----------

# DBTITLE 1,Constants
# Table paths
FEATURE_TABLE_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_features"

# Columns
PRIMARY_KEY_COL = "ID"
TIMESTAMP_COL = "Timestamp"
LABEL_COL = "Coffee_Intake_Binary"

# AutoML parameters
AUTOML_TIMEOUT_MINUTES = 10
AUTOML_EXPERIMENT_DIRECTORY = f"/Workspace/Users/{USER_EMAIL}/automl_experiments/"

today = date.today().strftime("%Y_%m_%d")
EXPERIMENT_NAME =f"coffee_automl_{today}"

# COMMAND ----------

training_data_df = spark.table(FEATURE_TABLE_PATH)

# COMMAND ----------

# DBTITLE 1,AutoML
summary = automl.classify(
    dataset=training_data_df,
    target_col=LABEL_COL,
    primary_metric="f1",
    # data_dir=  = None,
    experiment_dir=AUTOML_EXPERIMENT_DIRECTORY,
    experiment_name=EXPERIMENT_NAME,
    exclude_cols=[PRIMARY_KEY_COL, TIMESTAMP_COL],
    # exclude_frameworks=None,
    # feature_store_lookups=None,
    # imputers=None,
    pos_label="0",
    # time_col=None,
    # split_col=None,
    # sample_weight_col=None,
    # max_trials=None,
    timeout_minutes=AUTOML_TIMEOUT_MINUTES,  # Default value 120 minutes
)

# COMMAND ----------

print(f"Best trial notebook: {summary.best_trial.notebook_path}")
