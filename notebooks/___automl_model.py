# Databricks notebook source
# MAGIC %md
# MAGIC ### AutoML
# MAGIC
# MAGIC AutoML (Automated Machine Learning) simplifies the process of building machine learning models by automating tasks such as data preprocessing, feature engineering, model selection, and hyperparameter tuning. This enables users to quickly develop high-quality models with minimal manual intervention, making machine learning accessible to both experts and non-experts.
# MAGIC
# MAGIC Here, we employ Databricks AutoML to quickly get a baseline model and find a suited model type for the current setting.
# MAGIC
# MAGIC To save some time, we already ran an experiment for you:
# MAGIC
# MAGIC https://adb-1451829595406012.12.azuredatabricks.net/ml/experiments/1042389076180991
# MAGIC

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run ./___setup

# COMMAND ----------

# DBTITLE 1,Load Feature table
training_data_df = spark.table(
    "gtc25_ml_catalog.source_data.coffee_labeled_features"
)

# COMMAND ----------

# DBTITLE 1,AutoML
import databricks.automl as automl

# AutoML parameters
AUTOML_EXPERIMENT_DIRECTORY = (
    f"/Workspace/Users/{USER_EMAIL}/automl_experiments/"
)
EXPERIMENT_NAME = f"coffee_automl_{MY_SCHEMA}_2"

print("-" * 100)
print("Creating AutoML experiment")
print(f"\t-Experiment Directory:\t{AUTOML_EXPERIMENT_DIRECTORY}")
print(f"\t-Experiment Name:\t{EXPERIMENT_NAME}")
print("-" * 100, "\n")

summary = automl.classify(
    dataset=training_data_df,
    target_col="Coffee_Drinker",
    primary_metric="f1",
    experiment_dir=AUTOML_EXPERIMENT_DIRECTORY,
    experiment_name=EXPERIMENT_NAME,
    exclude_cols=["ID", "Timestamp"],
    pos_label="0",
    timeout_minutes=5,  # Default value 120 minutes, minimum 5 minutes
)
