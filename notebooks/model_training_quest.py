# Databricks notebook source
# MAGIC %md
# MAGIC # Training & Registering the model
# MAGIC Time to train and register the model!
# MAGIC
# MAGIC Work through the refreshed Optuna + MLflow workflow in quest form. Replace every `...` placeholder with real code before executing each quest.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialization

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run ./___setup

# COMMAND ----------

# DBTITLE 1,Imports
import mlflow
import mlflow.spark
import optuna
import pandas as pd
from databricks.feature_engineering import (
    FeatureEngineeringClient,
    FeatureLookup,
)
from mlflow import MlflowClient
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import functions as F
from pyspark.sql import types as T
from xgboost.spark import SparkXGBClassifier

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 1 · Define Feature Categories
# MAGIC **Goal:** detect numeric and categorical columns from the labeled dataset.
# MAGIC - Inspect the results.
# MAGIC - Are the contents of the variable lists accurate?
# MAGIC - Would you change anything?
# MAGIC
# MAGIC Need a nudge? Use the hint loader below.

# COMMAND ----------

# DBTITLE 1,Load hint for Quest 1
load_hint("model_training", "quest_1")

# COMMAND ----------

# DBTITLE 1,Feature categories
BASE_COLUMNS = ["Coffee_Intake_Binary", "ID", "Timestamp"]
# Keep label + identifiers out of the auto-generated feature lists
coffee_labeled_df = spark.table(f"{CATALOG}.{SCHEMA_WITH_SOURCE_DATA}.coffee_labeled_features")
data_schema_fields = coffee_labeled_df.schema.fields

numeric_cols = [
    field.name
    for field in data_schema_fields
    if field.dataType
    in [T.IntegerType(), T.DoubleType(), T.LongType(), T.FloatType()]
    and field.name not in BASE_COLUMNS
]

categorical_cols = [
    field.name
    for field in data_schema_fields
    if field.dataType == T.StringType() and field.name not in BASE_COLUMNS
]

all_feature_cols = numeric_cols + categorical_cols

print("Detected numeric columns:\n\t", numeric_cols)
print("\nDetected categorical columns:\n\t", categorical_cols)
coffee_labeled_df.limit(10).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 2 · Prepare the Feature Store Training Set
# MAGIC **Goal:** load the fact table and build a Feature Store training set that joins in features.
# MAGIC
# MAGIC You only need to replace the `...` placeholders.
# MAGIC
# MAGIC Need a nudge? Use the hint loader below.

# COMMAND ----------

# DBTITLE 1,Load hint for Quest 2
load_hint("model_training", "quest_2")

# COMMAND ----------

# DBTITLE 1,Loading the training set
fe = FeatureEngineeringClient()
labeled_fact_df = spark.table(f"{CATALOG}.{SCHEMA_WITH_SOURCE_DATA}.coffee_labeled_fact")

# Compose a Feature Lookup that pulls engineered columns by key + timestamp
feature_lookup = FeatureLookup(
    table_name=f"{CATALOG}.{SCHEMA_WITH_SOURCE_DATA}.coffee_labeled_features",
    feature_names=...,  # TODO replace placeholder
    lookup_key="ID",
    timestamp_lookup_key="Timestamp",
)

# Materialize a Feature Store training set with lookups and our label
training_set = fe.create_training_set(
    df=labeled_fact_df,
    feature_lookups=[...],  # TODO replace placeholder
    label="Coffee_Intake_Binary",
)

full_labeled_df = training_set.load_df()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 3 · Configure Splits, Pipeline, and MLflow
# MAGIC **Goal:** create data splits, build preprocessing stages, and configure the MLflow experiment.
# MAGIC
# MAGIC What do you thing is a good split?
# MAGIC
# MAGIC You only need to replace the `...` placeholders.
# MAGIC
# MAGIC Need a nudge? Use the hint loader below.

# COMMAND ----------

# DBTITLE 1,Load hint for Quest 3
load_hint("model_training", "quest_3")

# COMMAND ----------

# DBTITLE 1,Dataset splits & pipeline definition
train_df, valid_df, test_df = full_labeled_df.randomSplit(
    [..., ..., ...], seed=42  # TODO: replace placeholders
)
for split_name, split_df in [
    ("train", train_df),
    ("validation", valid_df),
    ("test", test_df),
]:
    print(f"{split_name.title()} split: {split_df.count():,} rows")

# Assemble preprocessing steps once so every model trial reuses them
STAGES = build_preprocessing_stages(categorical_cols, numeric_cols)

exp_id = init_experiment(f"/Workspace/Users/{USER_EMAIL}/coffee_hp_tuning_experiment")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 4 · Execute the Optuna Study
# MAGIC **Goal:** run Optuna with a head start!
# MAGIC
# MAGIC What starting parameters should we use, if any at all? How would you decide?
# MAGIC
# MAGIC The only thing you need to do is to fill the `seed_params` dictionary.

# COMMAND ----------

# DBTITLE 1,Load hint for Quest 4
load_hint("model_training", "quest_4")

# COMMAND ----------

# DBTITLE 1,Hyperparameter tuning
optuna.logging.set_verbosity(optuna.logging.ERROR)


# We wrap our model training in an objective function. This is a Trial
def objective(trial: optuna.Trial) -> float:
    run_name = (
        "seed_parameters_trial"
        if trial.number == 0
        else f"optuna_trial_{trial.number:03d}"
    )

    with mlflow.start_run(
        run_name=run_name,
        nested=True,
        tags={"mlflow.parentRunId": parent_run.info.run_id},
    ):
        params = {
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1.0, 10.0
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
        }

        xgb = SparkXGBClassifier(
            "Coffee_Intake_Binary"="Coffee_Intake_Binary",
            features_col="features",
            probability_col="probability",
            raw_"prediction"="rawPrediction",
            "prediction"="prediction",
            seed=42,
            tree_method="hist",
            eval_metric="logloss",
            **params,
        )

        pipeline = Pipeline(stages=[*STAGES, xgb])
        model = pipeline.fit(train_df)
        val_predictions = model.transform(valid_df).select(
            "Coffee_Intake_Binary", "prediction"
        )

        val_precision0, val_recall0, val_f10 = class_zero_metrics(
            val_predictions, "Coffee_Intake_Binary"="Coffee_Intake_Binary", pred_col="prediction"
        )

        mlflow.log_params(params)
        mlflow.log_metrics(
            {
                "validation_precision0": val_precision0,
                "validation_recall0": val_recall0,
                "validation_f10": val_f10,
            }
        )
        return val_f10


# TODO: choose your starting hyperparameters
seed_params = {
    "eta": ...,  # also known as learning rate
    "colsample_bytree": ...,
    "max_depth": ...,
    "min_child_weight": ...,
    "subsample": ...,
}

with mlflow.start_run(
    experiment_id=exp_id, run_name="parent_run_optuna_hp", nested=True
) as parent_run:

    # We define a study, which is a collection of trials
    study = optuna.create_study(
        direction="maximize", study_name="coffee_optuna_study"
    )
    # We seed the starting parameters
    study.enqueue_trial(seed_params, skip_if_exists=True)

    # We run the study
    study.optimize(
        objective,
        n_trials=5,
        n_jobs=-1,
        show_progress_bar=True,
    )

    # We keep the best parameters from the best trial
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_validation_f1", float(study.best_value))


print(f"\nBest validation F1 (class 0): {study.best_value:.4f}")
print("\nBest parameters:")
for key, value in study.best_params.items():
    print(f"  - {key}: {value}")
best_params = study.best_params

# Tuned model definition
best_xgb = SparkXGBClassifier(
    "Coffee_Intake_Binary"="Coffee_Intake_Binary",
    features_col="features",
    probability_col="probability",
    raw_"prediction"="rawPrediction",
    "prediction"="prediction",
    seed=42,
    tree_method="hist",
    eval_metric="logloss",
    **best_params,
)

best_pipeline = Pipeline(stages=[*STAGES, best_xgb])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 5 · Evaluate the Tuned Model
# MAGIC **Goal:** fit the best parameters on train+validation, score the test set, and report feature importances.
# MAGIC
# MAGIC You only need to replace the `...` placeholders.
# MAGIC
# MAGIC Need a nudge? Use the hint loader below.

# COMMAND ----------

# DBTITLE 1,Load hint for Quest 5
load_hint("model_training", "quest_5")

# COMMAND ----------

train_val_df = train_df.unionByName(...)  # TODO replace placeholder
print(f"Train + validation rows: {train_val_df.count():,}")

best_model = best_pipeline.fit(...)  # TODO replace placeholder

test_pred_df = best_model.transform(...)  # TODO replace placeholder
test_prec0, test_rec0, test_f10 = class_zero_metrics(
    test_pred_df, "Coffee_Intake_Binary", "prediction"
)

confusion_matrix_df = (
    test_pred_df.groupBy("prediction", "Coffee_Intake_Binary")
    .agg(F.count("*").alias("rows"))
    .orderBy("prediction", "Coffee_Intake_Binary")
)
confusion_matrix_pdf = confusion_matrix_df.toPandas()
display(confusion_matrix_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 6 · Register the Final Model
# MAGIC **Goal:** log artifacts to MLflow, and manage UC aliases.
# MAGIC
# MAGIC No need to do anything here, just study a bit the cell and run it.

# COMMAND ----------

# DBTITLE 1,Load hint for Quest 6
load_hint("model_training", "quest_6")

# COMMAND ----------

# DBTITLE 1,Register final model for production
client = MlflowClient()

with mlflow.start_run(run_name="coffee_xgb_best") as run:

    # Capture a schema example so MLflow can store model signature/input
    sample_inf_df = full_labeled_df.drop("Coffee_Intake_Binary").limit(1)
    sample_pred_df = best_model.transform(sample_inf_df).select(
        sample_inf_df.columns + ["prediction"]
    )

    signature = mlflow.models.infer_signature(sample_inf_df, sample_pred_df)

    mlflow_model_info = mlflow.spark.log_model(
        spark_model=best_model,
        artifact_path="spark-model-full-data",
        registered_model_name=f"{CATALOG}.{MY_SCHEMA}.coffee_xgb_model",
        pip_requirements=PIP_REQUIREMENTS,
        signature=signature,
    )

    versions = client.search_model_versions(f"name = '{f"{CATALOG}.{MY_SCHEMA}.coffee_xgb_model"}'")
    champion_version = versions[0].version

    client.set_registered_model_alias(
        f"{CATALOG}.{MY_SCHEMA}.coffee_xgb_model", "champion", champion_version
    )

    mlflow.log_params(best_params)
    mlflow.log_metrics(
        {
            "test_precision0": float(test_prec0),
            "test_recall0": float(test_rec0),
            "test_f10": float(test_f10),
        }
    )
    mlflow.log_table(confusion_matrix_pdf, "test_confusion_matrix.json")

print(f"Test precision0: {test_prec0:.4f}")
print(f"Test recall0: {test_rec0:.4f}")
print(f"Test F1 (class 0): {test_f10:.4f}")

print("Final XGBoost model trained on full dataset and logged to MLflow.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 7 · A Duel?!
# MAGIC
# MAGIC Now you have a model registered with the `"champion"` alias.
# MAGIC
# MAGIC Suppose new data comes in, in the form of `test_df`. How would you handle the logging a new model, the `"challenger"`?
# MAGIC
# MAGIC You can **use the previous cell's logic**, but what would you change?
# MAGIC
# MAGIC You do not need to use mlflow to log metrics and set tags for this task (lines `28-44` in previous cell).
# MAGIC
# MAGIC Write the logic below the final `TODO:` comment of the cell.

# COMMAND ----------

# DBTITLE 1,Load hint for Quest 7
load_hint("model_training", "quest_7")

# COMMAND ----------

# DBTITLE 1,Promote the challenger
# New data comes in!
new_data_for_training, new_data_for_testing = test_df.randomSplit(
    [0.5, 0.5], seed=42
)

updated_df = train_val_df.unionByName(new_data_for_training)
print(f"Updated data rows: {updated_df.count():,}")

# Load Champion model by using its alias
champion_model = mlflow.spark.load_model(f"models:/{f"{CATALOG}.{MY_SCHEMA}.coffee_xgb_model"}@champion")

# Evaluate the Champion model on the new data
champion_predictions_df = best_model.transform(new_data_for_testing)
champion_precision, champion_recall, champion_f1 = class_zero_metrics(
    champion_predictions_df, "Coffee_Intake_Binary", "prediction"
)
# Store current Champion info
champ_info = client.get_model_version_by_alias(f"{CATALOG}.{MY_SCHEMA}.coffee_xgb_model", "champion")
champion_version = champ_info.version

# Train the Challenger model on the updated data
challenger_model = best_pipeline.fit(updated_df)
challenger_predictions_df = challenger_model.transform(new_data_for_testing)

# Evaluate the Challenger model on the new data
challenger_precision, challenger_recall, challenger_f1 = class_zero_metrics(
    challenger_predictions_df, "Coffee_Intake_Binary", "prediction"
)

# Log Challenger model
with mlflow.start_run(run_name="coffee_xgb_best") as run:

    # Capture a signature snapshot for the challenger as well
    sample_inf_df = updated_df.drop("Coffee_Intake_Binary").limit(1)
    sample_pred_df = challenger_model.transform(sample_inf_df).select(
        sample_inf_df.columns + ["prediction"]
    )

    signature = mlflow.models.infer_signature(sample_inf_df, sample_pred_df)

    mlflow_model_info = mlflow.spark.log_model(
        spark_model=challenger_model,
        artifact_path="spark-model-full-data",
        registered_model_name=f"{CATALOG}.{MY_SCHEMA}.coffee_xgb_model",
        pip_requirements=PIP_REQUIREMENTS,
        signature=signature,
    )

    versions = client.search_model_versions(f"name = '{f"{CATALOG}.{MY_SCHEMA}.coffee_xgb_model"}'")
    challenger_version = versions[0].version
    client.set_registered_model_alias(
        f"{CATALOG}.{MY_SCHEMA}.coffee_xgb_model", "challenger", challenger_version
    )

    # TODO: Write promotion logic with aliases below

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bonus question!
# MAGIC Would you do anything different about the training of our final model?...
