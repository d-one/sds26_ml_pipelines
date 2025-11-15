# Databricks notebook source
# MAGIC %md
# MAGIC # Coffee Module 05 · Training the model
# MAGIC Time to train and register the model! Work through the refreshed Optuna + MLflow workflow in quest form. Replace every `...` placeholder with real code before executing each quest.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialization

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run ./00_setup

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

# DBTITLE 1,Constants
LABEL_COL = "Coffee_Intake_Binary"
PREDICTION_COL = "prediction"

MLFLOW_EXPERIMENT_NAME = (
    f"/Workspace/Users/{USER_EMAIL}/coffee_hp_tuning_experiment"
)
UC_MODEL_NAME = f"{CATALOG}.{MY_SCHEMA}.coffee_xgb_model"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 1 · Define Feature Categories
# MAGIC **Goal:** detect numeric and categorical columns from the labeled dataset.
# MAGIC
# MAGIC **Hints**
# MAGIC - Inspect the results.
# MAGIC - Are the contents of the list accurate?
# MAGIC - Would you change anything?

# COMMAND ----------

BASE_COLUMNS = ["Coffee_Intake_Binary", "ID", "Timestamp"]
coffee_labeled_df = spark.table(f"{CATALOG}.{MY_SCHEMA}.coffee_labeled")
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
coffee_labeled_df.limit(3).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 2 · Prepare the Feature Store Training Set
# MAGIC **Goal:** load the fact table and build a Feature Store training set that joins in features.
# MAGIC
# MAGIC **Hints**
# MAGIC - Use the feature list you configured in the previous quest

# COMMAND ----------

fe = FeatureEngineeringClient()
labeled_fact_df = spark.table(f"{CATALOG}.{MY_SCHEMA}.coffee_labeled_fact")

feature_lookup = FeatureLookup(
    table_name=f"{CATALOG}.{MY_SCHEMA}.coffee_features",
    feature_names=...,  # replace placeholder
    lookup_key="ID",
    timestamp_lookup_key="Timestamp",
)

training_set = fe.create_training_set(
    df=labeled_fact_df,
    feature_lookups=[...],  # replace placeholder
    label=LABEL_COL,
)

full_labeled_df = training_set.load_df()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 3 · Configure Splits, Pipeline, and MLflow
# MAGIC **Goal:** create data splits, build preprocessing stages, and configure the MLflow experiment.
# MAGIC
# MAGIC **Hints**
# MAGIC - Use three float numbers to do the splitting. What do you think is a good split?
# MAGIC

# COMMAND ----------

train_df, valid_df, test_df = full_labeled_df.randomSplit(
    [..., ..., ...], seed=42  # replace placeholders
)
for split_name, split_df in [
    ("train", train_df),
    ("validation", valid_df),
    ("test", test_df),
]:
    print(f"{split_name.title()} split: {split_df.count():,} rows")

indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in categorical_cols
]
encoder = OneHotEncoder(
    inputCols=[f"{c}_idx" for c in categorical_cols],
    outputCols=[f"{c}_ohe" for c in categorical_cols],
    handleInvalid="keep",
    dropLast=True,
)

assembler_input_cols = numeric_cols + [f"{c}_ohe" for c in categorical_cols]
assembler = VectorAssembler(
    inputCols=assembler_input_cols, outputCol="features", handleInvalid="keep"
)
STAGES = indexers + [encoder, assembler]
print("VectorAssembler inputs:", assembler_input_cols)

mlflow.autolog(disable=True)
exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
if exp is None:
    exp_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
else:
    exp_id = exp.experiment_id
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 4 · Execute the Optuna Study
# MAGIC **Goal:** run Optuna with a head start!
# MAGIC
# MAGIC **Hints**
# MAGIC - You can use `study.enqueue_trial(seed_params, skip_if_exists=True)` before `study.optimize` to start from a specific set of hyperparameters.
# MAGIC - AutoML experiment: https://adb-1451829595406012.12.azuredatabricks.net/ml/experiments/1514058481528333?o=1451829595406012

# COMMAND ----------

optuna.logging.set_verbosity(optuna.logging.ERROR)


def objective(trial: optuna.Trial) -> float:
    run_name = (
        "automl_parameters_trial"
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
            label_col=LABEL_COL,
            features_col="features",
            probability_col="probability",
            raw_prediction_col="rawPrediction",
            prediction_col=PREDICTION_COL,
            seed=42,
            tree_method="hist",
            eval_metric="logloss",
            **params,
        )

        pipeline = Pipeline(stages=[*STAGES, xgb])
        model = pipeline.fit(train_df)
        val_predictions = model.transform(valid_df).select(
            LABEL_COL, PREDICTION_COL
        )

        val_precision0, val_recall0, val_f10 = class_zero_metrics(
            val_predictions, label_col=LABEL_COL, pred_col=PREDICTION_COL
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
    study = optuna.create_study(
        direction="maximize", study_name="coffee_optuna_study"
    )
    study.enqueue_trial(seed_params, skip_if_exists=True)
    study.optimize(
        objective,
        n_trials=5,
        n_jobs=-1,
        show_progress_bar=True,
    )

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_validation_f1", float(study.best_value))
    mlflow.set_tags(
        tags={
            "project": "coffee_project_tag",
            "optimizer_engine": "coffee_hp_tuning_tag",
        }
    )

print(f"Best validation F1 (class 0): {study.best_value:.4f}")
print("Best parameters:")
for key, value in study.best_params.items():
    print(f"  - {key}: {value}")
best_params = study.best_params

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 5 · Evaluate the Tuned Model
# MAGIC **Goal:** fit the best parameters on train+validation, score the test set, and report feature importances.
# MAGIC
# MAGIC **Hints**
# MAGIC - Union train and validation (`train_df.unionByName(valid_df)`).
# MAGIC - Train with the new df, `train_val_df`
# MAGIC - Predict `test_df`
# MAGIC - Use `get_feature_name_mapping` with your version of `numeric_columns_list` and `categorical_columns_list`

# COMMAND ----------

train_val_df = train_df.unionByName(...)  # replace placeholder
print(f"Train + validation rows: {train_val_df.count():,}")

best_xgb = SparkXGBClassifier(
    label_col=LABEL_COL,
    features_col="features",
    probability_col="probability",
    raw_prediction_col="rawPrediction",
    prediction_col=PREDICTION_COL,
    seed=42,
    tree_method="hist",
    eval_metric="logloss",
    **best_params,
)

best_pipeline = Pipeline(stages=[*STAGES, best_xgb])
best_model = best_pipeline.fit(...)  # replace placeholder

test_pred_df = best_model.transform(...)  # replace placeholder
test_prec0, test_rec0, test_f10 = class_zero_metrics(
    test_pred_df, LABEL_COL, PREDICTION_COL
)

conf_mat_df = (
    test_pred_df.groupBy("prediction", LABEL_COL)
    .agg(F.count("*").alias("rows"))
    .orderBy("prediction", LABEL_COL)
)
conf_mat_pdf = conf_mat_df.toPandas()
display(conf_mat_pdf)

feature_name_mapping = get_feature_name_mapping(
    best_model,
    numeric_cols,
    categorical_cols,
)
feature_importances = best_model.stages[-1].get_feature_importances()

rows = []
for i in range(len(feature_name_mapping)):
    feature_key = f"f{i}"
    feature_name = feature_name_mapping.get(feature_key, feature_key)
    importance = float(feature_importances.get(f"f{i}", 0.0))
    rows.append(
        {
            "Feature_Index": feature_key,
            "Feature_Name": feature_name,
            "Importance": importance,
        }
    )

feature_importance_pdf = (
    pd.DataFrame(rows)
    .sort_values(by="Importance", ascending=False)
    .reset_index(drop=True)
)
display(feature_importance_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 6 · Register the Final Model
# MAGIC **Goal:** refit on the full dataset, log artifacts to MLflow, and manage UC aliases.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Register final model for production
client = MlflowClient()

final_xgb = SparkXGBClassifier(
    label_col=LABEL_COL,
    features_col="features",
    probability_col="probability",
    raw_prediction_col="rawPrediction",
    prediction_col=PREDICTION_COL,
    seed=42,
    tree_method="hist",
    eval_metric="logloss",
    **best_params,
)

final_pipeline = Pipeline(stages=[*STAGES, final_xgb])

with mlflow.start_run(run_name="coffee_xgb_best") as run:

    # Refit the tuned pipeline on the entire labelled dataset before registering it
    final_model = final_pipeline.fit(full_labeled_df)

    sample_inf_df = full_labeled_df.drop("Coffee_Intake_Binary").limit(1)
    sample_pred_df = final_model.transform(sample_inf_df).select(
        sample_inf_df.columns + ["prediction"]
    )

    signature = mlflow.models.infer_signature(sample_inf_df, sample_pred_df)

    mlflow_model_info = mlflow.spark.log_model(
        spark_model=final_model,
        artifact_path="spark-model-full-data",
        registered_model_name=UC_MODEL_NAME,
        pip_requirements=PIP_REQUIREMENTS,
        signature=signature,
        input_example=...,
    )

    versions = client.search_model_versions(f"name = '{UC_MODEL_NAME}'")
    champion_version = versions[0].version

    client.set_registered_model_alias(
        UC_MODEL_NAME, "champion", champion_version
    )

    mlflow.log_params(best_params)
    mlflow.log_metrics(
        {
            "test_precision0": float(test_prec0),
            "test_recall0": float(test_rec0),
            "test_f10": float(test_f10),
        }
    )
    mlflow.log_table(conf_mat_pdf, "test_confusion_matrix.json")
    mlflow.log_table(feature_importance_pdf, "xgb_feature_importances.json")
    mlflow.set_tags(
        tags={
            "project": "coffee_project_tag",
            "optimizer_engine": "coffee_hp_tuning_tag",
            "production_model": True,
        }
    )

print(f"Test precision0: {test_prec0:.4f}")
print(f"Test recall0: {test_rec0:.4f}")
print(f"Test F1 (class 0): {test_f10:.4f}")

print("Final XGBoost model trained on full dataset and logged to MLflow.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Bonus!**
# MAGIC
# MAGIC Now you have a model registered with the `"champion"` alias.
# MAGIC
# MAGIC Suppose new data comes in. How would you handle the logging a new model?
# MAGIC
# MAGIC You can **use the previous cell's logic**, but what would you change?
# MAGIC
# MAGIC You do not need to use mlflow to log metrics and set tags for this task (lines 45-61 in previous cell).
# MAGIC
# MAGIC **Hints**
# MAGIC - It can be done with the addition of just one line with conditional logic
# MAGIC - It has to do with aliases

# COMMAND ----------

# DBTITLE 1,Promote the challenger
# New data comes in!
holdout_df = spark.table(f"{CATALOG}.{MY_SCHEMA}.coffee_prod_holdout")

new_data_for_training, new_data_for_testing = holdout_df.randomSplit(
    [0.5, 0.5], seed=42
)

updated_full_labeled_df = full_labeled_df.unionByName(new_data_for_training)

# Champion model
champion_model = mlflow.spark.load_model(f"models:/{UC_MODEL_NAME}@champion")
champion_predictions_df = best_model.transform(new_data_for_testing)
champion_precision, champion_recall, champion_f1 = class_zero_metrics(
    champion_predictions_df, LABEL_COL, PREDICTION_COL
)

# Log Challenger model
with mlflow.start_run(run_name="coffee_xgb_best") as run:

    challenger_model = final_pipeline.fit(updated_full_labeled_df)
    challenger_predictions_df = challenger_model.transform(
        new_data_for_testing
    )
    challenger_precision, challenger_recall, challenger_f1 = (
        class_zero_metrics(
            challenger_predictions_df, LABEL_COL, PREDICTION_COL
        )
    )

    sample_inf_df = updated_full_labeled_df.drop("Coffee_Intake_Binary").limit(
        1
    )
    sample_pred_df = final_model.transform(sample_inf_df).select(
        sample_inf_df.columns + ["prediction"]
    )

    signature = mlflow.models.infer_signature(sample_inf_df, sample_pred_df)

    mlflow_model_info = mlflow.spark.log_model(
        spark_model=final_model,
        artifact_path="spark-model-full-data",
        registered_model_name=UC_MODEL_NAME,
        pip_requirements=PIP_REQUIREMENTS,
        signature=signature,
        input_example=sample_inf_df,
    )

    versions = client.search_model_versions(f"name = '{UC_MODEL_NAME}'")
    champion_version = versions[0].version

    # TODO: Write promotion logic with aliases below
