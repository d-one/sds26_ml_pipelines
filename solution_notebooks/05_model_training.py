# Databricks notebook source
# MAGIC %md
# MAGIC # 05 – Coffee Intake Classification Experiment
# MAGIC
# MAGIC This notebook orchestrates hyperparameter tuning for a Spark XGBoost classifier using Optuna and tracks the workflow with MLflow. All features are loaded directly from the Unity Catalog Feature Store so workshop participants can see how managed features flow into experimentation without relying on helper modules.

# COMMAND ----------

# MAGIC %md
# MAGIC ### How Optuna’s TPESampler works (light explanation)
# MAGIC
# MAGIC By default, Optuna uses the **TPE (Tree-structured Parzen Estimator) sampler**.
# MAGIC
# MAGIC TPE models **p(hyperparameters | score)**.
# MAGIC
# MAGIC It splits past trials into:
# MAGIC - **good trials** (top-performing N%) → `l(x)`
# MAGIC - **bad trials** (remaining trials) → `g(x)`
# MAGIC
# MAGIC Then it selects new hyperparameters `x` that **maximize the ratio**:
# MAGIC
# MAGIC > `l(x) / g(x)`
# MAGIC > *i.e., how likely x is to come from the “good” distribution rather than the “bad” one*
# MAGIC
# MAGIC So in effect, TPE tries to **sample more often from regions of the search space that look like past strong performers**.
# MAGIC
# MAGIC Optuna also supports other samplers (e.g., `RandomSampler`, `CmaEsSampler`, `GridSampler`, etc.), but TPE is the default and most commonly used.
# MAGIC

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
# Define random seed for reproducibility
RANDOM_SEED = 42

# Table paths
COFFEE_FACT_TABLE_PATH = f"{CATALOG}.{SCHEMA_WITH_SOURCE_DATA}.coffee_labeled_fact"
FEATURE_TABLE_PATH = f"{CATALOG}.{SCHEMA_WITH_SOURCE_DATA}.coffee_labeled_features"

# Columns
PRIMARY_KEY_COL = "ID"
TIMESTAMP_COL = "Timestamp"
LABEL_COL = "Coffee_Intake_Binary"
PREDICTION_COL = "prediction"

# mlflow parameters
MLFLOW_EXPERIMENT_NAME = f"/Workspace/Users/{USER_EMAIL}/coffee_hp_tuning_experiment"
PARENT_EXPERIMENT_NAME = "parent_run_optuna_hp"
PROJECT_TAG = "coffee_project_tag"
HYPERPARAMETER_TUNING_TAG = "coffee_hp_tuning_tag"
UC_MODEL_NAME = f"{CATALOG}.{MY_SCHEMA}.coffee_xgb_model"
PIP_REQUIREMENTS = [
    "mlflow==3.6.0",
    "pyspark==3.5.2",
    "scikit-learn==1.4.2",
    "xgboost==2.0.3"
]

# Optuna parameters
OPTUNA_TRIALS = 5
OPTUNA_JOBS = -1  # Number of parallel jobs
STUDY_NAME = "coffee_optuna_study"

# COMMAND ----------

# DBTITLE 1,Column categories
BASE_COLUMNS = [LABEL_COL, PRIMARY_KEY_COL, TIMESTAMP_COL]
data_schema_fields = spark.table(FEATURE_TABLE_PATH).schema.fields

# Automatically detect numeric columns
numeric_cols = [
    field.name
    for field in data_schema_fields
    if field.dataType
    in [T.IntegerType(), T.DoubleType(), T.LongType(), T.FloatType()]
    and field.name not in BASE_COLUMNS
]

# Automatically detect string/categorical columns
categorical_cols = [
    field.name
    for field in data_schema_fields
    if field.dataType == T.StringType() and field.name not in BASE_COLUMNS
]

all_feature_cols = numeric_cols + categorical_cols

print("Detected numeric columns:\n\t", numeric_cols)
print("\nDetected categorical columns:\n\t", categorical_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparations

# COMMAND ----------

# DBTITLE 1,Feature Lookups
fe = FeatureEngineeringClient()

labeled_fact_df = spark.table(COFFEE_FACT_TABLE_PATH)

feature_lookup = FeatureLookup(
    table_name=FEATURE_TABLE_PATH,
    feature_names=all_feature_cols,
    lookup_key=PRIMARY_KEY_COL,
    timestamp_lookup_key=TIMESTAMP_COL,
)

training_set = fe.create_training_set(
    df=labeled_fact_df,
    feature_lookups=[feature_lookup],
    label=LABEL_COL,
)

full_labeled_df = training_set.load_df()

# COMMAND ----------

# DBTITLE 1,Train-Validate-Test split
train_df, valid_df, test_df = full_labeled_df.randomSplit(
    [0.6, 0.2, 0.2], seed=RANDOM_SEED
)

for split_name, split_df in [
    ("train", train_df),
    ("validation", valid_df),
    ("test", test_df),
]:
    print(f"{split_name.title()} split: {split_df.count():,} rows")

# COMMAND ----------

# DBTITLE 1,Pipeline stages
# 1️⃣ StringIndexer stages (one per categorical column)
indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in categorical_cols
]

# 2️⃣ OneHotEncoder (multi-column, for all indexed cols at once)
encoder = OneHotEncoder(
    inputCols=[f"{c}_idx" for c in categorical_cols],
    outputCols=[f"{c}_ohe" for c in categorical_cols],
    handleInvalid="keep",
    dropLast=False,
)

# 3️⃣ VectorAssembler
assembler_input_cols = numeric_cols + [f"{c}_ohe" for c in categorical_cols]
assembler = VectorAssembler(
    inputCols=assembler_input_cols, outputCol="features", handleInvalid="keep"
)

# 4️⃣ Combine stages
STAGES = indexers + [encoder, assembler]

print("VectorAssembler input columns:", assembler_input_cols)

# COMMAND ----------

# DBTITLE 1,Create mlflow experiment
mlflow.set_registry_uri("databricks-uc")
# Disable autologging (because it can create extra experiments when fitting models in final run)
mlflow.autolog(disable=True)

exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
if exp is None:
    # If it doesn't exist, create it
    exp_id = mlflow.create_experiment(
        MLFLOW_EXPERIMENT_NAME
    )  # parent already exists
else:
    exp_id = exp.experiment_id

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter tuning study

# COMMAND ----------

# DBTITLE 1,Callback function
# override Optuna's default logging to ERROR only
optuna.logging.set_verbosity(optuna.logging.ERROR)

# define a logging callback that will report on only new challenger parameter configurations if a
# trial has usurped the state of 'best conditions'


def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.

    Note: This callback is not intended for use in distributed computing systems such as Spark
    or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
    workers or agents.
    The race conditions with file system state management for distributed trials will render
    inconsistent values with this callback.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (
                abs(winner - study.best_value) / study.best_value
            ) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(
                f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}"
            )


# COMMAND ----------

# DBTITLE 1,Objective function

def objective(trial: optuna.Trial) -> float:

    run_name = (
        "automl_parameters_trial"
        if trial.number == 0
        else f"optuna_trial_{trial.number:03d}"
    )

    with mlflow.start_run(
        run_name=run_name,
        nested=True,
        tags={
            "mlflow.parentRunId": parent_run.info.run_id
        },  # <-- link to parent
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
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-3, 10.0, log=True
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "num_round": trial.suggest_int("num_round", 100, 1200),
        }

        xgb = SparkXGBClassifier(
            label_col=LABEL_COL,
            features_col="features",
            probability_col="probability",
            raw_prediction_col="rawPrediction",
            prediction_col=PREDICTION_COL,
            seed=RANDOM_SEED,
            tree_method="hist",
            eval_metric="logloss",
            **params,
        )

        pipeline = Pipeline(stages=[*STAGES, xgb])
        print(pipeline)
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

        # trial.set_user_attr("run_id", mlflow.active_run().info.run_id)
        return val_f10


# COMMAND ----------

# DBTITLE 1,AutoML parameters
# Seed the optimisation with AutoML-derived hyperparameters to jump-start the search.
automl_seed_params = {
    "eta": 0.05759496965676729,
    "colsample_bytree": 0.6263993741226758,
    "max_depth": 9,
    "min_child_weight": 5.0,
    "subsample": 0.6616262667209235,
    "reg_lambda": 1.0,
    "reg_alpha": 1e-4,
    "num_round": 1079,
}

# COMMAND ----------

# MAGIC %md
# MAGIC With the above, AutoML produced a model with:
# MAGIC - f1_val  = 0.9296482412060302
# MAGIC - f1_test = 0.9495412844036697
# MAGIC

# COMMAND ----------

# DBTITLE 1,Run Optuna study
# Start a new Optuna run

with mlflow.start_run(
    experiment_id=exp_id, run_name=PARENT_EXPERIMENT_NAME, nested=True
) as parent_run:

    study = optuna.create_study(direction="maximize", study_name=STUDY_NAME)

    # Try the AutoML parameters in the first trial
    study.enqueue_trial(automl_seed_params, skip_if_exists=True)

    study.optimize(
        objective,
        n_trials=OPTUNA_TRIALS,
        n_jobs=OPTUNA_JOBS,
        show_progress_bar=True,
        callbacks=[champion_callback],
    )

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_validation_f1", float(study.best_value))
    # mlflow.log_dict(study.best_params, "best_params.json")
    mlflow.set_tags(
        tags={
            "project": PROJECT_TAG,
            "optimizer_engine": HYPERPARAMETER_TUNING_TAG,
        }
    )

print(f"Best validation F1 (class 0): {study.best_value:.4f}")
print("Best parameters:")
for key, value in study.best_params.items():
    print(f"  - {key}: {value}")
best_params = study.best_params

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final training with best parameters

# COMMAND ----------

# DBTITLE 1,Evaluate on test set
# Combine train and validate splits
train_val_df = train_df.unionByName(valid_df)
print(f"Train + validation rows: {train_val_df.count():,}")

# Load model class with best parameters from Optuna study
best_xgb = SparkXGBClassifier(
    label_col=LABEL_COL,
    features_col="features",
    probability_col="probability",
    raw_prediction_col="rawPrediction",
    prediction_col=PREDICTION_COL,
    seed=RANDOM_SEED,
    tree_method="hist",
    eval_metric="logloss",
    **best_params,
)

# Train the model
best_pipeline = Pipeline(stages=[*STAGES, best_xgb])
best_model = best_pipeline.fit(train_val_df)

test_pred_df = best_model.transform(test_df)

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

# COMMAND ----------

# DBTITLE 1,Feature importance
# Create feature importance mapping
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

# Create a Pandas DataFrame
feature_importance_pdf = pd.DataFrame(rows)

# Sort descending by importance
feature_importance_pdf = feature_importance_pdf.sort_values(
    by="Importance", ascending=False
).reset_index(drop=True)

display(feature_importance_pdf)

# COMMAND ----------

print(len(feature_importances))
feature_importances

# COMMAND ----------

feature_name_mapping

# COMMAND ----------

# DBTITLE 1,Register final model for production
client = MlflowClient()
mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name="coffee_xgb_best") as run:

    # Refit the tuned pipeline on the entire labelled dataset before registering it
    final_xgb = SparkXGBClassifier(
        label_col=LABEL_COL,
        features_col="features",
        probability_col="probability",
        raw_prediction_col="rawPrediction",
        prediction_col=PREDICTION_COL,
        seed=RANDOM_SEED,
        tree_method="hist",
        eval_metric="logloss",
        **best_params,
    )

    final_pipeline = Pipeline(stages=[*STAGES, final_xgb])
    final_model = final_pipeline.fit(full_labeled_df)

    sample_inf_df = full_labeled_df.drop("Coffee_Intake_Binary").limit(1)
    sample_pred_df = final_model.transform(sample_inf_df).select(
        sample_inf_df.columns + ["prediction"]
    )

    signature = mlflow.models.infer_signature(sample_inf_df, sample_pred_df)

    fe_model_info = fe.log_model(
        artifact_path="spark-model-full-data",
        model=final_model,
        flavor=mlflow.spark,
        training_set=training_set,
        registered_model_name=UC_MODEL_NAME,
        signature=signature,
        infer_input_example=sample_inf_df.toPandas(),
    )

    mlflow_model_info = mlflow.spark.log_model(
        spark_model=final_model,
        artifact_path="spark-model-full-data",
        registered_model_name=UC_MODEL_NAME,
        signature=signature,
        input_example=sample_inf_df.toPandas(),
        pip_requirements=PIP_REQUIREMENTS,
    )

    versions = client.search_model_versions(f"name = '{UC_MODEL_NAME}'")
    champion_version = versions[0].version
    lineage = versions[1].version

    client.set_registered_model_alias(
        UC_MODEL_NAME, "champion", champion_version
    )
    # Delete the model version with alias "lineage"
    try:
        lineage_version = client.get_model_version_by_alias(
            UC_MODEL_NAME, "lineage"
        ).version
        client.delete_model_version(
            name=UC_MODEL_NAME, version=lineage_version
        )
    except:
        print("A model version with lineage does not exist.")

    client.set_registered_model_alias(UC_MODEL_NAME, "lineage", lineage)

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
            "project": PROJECT_TAG,
            "optimizer_engine": HYPERPARAMETER_TUNING_TAG,
            "production_model": True,
        }
    )

print(f"Test precision0: {test_prec0:.4f}")
print(f"Test recall0: {test_rec0:.4f}")
print(f"Test F1 (class 0): {test_f10:.4f}")

print("Final XGBoost model trained on full dataset and logged to MLflow.")
