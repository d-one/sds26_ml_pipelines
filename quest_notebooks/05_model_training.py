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
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from mlflow import MlflowClient
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import functions as F
from pyspark.sql import types as T
from xgboost.spark import SparkXGBClassifier

# COMMAND ----------

# DBTITLE 1,Constants
RANDOM_SEED = 42

COFFEE_LABELED_DATA_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_labeled"
COFFEE_FACT_TABLE_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_labeled_fact"
FEATURE_TABLE_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_features"

PRIMARY_KEY_COL = "ID"
TIMESTAMP_COL = "Timestamp"
LABEL_COL = "Coffee_Intake_Binary"
PREDICTION_COL = "prediction"

MLFLOW_EXPERIMENT_NAME = f"/Workspace/Users/{USER_EMAIL}/coffee_hp_tuning_experiment"
PARENT_EXPERIMENT_NAME = "parent_run_optuna_hp"
PROJECT_TAG = "coffee_project_tag"
HYPERPARAMETER_TUNING_TAG = "coffee_hp_tuning_tag"
UC_MODEL_NAME = f"{CATALOG}.{MY_SCHEMA}.coffee_xgb_model"
PIP_REQUIREMENTS = [
    "mlflow==3.6.0",
    "pyspark==3.5.2",
    "scikit-learn==1.4.2",
    "xgboost==2.0.3",
]

OPTUNA_TRIALS = 5
OPTUNA_JOBS = -1
STUDY_NAME = "coffee_optuna_study"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 1 · Define Feature Categories
# MAGIC **Goal:** detect numeric and categorical columns from the labeled dataset.
# MAGIC
# MAGIC **Hints**
# MAGIC - Read `COFFEE_LABELED_DATA_PATH` to access the schema.
# MAGIC - Exclude `LABEL_COL`, `PRIMARY_KEY_COL`, and `TIMESTAMP_COL` from both lists.
# MAGIC - Use the `pyspark.sql.types` classes (`T.IntegerType`, etc.) for comparisons.

# COMMAND ----------

BASE_COLUMNS = [LABEL_COL, PRIMARY_KEY_COL, TIMESTAMP_COL]
data_schema_fields = spark.table(COFFEE_LABELED_DATA_PATH).schema.fields

... = [                                                                                 # replace placeholder
    field.name
    for field in data_schema_fields
    if field.dataType
    in [T.IntegerType(), T.DoubleType(), T.LongType(), T.FloatType()]
    and field.name not in BASE_COLUMNS
]

... = [                                                                                 # replace placeholder
    field.name
    for field in data_schema_fields
    if field.dataType == T.StringType() and field.name not in BASE_COLUMNS
]

all_feature_cols = ... + ...                                                            # replace placeholders

print("Detected numeric columns:\n\t", ...)
print("\nDetected categorical columns:\n\t", ...)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 2 · Prepare the Feature Store Training Set
# MAGIC **Goal:** load the fact table and build a Feature Store training set that joins in features.
# MAGIC
# MAGIC **Hints**
# MAGIC - Instantiate `FeatureEngineeringClient()` and read `COFFEE_FACT_TABLE_PATH`.
# MAGIC - Configure a `FeatureLookup` with the `FEATURE_TABLE_PATH`, feature list, ID, and timestamp.
# MAGIC - Call `training_set.load_df()` to materialize `full_labeled_df`.

# COMMAND ----------

# TODO: Materialize the Feature Store training set.
fe = FeatureEngineeringClient()
labeled_fact_df = spark.table(...)                                                      # replace placeholder

feature_lookup = FeatureLookup(
    table_name=FEATURE_TABLE_PATH,
    feature_names=...,                                                                  # replace placeholder
    lookup_key=PRIMARY_KEY_COL,
    timestamp_lookup_key=TIMESTAMP_COL,
)

training_set = fe.create_training_set(
    df=labeled_fact_df,
    feature_lookups=[...],                                                              # replace placeholder
    label=LABEL_COL,
)

full_labeled_df = training_set.load_df()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 3 · Configure Splits, Pipeline, and MLflow
# MAGIC **Goal:** create train/validation/test splits, build preprocessing stages, and configure the MLflow experiment.
# MAGIC
# MAGIC **Hints**
# MAGIC - Use `randomSplit([0.6, 0.2, 0.2], seed=RANDOM_SEED)`.
# MAGIC - Pipeline stages follow the pattern: StringIndexer → OneHotEncoder → VectorAssembler. 
# MAGIC     - StringIndexer and OneHotEncoder use your version of `categorical_columns_list`
# MAGIC     - VectorAssembler uses a `assembler_input_cols` which is the concatenation of your version of `numeric_columns_list` + a list with ohe columns
# MAGIC - Set the MLflow registry URI to `databricks-uc`, disable autolog, and ensure the experiment exists.

# COMMAND ----------

train_df, valid_df, test_df = full_labeled_df.randomSplit(
    [..., ..., ...], seed=RANDOM_SEED                                                    # replace placeholders
)
for split_name, split_df in [
    ("train", train_df),
    ("validation", valid_df),
    ("test", test_df),
]:
    print(f"{split_name.title()} split: {split_df.count():,} rows")

indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in ...                                                                          # replace placeholder
]
encoder = OneHotEncoder(
    inputCols=[f"{c}_idx" for c in ...],                                                  # replace placeholders
    outputCols=[f"{c}_ohe" for c in ...],
    handleInvalid="keep",
    dropLast=False,
)
assembler_input_cols = ... + [f"{c}_ohe" for c in ...]                                    # replace placeholders
assembler = VectorAssembler(
    inputCols=assembler_input_cols, outputCol="features", handleInvalid="keep"
)
STAGES = indexers + [encoder, assembler]
print("VectorAssembler inputs:", assembler_input_cols)

mlflow.set_registry_uri("...")                                                            # replace placeholder
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
# MAGIC **Goal:** run Optuna with the refreshed objective function and AutoML seed parameters.
# MAGIC
# MAGIC **Hints**
# MAGIC - Use `study.enqueue_trial(automl_seed_params, skip_if_exists=True)` before `study.optimize` to start from the best set of AutoML parameters.

# COMMAND ----------

optuna.logging.set_verbosity(optuna.logging.ERROR)


def champion_callback(study, frozen_trial):
    winner = study.user_attrs.get("winner", None)
    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (
                abs(winner - study.best_value) / study.best_value
            ) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} "
                f"with {improvement_percent: .4f}% improvement"
            )
        else:
            print(
                f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}"
            )


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

with mlflow.start_run(
    experiment_id=exp_id, run_name=PARENT_EXPERIMENT_NAME, nested=True
) as parent_run:
    study = optuna.create_study(direction="maximize", study_name=STUDY_NAME)
    study.enqueue_trial(..., skip_if_exists=True)                                              # replace placeholder
    study.optimize(
        objective,
        n_trials=OPTUNA_TRIALS,
        n_jobs=OPTUNA_JOBS,
        show_progress_bar=True,
        callbacks=[champion_callback],
    )

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_validation_f1", float(study.best_value))
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
# MAGIC ## Quest 5 · Evaluate the Tuned Model
# MAGIC **Goal:** fit the best parameters on train+validation, score the test set, and report feature importances.
# MAGIC
# MAGIC **Hints**
# MAGIC - Union train and validation (`train_df.unionByName(valid_df)`).
# MAGIC - Train with the new df, `train_val_df`
# MAGIC - Predict `test_df`
# MAGIC - Use `get_feature_name_mapping` with your version of `numeric_columns_list` and `categorical_columns_list`

# COMMAND ----------

# TODO: Fit the best parameters and evaluate on the test set.
train_val_df = train_df.unionByName(...)                                                         # replace placeholder
print(f"Train + validation rows: {train_val_df.count():,}")

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

best_pipeline = Pipeline(stages=[*STAGES, best_xgb])
best_model = best_pipeline.fit(...)                                                              # replace placeholder

test_pred_df = best_model.transform(...)                                                         # replace placeholder
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

feature_name_mapping = get_feature_name_mapping(                                                 # replace placeholders
    best_model,
    ..., # numeric columns list
    ..., # categorical columns list
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

feature_importance_pdf = pd.DataFrame(rows).sort_values(
    by="Importance", ascending=False
).reset_index(drop=True)
display(feature_importance_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 6 · Register the Final Model
# MAGIC **Goal:** refit on the full dataset, log artifacts to MLflow, and manage UC aliases.
# MAGIC
# MAGIC **Hints**
# MAGIC - Make sure to log the correct metrics/tables!

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
            "test_precision0": float(...),                                                   # replace placeholders
            "test_recall0": float(...),
            "test_f10": float(...),
        }
    )
    mlflow.log_table(..., "test_confusion_matrix.json")                                      # replace placeholders
    mlflow.log_table(..., "xgb_feature_importances.json")
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
