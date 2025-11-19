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

exp_id = setup_experiment(MLFLOW_EXPERIMENT_NAME)

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
alias = .......  # Replace with the alias of the model
CHAMPION_MODEL_URI = f"models:/{MODEL_PATH}{alias}"

model = mlflow.spark.load_model(CHAMPION_MODEL_URI)

# COMMAND ----------

# DBTITLE 1,Get predictions
predictions_df = model.transform(holdout_df)
display(predictions_df.limit(10))

# Save predictions in a table
PREDICTIONS_TABLE_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_predictions"
predictions_df.write.format("delta").mode("overwrite").saveAsTable(PREDICTIONS_TABLE_PATH)
print(f"DataFrame predictions_df has been written to table:\n\t- {PREDICTIONS_TABLE_PATH}")

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
# MAGIC ## Oh, how quickly time flies!
# MAGIC
# MAGIC We use our magic remote control to fast forward time, to a point where we have gathered the true labels for the holdout dataset.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src="../img/fast_forward.png" width="700">

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 3 · Evaluate the model's performance in the production data
# MAGIC
# MAGIC The below cell uses MLFlow Evaluate to assess the performance of our model in production.
# MAGIC
# MAGIC Run the following 2 cells to load the hint and run the code, and then try to answer the question:
# MAGIC
# MAGIC > Which metrics where logged and where?
# MAGIC

# COMMAND ----------

# DBTITLE 1,Load hint for Quest 3
load_hint("predictions", "quest_3")

# COMMAND ----------

# DBTITLE 1,mlflow evaluate
client = MlflowClient()

version = client.get_model_version_by_alias(name=MODEL_PATH, alias="champion")

run_name = f"coffee_xgb_model_v{version.version}_evaluation"

with mlflow.start_run(run_name=run_name):

    # Evaluate the model
    results = mlflow.models.evaluate(
        CHAMPION_MODEL_URI,
        holdout_df,
        targets="Coffee_Drinker",
        model_type="classifier",
        evaluators=["default"],
        evaluator_config={"default": {"pos_label": 0}}
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 4 · A Duel for the crown!
# MAGIC
# MAGIC Days passed (in theory) and now we have a bigger labeled dataset.
# MAGIC
# MAGIC We are going to train a new version of the model, using the updated dataset, and compare its performance with a previous champion.
# MAGIC
# MAGIC The code in the following cell:
# MAGIC - Unites the dataset into one big Spark DataFrame, and then splits it into a Train DataFrame and a Test DataFrame.
# MAGIC - Creates an Experiment
# MAGIC - Trains a new version of the model and evaluates it on the Test set.
# MAGIC - Evaluates the previous champion on the Test set.
# MAGIC - Compares the two models and decides who is the new champion!
# MAGIC
# MAGIC
# MAGIC **Question 1**
# MAGIC > Do you think this is a fair comparison?
# MAGIC
# MAGIC **Question 2**
# MAGIC > What considerations should we take into account when changing the production model?
# MAGIC
# MAGIC `No hints for the last Quest!
# MAGIC Share your thoughts with us in an open discussion.`
# MAGIC

# COMMAND ----------

# DBTITLE 1,Finding the new champion
# Union all the available data
coffee_labeled_df = spark.table(f"{CATALOG}.{SCHEMA_WITH_SOURCE_DATA}.coffee_labeled_features")
holdout_df = spark.table(HOLDOUT_TABLE_PATH)
all_available_data_df = coffee_labeled_df.unionByName(holdout_df)

# Split the data
train_df, test_df = all_available_data_df.randomSplit([0.8, 0.2], seed=42)

MODEL_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_xgb_model"

with mlflow.start_run(run_name="coffee_model_duel"):

    # Train a new model to challenge the existing champion
    challenger_xgb_params = {
        "label_col": "Coffee_Drinker",
        "features_col": "features",
        "probability_col": "probability",
        "raw_prediction": "rawPrediction",
        "prediction": "prediction",
        "seed": 42,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "eta": 0.05,
        "colsample_bytree": 0.6,
        "max_depth": 9,
        "min_child_weight": 5.0,
        "subsample": 0.65,
    }

    STAGES = build_preprocessing_stages()
    pipeline = Pipeline(stages=[*STAGES, SparkXGBClassifier(**challenger_xgb_params)])
    challenger_model = pipeline.fit(train_df)

    challenger_predictions = challenger_model.transform(test_df).select("Coffee_Drinker", "prediction")
    _, _, challenger_test_f1 = class_zero_metrics(df=challenger_predictions, label_col="Coffee_Drinker", pred_col="prediction")

    sample_inf_df = train_df.drop("Coffee_Drinker").limit(1)
    sample_pred_df = challenger_model.transform(sample_inf_df).select(sample_inf_df.columns + ["prediction"])
    signature = mlflow.models.infer_signature(sample_inf_df, sample_pred_df)

    mlflow.spark.log_model(
        spark_model=challenger_model,
        artifact_path="spark-model-full-data",
        registered_model_name=MODEL_PATH,
        pip_requirements=PIP_REQUIREMENTS,
        signature=signature,
    )

    # Set latest model's alias to "challenger"
    versions = client.search_model_versions(f"name = '{MODEL_PATH}'")
    challenger_version = versions[-1].version  # Latest version is the challenger
    client.set_registered_model_alias(MODEL_PATH, "challenger", challenger_version)


    # Load & evaluate the Champion model
    champion_model = mlflow.spark.load_model(f"models:/{MODEL_PATH}@champion")

    champion_predictions = champion_model.transform(test_df).select("Coffee_Drinker", "prediction")
    _, _, champion_test_f1 = class_zero_metrics(df=champion_predictions, label_col="Coffee_Drinker", pred_col="prediction")

    champion_version_info = client.get_model_version_by_alias(MODEL_PATH, "champion")
    champion_version = champion_version_info.version

    # Duel of champions
    if challenger_test_f1 > champion_test_f1:
        # The previous champion loses his title
        client.set_registered_model_alias(MODEL_PATH, "challenger", champion_version)
        # And we have a new champion
        client.set_registered_model_alias(MODEL_PATH, "champion", challenger_version)

