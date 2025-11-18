# Databricks notebook source
import os
import re
from typing import Tuple

from pyspark.sql import functions as F

# --- Defaults ---
CATALOG = "gtc25_ml_catalog"
SCHEMA_WITH_SOURCE_DATA = "source_data"

# --- User info ---
cwd = os.getcwd()
parts = cwd.split("/Users/")
USER_EMAIL = parts[1].split("/")[0] if len(parts) > 1 else None
MY_NAME = USER_EMAIL.split("@")[0]
MY_NAME = re.sub(r"[^a-zA-Z0-9_]", "_", MY_NAME)

MY_SCHEMA = MY_NAME

PIP_REQUIREMENTS = [
    "mlflow==3.6.0",
    "pyspark==3.5.2",
    "scikit-learn==1.4.2",
    "xgboost==2.0.3",
]

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{MY_SCHEMA}")

print("The following variables have been set:")
print(f"- USER_EMAIL:".ljust(30) + f"{USER_EMAIL}")
print(f"- CATALOG:".ljust(30) + f"{CATALOG}")
print(f"- SCHEMA_WITH_SOURCE_DATA:".ljust(30) + f"{SCHEMA_WITH_SOURCE_DATA}")
print(f"- MY_SCHEMA:".ljust(30) + f"{MY_SCHEMA}")
print(f"- PIP_REQUIREMENTS")

# COMMAND ----------

# DBTITLE 1,Evaluation function

def class_zero_metrics(
    df, label_col: str, pred_col: str
) -> Tuple[float, float, float]:
    label = F.col(label_col)
    prediction = F.col(pred_col)

    tp0 = df.filter((label == 0) & (prediction == 0)).count()
    fp0 = df.filter((label == 1) & (prediction == 0)).count()
    fn0 = df.filter((label == 0) & (prediction == 1)).count()

    prec0 = tp0 / (tp0 + fp0) if (tp0 + fp0) > 0 else 0.0
    rec0 = tp0 / (tp0 + fn0) if (tp0 + fn0) > 0 else 0.0
    f1_0 = (2 * prec0 * rec0 / (prec0 + rec0)) if (prec0 + rec0) > 0 else 0.0
    return float(prec0), float(rec0), float(f1_0)


# COMMAND ----------

# DBTITLE 1,Preprocessing pipeline function
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

def build_preprocessing_stages(categorical_cols, numeric_cols):
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
    stages = indexers + [encoder, assembler]
    print("\n1. String Indexers, these convert text categories into numeric indices")
    for c in categorical_cols:
        print(f"   StringIndexer: {c} -> {c}_idx")
    print()

    print("2. One Hot Encoders, these turn each index into a vector of binary flags")
    for c in categorical_cols:
        print(f"   OneHotEncoder: {c}_idx -> {c}_ohe")
    print()

    print("3. Vector Assembler, this gathers all numeric and encoded features into a single feature vector")
    all_inputs = numeric_cols + [c + "_ohe" for c in categorical_cols]
    print(f"   VectorAssembler: {all_inputs} -> features\n")
    return stages


# COMMAND ----------

# DBTITLE 1,Experiment setup function
import mlflow

def init_experiment(experiment_name):
    mlflow.autolog(disable=True)

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = mlflow.create_experiment(experiment_name)
    else:
        exp_id = exp.experiment_id

    mlflow.set_experiment(experiment_name)
    return exp_id

# COMMAND ----------


def _drop_table_if_exists(table_path: str, label: str) -> None:
    if not table_path:
        return
    try:
        spark.sql(f"DROP TABLE IF EXISTS {table_path}")
        print(f"Dropped {label}: {table_path}")
    except Exception as exc:
        print(f"Unable to drop {label} ({table_path}): {exc}")


def _delete_workspace_dir(path: str) -> None:
    if not path:
        return
    workspace_utils = globals().get("dbutils", None)
    if workspace_utils is None:
        print(f"dbutils is unavailable; cannot remove workspace path {path}.")
        return

    normalized = path.rstrip("/")
    candidates = {normalized}
    if normalized.startswith("/Workspace/"):
        candidates.add("/" + normalized.split("/Workspace/", 1)[1])

    for candidate in candidates:
        try:
            workspace_utils.workspace.delete(candidate, True)
            print(f"Deleted workspace directory: {candidate}")
        except Exception as exc:
            if "RESOURCE_DOES_NOT_EXIST" in str(exc):
                print(f"Workspace directory already missing: {candidate}")
            else:
                print(
                    f"Unable to delete workspace directory {candidate}: {exc}"
                )


def _delete_named_mlflow_experiment(experiment_name: str) -> None:
    if not experiment_name:
        return
    try:
        import mlflow
        from mlflow import MlflowClient
    except Exception as exc:
        print(f"Unable to load MLflow for experiment cleanup: {exc}")
        return

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"MLflow experiment not found: {experiment_name}")
        return

    try:
        MlflowClient().delete_experiment(experiment.experiment_id)
        print(f"Deleted MLflow experiment: {experiment_name}")
    except Exception as exc:
        print(f"Unable to delete MLflow experiment {experiment_name}: {exc}")


def _delete_mlflow_experiments_matching(substring: str) -> None:
    if not substring:
        return
    try:
        from mlflow import MlflowClient
        from mlflow.entities import ViewType
    except Exception as exc:
        print(f"Unable to scan MLflow experiments: {exc}")
        return

    client = MlflowClient()
    try:
        experiments = client.search_experiments(view_type=ViewType.ALL)
    except Exception as exc:
        print(f"Unable to list MLflow experiments: {exc}")
        return

    matches = [
        exp for exp in experiments if exp.name and substring in exp.name
    ]
    if not matches:
        print(f"No MLflow experiments matched substring: {substring}")
        return

    for exp in matches:
        try:
            client.delete_experiment(exp.experiment_id)
            print(f"Deleted MLflow experiment: {exp.name}")
        except Exception as exc:
            print(f"Unable to delete MLflow experiment {exp.name}: {exc}")


def _delete_registered_model(model_name: str) -> None:
    if not model_name:
        return
    try:
        import mlflow
        from mlflow import MlflowClient
    except Exception as exc:
        print(f"Unable to load MLflow for model cleanup: {exc}")
        return

    try:
        mlflow.set_registry_uri("databricks-uc")
        MlflowClient().delete_registered_model(name=model_name)
        print(f"Deleted Unity Catalog model: {model_name}")
    except Exception as exc:
        if "RESOURCE_DOES_NOT_EXIST" in str(exc):
            print(f"Unity Catalog model already missing: {model_name}")
        else:
            print(f"Unable to delete model {model_name}: {exc}")


def _get_feature_engineering_client():
    try:
        from databricks.feature_engineering import FeatureEngineeringClient
    except Exception as exc:
        print(f"FeatureEngineeringClient unavailable: {exc}")
        return None
    try:
        return FeatureEngineeringClient()
    except Exception as exc:
        print(f"Unable to create FeatureEngineeringClient: {exc}")
        return None


def _delete_online_store(store_name: str) -> None:
    if not store_name:
        return
    fe_client = _get_feature_engineering_client()
    if not fe_client:
        return

    delete_fn = getattr(fe_client, "delete_online_store", None)
    if not callable(delete_fn):
        print("FeatureEngineeringClient has no delete_online_store method.")
        return

    try:
        delete_fn(name=store_name)
        print(f"Deleted online feature store: {store_name}")
    except Exception as exc:
        if "RESOURCE_DOES_NOT_EXIST" in str(exc):
            print(f"Online store already missing: {store_name}")
        else:
            print(f"Unable to delete online store {store_name}: {exc}")


def _drop_feature_table(table_name: str) -> None:
    if not table_name:
        return
    fe_client = _get_feature_engineering_client()
    table_fn = None
    if fe_client:
        table_fn = getattr(fe_client, "drop_table", None) or getattr(
            fe_client, "delete_table", None
        )
    if callable(table_fn):
        try:
            table_fn(name=table_name)
            print(f"Dropped feature store table: {table_name}")
            return
        except Exception as exc:
            print(
                f"Unable to drop feature table via client {table_name}: {exc}"
            )
    _drop_table_if_exists(table_name, "feature table")


def cleanup1() -> None:
    """Undo tables created in 01_data_ingest."""
    _drop_table_if_exists(
        f"{CATALOG}.{MY_SCHEMA}.coffee_labeled", "labeled training table"
    )
    _drop_table_if_exists(
        f"{CATALOG}.{MY_SCHEMA}.coffee_prod_holdout",
        "production holdout table",
    )


def cleanup2() -> None:
    """Module 02 is read-only; nothing to undo."""
    print("Module 02 does not create persistent assets—nothing to clean.")


def cleanup3() -> None:
    """Remove feature store artifacts from 03_feature_store."""
    _drop_table_if_exists(
        f"{CATALOG}.{MY_SCHEMA}.coffee_labeled_fact", "fact table"
    )
    _drop_feature_table(f"{CATALOG}.{MY_SCHEMA}.coffee_features")
    _drop_table_if_exists(
        f"{CATALOG}.{MY_SCHEMA}.online_store_coffee", "online feature table"
    )
    _delete_online_store(f"{MY_NAME.replace('_', '-')}-online-store")


def cleanup4() -> None:
    """Remove AutoML experiments and notebooks."""
    _delete_mlflow_experiments_matching("coffee_automl_")
    _delete_mlflow_experiments_matching("automl_experiments")
    _delete_workspace_dir(f"/Workspace/Users/{USER_EMAIL}/automl_experiments")


def cleanup5() -> None:
    """Remove Optuna experiment artifacts and registered model."""
    _delete_named_mlflow_experiment(
        f"/Workspace/Users/{USER_EMAIL}/coffee_hp_tuning_experiment"
    )
    _delete_registered_model(f"{CATALOG}.{MY_SCHEMA}.coffee_xgb_model")


def cleanup6() -> None:
    """Remove prediction simulation outputs."""
    _drop_table_if_exists(
        f"{CATALOG}.{MY_SCHEMA}.coffee_prod_predictions",
        "production predictions table",
    )
    _delete_named_mlflow_experiment(
        f"/Workspace/Users/{USER_EMAIL}/coffee_prod_predictions"
    )


def cleanup7() -> None:
    """Delete the Databricks Model Serving endpoint."""
    model_name = "coffee_xgb_model"
    endpoint_name = f"{MY_NAME}_{model_name}_endpoint"
    try:
        import mlflow.deployments
    except Exception as exc:
        print(f"Unable to load MLflow deployments client: {exc}")
        return

    try:
        client = mlflow.deployments.get_deploy_client("databricks")
        client.delete_endpoint(endpoint_name)
        print(f"Deleted serving endpoint: {endpoint_name}")
    except Exception as exc:
        if "RESOURCE_DOES_NOT_EXIST" in str(exc) or "not found" in str(exc):
            print(f"Serving endpoint already missing: {endpoint_name}")
        else:
            print(f"Unable to delete serving endpoint {endpoint_name}: {exc}")


def clean_all() -> None:
    """Run every cleanup helper in dependency order."""
    for fn in (
        cleanup7,
        cleanup6,
        cleanup5,
        cleanup4,
        cleanup3,
        cleanup2,
        cleanup1,
    ):
        try:
            fn()
        except Exception as exc:
            print(f"Cleanup '{fn.__name__}' failed: {exc}")


# COMMAND ----------

# DBTITLE 1,HINTS
HINTS = {
    ("predictions","quest_1"): 
      """
        <details class="hintbox">
          <summary>Show me the hint!</summary>
          <p>If you navigate to Catalog > gtc25_ml_catalog > gtc2025_(your_user_number) > Models and click on the coffee_xgb_model, you will see the alias of the latest version of the model.</p>
        </details>
        <details class="hintbox">
          <summary>Just show me the answer… 🫠</summary>
          <pre><code>alias = "@champion"</code></pre>
        </details>
        """,
    ("predictions","quest_2"):
      """
        <details class="hintbox">
          <summary>Show me the hint!</summary>
          <p class="answer"><strong>Hint for Question 1</strong></p>
          <p>Use the following commands and try to identify the extra columns</p>
          <pre>
            <code>
            print(predictions_df.columns)
            print(holdout_df.columns)
            </code>
          </pre>
          <p class="answer"><strong>Hint for Question 2</strong></p>
          <p>Use the following command to see a sample of the data</p>
          <pre>
            <code>
            holdout_df.display()
            </code>
          </pre>
          <p>Additional reading</p>
          <ul>
            <li><a href="https://spark.apache.org/docs/latest/ml-features.html#stringindexer">StringIndexer</a></li>
            <li><a href="https://spark.apache.org/docs/latest/ml-features.html#onehotencoder">OneHotEncoder</a></li>
            <li><a href="https://spark.apache.org/docs/latest/ml-features.html#vectorassembler">VectorAssembler</a></li>
            <li><a href="https://xgboost.readthedocs.io/en/stable/tutorials/spark_estimator.html#sparkxgbclassifier">SparkXGBClassifier</a></li>
          </ul>
        </details>
        <details class="hintbox">
          <summary>Just show me the answer… 🫠</summary>
          <p class="answer"><strong>Answer to Question 1</strong></p>
          <p>These columns were added by the model:</p>
          <ul>
            <li>Gender_idx</li>
            <li>Sleep_Quality_idx</li>
            <li>Stress_Level_idx</li>
            <li>Health_Issues_idx</li>
            <li>Occupation_idx</li>
            <li>Breakfast_Type_idx</li>
            <li>Gender_ohe</li>
            <li>Sleep_Quality_ohe</li>
            <li>Stress_Level_ohe</li>
            <li>Health_Issues_ohe</li>
            <li>Occupation_ohe</li>
            <li>Breakfast_Type_ohe</li>
            <li>features</li>
            <li>rawPrediction</li>
            <li>prediction</li>
            <li>probability</li>
          </ul>

          <p class="answer"><strong>Answer to Question 2</strong></p>
          <p>What these columns represent:</p>

          <ul>
            <li><strong>“_idx” columns:</strong> Numeric category indices created by <em>StringIndexer</em>, used as inputs to the OneHotEncoder.</li>
            <li><strong>“_ohe” columns:</strong> One-hot encoded vectors produced by <em>OneHotEncoder</em>, used as inputs by the VectorAssembler.</li>
            <li><strong>“features”:</strong> Final feature vector assembled by <em>VectorAssembler</em> and used to train the model.</li>
            <li><strong>“rawPrediction”:</strong> Raw model output scores (margins) for each class.</li>
            <li><strong>“prediction”:</strong> Final predicted class label.</li>
            <li><strong>“probability”:</strong> Probability distribution across all classes.</li>
          </ul>

        </details>
        """,
    ("predictions","quest_3",):
      """
          <details class="hintbox">
            <summary>Show me the hint!</summary>

            <p class="answer"><strong>Hint for Question 1</strong></p>
            <p>When you run the cell below there is an output...check it out!</p>

            <p class="answer"><strong>Hint for Question 2</strong></p>
            <p>Open the Experiment created by mlflow and take a look around!</p>

          </details>

          <details class="hintbox">
            <summary>Just show me the answer… 🫠</summary>

            <p class="answer"><strong>Answer to Question 1</strong></p>
            <p>When you run the following code, MLflow creates the output:</p>
            <p><code>Logged 1 run to an experiment in MLflow.</code></p>
            <p>You can see it in the <a href="https://adb-1451829595406012.12.azuredatabricks.net/ml/experiments">Experiments</a> page.</p>
            <p>The "run" and "experiment" are links to the corresponding resources. Clicking on them will take you to the Experiments page where you can see the results.</p>

            <p class="answer"><strong>Answer to Question 2</strong></p>
            <p>The metrics that were logged for evaluating the model are:</p>
            <ul>
              <li>true_negatives</li>
              <li>false_positives</li>
              <li>false_negatives</li>
              <li>true_positives</li>
              <li>example_count</li>
              <li>accuracy_score</li>
              <li>recall_score</li>
              <li>precision_score</li>
              <li>f1_score</li>
            </ul>

          </details>
        """,
    (
        "model_training",
        "quest_1",
    ): """
        <details class="hintbox">
          <summary>Show me the hint!</summary>
          <ul>
            <li>Look at the columns: <code><span class="string">"Alcohol_Consumption"</span></code>  and <code><span class="string">"Smoking"</span></code></li>
            <li>Do they look like numeric columns?</li>
          </ul>
        </details>
        <details class="hintbox">
          <summary>Just show me the answer… 🫠</summary>
          <p>Run this block of code <b> after the execution of the cell below</b> before moving on:</p>
          <div class="code-block">
            <span class="variable">numeric_cols</span><span class="operator">.</span><span class="function">remove</span><span class="bracket">(</span><span class="string">"Alcohol_Consumption"</span><span class="bracket">)<br></span>
            <span class="variable">numeric_cols</span><span class="operator">.</span><span class="function">remove</span><span class="bracket">(</span><span class="string">"Smoking"</span><span class="bracket">)<br></span>
            <span class="variable">categorical_cols</span> <span class="operator">+=</span> <span class="bracket">[</span><span class="string">"Alcohol_Consumption"</span><span class="operator">,</span> <span class="string">"Smoking"</span><span class="bracket">]</span>
          </div>
        </details>
        """,
    (
        "model_training",
        "quest_2",
    ): """
        <details class="hintbox">
          <summary>Show me the hint!</summary>
          <p>Reuse the feature list you derived in Quest 1.</p>
        </details>
        <details class="hintbox">
          <summary>Just show me the answer… 🫠</summary>
          <div class="code-block">
            <pre><span class="variable">feature_names</span><span class="operator">=</span><span class="variable">all_feature_cols</span>
<span class="variable">feature_lookups</span><span class="operator">=</span><span class="bracket">[</span><span class="variable">feature_lookup</span><span class="bracket">]</span></pre>
          </div>
        </details>

        """,
    (
        "model_training",
        "quest_3",
    ): """
        <details class="hintbox">
          <summary>Show me the hint!</summary>
          <p>Use three float numbers in the placeholder list.</p>
        </details>
        <details class="hintbox">
          <summary>Just show me the answer… 🫠</summary>
          <div class="code-block">
            <pre><span class="variable">train_df</span><span class="operator">,</span> <span class="variable">valid_df</span><span class="operator">,</span> <span class="variable">test_df</span> <span class="operator">=</span> <span class="variable">full_labeled_df</span><span class="operator">.</span><span class="function">randomSplit</span><span class="bracket">(</span>
    <span class="bracket">[</span><span class="comment">0.6</span><span class="operator">,</span> <span class="comment">0.2</span><span class="operator">,</span> <span class="comment">0.2</span><span class="bracket">]</span><span class="operator">,</span> <span class="variable">seed</span><span class="operator">=</span><span class="comment">42</span>
<span class="bracket">)</span></pre>
          </div>
        </details>
        """,
    (
        "model_training",
        "quest_4",
    ): """
        <details class="hintbox">
          <summary>Show me the hint!</summary>
          <p>Review the AutoML experiment: <a href="https://adb-1451829595406012.12.azuredatabricks.net/?o=1451829595406012#mlflow/experiments/895133209892610" target="_blank">Coffee AutoML run</a>.</p>
        </details>
        <details class="hintbox">
          <summary>Just show me the answer… 🫠</summary>
          <div class="code-block">
            <pre><span class="variable">seed_params</span> <span class="operator">=</span> <span class="bracket">{</span>
    <span class="string">"eta"</span><span class="operator">:</span> <span class="comment">0.05759496965676729</span><span class="operator">,</span>
    <span class="string">"colsample_bytree"</span><span class="operator">:</span> <span class="comment">0.6263993741226758</span><span class="operator">,</span>
    <span class="string">"max_depth"</span><span class="operator">:</span> <span class="comment">9</span><span class="operator">,</span>
    <span class="string">"min_child_weight"</span><span class="operator">:</span> <span class="comment">5.0</span><span class="operator">,</span>
    <span class="string">"subsample"</span><span class="operator">:</span> <span class="comment">0.6616262667209235</span><span class="operator">,</span>
<span class="bracket">}</span></pre>
          </div>
        </details>
        """,
    (
        "model_training",
        "quest_5",
    ): """
        <details class="hintbox">
          <summary>Show me the hint!</summary>
          <ul>
            <li>fill unionByName(...) with the df what we used for <b>validating</b> hyperparameters.</li>
            <li>fit(...) <b>trains</b> the model.</li>
            <li>transform(...) makes predictions to <b>test</b> the model.</li>
          </ul>
        </details>
        <details class="hintbox">
          <summary>Just show me the answer… 🫠</summary>
          <div class="code-block">
            <pre><span class="variable">train_val_df</span> <span class="operator">=</span> <span class="variable">train_df</span><span class="operator">.</span><span class="function">unionByName</span><span class="bracket">(</span><span class="variable">valid_df</span><span class="bracket">)</span> 
<span class="function">print</span><span class="bracket">(</span><span class="string">f"Train + validation rows: {train_val_df.count():,}"</span><span class="bracket">)</span>

<span class="variable">best_model</span> <span class="operator">=</span> <span class="variable">best_pipeline</span><span class="operator">.</span><span class="function">fit</span><span class="bracket">(</span><span class="variable">train_val_df</span><span class="bracket">)</span>  

<span class="variable">test_pred_df</span> <span class="operator">=</span> <span class="variable">best_model</span><span class="operator">.</span><span class="function">transform</span><span class="bracket">(</span><span class="variable">test_df</span><span class="bracket">)</span>  
<span class="variable">test_prec0</span><span class="operator">,</span> <span class="variable">test_rec0</span><span class="operator">,</span> <span class="variable">test_f10</span> <span class="operator">=</span> <span class="function">class_zero_metrics</span><span class="bracket">(</span>
    <span class="variable">test_pred_df</span><span class="operator">,</span> <span class="variable">LABEL_COL</span><span class="operator">,</span> <span class="variable">PREDICTION_COL</span>
<span class="bracket">)</span></pre>
          </div>
        </details>
        """,
    (
        "model_training",
        "quest_6",
    ): """
        <details class="hintbox">
          <summary>Show me the hint!</summary>
          <p>Take a break 🏖️! Just run the cell!</p>
        </details>
        <details class="hintbox">
          <summary>Just show me the answer… 🫠</summary>
          <p>Seriously, just run it 🚀</p>
        </details>
        """,
    (
        "model_training",
        "quest_7",
    ): """
        <details class="hintbox">
          <summary>Show me the hint!</summary>
          <ul>
            <li>It could be done with the addition of just one line with conditional logic</li>
            <li>There is no room for two champions...⚔️ </li>
          </ul>
        </details>
<details class="hintbox">
          <summary>Just show me the answer… 🫠</summary>
          <div class="code-block">
            <pre><span class="keyword">if</span> <span class="variable">challenger_f1</span> <span class="operator">&gt;</span> <span class="variable">champion_f1</span><span class="operator">:</span>
            <span class="comment"># This automatically removes the champion alias from the previous version</span>
            <span class="variable">client</span><span class="operator">.</span><span class="function">set_registered_model_alias</span><span class="bracket">(</span>
                <span class="variable">UC_MODEL_NAME</span><span class="operator">,</span> <span class="string">"champion"</span><span class="operator">,</span> <span class="variable">challenger_version</span>
            <span class="bracket">)</span>
            <span class="comment"># Mark the previous champion</span>
            <span class="variable">client</span><span class="operator">.</span><span class="function">set_registered_model_alias</span><span class="bracket">(</span>
                <span class="variable">UC_MODEL_NAME</span><span class="operator">,</span> <span class="string">"previous_champion"</span><span class="operator">,</span> <span class="variable">champion_version</span>
            <span class="bracket">)</span>
            <span class="function">print</span><span class="bracket">(</span><span class="string">f"Challenger wins! Model version {challenger_version} is now the new champion."</span><span class="bracket">)</span></pre>
            <pre><span class="keyword">else</span><span class="operator">:</span>
            <span class="function">print</span><span class="bracket">(</span><span class="string">f"Champion wins! Model version {champion_version} remains the champion."</span><span class="bracket">)</span></pre>
          </div>
        </details>
        """,
}

# COMMAND ----------

# DBTITLE 1,load_hint

def load_hint(notebook, quest_id):
    base_css = """
    <style>

    * {
      font-size: 16px;
    }

    .hintbox summary {
      cursor: pointer;
      font-weight: bold;
      margin-bottom: 5px;
    }
    .hintbox {
      margin: 10px 0;
      padding: 10px;
      border: 1px solid #ff00ff;
      border-radius: 6px;
    }

    .hintbox ul {
      padding-left: 20px;
    }

    /* Make list bullets magenta (not the text) */
    .hintbox ul li {
      list-style-type: disc;
    }

    .hintbox ul li::marker {
      color: #ff00ff;
    }
    .hintbox code {
      background: #f5f5f5;
      display: block;
      padding: 10px;
      border-radius: 4px;
      white-space: pre;
    }

    /* New magenta paragraph class */
    .answer {
      color: #ff00ff;
      text-decoration: underline;
    }

    /* Code block styling */
    .code-block {
      background: #f0f0f0;
      padding: 20px;
      border-radius: 8px;
      margin: 10px 0;
      overflow-x: auto;
    }

    .code-block pre {
      margin: 0;
      font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
      font-size: 14px;
      line-height: 1.6;
      color: #5c5c5c;
    }

    .comment {
        color: #008000;
        font-weight: bold;
    }

    .keyword {
        color: #0000ff;
    }

    .function {
        color: #795e26;
    }

    .string {
        color: #c72e0f;
    }

    .variable {
        color: #5c5c5c;
    }

    .operator {
        color: #5c5c5c;
    }

    .bracket {
        color: #0000ff;
    }
    </style>
    """

    body = HINTS.get((notebook, quest_id))
    if not body:
        displayHTML("No hint available.")
        return

    displayHTML(base_css + body)
