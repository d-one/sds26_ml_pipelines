# Databricks notebook source
# DBTITLE 1,Settings & Configuration
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
MY_SCHEMA = USER_EMAIL.split("@")[0]
MY_SCHEMA = re.sub(r"[^a-zA-Z0-9_]", "_", MY_SCHEMA)

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

# COMMAND ----------

# DBTITLE 1,Column categories
NUMERICAL_COLUMNS = [
    "Age",
    "Sleep_Hours",
    "BMI",
    "Heart_Rate",
    "Physical_Activity_Hours",
    "Work_Hours_Per_Week",
    "Tea_Consumption_Per_Day_ml",
    "Energy_Drink_Consumption_ml",
]
CATEGORICAL_COLUMNS = [
    "Gender",
    "Sleep_Quality",
    "Stress_Level",
    "Health_Issues",
    "Occupation",
    "Breakfast_Type",
    "Alcohol_Consumption",
    "Smoking",
]

FEATURE_COLUMNS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS

# COMMAND ----------

# DBTITLE 1,Function: class_zero_metrics

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

# DBTITLE 1,Function: build_preprocessing_stages
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler


def build_preprocessing_stages():
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in CATEGORICAL_COLUMNS
    ]
    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in CATEGORICAL_COLUMNS],
        outputCols=[f"{c}_ohe" for c in CATEGORICAL_COLUMNS],
        handleInvalid="keep",
        dropLast=True,
    )

    assembler_input_cols = NUMERICAL_COLUMNS + [
        f"{c}_ohe" for c in CATEGORICAL_COLUMNS
    ]
    assembler = VectorAssembler(
        inputCols=assembler_input_cols,
        outputCol="features",
        handleInvalid="keep",
    )
    stages = indexers + [encoder, assembler]
    print(
        "\n1. String Indexers, these convert text categories into numeric indices"
    )
    for c in CATEGORICAL_COLUMNS:
        print(f"   StringIndexer: {c} -> {c}_idx")
    print()

    print(
        "2. One Hot Encoders, these turn each index into a vector of binary flags"
    )
    for c in CATEGORICAL_COLUMNS:
        print(f"   OneHotEncoder: {c}_idx -> {c}_ohe")
    print()

    print(
        "3. Vector Assembler, this gathers all numeric and encoded features into a single feature vector"
    )
    all_inputs = NUMERICAL_COLUMNS + [c + "_ohe" for c in CATEGORICAL_COLUMNS]
    print(f"   VectorAssembler: {all_inputs} -> features\n")
    return stages


# COMMAND ----------

# DBTITLE 1,Function: setup_experiment
import mlflow


def setup_experiment(experiment_name):
    mlflow.autolog(disable=True)

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = mlflow.create_experiment(experiment_name)
    else:
        exp_id = exp.experiment_id

    mlflow.set_experiment(experiment_name)
    return exp_id


# COMMAND ----------

# DBTITLE 1,Functions: clean up functions
# Databricks Schema Complete Cleanup Function
# Cleans up all tables, views, functions, and registered models in a schema

from mlflow import MlflowClient


def cleanup_databricks_schema(catalog, schema, dry_run=False):
    """
    Clean up all objects in a Databricks Unity Catalog schema.

    Parameters:
    -----------
    catalog : str
        The catalog name (e.g., 'gtc25_ml_catalog')
    schema : str
        The schema name (e.g., 'michalis_kalligas')
    dry_run : bool, optional
        If True, only shows what would be deleted without actually deleting (default: False)

    Returns:
    --------
    dict
        Summary of deleted objects
    """
    full_schema = f"{catalog}.{schema}"
    summary = {
        "tables": 0,
        "views": 0,
        "functions": 0,
        "models": 0,
        "model_versions": 0,
        "errors": [],
    }

    print("=" * 70)
    print(f"{'DRY RUN - ' if dry_run else ''}Cleanup of {full_schema}")
    print("=" * 70)

    # Set the current catalog to avoid context issues
    try:
        spark.sql(f"USE CATALOG {catalog}")
        print(f"Set current catalog to: {catalog}")
    except Exception as e:
        print(f"Warning: Could not set catalog: {e}")

    # ==================================================================
    # PART 1: Clean up Tables and Views
    # ==================================================================
    print(
        f"\n{'[DRY RUN] ' if dry_run else ''}Cleaning up tables and views..."
    )

    try:
        tables_df = spark.sql(f"SHOW TABLES IN {full_schema}")
        tables_list = tables_df.collect()

        print(f"Found {len(tables_list)} table/view objects")

        for row in tables_list:
            table_name = row.tableName
            is_temp = row.isTemporary

            if not is_temp:
                full_name = f"{full_schema}.{table_name}"

                if dry_run:
                    print(f"  [DRY RUN] Would drop: {full_name}")
                    summary["tables"] += 1
                else:
                    try:
                        # Try dropping as table first
                        spark.sql(f"DROP TABLE IF EXISTS {full_name}")
                        print(f"  ✓ Dropped table: {full_name}")
                        summary["tables"] += 1
                    except:
                        try:
                            # If that fails, try as view
                            spark.sql(f"DROP VIEW IF EXISTS {full_name}")
                            print(f"  ✓ Dropped view: {full_name}")
                            summary["views"] += 1
                        except Exception as e:
                            error_msg = (
                                f"Failed to drop {full_name}: {str(e)[:100]}"
                            )
                            print(f"  ✗ {error_msg}")
                            summary["errors"].append(error_msg)
    except Exception as e:
        error_msg = f"Error listing tables: {str(e)[:100]}"
        print(f"✗ {error_msg}")
        summary["errors"].append(error_msg)

    # ==================================================================
    # PART 2: Clean up Functions
    # ==================================================================
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Cleaning up functions...")

    try:
        # Make sure we're using the right catalog and schema
        spark.sql(f"USE CATALOG {catalog}")
        spark.sql(f"USE SCHEMA {schema}")

        functions_df = spark.sql(f"SHOW FUNCTIONS IN {full_schema}")
        functions_list = functions_df.collect()

        schema_functions = [
            row.function
            for row in functions_list
            if full_schema in row.function
        ]
        print(f"Found {len(schema_functions)} functions")

        for func_name in schema_functions:
            if dry_run:
                print(f"  [DRY RUN] Would drop function: {func_name}")
                summary["functions"] += 1
            else:
                try:
                    spark.sql(f"DROP FUNCTION IF EXISTS {func_name}")
                    print(f"  ✓ Dropped function: {func_name}")
                    summary["functions"] += 1
                except Exception as e:
                    error_msg = (
                        f"Failed to drop function {func_name}: {str(e)[:100]}"
                    )
                    print(f"  ✗ {error_msg}")
                    summary["errors"].append(error_msg)
    except Exception as e:
        print(f"No functions found or error: {str(e)[:200]}")

    # ==================================================================
    # PART 3: Clean up Registered Models
    # ==================================================================
    print(
        f"\n{'[DRY RUN] ' if dry_run else ''}Cleaning up registered models..."
    )

    try:
        client = MlflowClient()

        # Get all registered models and filter by our schema
        print("  Searching for models...")
        all_models = client.search_registered_models(max_results=10000)

        # Filter to only models in our specific schema
        schema_prefix = f"{full_schema}."
        schema_models = []

        for model in all_models:
            if model.name.startswith(schema_prefix):
                schema_models.append(model)
                print(f"  Found: {model.name}")

        print(f"\nTotal models found in {full_schema}: {len(schema_models)}")

        if len(schema_models) == 0:
            print("  No models to delete")
        else:
            # List all models that will be affected
            print(
                f"\n  Models to be {'deleted' if not dry_run else 'affected'}:"
            )
            for model in schema_models:
                print(f"    - {model.name}")

            if dry_run:
                print(
                    f"\n  [DRY RUN] Would process {len(schema_models)} models"
                )

            # Process each model
            for model in schema_models:
                model_name = model.name

                # CRITICAL SAFETY CHECK
                if not model_name.startswith(schema_prefix):
                    error_msg = f"SAFETY CHECK FAILED: {model_name} doesn't start with {schema_prefix}"
                    print(f"  ✗ {error_msg}")
                    summary["errors"].append(error_msg)
                    continue

                try:
                    # Get all versions
                    versions = client.search_model_versions(
                        f"name='{model_name}'"
                    )
                    print(
                        f"\n  Model: {model_name} ({len(versions)} versions)"
                    )

                    if dry_run:
                        print(
                            f"    [DRY RUN] Would delete {len(versions)} versions and the model"
                        )
                        summary["models"] += 1
                        summary["model_versions"] += len(versions)
                    else:
                        # Delete all versions first
                        for version in versions:
                            try:
                                client.delete_model_version(
                                    name=model_name, version=version.version
                                )
                                print(
                                    f"    ✓ Deleted version {version.version}"
                                )
                                summary["model_versions"] += 1
                            except Exception as e:
                                error_msg = f"Failed to delete version {version.version}: {str(e)[:100]}"
                                print(f"    ✗ {error_msg}")
                                summary["errors"].append(error_msg)

                        # Delete the registered model
                        client.delete_registered_model(model_name)
                        print(f"  ✓ Deleted registered model: {model_name}")
                        summary["models"] += 1

                except Exception as e:
                    error_msg = (
                        f"Failed to process model {model_name}: {str(e)[:100]}"
                    )
                    print(f"  ✗ {error_msg}")
                    summary["errors"].append(error_msg)

    except Exception as e:
        error_msg = f"Error cleaning up models: {str(e)[:200]}"
        print(f"✗ {error_msg}")
        summary["errors"].append(error_msg)

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 70)
    print(f"{'DRY RUN ' if dry_run else ''}CLEANUP SUMMARY")
    print("=" * 70)
    print(f"Tables dropped:           {summary['tables']}")
    print(f"Views dropped:            {summary['views']}")
    print(f"Functions dropped:        {summary['functions']}")
    print(f"Models deleted:           {summary['models']}")
    print(f"Model versions deleted:   {summary['model_versions']}")
    print(f"Errors encountered:       {len(summary['errors'])}")

    if summary["errors"]:
        print("\nErrors:")
        for error in summary["errors"][:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(summary["errors"]) > 10:
            print(f"  ... and {len(summary['errors']) - 10} more errors")

    if dry_run:
        print("\n⚠️  This was a DRY RUN - no objects were actually deleted")
        print("   Run with dry_run=False to perform actual deletion")
    else:
        print(f"\n✓ Cleanup of {full_schema} completed!")

    print("=" * 70)

    return summary


# ==================================================================
# USAGE EXAMPLES
# ==================================================================

# Example 1: Dry run to see what would be deleted (RECOMMENDED FIRST)
# summary = cleanup_databricks_schema("gtc25_ml_catalog", "michalis_kalligas", dry_run=True)

# Example 2: Actual cleanup
# summary = cleanup_databricks_schema("gtc25_ml_catalog", "michalis_kalligas")

# Example 3: Cleanup another schema
# summary = cleanup_databricks_schema("my_catalog", "my_schema", dry_run=True)

# COMMAND ----------

# DBTITLE 1,HINTS
HINTS = {
    (
        "predictions",
        "quest_1",
    ): """
        <details class="hintbox">
          <summary>Show me the hint!</summary>
          <p>If you navigate to Catalog > gtc25_ml_catalog > gtc2025_(your_user_number) > Models and click on the coffee_xgb_model, you will see the alias of the latest version of the model.</p>
        </details>
        <details class="hintbox">
          <summary>Just show me the answer… 🫠</summary>
          <pre><code>alias = "@champion"</code></pre>
        </details>
        """,
    (
        "predictions",
        "quest_2",
    ): """
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
    (
        "predictions",
        "quest_3",
    ): """
          <details class="hintbox">
            <summary>Show me the hint!</summary>

            <p>Open the Experiment created by mlflow and take a look around!</p>

          </details>

          <details class="hintbox">
            <summary>Just show me the answer… 🫠</summary>

            <p>When you run the following code, MLflow creates the output:</p>
            <p><code>Logged 1 run to an experiment in MLflow.</code></p>
            <p>You can see it in the <a href="https://adb-1451829595406012.12.azuredatabricks.net/ml/experiments">Experiments</a> page.</p>
            <p>The "run" and "experiment" are links to the corresponding resources. Clicking on them will take you to the Experiments page where you can see the results.</p>

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
          <p>Use three decimal numbers in the placeholder list.</p>
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
        "quest_2",
    ): """
        <details class="hintbox">
          <summary>Show me the hint!</summary>
          <p>Review the AutoML experiment: <a href="https://adb-1451829595406012.12.azuredatabricks.net/ml/experiments/2194677846827561?o=1451829595406012" target="_blank">Coffee AutoML run</a>.</p>
          <p>You can click a model run with a suitable type (XGBoost) and search for parameters in the Overview tab! 🔎</p>
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
        "quest_3",
    ): """
        <details class="hintbox">
          <summary>Show me the hint!</summary>
          <ul>
            <li>fill unionByName(...) with the df what we used for <b>validating</b> hyperparameters.</li>
            <li>fit(...) <b>trains</b> the model on the <b>union</b> dataframe.</li>
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
        "quest_4",
    ): """
        <details class="hintbox">
          <summary>Show me the hint!</summary>
          <p class="answer"><strong>Hint for Question 1</strong></p>
          <p>It follows the standard three-level namespace of Unity Catalog.</p>
          <p class="answer"><strong>Hint for Question 2</strong></p>
          <p>What would happen if you had to train a new model? Or revert to an old one?</p>

        </details>
        <details class="hintbox">
          <summary>Just show me the answer… 🫠</summary>
          <p class="answer"><strong>Answer to Question 1</strong></p>
          <p>Navigate to Catalog > gtc25_ml_catalog > gtc2025_(your_user_number) > Models and click on the coffee_xgb_model</p>
          <p class="answer"><strong>Answer to Question 2</strong></p>
          <p>We use aliases on Databricks models so that production systems always point to a stable name like "champion" instead of a specific version, which lets us update or roll back models instantly without changing any downstream code.</p>

        </details>
        """,
}

# COMMAND ----------

# DBTITLE 1,Function: load_hint

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
