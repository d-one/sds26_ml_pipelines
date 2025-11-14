# Databricks notebook source
# MAGIC %md
# MAGIC # Coffee Module 07 · Model Serving
# MAGIC Deploy the registered XGBoost model to a Databricks Model Serving endpoint. Fill in each `...` placeholder before running the cell.

# COMMAND ----------

# DBTITLE 1,Run setup
# MAGIC %run ./00_setup

# COMMAND ----------

# DBTITLE 1,Imports & Constants
import mlflow
import mlflow.deployments
from mlflow.tracking import MlflowClient

MODEL_NAME = "coffee_xgb_model"
MODEL_PATH = f"{CATALOG}.{MY_SCHEMA}.{MODEL_NAME}"
CHAMPION_MODEL_URI = f"models:/{MODEL_PATH}@champion"

ENDPOINT_NAME = f"{MY_NAME}_{MODEL_NAME}_endpoint"
SERVED_ENTITY_NAME = f"{MY_NAME}_coffee_endpoint"
ENDPOINT_DESCRIPTION = "Endpoint that predicts whether a person drinks coffee."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 1 · Retrieve the Champion Metadata
# MAGIC **Goal:** grab the champion model version via the MLflow client.
# MAGIC
# MAGIC **Hints**
# MAGIC - Replace the alias placeholder with `"champion"` when calling `get_model_version_by_alias`.

# COMMAND ----------

# DBTITLE 1,Create a serving endpoint using MLflow Deployments SDK
mlflow_client = MlflowClient()
model = mlflow_client.get_model_version_by_alias(name=MODEL_PATH, alias="...")                     # replace placeholder
model_version = model.version

client = mlflow.deployments.get_deploy_client("databricks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 2 · Define the Endpoint Configuration
# MAGIC **Goal:** build the `endpoint_config` dictionary with the updated metadata.
# MAGIC
# MAGIC **Hints**
# MAGIC - Ensure `ENDPOINT_NAME` stays below 64 characters.
# MAGIC - Populate `entity_name` with `MODEL_PATH` and `entity_version` with `model_version`.
# MAGIC - Keep `workload_size="Small"` and `scale_to_zero_enabled=True` to automatically shut the serving endpoint down when idle to save compute, and spin it back up when traffic arrives.

# COMMAND ----------

if len(ENDPOINT_NAME) >= ...:                                                                       # replace placeholder
    raise Exception(
        f"Endpoint name '{ENDPOINT_NAME}' is too long. Must be less than 63 characters."
    )
endpoint_config = {
    "description": ENDPOINT_DESCRIPTION,
    "served_entities": [
        {
            "name": SERVED_ENTITY_NAME,
            "entity_name": ...,                                                                     # replace placeholders
            "entity_version": ...,
            "workload_size": "...",
            "scale_to_zero_enabled": ...,
        }  # ,  We can serve more than one entities at once
        #  {
        #     "name":"challenger",
        #     "entity_name":"catalog.schema.model-B",
        #     "entity_version":"1",
        #     "workload_size":"Small",
        #     "scale_to_zero_enabled":true
        #  }
    ],
    "traffic_config": {
        "routes": [
            {
                "served_model_name": SERVED_ENTITY_NAME,
                "traffic_percentage": "100",
            }  # ,
            # {
            #    "served_model_name":"challenger",
            #    "traffic_percentage":"10" // was 90%/10% with the above
            # }
        ]
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 3 · Deploy or Update the Endpoint
# MAGIC **Goal:** use the MLflow Deployments client to create or update the serving endpoint.
# MAGIC
# MAGIC **Hints**
# MAGIC - Call `client.get_endpoint(ENDPOINT_NAME)`

# COMMAND ----------

try:
    existing = client.get_endpoint(...)                                                               # replace placeholder
    print(f"✅ Endpoint '{ENDPOINT_NAME}' already exists. Updating it...")

    client.update_endpoint(
        ENDPOINT_NAME,
        config=endpoint_config,
    )
except Exception as e:
    # The Databricks API throws an error if endpoint does not exist
    if "RESOURCE_DOES_NOT_EXIST" in str(e) or "does not exist" in str(e):
        print(f"🆕 Endpoint '{ENDPOINT_NAME}' not found. Creating it...")
        client.create_endpoint(
            name=ENDPOINT_NAME,
            config=endpoint_config,
            route_optimized=False,
        )
    else:
        raise e
