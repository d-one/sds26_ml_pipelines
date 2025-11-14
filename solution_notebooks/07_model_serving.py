# Databricks notebook source
# MAGIC %md
# MAGIC # Model Serving and Production Querying
# MAGIC - This notebook demonstrates how to serve a registered machine learning model using Databricks.
# MAGIC - The workflow includes:
# MAGIC     1. Deploying the **registered model** to a production-ready serving endpoint.
# MAGIC     2. **Querying the model** to obtain predictions, simulating production inference scenarios.
# MAGIC - The notebook provides step-by-step guidance for both serving and querying, ensuring a smooth transition from model registration to real-time production use.
# MAGIC
# MAGIC Documentation: https://docs.databricks.com/aws/en/machine-learning/model-serving/manage-serving-endpoints

# COMMAND ----------

# DBTITLE 1,Run setup
# MAGIC %run ./00_setup

# COMMAND ----------

# DBTITLE 1,Load model
import mlflow
import mlflow.deployments
from mlflow.tracking import MlflowClient

MODEL_NAME = "coffee_xgb_model"
MODEL_PATH = f"{CATALOG}.{MY_SCHEMA}.{MODEL_NAME}"
CHAMPION_MODEL_URI = f"models:/{MODEL_PATH}@champion"

ENDPOINT_NAME = f"{MY_NAME}_{MODEL_NAME}_endpoint"
SERVED_ENTITY_NAME = f"{MY_NAME}_coffee_endpoint"
ENDPOINT_DESCRIPTION = "This endpoint serves the model for predicting whether a person drinks coffee or not."

# Get the version of the model to be served
mlflow_client = MlflowClient()

model = mlflow_client.get_model_version_by_alias(name=MODEL_PATH, alias="champion")
model_version = model.version

# COMMAND ----------

# DBTITLE 1,Create a serving endpoint using MLflow Deployments SDK
client = mlflow.deployments.get_deploy_client("databricks")

# COMMAND ----------

# DBTITLE 1,Set endpoint parameters
# Endpoint name must be maximum 63 characters, and alphanumeric with hyphens and underscores allowed in between.
if len(ENDPOINT_NAME) >= 64:
    raise Exception(
        f"Endpoint name '{ENDPOINT_NAME}' is too long. Must be less than 63 characters."
    )
endpoint_config = {
    "description": ENDPOINT_DESCRIPTION,
    "served_entities": [
        {
            "name": SERVED_ENTITY_NAME,
            "entity_name": MODEL_PATH,
            "entity_version": model_version,
            "workload_size": "Small",
            "scale_to_zero_enabled": True,
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

# DBTITLE 1,Deploy or update endpoint
# Try to fetch the endpoint (if it already exists)
try:
    existing = client.get_endpoint(ENDPOINT_NAME)
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
