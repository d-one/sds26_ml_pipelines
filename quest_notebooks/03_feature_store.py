# Databricks notebook source
# MAGIC %md
# MAGIC # Coffee Module 03 · Feature Engineering
# MAGIC This module mirrors the refreshed feature-store workflow. Each quest cell contains `...` placeholders—replace them with working code before running.

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run ./00_setup

# COMMAND ----------

# DBTITLE 1,Imports
from databricks.feature_engineering import FeatureEngineeringClient

# COMMAND ----------

# DBTITLE 1,Constants
COFFEE_LABELED_DATA_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_labeled"
COFFEE_FACT_TABLE_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_labeled_fact"
FEATURE_TABLE_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_features"

PRIMARY_KEY_COL = "ID"
TIMESTAMP_COL = "Timestamp"
LABEL_COL = "Coffee_Intake_Binary"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 1 · Load the Labeled Dataset
# MAGIC **Goal:** hydrate `coffee_df` from the Unity Catalog table.
# MAGIC
# MAGIC **Hints**
# MAGIC - Use `spark.table(COFFEE_LABELED_DATA_PATH)`.
# MAGIC - `display(df.limit(10))` ensures the schema looks correct.

# COMMAND ----------

coffee_df = spark.table(...)                                                                 # replace placeholder
display(coffee_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 2 · Derive Fact and Feature Views
# MAGIC **Goal:** create `feature_df` (deduped by entity) and `fact_labeled_df` (label-only view).
# MAGIC
# MAGIC **Hints**
# MAGIC - Deduplicate on the primary key for the feature view.
# MAGIC - Create the fact view by selecting the ID, timestamp, and label columns.

# COMMAND ----------

feature_df = coffee_df.dropDuplicates([...])                                                  # replace placeholders
fact_labeled_df = coffee_df.select(..., ..., ...)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 3 · Register the Feature Table
# MAGIC **Goal:** ensure the Feature Store table exists with the proper schema and keys.
# MAGIC
# MAGIC **Hints**
# MAGIC - Instantiate `FeatureEngineeringClient()` once and reuse it.
# MAGIC - Only call `fe.create_table` when the Feature Store table is missing.
# MAGIC - Specify both `primary_keys` and `timestamp_keys`.

# COMMAND ----------

fe = FeatureEngineeringClient()

if not spark.catalog.tableExists(...):                                                         # replace placeholders
    fe.create_table(
        name=FEATURE_TABLE_PATH,
        primary_keys=[..., ...],
        schema=feature_df.schema,
        description="Feature set derived from the EDA-transformed coffee dataset.",
        timestamp_keys=[...],
    )
    print(f"Created Feature Store table: {FEATURE_TABLE_PATH}")
else:
    print(f"Feature Store table already exists: {FEATURE_TABLE_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 4 · Persist the Fact Table
# MAGIC **Goal:** write `fact_labeled_df` to Unity Catalog.
# MAGIC
# MAGIC **Hints**
# MAGIC - Use Delta overwrite mode with `overwriteSchema=True`.

# COMMAND ----------

fact_labeled_df.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(COFFEE_FACT_TABLE_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 5 · Write Features and Enable CDF
# MAGIC **Goal:** push the deduplicated features to the Feature Store and enable Change Data Feed.
# MAGIC
# MAGIC **Hints**
# MAGIC - Use `fe.write_table(..., mode="merge")`.
# MAGIC - Run an `ALTER TABLE` statement to enable Change Data Feed.

# COMMAND ----------

fe.write_table(
    name=FEATURE_TABLE_PATH,
    df=...,                                                                                     # replace placeholder
    mode="merge",
)
print(
    f"Feature table '{FEATURE_TABLE_PATH}' updated with {feature_df.count():,} records."
)

spark.sql(
    f"ALTER TABLE {FEATURE_TABLE_PATH} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 6 · Publish to the Online Store
# MAGIC **Goal:** create (or reuse) an online store and publish the feature table.
# MAGIC
# MAGIC **Hints**
# MAGIC - Replace underscores in `MY_NAME` before using it within the store name.
# MAGIC - Use `fe.create_online_store` inside a try/except so reruns reuse existing stores.
# MAGIC - Call `fe.publish_table` with the offline table name and desired `online_table_name`.

# COMMAND ----------

online_store_name = f"{MY_NAME.replace('_', '-')}-online-store"
try:
    online_store = fe.create_online_store(
        name=...,                                                                               # replace placeholder
        capacity="CU_2",  # Valid options: "CU_1", "CU_2", "CU_4", "CU_8"
    )
except Exception:
    online_store = fe.get_online_store(name=...)                                                # replace placeholder
    print("Online store already exists; reusing it.")

# Publish the feature table to the online store
online_table_name = f"{CATALOG}.{MY_SCHEMA}.online_store_coffee"
fe.publish_table(
    online_store=online_store,
    source_table_name=FEATURE_TABLE_PATH,
    online_table_name=online_table_name,
)
