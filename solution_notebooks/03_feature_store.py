# Databricks notebook source
# MAGIC %md
# MAGIC ### Why Write to the Feature Store?
# MAGIC - **Requirements:** Each feature table must include a stable primary key, a clearly defined feature timestamp, and a well-defined schema to support point-in-time correct joins.
# MAGIC - **Strengths vs. Delta Tables:** The Feature Store automates feature lineage, versioning, and online/offline consistency, giving stronger governance and easier reuse compared with ad hoc Delta tables.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Initialization

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run ./00_setup

# COMMAND ----------

# DBTITLE 1,Imports
from databricks.feature_engineering import FeatureEngineeringClient

# COMMAND ----------

# DBTITLE 1,Constants
# Table paths
COFFEE_FULL_DATA_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_labeled"
COFFEE_FACT_TABLE_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_labeled_fact"
FEATURE_TABLE_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_features"

# Columns
PRIMARY_KEY_COL = "ID"
TIMESTAMP_COL = "Timestamp"
LABEL_COL = "Coffee_Intake_Binary"

# COMMAND ----------

# DBTITLE 1,Load tables
coffee_df = spark.table(COFFEE_FULL_DATA_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Fact & Features Table

# COMMAND ----------

feature_df = coffee_df.dropDuplicates([PRIMARY_KEY_COL])
fact_labeled_df = coffee_df.select(PRIMARY_KEY_COL, TIMESTAMP_COL, LABEL_COL)

# COMMAND ----------

# DBTITLE 1,Feature store creation
fe = FeatureEngineeringClient()

if not spark.catalog.tableExists(FEATURE_TABLE_PATH):
    fe.create_table(
        name=FEATURE_TABLE_PATH,
        primary_keys=[PRIMARY_KEY_COL, TIMESTAMP_COL],
        schema=feature_df.schema,
        description="Feature set derived from the EDA-transformed coffee dataset.",
        timestamp_keys=[TIMESTAMP_COL],
    )
    print(f"\nCreated Feature Store table: {FEATURE_TABLE_PATH}")


# COMMAND ----------

# MAGIC %md
# MAGIC #### Output

# COMMAND ----------

fact_labeled_df.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(COFFEE_FACT_TABLE_PATH)

# COMMAND ----------

# DBTITLE 1,Write features table
fe.write_table(
    name=FEATURE_TABLE_PATH,
    df=feature_df,
    mode="merge",
)
print(
    f"\nFeature Store table '{FEATURE_TABLE_PATH}' updated with {feature_df.count():,} records."
)

# COMMAND ----------

# DBTITLE 1,Enable Change Data Feed
spark.sql(f"ALTER TABLE {FEATURE_TABLE_PATH} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")

# COMMAND ----------

# DBTITLE 1,--- (delete online store if already exists)
# To check if the online store exists and to delete it
# online_store = fe.get_online_store(name=f"{MY_NAME}-online-store")
# fe.delete_online_store(name=f"{MY_NAME}-online-store")

# COMMAND ----------

# Create an online store with specified capacity
MY_NAME = MY_NAME.replace("_", "-")
try:
    online_store = fe.create_online_store(
        name=f"{MY_NAME}-online-store",
        capacity="CU_2"  # Valid options: "CU_1", "CU_2", "CU_4", "CU_8"
    )
except Exception as e:
    print(e)
    print("Online store already exists")
    online_store = fe.get_online_store(name=f"{MY_NAME}-online-store")

# Publish the feature table to the online store
online_table_name = f"{CATALOG}.{MY_SCHEMA}.online_store_coffee"
fe.publish_table(
    online_store=online_store,
    source_table_name=FEATURE_TABLE_PATH,
    online_table_name=online_table_name
)
