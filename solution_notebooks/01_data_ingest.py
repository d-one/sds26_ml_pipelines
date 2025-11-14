# Databricks notebook source
# MAGIC %md
# MAGIC ### Initialization

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run ./00_setup

# COMMAND ----------

# DBTITLE 1,Imports
import pyspark.sql.functions as F

# COMMAND ----------

# DBTITLE 1,Constants
# Table paths
COFFEE_RAW_DATA_PATH = f"{CATALOG}.{SCHEMA_WITH_SOURCE_DATA}.coffee_raw"
GNI_PER_CAPITA_PATH = f"{CATALOG}.{SCHEMA_WITH_SOURCE_DATA}.gni_per_capita_data"
COFFEE_LABELED_DATA_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_labeled"
HOLDOUT_TABLE_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_prod_holdout"

# Columns
PRIMARY_KEY_COL = "ID"
TIMESTAMP_COL = "Timestamp"
LABEL_COL = "Coffee_Intake_Binary"

HOLDOUT_FRACTION = 0.1

# COMMAND ----------

# DBTITLE 1,Load tables
coffee_raw_df = spark.table(COFFEE_RAW_DATA_PATH)
gni_pc_df = spark.table(GNI_PER_CAPITA_PATH)

display(coffee_raw_df.limit(10))
display(gni_pc_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pre-processing

# COMMAND ----------

# DBTITLE 1,Find mismatches
countries_of_coffee_df = coffee_raw_df.select("Country").distinct()
countries_of_gni_df = gni_pc_df.select("country").distinct()
print(
    f"DataFrame coffee_df contains -{countries_of_coffee_df.count()}- countries"
)
print(
    f"DataFrame gni_pc_df contains -{countries_of_gni_df.count()}- countries"
)

# Find the mismatches in country names between the two DataFrames
join_to_find_mismatches_df = countries_of_coffee_df.join(
    countries_of_gni_df,
    countries_of_coffee_df.Country == gni_pc_df.country,
    "left_anti",
).collect()

print(
    "\nA left-anti join reveals which countries are in coffee_df but not in gni_pc_df"
)
missing_countries = [row["Country"] for row in join_to_find_mismatches_df]
for country in missing_countries:
    print(f"\t- {country}")

# COMMAND ----------

# DBTITLE 1,Fix mismatches
gni_pc_df = gni_pc_df.withColumn(
    "country",
    F.when(gni_pc_df.country == "United States", "USA")
    .when(gni_pc_df.country == "Korea, Rep.", "South Korea")
    .when(gni_pc_df.country == "United Kingdom", "UK")
    .otherwise(gni_pc_df.country),
)

# COMMAND ----------

# DBTITLE 1,Enrich Coffee Consumption data with GNI per Capita data
enriched_coffee_df = (
    coffee_raw_df.join(gni_pc_df, coffee_raw_df.Country == gni_pc_df.country, "left")
    .drop("country")
    .withColumnRenamed("gni_pc", "GNI_Per_Capita")
)
print(enriched_coffee_df.count())
display(enriched_coffee_df.limit(10))

# COMMAND ----------

# DBTITLE 1,Create holdout table
# Convert timestamp to numeric for quantile calculation
holdout_cutoff = enriched_coffee_df.selectExpr(
    f"unix_timestamp({TIMESTAMP_COL}) as ts_numeric"
).approxQuantile("ts_numeric", [1 - HOLDOUT_FRACTION], 0.00001)[0]

prod_holdout_df = enriched_coffee_df.filter(F.unix_timestamp(F.col(TIMESTAMP_COL)) > holdout_cutoff)
print(prod_holdout_df.count())

# COMMAND ----------

# DBTITLE 1,Create labeled table
labeled_df = (
    enriched_coffee_df
    .join(
        prod_holdout_df.select(PRIMARY_KEY_COL, TIMESTAMP_COL),
        on=[PRIMARY_KEY_COL, TIMESTAMP_COL],
        how="left_anti"
    )
)
print(labeled_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Output

# COMMAND ----------

# DBTITLE 1,Write to Unity Catalog Delta Table
labeled_df.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(COFFEE_LABELED_DATA_PATH)
print(
    f"DataFrame labeled_df has been written to table:\n\t- {COFFEE_LABELED_DATA_PATH}"
)

# COMMAND ----------

prod_holdout_df.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(HOLDOUT_TABLE_PATH)
print(
    f"DataFrame prod_holdout_df has been written to table:\n\t- {HOLDOUT_TABLE_PATH}"
)
