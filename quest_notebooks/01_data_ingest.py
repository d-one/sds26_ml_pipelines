# Databricks notebook source
# MAGIC %md
# MAGIC # Coffee Module 01 · Data Ingestion Workflow
# MAGIC This guided notebook mirrors the updated ingestion pipeline while turning each step into a fill-in-the-blanks exercise. Use the hints to stay on track and replace every placeholder before running.
# MAGIC
# MAGIC **How to play**
# MAGIC - Every quest cell contains `...` placeholders: replace them with working code before running.
# MAGIC - The hints reference the APIs/columns you will need; consult them if you get stuck.

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run ./00_setup

# COMMAND ----------

# DBTITLE 1,Imports
import pyspark.sql.functions as F

# COMMAND ----------

# DBTITLE 1,Constants
COFFEE_RAW_DATA_PATH = f"{CATALOG}.{SCHEMA_WITH_SOURCE_DATA}.coffee_raw"
GNI_PER_CAPITA_PATH = (
    f"{CATALOG}.{SCHEMA_WITH_SOURCE_DATA}.gni_per_capita_data"
)
COFFEE_LABELED_DATA_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_labeled"
HOLDOUT_TABLE_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_prod_holdout"

PRIMARY_KEY_COL = "ID"
TIMESTAMP_COL = "Timestamp"
LABEL_COL = "Coffee_Intake_Binary"
HOLDOUT_FRACTION = 0.1

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 1 · Load Source Tables
# MAGIC **Goal:** materialize `coffee_raw_df` and `gni_pc_df`.
# MAGIC
# MAGIC **Hints**
# MAGIC - Use `spark.table(<Unity Catalog path>)` for both tables.

# COMMAND ----------

coffee_raw_df = spark.table(...)                                                   # replace placeholders
gni_pc_df = spark.table(...)

display(coffee_raw_df.limit(10))
display(gni_pc_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 2 · Identify Country Mismatches
# MAGIC **Goal:** compare the distinct country lists and print unmapped entries.
# MAGIC
# MAGIC **Hints**
# MAGIC - Use `select("Country").distinct()` and `select("country").distinct()`.
# MAGIC - Join on equality with `how="left_anti"` to find the mismatches.
# MAGIC - Collect the result so you can iterate and print.

# COMMAND ----------

countries_of_coffee_df = coffee_raw_df.select("...").distinct()                     # replace placeholders
countries_of_gni_df = gni_pc_df.select("...").distinct()
print(
    f"coffee_raw_df contains {countries_of_coffee_df.count()} distinct countries."
)
print(f"gni_pc_df contains {countries_of_gni_df.count()} distinct countries.")

join_to_find_mismatches_df = countries_of_coffee_df.join(
    countries_of_gni_df,
    countries_of_coffee_df.Country == gni_pc_df.country,
    "...",                                                                          # replace placeholder
).collect()

print("\nCountries present in coffee_raw_df but missing in gni_pc_df:")
missing_countries = [row["..."] for row in join_to_find_mismatches_df]
for country in missing_countries:
    print(f"\t- {country}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 3 · Standardize Country Labels
# MAGIC **Goal:** align the mismatched country names using conditional logic.
# MAGIC
# MAGIC **Hints**
# MAGIC Countries to normalize:
# MAGIC - "United States"
# MAGIC - "Korea, Rep."
# MAGIC - "United Kingdom"

# COMMAND ----------

gni_pc_df = gni_pc_df.withColumn(
    "country",
    F.when(gni_pc_df.country == "...", "...")                                       # replace placeholders
    .when(gni_pc_df.country == "...", "...")                                      
    .when(gni_pc_df.country == "...", "...")                                       
    .otherwise(...),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 4 · Create the Enriched Dataset
# MAGIC **Goal:** join the coffee data with GNI per capita and rename the metric column.
# MAGIC
# MAGIC **Hints**
# MAGIC - Use a `left` join on the country fields.
# MAGIC - Drop the duplicate `country` column and rename `gni_pc` to `GNI_Per_Capita`.

# COMMAND ----------

enriched_coffee_df = (
    coffee_raw_df.join(
        gni_pc_df, coffee_raw_df.Country == gni_pc_df.country, "..."                # replace placeholders
    )
    .drop("...")                                                                    
    .withColumnRenamed("...", "GNI_Per_Capita")                                     
)
display(enriched_coffee_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 5 · Build the Production Holdout
# MAGIC **Goal:** carve out the most recent ~10% of records based on the timestamp.
# MAGIC
# MAGIC **Hints**
# MAGIC - Use `selectExpr(f"unix_timestamp({TIMESTAMP_COL}) as ts_numeric")`.
# MAGIC - Filter with `F.unix_timestamp(F.col(TIMESTAMP_COL)) > holdout_cutoff`.

# COMMAND ----------

holdout_cutoff = enriched_coffee_df.selectExpr(
    f"...({TIMESTAMP_COL}) as ts_numeric"                                            # replace placeholders
).approxQuantile("...", [1 - HOLDOUT_FRACTION], 0.00001)[0]  

prod_holdout_df = enriched_coffee_df.filter(
    F.unix_timestamp(F.col(TIMESTAMP_COL)) > holdout_cutoff
)
print(f"Prod holdout rows: {prod_holdout_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 6 · Assemble the Labeled Training Set
# MAGIC **Goal:** remove holdout keys from the enriched dataset to form `labeled_df`.
# MAGIC
# MAGIC **Hints**
# MAGIC - Select the key columns from `prod_holdout_df` before joining.
# MAGIC - Join with `how="left_anti"` so only non-holdout rows remain.

# COMMAND ----------

labeled_df = enriched_coffee_df.join(
    prod_holdout_df.select(..., ...),                                                # replace placeholders
    on=[PRIMARY_KEY_COL, TIMESTAMP_COL],
    how="...",                                                                      
)
print(f"Labeled training rows: {labeled_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 7 · Publish the Tables
# MAGIC **Goal:** write both `labeled_df` and `prod_holdout_df` to Unity Catalog.

# COMMAND ----------

# DBTITLE 1,Write to Unity Catalog Delta Table
labeled_df.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(COFFEE_LABELED_DATA_PATH)
print(
    f"DataFrame labeled_df has been written to table:\n\t- {COFFEE_LABELED_DATA_PATH}"
)


prod_holdout_df.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(HOLDOUT_TABLE_PATH)
print(
    f"DataFrame prod_holdout_df has been written to table:\n\t- {HOLDOUT_TABLE_PATH}"
)
