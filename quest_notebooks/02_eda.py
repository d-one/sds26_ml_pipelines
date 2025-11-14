# Databricks notebook source
# MAGIC %md
# MAGIC # Coffee Module 02 · Exploratory Analysis
# MAGIC This module keeps the EDA workflow aligned with the latest labeled table while framing each step as a fill-in-the-blanks exercise. Each code cell includes `...` placeholders—fill them in before running.

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run ./00_setup

# COMMAND ----------

# DBTITLE 1,Imports
import matplotlib.pyplot as plt
import pandas as pd
import pyspark.sql.functions as F
from ydata_profiling import ProfileReport

plt.style.use("seaborn-v0_8")

# COMMAND ----------

# DBTITLE 1,Constants
COFFEE_LABELED_DATA_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_labeled"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 1 · Load the Labeled Dataset
# MAGIC **Goal:** create `coffee_df` from the Unity Catalog table.
# MAGIC
# MAGIC **Hints**
# MAGIC - Use `spark.table(COFFEE_LABELED_DATA_PATH)`.
# MAGIC - Add `display(coffee_df.limit(10))` to preview the dataset.

# COMMAND ----------

coffee_df = spark.table(...)                                                                 # replace placeholder
display(coffee_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 2 · Generate the Profiling Report
# MAGIC **Goal:** render a ydata-profiling report for the labeled data.
# MAGIC
# MAGIC **Hints**
# MAGIC - Use True/False to enable/disable each correlation method.

# COMMAND ----------

df_profile = ProfileReport(
    coffee_df.toPandas(),
    correlations={
        "auto": {"calculate": ...},                                                         # replace placeholders  
        "pearson": {"calculate": ...},
        "spearman": {"calculate": ...},
        "kendall": {"calculate": ...},
        "phi_k": {"calculate": ...},
        "cramers": {"calculate": ...},
    },
    title="Profiling Report", progress_bar=False, infer_dtypes=False)
profile_html = df_profile.to_html()
displayHTML(profile_html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 3 · Aggregate Sleep and Stress Metrics
# MAGIC **Goal:** reproduce the cohort-level aggregations for sleep, stress, and BMI.
# MAGIC
# MAGIC **Hints**
# MAGIC - Group by `Coffee_Intake_Binary` for the first aggregation.
# MAGIC - Use `F.when(F.col("Sleep_Hours") < 7, 1)` inside `avg`.
# MAGIC - For the second aggregation, group by `(Coffee_Intake_Binary, Stress_Level, Sleep_Quality)` and order by `record_count`.

# COMMAND ----------

sleep_stress_df = (
    coffee_df.groupBy("...")                                                                 # replace placeholders
    .agg(
        F.avg("...").alias("avg_sleep_hours"),
        F.avg(F.when(F.col("...") < 7, 1).otherwise(0)).alias("share_short_sleep"),
        F.avg("...").alias("avg_bmi"),
    )
)
display(sleep_stress_df)

stress_quality_df = (
    coffee_df
    .groupBy("...","...","...")                                                              # replace placeholders
    .agg(F.count("*").alias("record_count"))
    .orderBy("Coffee_Intake_Binary", F.col("record_count").desc())
)
display(stress_quality_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quest 4 · Visualize Cohort Metrics
# MAGIC **Goal:** convert the aggregated metrics to a compact visualization.

# COMMAND ----------

sleep_stress_pd = sleep_stress_df.toPandas()

if not sleep_stress_pd.empty:
    sleep_stress_pd["Coffee_Intake_Binary"] = sleep_stress_pd[
        "Coffee_Intake_Binary"
    ].map({0: "Non-Drinkers", 1: "Drinkers"})
    metrics = [
        ("avg_sleep_hours", "Average Sleep Hours", "#88b04b"),
        ("share_short_sleep", "Share Short Sleep (<7h)", "#ffaf87"),
        ("avg_bmi", "Average BMI", "#6a5acd"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, (metric_column, title, color) in zip(axes, metrics):
        ax.bar(
            sleep_stress_pd["Coffee_Intake_Binary"],
            sleep_stress_pd[metric_column],
            color=color,
        )
        ax.set_title(title)
        ax.set_ylabel(metric_column.replace("_", " ").title())
        for idx, value in enumerate(sleep_stress_pd[metric_column]):
            if pd.notnull(value):
                ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    display(fig)
    plt.close(fig)
else:
    print("Sleep and stress aggregation is empty; skipping visual comparison.")
