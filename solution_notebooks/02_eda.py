# Databricks notebook source
# MAGIC %md
# MAGIC # Data Exploration
# MAGIC - This notebook performs exploratory data analysis (EDA) on the dataset.
# MAGIC - Two approaches are used:
# MAGIC     1. The **AutoML** generated EDA notebook, which uses the automated Profiling Report. This approach requires the following libraries, which for this workshop are pre-installed on the clusters.
# MAGIC
# MAGIC         `%pip install --no-deps ydata-profiling==4.8.3 visions==0.7.6 # pandas==2.2.3 tzdata==2024.2`
# MAGIC
# MAGIC     2. **Custom EDA** approaches developed by us.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initialization

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run ./00_setup

# COMMAND ----------

# DBTITLE 1,Imports
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
import pyspark.sql.functions as F

plt.style.use("seaborn-v0_8")

# COMMAND ----------

# DBTITLE 1,Constants
# Table paths
COFFEE_LABELED_DATA_PATH = f"{CATALOG}.{MY_SCHEMA}.coffee_labeled"

# COMMAND ----------

# DBTITLE 1,Load table
coffee_df = spark.table(COFFEE_LABELED_DATA_PATH)
display(coffee_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA

# COMMAND ----------

# MAGIC %md
# MAGIC #### AutoML EDA

# COMMAND ----------

# DBTITLE 1,Profiling Results
from ydata_profiling import ProfileReport

df_profile = ProfileReport(coffee_df.toPandas(),
                           correlations={
                               "auto": {"calculate": True},
                               "pearson": {"calculate": True},
                               "spearman": {"calculate": True},
                               "kendall": {"calculate": True},
                               "phi_k": {"calculate": True},
                               "cramers": {"calculate": True},
                           }, title="Profiling Report", progress_bar=False, infer_dtypes=False)
profile_html = df_profile.to_html()

displayHTML(profile_html)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Custom EDA

# COMMAND ----------

# MAGIC %md
# MAGIC Compare sleep, stress, and BMI patterns across the coffee intake cohorts.

# COMMAND ----------

# DBTITLE 1,Coffee Intake vs. Sleep & Stress
sleep_stress_df = (
    coffee_df
    .groupBy("Coffee_Intake_Binary")
    .agg(
        F.avg("Sleep_Hours").alias("avg_sleep_hours"),
        F.avg(
            F.when(F.col("Sleep_Hours") < 7, 1)
            .otherwise(0)
        ).alias("share_short_sleep"),
        F.avg("BMI").alias("avg_bmi"),
    )
)
display(sleep_stress_df)

stress_quality_df = (
    coffee_df
    .groupBy("Coffee_Intake_Binary", "Stress_Level", "Sleep_Quality")
    .agg(F.count("*").alias("record_count"))
    .orderBy("Coffee_Intake_Binary", F.col("record_count").desc())
)
display(stress_quality_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Compare sleep and stress indicators across coffee preference segments using compact visuals.

# COMMAND ----------

# DBTITLE 1,Visualize Coffee Intake vs. Sleep & Stress
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
