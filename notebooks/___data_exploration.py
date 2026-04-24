# Databricks notebook source
# MAGIC %md
# MAGIC # Data Exploration
# MAGIC - This notebook performs exploratory data analysis (EDA) on the dataset.
# MAGIC - Two approaches are used:
# MAGIC     1. The **AutoML** generated EDA notebook, which uses the automated Profiling Report. This approach requires the following libraries, which for this workshop are pre-installed on the clusters.

# COMMAND ----------

# DBTITLE 1,Data Exploration
from ydata_profiling import ProfileReport

coffee_df = spark.table("sds26_ml_catalog.source_data.coffee_labeled_features")
display(coffee_df.limit(10))

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
