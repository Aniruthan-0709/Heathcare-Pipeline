import pyspark.sql.functions as F
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("BronzeToSilver").getOrCreate()

# Load Bronze
df_bronze = spark.read.csv("s3://healthcare-data-lake-07091998-csk/bronze/uci_diabetes.csv", header=True, inferSchema=True)

# 1. Remove PHI (Safe Harbor)
df_silver = (
    df_bronze
    .withColumn("patient_id_hashed", F.sha2(F.col("patient_nbr").cast("string"), 256))
    .drop("patient_nbr")
)

# 2. Impute & Clean
df_silver = df_silver.fillna({"medical_specialty": "Missing"})

# 3. Feature Engineering
df_silver = df_silver.withColumn("num_visits", 
                                 F.col("number_outpatient") + F.col("number_inpatient") + F.col("number_emergency"))

df_silver = df_silver.withColumn("high_risk",
                                 F.when((F.col("age").isin("[60-70)", "[70-80)", "[80-90)", "[90-100)")) &
                                        (F.col("a1cresult") == ">8"), 1).otherwise(0))

# 4. Partition & Write to S3 (Parquet)
df_silver.write.mode("overwrite") \
    .partitionBy("readmitted") \
    .parquet("s3://healthcare-data-lake-07091998-csk/silver/")
