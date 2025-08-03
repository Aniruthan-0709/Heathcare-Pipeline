import pyspark.sql.functions as F
from pyspark.sql import SparkSession
import boto3

# Initialize Spark
spark = SparkSession.builder.appName("BronzeToSilver").getOrCreate()

# S3 paths
bronze_path = "s3://healthcare-data-lake-07091998-csk/bronze/uci_diabetes.csv"
silver_folder = "s3://healthcare-data-lake-07091998-csk/silver/"

# Create Silver folder explicitly using boto3 (if not exists)
s3 = boto3.client("s3")
bucket_name = "healthcare-data-lake-07091998-csk"
silver_prefix = "silver/"

# Check if the prefix exists
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=silver_prefix)
if 'Contents' not in response:
    # Create an empty placeholder file to establish the folder
    s3.put_object(Bucket=bucket_name, Key=f"{silver_prefix}_placeholder")

print(f"Ensured {silver_folder} exists.")

# Load Bronze
df_bronze = spark.read.csv(bronze_path, header=True, inferSchema=True)

# 1. Remove PHI (Safe Harbor)
df_silver = (
    df_bronze
    .withColumn("patient_id_hashed", F.sha2(F.col("patient_nbr").cast("string"), 256))
    .drop("patient_nbr")
)

# 2. Impute & Clean
df_silver = df_silver.fillna({"medical_specialty": "Missing"})

# 3. Feature Engineering
df_silver = df_silver.withColumn(
    "num_visits",
    F.col("number_outpatient") + F.col("number_inpatient") + F.col("number_emergency")
)

df_silver = df_silver.withColumn(
    "high_risk",
    F.when(
        (F.col("age").isin("[60-70)", "[70-80)", "[80-90)", "[90-100)")) & 
        (F.col("a1cresult") == ">8"),
        1
    ).otherwise(0)
)

# 4. Partition & Write to S3 (Parquet)
df_silver.write.mode("overwrite") \
    .partitionBy("readmitted") \
    .parquet(silver_folder)

print(f"Silver data written successfully to {silver_folder}")
