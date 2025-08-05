import pyspark.sql.functions as F
from pyspark.sql import SparkSession
import boto3
import json

# Initialize Spark
spark = SparkSession.builder.appName("BronzeToSilver_CDC").getOrCreate()

# S3 Paths
bucket_name = "healthcare-data-lake-07091998-csk"
bronze_prefix = "bronze/"
silver_prefix = "silver/"
metadata_delta_key = "metadata/delta.json"

bronze_path = f"s3://{bucket_name}/{bronze_prefix}"
silver_path = f"s3://{bucket_name}/{silver_prefix}"

# Initialize boto3 S3 client
s3 = boto3.client("s3")

# --- Step 1: Retrieve last_run_ts from metadata/delta.json ---
def get_last_run_ts():
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=metadata_delta_key)
        data = json.loads(obj["Body"].read())
        last_run_ts = data.get("last_run_ts", "1900-01-01 00:00:00")
        print(f"✅ Found delta.json. Last Run Timestamp: {last_run_ts}")
        return last_run_ts
    except s3.exceptions.NoSuchKey:
        # Initialize if not present
        default_ts = "1900-01-01 00:00:00"
        s3.put_object(
            Bucket=bucket_name,
            Key=metadata_delta_key,
            Body=json.dumps({"last_run_ts": default_ts}, indent=4),
            ServerSideEncryption="aws:kms"
        )
        print(f"⚠️ No delta.json found. Created new one with {default_ts}")
        return default_ts

last_run_ts = get_last_run_ts()

# --- Step 2: Ensure Silver folder exists ---
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=silver_prefix)
if 'Contents' not in response:
    s3.put_object(Bucket=bucket_name, Key=f"{silver_prefix}_placeholder")
    print(f"✅ Created placeholder for Silver folder: {silver_prefix}")

# --- Step 3: Identify Bronze file for this run ---
file_safe_ts = last_run_ts.replace(" ", "_").replace(":", "-")
bronze_file = f"{bronze_path}uci_raw_data_{file_safe_ts}.csv"
print(f"📥 Reading Bronze file: {bronze_file}")

# --- Step 4: Read Bronze file ---
df_bronze = spark.read.csv(bronze_file, header=True, inferSchema=True)

# --- Step 5: Cleaning & Feature Engineering ---
df_silver = df_bronze.withColumn(
    "patient_id_hashed",
    F.sha2(
        F.concat_ws("_",
                    F.col("race"),
                    F.col("gender"),
                    F.col("age"),
                    F.col("admission_type_id").cast("string"),
                    F.col("time_in_hospital").cast("string")
                   ), 256)
).fillna({"medical_specialty": "Missing"}) \
 .withColumn("num_visits",
             F.col("number_outpatient") + F.col("number_inpatient") + F.col("number_emergency")) \
 .withColumn("high_risk",
             F.when(
                 (F.col("age").isin("[60-70)", "[70-80)", "[80-90)", "[90-100)")) &
                 (F.col("A1Cresult").isin(">8", "Norm")),
                 1
             ).otherwise(0))

# --- Step 6: Save Silver as versioned file ---
silver_file = f"{silver_path}uci_raw_data_{file_safe_ts}"
df_silver.write.mode("overwrite").parquet(silver_file)
print(f"✅ Silver data written successfully to: {silver_file}")
