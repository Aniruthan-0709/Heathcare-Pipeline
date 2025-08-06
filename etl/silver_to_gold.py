import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType
from datetime import datetime
import boto3
import json

# ----------------------------
# Glue Job Init
# ----------------------------
args = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# ----------------------------
# S3 Config
# ----------------------------
bucket_name = "healthcare-data-lake-07091998-csk"
gold_path = f"s3://{bucket_name}/gold/"
delta_key = "metadata/delta.json"
s3 = boto3.client("s3")

# ----------------------------
# Step 1: Get last_run_ts from delta.json
# ----------------------------
def fetch_last_run_ts():
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=delta_key)
        data = json.loads(obj['Body'].read())
        ts = data.get("last_run_ts", "1900-01-01 00:00:00")
        print(f"✅ Loaded last_run_ts from delta.json: {ts}")
        return ts
    except s3.exceptions.NoSuchKey:
        print("⚠️ No delta.json found. Initializing with default timestamp.")
        default_ts = "1900-01-01 00:00:00"
        s3.put_object(
            Bucket=bucket_name,
            Key=delta_key,
            Body=json.dumps({"last_run_ts": default_ts}, indent=4),
            ServerSideEncryption="aws:kms"
        )
        return default_ts

last_run_ts = fetch_last_run_ts()

# ----------------------------
# Step 2: Build Silver Path from last_run_ts
# ----------------------------
def format_ts_for_silver(ts: str) -> str:
    return ts.replace(" ", "_").replace(":", "-")

formatted_ts = format_ts_for_silver(last_run_ts)
silver_versioned_path = f"s3://{bucket_name}/silver/uci_raw_data_{formatted_ts}/"
print(f"✅ Reading Silver version from: {silver_versioned_path}")

# ----------------------------
# Step 3: Read Silver Data (CDC)
# ----------------------------
df = spark.read.parquet(silver_versioned_path)

if "last_updated_ts" in df.columns:
    df = df.filter(F.col("last_updated_ts") > F.lit(last_run_ts))
print(f"✅ Rows after CDC filtering: {df.count()}")

# ----------------------------
# Step 4: Feature Engineering
# ----------------------------
age_map = {"[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
           "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
           "[80-90)": 85, "[90-100)": 95}
gender_map = {"male": 1, "female": 0}
readmission_flag_map = {"<30": 1, ">30": 0, "NO": 0}
a1c_encoded_map = {">8": 2, "7-8": 1, "Norm": 0}
glu_encoded_map = {">300": 3, ">200": 2, "Norm": 0}
medication_encoding = {"Up": 1, "Down": -1, "Steady": 0, "No": 0}

age_udf = F.udf(lambda x: age_map.get(x, None), IntegerType())
df = df.withColumn("age_num", age_udf(F.col("age")))
df = df.withColumn("gender_num", 
                   F.when(F.lower(F.col("gender")) == "male", 1)
                    .when(F.lower(F.col("gender")) == "female", 0)
                    .otherwise(None))
df = df.withColumn("readmission_flag", F.when(F.col("readmitted") == "<30", 1).otherwise(0))
df = df.withColumn("a1c_encoded",
                   F.when(F.col("A1Cresult") == ">8", 2)
                    .when(F.col("A1Cresult") == "7-8", 1)
                    .when(F.col("A1Cresult") == "Norm", 0)
                    .otherwise(None))
df = df.withColumn("glu_encoded",
                   F.when(F.col("max_glu_serum") == ">300", 3)
                    .when(F.col("max_glu_serum") == ">200", 2)
                    .when(F.col("max_glu_serum") == "Norm", 0)
                    .otherwise(None))

# Medication Encoding
med_cols = [
    "metformin","repaglinide","nateglinide","chlorpropamide","glimepiride",
    "glipizide","glyburide","pioglitazone","rosiglitazone","acarbose",
    "miglitol","troglitazone","tolazamide","insulin"
]
for col in med_cols:
    df = df.withColumn(f"{col}_enc",
                       F.when(F.col(col) == "Up", 1)
                        .when(F.col(col) == "Down", -1)
                        .when(F.col(col) == "Steady", 0)
                        .otherwise(0))

# Comorbidity count
df = df.withColumn("comorbidity_count",
                   F.expr("size(array_remove(array(diag_1, diag_2, diag_3), null))"))

# Drop non-ML columns
drop_cols = ["race", "gender", "age", "readmitted", "max_glu_serum", 
             "A1Cresult", "medical_specialty", "payer_code"]
df = df.drop(*[c for c in drop_cols if c in df.columns])

# Add ETL Timestamp
etl_load_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
df = df.withColumn("etl_load_ts", F.lit(etl_load_ts))

# ----------------------------
# Step 5: Write Gold Incrementally
# ----------------------------
df.write.mode("append") \
    .partitionBy("readmission_flag") \
    .format("parquet") \
    .option("compression", "snappy") \
    .save(gold_path)
print(f"✅ Incremental Gold written to {gold_path}")

# ----------------------------
# Step 6: Save Metadata to metadata/silver_to_gold/{prev_ts}/
# ----------------------------
feature_mappings = {
    "age_map": age_map,
    "gender_map": gender_map,
    "readmission_flag_map": readmission_flag_map,
    "a1c_encoded_map": a1c_encoded_map,
    "glu_encoded_map": glu_encoded_map,
    "medication_encoding": medication_encoding,
    "etl_load_ts": etl_load_ts
}
schema_dict = {field.name: field.dataType.simpleString() for field in df.schema.fields}

prev_ts_folder = format_ts_for_silver(last_run_ts)
metadata_path = f"metadata/silver_to_gold/{prev_ts_folder}/"

s3.put_object(
    Bucket=bucket_name,
    Key=f"{metadata_path}feature_mappings.json",
    Body=json.dumps(feature_mappings, indent=4),
    ServerSideEncryption="aws:kms"
)
s3.put_object(
    Bucket=bucket_name,
    Key=f"{metadata_path}data_types.json",
    Body=json.dumps(schema_dict, indent=4),
    ServerSideEncryption="aws:kms"
)
print(f"✅ Metadata saved under: {metadata_path}")

# ----------------------------
# Step 7: Update delta.json with new timestamp
# ----------------------------
new_run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
s3.put_object(
    Bucket=bucket_name,
    Key=delta_key,
    Body=json.dumps({"last_run_ts": new_run_ts}, indent=4),
    ServerSideEncryption="aws:kms"
)
print(f"✅ delta.json updated with last_run_ts: {new_run_ts}")

# ----------------------------
# Commit Glue Job
# ----------------------------
job.commit()
