import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType
from datetime import datetime

# Glue Job Init
args = getResolvedOptions(sys.argv, ["JOB_NAME", "LAST_RUN_TS"])  # Added CDC parameter
last_run_ts = args["LAST_RUN_TS"]  # Timestamp of last ETL run (passed in via EventBridge or ParamStore)

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# Paths
silver_path = "s3://healthcare-data-lake-07091998-csk/silver/"
gold_path = "s3://healthcare-data-lake-07091998-csk/gold/"

# --- Read Silver Layer ---
df = spark.read.parquet(silver_path)

# CDC Filter: Process only rows updated after LAST_RUN_TS
if "last_updated_ts" in df.columns:
    df = df.filter(F.col("last_updated_ts") > F.lit(last_run_ts))

# --- Feature Engineering (Same Logic as Before) ---

# 1. Age midpoint
age_map = {"[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
           "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
           "[80-90)": 85, "[90-100)": 95}
age_udf = F.udf(lambda x: age_map.get(x, None), IntegerType())
df = df.withColumn("age_num", age_udf(F.col("age")))

# 2. Encode gender
df = df.withColumn("gender_num",
                   F.when(F.lower(F.col("gender")) == "male", 1)
                    .when(F.lower(F.col("gender")) == "female", 0)
                    .otherwise(None))

# 3. Readmission flag (<30 = 1, else 0)
df = df.withColumn("readmission_flag", F.when(F.col("readmitted") == "<30", 1).otherwise(0))

# 4. Encode lab results
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

# 5. Encode medications
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

# 6. Comorbidity count
df = df.withColumn("comorbidity_count",
                   F.expr("size(array_remove(array(diag_1, diag_2, diag_3), null))"))

# --- Drop non-ML columns ---
drop_cols = ["race", "gender", "age", "readmitted", "max_glu_serum", "A1Cresult", "medical_specialty", "payer_code"]
df = df.drop(*[c for c in drop_cols if c in df.columns])

# Add ETL Load Timestamp
df = df.withColumn("etl_load_ts", F.lit(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# --- Write Gold Layer Incrementally ---
df.write.mode("append") \   # APPEND for CDC
    .partitionBy("readmission_flag") \
    .format("parquet") \
    .option("compression", "snappy") \
    .save(gold_path)

print(f"Incremental Gold data appended to {gold_path}")
job.commit()
