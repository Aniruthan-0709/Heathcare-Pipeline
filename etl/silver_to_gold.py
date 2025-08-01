import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType

# Glue Job Init
args = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# Paths
silver_path = "s3://healthcare-data-lake-07091998-csk/silver/"
gold_path = "s3://healthcare-data-lake-07091998-csk/gold/"

# Read Silver Layer
df = spark.read.parquet(silver_path)

# --- Feature Engineering ---

# 1. Age midpoint
age_map = {"[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
           "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
           "[80-90)": 85, "[90-100)": 95}
age_udf = F.udf(lambda x: age_map.get(x, None), IntegerType())
df = df.withColumn("age_num", age_udf(F.col("age")))

# 2. Encode gender
df = df.withColumn("gender_num", F.when(F.lower(F.col("gender")) == "male", 1)
                                     .when(F.lower(F.col("gender")) == "female", 0)
                                     .otherwise(None))

# 3. Readmission flag
df = df.withColumn("readmission_flag", F.when(F.col("readmitted") == "<30", 1).otherwise(0))

# 4. Encode lab results
df = df.withColumn("a1c_encoded", 
                   F.when(F.col("a1cresult") == ">8", 2)
                    .when(F.col("a1cresult") == "7-8", 1)
                    .when(F.col("a1cresult") == "Norm", 0)
                    .otherwise(None))
df = df.withColumn("glu_encoded",
                   F.when(F.col("max_glu_serum") == ">200", 2)
                    .when(F.col("max_glu_serum") == ">300", 3)
                    .when(F.col("max_glu_serum") == "Norm", 0)
                    .otherwise(None))

# 5. Encode medications (Up=1, Down=-1, Steady=0, No=0)
med_cols = ["metformin","insulin","glipizide","glyburide","pioglitazone","rosiglitazone"]
for col in med_cols:
    df = df.withColumn(f"{col}_enc", 
                       F.when(F.col(col) == "Up", 1)
                        .when(F.col(col) == "Down", -1)
                        .when(F.col(col) == "Steady", 0)
                        .otherwise(0))

# 6. Comorbidity count (number of non-null diagnoses)
df = df.withColumn("comorbidity_count",
                   F.expr("size(array_remove(array(diag_1, diag_2, diag_3), null))"))

# --- Drop non-ML or PHI-safe columns ---
drop_cols = ["first_name", "last_name", "dob", "patient_id_hashed", "gender", "age", "readmitted",
             "max_glu_serum", "a1cresult"]
df = df.drop(*[c for c in drop_cols if c in df.columns])

# --- Write Gold Layer ---
df.write.mode("overwrite") \
    .partitionBy("readmission_flag") \
    .format("parquet") \
    .option("compression", "snappy") \
    .save(gold_path)

print(f"🎉 Gold ML-ready data written to {gold_path}")
job.commit()
