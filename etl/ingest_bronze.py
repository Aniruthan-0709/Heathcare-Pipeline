import pandas as pd
import boto3
import json
from datetime import datetime
from ucimlrepo import fetch_ucirepo
import numpy as np

# --- S3 Config ---
bucket_name = "healthcare-data-lake-07091998-csk"
metadata_prefix = "metadata/"
delta_key = f"{metadata_prefix}delta.json"

s3 = boto3.client("s3")

# ------------------------
# 1. Load UCI Diabetes Readmission Dataset
# ------------------------
uci_data = fetch_ucirepo(id=296)
df_uci = pd.concat([uci_data.data.features, uci_data.data.targets], axis=1)
print(f"✅ UCI Dataset Loaded: {df_uci.shape}")

# Clean Gender: Assign females where pregnancies > 0, else random male/female
df_uci['gender'] = np.where(df_uci['number_outpatient'] > 0, 'female', 
                            np.random.choice(['male', 'female'], size=len(df_uci)))

# ------------------------
# 2. Ensure metadata/delta.json exists (create if missing)
# ------------------------
def ensure_delta_json():
    try:
        s3.head_object(Bucket=bucket_name, Key=delta_key)
        # If file exists, read it
        obj = s3.get_object(Bucket=bucket_name, Key=delta_key)
        data = json.loads(obj['Body'].read())
        last_run_ts = data.get("last_run_ts", "1900-01-01 00:00:00")
        print(f"✅ Found delta.json. Last Run Timestamp: {last_run_ts}")
        return last_run_ts
    except s3.exceptions.ClientError:
        # If missing, create it with default timestamp
        default_ts = "1900-01-01 00:00:00"
        s3.put_object(
            Bucket=bucket_name,
            Key=delta_key,
            Body=json.dumps({"last_run_ts": default_ts}, indent=4),
            ServerSideEncryption="aws:kms"
        )
        print(f"⚠️ No delta.json found. Created new one with last_run_ts: {default_ts}")
        return default_ts

last_run_ts = ensure_delta_json()

# ------------------------
# 3. Save Bronze file with versioned name (using last_run_ts)
# ------------------------
file_safe_ts = last_run_ts.replace(" ", "_").replace(":", "-")
bronze_key = f"bronze/uci_raw_data_{file_safe_ts}.csv"

# Save locally
df_uci.to_csv("uci_diabetes_raw.csv", index=False)
print("✅ Saved raw PHI dataset locally")

# Upload to Bronze
s3.upload_file(
    Filename="uci_diabetes_raw.csv",
    Bucket=bucket_name,
    Key=bronze_key,
    ExtraArgs={"ServerSideEncryption": "aws:kms"}
)
print(f"✅ Uploaded securely to s3://{bucket_name}/{bronze_key}")
