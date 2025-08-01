import pandas as pd
import boto3
from ucimlrepo import fetch_ucirepo

# ------------------------
# 1. Load UCI Diabetes Readmission Dataset
# ------------------------
uci_data = fetch_ucirepo(id=296)
df_uci = pd.concat([uci_data.data.features, uci_data.data.targets], axis=1)
print(f"UCI Dataset Loaded: {df_uci.shape}")

# Clean Gender: Assign females where pregnancies > 0, else random male/female
import numpy as np
df_uci['gender'] = np.where(df_uci['number_outpatient'] > 0, 'female', 
                            np.random.choice(['male', 'female'], size=len(df_uci)))

# ------------------------
# 2. Save Locally
# ------------------------
df_uci.to_csv("uci_diabetes_raw.csv", index=False)
print("✅ Saved raw PHI dataset locally")

# ------------------------
# 3. Upload to HIPAA Bronze (Encrypted)
# ------------------------
s3 = boto3.client("s3")
bronze_bucket = "healthcare-data-lake-07091998-csk"
bronze_key = "bronze/uci_diabetes_raw.csv"

s3.upload_file(
    Filename="uci_diabetes_raw.csv",
    Bucket=bronze_bucket,
    Key=bronze_key,
    ExtraArgs={"ServerSideEncryption": "aws:kms"}
)
print(f"Uploaded securely to s3://{bronze_bucket}/{bronze_key}")
