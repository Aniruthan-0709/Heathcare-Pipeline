import pandas as pd
import requests
import boto3
import numpy as np
from ucimlrepo import fetch_ucirepo

# ------------------------
# 1. Fetch UCI Diabetes Dataset
# ------------------------
diabetes_data = fetch_ucirepo(id=296)
X = diabetes_data.data.features
y = diabetes_data.data.targets

# Combine features & targets into one DataFrame
df_uci = pd.concat([X, y], axis=1)
df_uci.columns = [col.lower().replace(" ", "_") for col in df_uci.columns]

# Gender assignment based on pregnancies column if available
if "number_of_pregnancies" in df_uci.columns:
    df_uci["gender"] = np.where(df_uci["number_of_pregnancies"] > 0, "female",
                                np.random.choice(["male", "female"], size=len(df_uci)))
else:
    df_uci["gender"] = np.random.choice(["male", "female"], size=len(df_uci))

print(f"✅ UCI Diabetes Dataset Loaded: {df_uci.shape}")

# ------------------------
# 2. Fetch FHIR Patient Data
# ------------------------
FHIR_URL = "https://hapi.fhir.org/baseR4/Patient?_count=200"
response = requests.get(FHIR_URL)
patients = response.json()

records = []
for entry in patients.get("entry", []):
    r = entry["resource"]
    records.append([
        r.get("id", ""),
        r.get("name", [{}])[0].get("given", [""])[0] or "John",
        r.get("name", [{}])[0].get("family", "") or "Doe",
        r.get("birthDate", "") or "1970-01-01",
        r.get("gender", "") or np.random.choice(["male", "female"])
    ])

df_fhir = pd.DataFrame(records, columns=["patient_id", "first_name", "last_name", "dob", "gender"])
df_fhir["dob"] = pd.to_datetime(df_fhir["dob"], errors="coerce").fillna(pd.Timestamp("1970-01-01"))

# Match FHIR row count to UCI dataset
df_fhir = df_fhir.sample(n=len(df_uci), replace=True).reset_index(drop=True)

print(f"✅ FHIR Patient Data Loaded: {df_fhir.shape}")

# ------------------------
# 3. Merge and Save
# ------------------------
df_combined = pd.concat([df_uci.reset_index(drop=True), df_fhir], axis=1)
df_combined.to_csv("uci_fhir_raw.csv", index=False)
print("✅ Combined raw PHI dataset created")

# ------------------------
# 4. Upload to HIPAA S3 Bronze
# ------------------------
bronze_bucket = "healthcare-data-lake-07091998-csk"
bronze_key = "bronze/uci_fhir_raw.csv"

s3 = boto3.client("s3")
s3.upload_file(
    Filename="uci_fhir_raw.csv",
    Bucket=bronze_bucket,
    Key=bronze_key,
    ExtraArgs={"ServerSideEncryption": "aws:kms"}
)
print(f"🎉 Uploaded to s3://{bronze_bucket}/{bronze_key}")
