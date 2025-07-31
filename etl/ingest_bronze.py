import pandas as pd
import requests
import boto3

# ------------------------
# 1. Load UCI Diabetes Readmission Dataset
# ------------------------
uci_url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df_uci = pd.read_csv(uci_url)
print(f"UCI Diabetes Dataset Loaded. Shape: {df_uci.shape}")

# ------------------------
# 2. Fetch FHIR Patient Data (HL7 FHIR Standard)
# ------------------------
FHIR_URL = "https://hapi.fhir.org/baseR4/Patient?_count=50"
response = requests.get(FHIR_URL)
patients = response.json()

records = []
for entry in patients.get("entry", []):
    resource = entry["resource"]
    patient_id = resource.get("id", "")
    name = resource.get("name", [{}])[0].get("given", [""])[0]
    family_name = resource.get("name", [{}])[0].get("family", "")
    dob = resource.get("birthDate", "")
    gender = resource.get("gender", "")
    records.append([patient_id, name, family_name, dob, gender])

df_fhir = pd.DataFrame(records, columns=["patient_id", "first_name", "last_name", "dob", "gender"])
print(f"FHIR Patient Data Loaded. Shape: {df_fhir.shape}")

# ------------------------
# 3. Combine and Save
# ------------------------
df_combined = pd.concat([df_uci.reset_index(drop=True), df_fhir.reset_index(drop=True)], axis=1)
df_combined.to_csv("uci_fhir_raw.csv", index=False)
print("Combined raw PHI dataset created and saved as uci_fhir_raw.csv")

# ------------------------
# 4. Upload to HIPAA S3 Bronze (Encrypted)
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
print(f"Uploaded to s3://{bronze_bucket}/{bronze_key}")
