import os
import boto3
import requests
import pandas as pd

# 1. CMS Claims Dataset URLs (Synthetic PUF)
CMS_DATASETS = {
    "inpatient": "https://downloads.cms.gov/files/medicare-inpatient-synthetic-puf.csv",
    "outpatient": "https://downloads.cms.gov/files/medicare-outpatient-synthetic-puf.csv",
    "partd": "https://downloads.cms.gov/files/medicare-partd-synthetic-puf.csv"
}

local_dir = "cms_claims_raw"
os.makedirs(local_dir, exist_ok=True)

print("Downloading CMS Medicare Claims data...")

# Download each dataset
for name, url in CMS_DATASETS.items():
    file_path = os.path.join(local_dir, f"{name}.csv")
    response = requests.get(url)
    with open(file_path, "wb") as f:
        f.write(response.content)
    print(f"Downloaded {name} claims: {file_path}")

# 2. Upload to HIPAA S3 Bronze
bronze_bucket = "healthcare-data-lake-07091998-csk"
session = boto3.Session()
s3 = session.client("s3")

print("Uploading to HIPAA S3 Bronze (encrypted)...")

for file in os.listdir(local_dir):
    file_path = os.path.join(local_dir, file)
    s3_key = f"bronze/cms_claims/{file}"
    s3.upload_file(
        Filename=file_path,
        Bucket=bronze_bucket,
        Key=s3_key,
        ExtraArgs={"ServerSideEncryption": "aws:kms"}
    )
    print(f"Uploaded to s3://{bronze_bucket}/{s3_key}")

print("CMS Claims ingestion completed (Bronze layer).")
