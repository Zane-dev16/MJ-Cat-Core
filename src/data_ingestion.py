import s3fs
import os
from tqdm import tqdm
import yaml
from loguru import logger
from rich.logging import RichHandler

logger.remove()
logger.add(RichHandler(), level="INFO")
logger.add("logs/data_ingestion.log", rotation="5 MB", level="INFO", )
logger.info("Starting data ingestion...")

with open("params.yaml") as f:
    params = yaml.safe_load(f)

S3_BUCKET = params["s3_bucket"]
LOCAL_DIR = "data"

fs = s3fs.S3FileSystem(anon=False)  

def download_folder_simple(s3_bucket, folder_name, local_dir):
    """Download all files in a specific S3 folder (positive or negative)"""
    s3_path = f"{s3_bucket}/{folder_name}"
    files = fs.ls(s3_path)
    logger.info(f"Found {len(files)} files in {s3_path}")

    for file_key in tqdm(files, desc=f"Downloading {folder_name}"):
        if os.path.exists(os.path.join(local_dir, folder_name, os.path.basename(file_key))):
            logger.info(f"Skipping {file_key} (already exists)")
            continue
        file_name = os.path.basename(file_key)
        local_path = os.path.join(local_dir, folder_name, file_name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            fs.get(file_key, local_path)
        except Exception as e:
            logger.error(f"Failed to download {file_key}: {e}")

if __name__ == "__main__":
    for folder in ["dataset/negative", "dataset/positive"]:
        download_folder_simple(S3_BUCKET, folder, LOCAL_DIR)

    logger.info("Data ingestion completed successfully!")

