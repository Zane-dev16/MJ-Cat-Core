from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from loguru import logger
from rich.logging import RichHandler
from tqdm import tqdm

logger.remove()
logger.add(RichHandler(), level="INFO")
logger.add("logs/split_dataset.log", rotation="5 MB", level="INFO")

INPUT_DIR = "data/dataset"  
OUTPUT_DIR = "data/splits"

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

def split_folder_by_song(folder_path, output_dir):
    files = list(Path(folder_path).glob("*.mp3"))
    labels = folder_path.name  

    train_files, temp_files = train_test_split(files, test_size=SPLIT_RATIOS["val"]+SPLIT_RATIOS["test"], random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=SPLIT_RATIOS["test"]/(SPLIT_RATIOS["val"]+SPLIT_RATIOS["test"]), random_state=42)

    splits = {"train": train_files, "val": val_files, "test": test_files}

    for split_name, split_files in splits.items():
        split_folder = Path(output_dir) / split_name / labels
        split_folder.mkdir(parents=True, exist_ok=True)
        for file_path in tqdm(split_files, desc=f"Saving {split_name} split", unit="file"):
            try: 
                shutil.copy(file_path, split_folder / file_path.name)
            except Exception as e:
                logger.error(f"Failed to copy {file_path}: {e}")
        logger.info(f"Saved {len(split_files)} files to {split_folder}")

if __name__ == "__main__":
    for folder_name in ["positive", "negative"]:
        folder_path = Path(INPUT_DIR) / folder_name
        split_folder_by_song(folder_path, OUTPUT_DIR)

    logger.info("Song-level dataset splitting completed!")

