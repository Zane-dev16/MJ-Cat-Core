from pathlib import Path
import numpy as np
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from tqdm import tqdm
from loguru import logger
from rich.logging import RichHandler
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = Path("data/prepared")
EMBED_DIR = Path("data/embeddings")
VGGISH_MODEL_URL = "https://tfhub.dev/google/vggish/1"
SR = 16000
RANDOM_STATE = 42
MAX_WORKERS = 8

logger.remove()
logger.add(RichHandler(), level="INFO")
logger.add("logs/extract_embeddings.log", rotation="5 MB", level="INFO")

def waveform_to_tensor(waveform: np.ndarray) -> tf.Tensor:
    waveform = np.array(waveform, dtype=np.float32)
    waveform = tf.convert_to_tensor(waveform)
    return waveform

def extract_embedding(file_path: Path, model) -> np.ndarray:
    """
    Load audio and compute VGGish embedding features.
    Returns a concatenated vector of mean, std, and max across time.
    """
    try:
        audio, sr = librosa.load(file_path, sr=SR, mono=True)
        input_tensor = waveform_to_tensor(audio)
        embedding = model(input_tensor)
        embedding = embedding.numpy()
        feature_vector = np.concatenate([
            np.mean(embedding, axis=0),
            np.std(embedding, axis=0),
            np.max(embedding, axis=0)
        ])
        return feature_vector
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return None

def load_data_from_split(split_name: str, model) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all embeddings for a data split.
    Automatically detects labels from subfolders.
    """
    X, y = [], []
    split_path = DATA_DIR / split_name
    if not split_path.exists():
        logger.warning(f"Split folder {split_path} does not exist.")
        return np.array(X), np.array(y)
    
    labels = [f.name for f in split_path.iterdir() if f.is_dir()]
    logger.info(f"Found labels for {split_name}: {labels}")
    
    for label in labels:
        folder = split_path / label
        files = list(folder.glob("**/*.wav"))
        logger.info(f"Processing {len(files)} files in {split_name}/{label}")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(extract_embedding, f, model): f for f in files}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{split_name}/{label}"):
                embedding = future.result()
                if embedding is not None:
                    X.append(embedding)
                    y.append(label)
                    
    label_map = {label: idx for idx, label in enumerate(sorted(set(y)))}
    y_int = np.array([label_map[label] for label in y])
    return np.array(X), y_int

def save_embeddings(X: np.ndarray, y: np.ndarray, split_name: str):
    """
    Save embeddings and labels to disk.
    """
    split_dir = EMBED_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / f"X_{split_name}.npy", X)
    np.save(split_dir / f"y_{split_name}.npy", y)
    logger.info(f"Saved embeddings for {split_name} to {split_dir}")

if __name__ == "__main__":
    logger.info("Loading VGGish model from TensorFlow Hub...")
    vggish_model = hub.load(VGGISH_MODEL_URL)

    for split in ["train", "val", "test"]:
        X, y = load_data_from_split(split, vggish_model)
        save_embeddings(X, y, split)

