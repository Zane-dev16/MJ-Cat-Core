from pathlib import Path
import numpy as np
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from loguru import logger
from rich.logging import RichHandler
import bentoml
from tqdm import tqdm
import io

DATA_DIR = Path("data/prepared")
EMBED_DIR = Path("data/embeddings")
VGGISH_MODEL_URL = "https://tfhub.dev/google/vggish/1"
SR = 16000
MAX_WORKERS = 8

logger.remove()
logger.add(RichHandler(), level="INFO")
logger.add("logs/extract_embeddings.log", rotation="5 MB", level="INFO")

class AudioPreprocessor:
    """
    Preprocessor for raw audio inputs.
    Accepts:
      - Path-like objects
      - Bytes (common for HTTP uploads)
    Returns a TensorFlow tensor suitable for VGGish.
    """
    def __init__(self, sample_rate: int = SR):
        self.sample_rate = sample_rate

    def __call__(self, audio_input):
        if isinstance(audio_input, (str, Path)):
            waveform, _ = librosa.load(audio_input, sr=self.sample_rate, mono=True)
        elif isinstance(audio_input, bytes):
            waveform, _ = librosa.load(io.BytesIO(audio_input), sr=self.sample_rate, mono=True)
        else:
            raise ValueError(f"Unsupported input type: {type(audio_input)}")
        return tf.convert_to_tensor(np.array(waveform, dtype=np.float32))

def extract_embedding(file_path: Path, preprocessor: AudioPreprocessor, vggish_model):
    try:
        tensor = preprocessor(file_path)
        embedding = vggish_model(tensor).numpy()
        feature_vector = np.concatenate([
            np.mean(embedding, axis=0),
            np.std(embedding, axis=0),
            np.max(embedding, axis=0)
        ])
        return feature_vector
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return None

def load_data_from_split(split_name: str, preprocessor: AudioPreprocessor, vggish_model):
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
        for f in tqdm(files, desc=f"Processing {split_name}/{label}"):
            embedding = extract_embedding(f, preprocessor, vggish_model)
            if embedding is not None:
                X.append(embedding)
                y.append(label)
            else:
                logger.error(f"Failed to extract embedding for {f}")
                exit(1)


    label_map = {label: idx for idx, label in enumerate(sorted(set(y)))}
    y_int = np.array([label_map[label] for label in y])
    return np.array(X), y_int

def save_embeddings(X: np.ndarray, y: np.ndarray, split_name: str):
    split_dir = EMBED_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / f"X_{split_name}.npy", X)
    np.save(split_dir / f"y_{split_name}.npy", y)
    logger.info(f"Saved embeddings for {split_name} to {split_dir}")

if __name__ == "__main__":
    preprocessor = AudioPreprocessor()
    vggish_model = hub.load(VGGISH_MODEL_URL)

    bentoml.picklable_model.save_model("audio_preprocessor", preprocessor)
    bentoml.tensorflow.save_model("vggish_model", vggish_model)

    for split in ["train", "val", "test"]:
        X, y = load_data_from_split(split, preprocessor, vggish_model)
        save_embeddings(X, y, split)

