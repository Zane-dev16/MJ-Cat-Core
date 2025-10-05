from pathlib import Path
import numpy as np
import librosa
import tensorflow_hub as hub
from tqdm import tqdm
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow
import mlflow.lightgbm
from loguru import logger
from rich.logging import RichHandler
import joblib
import dagshub
import os

MODEL_PATH = "vggish_lgbm_model.pkl"

logger.remove()
logger.add(RichHandler(), level="INFO")
logger.add("logs/train_lgbm.log", rotation="5 MB", level="INFO")

dagshub.init(repo_owner='Zane-dev16', repo_name='MJ-Cat-Core', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Zane-dev16/MJ-Cat-Core.mlflow")

DATA_DIR = Path("data/prepared")
EMBED_DIR = Path("data/embeddings")
VGGISH_MODEL_URL = "https://tfhub.dev/google/vggish/1"
RANDOM_STATE = 42

def waveform_to_examples(waveform, sr=16000):
    """Convert waveform to VGGish input (batch of 1D float32 arrays)"""
    waveform = np.array(waveform, dtype=np.float32)
    return waveform

def extract_embedding(file_path, model):
    """Load audio and get VGGish embedding"""
    try:
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        input_tensor = waveform_to_examples(audio, sr)
        embedding = model(input_tensor).numpy()
        mean_embedding = np.mean(embedding, axis=0)
        std_embedding = np.std(embedding, axis=0)
        max_embedding = np.max(embedding, axis=0)
        richer_feature_vector = np.concatenate([mean_embedding, std_embedding, max_embedding])
        return richer_feature_vector
    except Exception as e:
        logger.error(f"Failed to extract embedding for {file_path}: {e}")
        return None

def load_data_from_split(split_name, model):
    X, y = [], []
    for label in ["positive", "negative"]:
        folder = DATA_DIR / split_name / label
        if not folder.exists():
            logger.warning(f"Folder {folder} does not exist. Skipping.")
            continue
        files = list(folder.glob("**/*.wav"))
        for f in tqdm(files, desc=f"Processing {split_name}/{label}"):
            embedding = extract_embedding(f, model)
            if embedding is not None:
                X.append(embedding)
                y.append(1 if label == "positive" else 0)
    return np.array(X), np.array(y)

def save_embeddings(X_train, y_train, X_valid, y_valid):
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(EMBED_DIR / "embeddings.npz",
             X_train=X_train, y_train=y_train,
             X_valid=X_valid, y_valid=y_valid)
    logger.info(f"Saved embeddings to {EMBED_DIR/'embeddings.npz'}")

def load_embeddings():
    data = np.load(EMBED_DIR / "embeddings.npz")
    logger.info(f"Loaded embeddings from {EMBED_DIR/'embeddings.npz'}")
    return data["X_train"], data["y_train"], data["X_valid"], data["y_valid"]

if EMBED_DIR.exists() and (EMBED_DIR / "embeddings.npz").exists():
    logger.info("Embeddings folder found. Loading existing embeddings...")
    X_train, y_train, X_valid, y_valid = load_embeddings()
else:
    logger.info("No embeddings found. Extracting new embeddings using VGGish...")
    logger.info("Loading VGGish model...")
    vggish_model = hub.load(VGGISH_MODEL_URL)
    
    X_train, y_train = load_data_from_split("train", vggish_model)
    X_valid, y_valid = load_data_from_split("val", vggish_model)
    
    save_embeddings(X_train, y_train, X_valid, y_valid)

logger.info(f"Training set: {X_train.shape}, Validation set: {X_valid.shape}")

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "random_state": RANDOM_STATE,
}

logger.info("Starting MLflow run...")
with mlflow.start_run(run_name="vggish_lgbm"):
    mlflow.log_params(params)
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
    
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
        ],
    )
    
    y_pred_proba = model.predict(X_valid)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    acc = accuracy_score(y_valid, y_pred)
    auc = roc_auc_score(y_valid, y_pred_proba)
    
    logger.info(f"Validation Accuracy: {acc:.4f}")
    logger.info(f"Validation AUC: {auc:.4f}")
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("auc", auc)
    
    joblib.dump(model, MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH, artifact_path="vggish_lgbm_model")

logger.info("✅ Training completed and model logged to MLflow.")

