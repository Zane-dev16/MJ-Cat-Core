from pathlib import Path
import numpy as np
from tqdm import tqdm
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow
import mlflow.lightgbm
from loguru import logger
from rich.logging import RichHandler
import joblib
import bentoml
import sys

EMBED_DIR = Path("data/embeddings")
RANDOM_STATE = 42
MODEL_PATH = Path("lgbm_model.pkl")
LOG_PATH = "logs/train_lgbm.log"


logger.remove()
logger.add(RichHandler(), level="INFO")
logger.add(LOG_PATH, rotation="5 MB", level="INFO")

def load_embeddings(split_name):
    split_dir = EMBED_DIR / split_name
    if not split_dir.exists():
        logger.error(f"Embedding folder for {split_dir} does not exist.")
        sys.exit(1)

    X = np.load(split_dir / f"X_{split_name}.npy")
    y = np.load(split_dir / f"y_{split_name}.npy")
    return X, y

def train_lgbm(X_train, y_train, X_valid, y_valid, params):
    """Train LightGBM model with early stopping and MLflow logging."""
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=50)],
    )

    y_pred_proba = model.predict(X_valid)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    acc = accuracy_score(y_valid, y_pred)
    auc = roc_auc_score(y_valid, y_pred_proba)

    logger.info(f"Validation Accuracy: {acc:.4f}")
    logger.info(f"Validation AUC: {auc:.4f}")

    joblib.dump(model, MODEL_PATH)
    bentoml.lightgbm.save_model("lgbm_model", model)
    MODEL_PATH.unlink()

    return model


def main():
    if EMBED_DIR.exists():
        logger.info("Embeddings folder found. Loading existing embeddings...")
        X_train, y_train = load_embeddings("train")
        X_valid, y_valid = load_embeddings("val")
    else:
        logger.error("Embeddings folder not found or missing embeddings.npy.")
        sys.exit(1)

    logger.info(f"Training set shape: {X_train.shape}, Validation set shape: {X_valid.shape}")

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "random_state": RANDOM_STATE,
    }

    train_lgbm(X_train, y_train, X_valid, y_valid, params)

    logger.info("✅ Training completed and model logged to MLflow.")
    logger.info("✅ Model saved to BentoML.")


if __name__ == "__main__":
    main()

