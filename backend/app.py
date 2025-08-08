"""
Three-layer TouristBot backend (L1: CSV lookup, L2: pretrained pickles, L3: mBERT fallback)
Drop this file into your project `backend/` folder along with:
 - `tourism_dataset.csv` (columns at least: 'question' and 'answer')
 - `LR Model.pkl` and/or `RF Model.pkl` (scikit-learn models)
 - optionally `vectorizer.pkl` (TF-IDF vectorizer from Colab)

Usage:
    python app.py

Endpoints:
    POST /chat  -> JSON {"message":"..."}
    GET  /health -> simple healthcheck

Notes:
 - This script will try to load saved vectorizer/model pickles if they exist.
 - If vectorizer pickle is missing, it will fit a new TF-IDF on the CSV (recommended: save vectorizer when training in Colab).
 - L1 uses TF-IDF cosine similarity + RapidFuzz token ratio for typo-tolerance.
 - L2 uses LR and RF (if present). If both models exist and agree with high confidence, returns that.
 - L3 uses sentence-transformers (distiluse-base-multilingual-cased-v2) for semantic similarity.

Install requirements (example):
    pip install flask pandas scikit-learn sentence-transformers rapidfuzz joblib

"""

import os
import json
import logging
from typing import Optional, Tuple

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Rapidfuzz for fuzzy matching (fast)
from rapidfuzz import fuzz

# sentence-transformers for mBERT embeddings
from sentence_transformers import SentenceTransformer, util

# Optional: use joblib for faster sklearn model loading
# from joblib import load

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("touristbot_backend")

# ========== CONFIG ===========
CSV_PATH = "tourism_dataset.csv"  # your dataset
VECTORIZER_PKL = "vectorizer.pkl"  # optional: TF-IDF vectorizer from Colab
LR_PKL = "LR Model.pkl"
RF_PKL = "RF Model.pkl"

# Matching thresholds (tune as needed)
L1_TFIDF_THRESHOLD = 0.70    # cosine similarity threshold for dataset match
L1_FUZZY_THRESHOLD = 70      # rapidfuzz token ratio threshold (0-100)
L2_PROBA_THRESHOLD = 0.60    # min probability for L2 model to be confident
L3_EMBED_THRESH = 0.55       # cosine threshold for sentence-transformers

# sentence-transformers model (semantic multilingual)
MBERT_MODEL_NAME = "distiluse-base-multilingual-cased-v2"

# ========== APP ===========
app = Flask(__name__)

# Global holders
dataset_df: Optional[pd.DataFrame] = None
vectorizer: Optional[TfidfVectorizer] = None
tfidf_matrix = None
lr_model = None
rf_model = None
mbert_model = None
mbert_embeddings = None  # torch tensors


# ========== UTIL / LOADERS ===========

def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    # Minimal expected columns
    expected = ["question", "answer"]
    for col in expected:
        if col not in df.columns:
            raise ValueError(f"Dataset must contain column: '{col}'")
    df = df.dropna(subset=["question", "answer"]).reset_index(drop=True)
    return df


def try_load_pickle(path: str):
    if os.path.exists(path):
        logger.info(f"Loading pickle: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    logger.info(f"Pickle not found: {path}")
    return None


def prepare_vectorizer_and_matrix(df: pd.DataFrame):
    global vectorizer, tfidf_matrix
    vec = try_load_pickle(VECTORIZER_PKL)
    if vec is not None:
        vectorizer = vec
        logger.info("Using loaded vectorizer from pickle.")
    else:
        logger.info("Fitting new TF-IDF vectorizer on dataset questions (saving recommended).")
        vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=1)
        vectorizer.fit(df["question"].values.astype("U"))
    tfidf_matrix = vectorizer.transform(df["question"].values.astype("U"))


def load_models():
    global lr_model, rf_model
    lr_model = try_load_pickle(LR_PKL)
    rf_model = try_load_pickle(RF_PKL)


def init_mbert_embeddings(df: pd.DataFrame):
    global mbert_model, mbert_embeddings
    try:
        logger.info(f"Loading mBERT model '{MBERT_MODEL_NAME}' (this may take a while)...")
        mbert_model = SentenceTransformer(MBERT_MODEL_NAME)
        # Precompute embeddings for dataset questions (and maybe answers)
        questions = df["question"].astype(str).tolist()
        # Returns torch tensors; utility functions accept them
        mbert_embeddings = mbert_model.encode(questions, convert_to_tensor=True, show_progress_bar=True)
        logger.info("mBERT embeddings ready.")
    except Exception as e:
        logger.exception("Failed to load or compute mBERT embeddings: %s", e)
        mbert_model = None
        mbert_embeddings = None


# ========== LAYER 1: DATASET SEARCH (TF-IDF + fuzzy) ===========

def l1_search(user_message: str) -> Optional[Tuple[str, float]]:
    """Try to find a matching answer from the CSV dataset.
    Returns (answer, score) if found with confidence; otherwise None.
    """
    global tfidf_matrix, vectorizer, dataset_df
    if tfidf_matrix is None or vectorizer is None:
        return None

    # TF-IDF cosine similarity
    q_vec = vectorizer.transform([user_message])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    # Also check fuzzy text ratio on raw text (typo tolerance / token similarity)
    fuzzy_scores = dataset_df["question"].astype(str).apply(lambda x: fuzz.token_set_ratio(x, user_message))
    best_fuzzy_idx = int(fuzzy_scores.idxmax())
    best_fuzzy_score = float(fuzzy_scores.max())

    logger.debug("L1 TF-IDF best score: %s (idx=%s)", best_score, best_idx)
    logger.debug("L1 Fuzzy best score: %s (idx=%s)", best_fuzzy_score, best_fuzzy_idx)

    # Choose winner between TF-IDF and fuzzy based on thresholds
    if best_score >= L1_TFIDF_THRESHOLD:
        answer = dataset_df.loc[best_idx, "answer"]
        return str(answer), best_score

    if best_fuzzy_score >= L1_FUZZY_THRESHOLD:
        answer = dataset_df.loc[best_fuzzy_idx, "answer"]
        # normalize fuzzy score to 0-1 for return
        return str(answer), best_fuzzy_score / 100.0

    return None


# ========== LAYER 2: PRETRAINED PICKLE MODELS (LR / RF) ===========

def l2_predict(user_message: str) -> Optional[Tuple[str, float]]:
    """Use loaded scikit-learn models to predict an answer.
    Returns (answer, confidence) or None.
    Assumes models accept raw string input transformed by the same TF-IDF vectorizer.
    """
    global lr_model, rf_model, vectorizer
    if vectorizer is None:
        return None

    q_vec = vectorizer.transform([user_message])

    candidates = []

    # Helper to extract proba if available
    def model_predict_with_proba(model, name):
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(q_vec)
                top_idx = int(np.argmax(probs[0]))
                top_proba = float(np.max(probs[0]))
                pred = model.classes_[top_idx]
                logger.debug("%s prediction: %s (proba=%s)", name, pred, top_proba)
                return str(pred), top_proba
            else:
                pred = model.predict(q_vec)[0]
                logger.debug("%s prediction (no proba): %s", name, pred)
                # unknown confidence
                return str(pred), 0.5
        except Exception as e:
            logger.exception("Model %s prediction failed: %s", name, e)
            return None

    if lr_model is not None:
        res = model_predict_with_proba(lr_model, "LR")
        if res:
            candidates.append(("LR", res[0], res[1]))

    if rf_model is not None:
        res = model_predict_with_proba(rf_model, "RF")
        if res:
            candidates.append(("RF", res[0], res[1]))

    if not candidates:
        return None

    # If both models present and same prediction, pick it if avg proba high enough
    preds = [c[1] for c in candidates]
    if len(preds) > 1 and all(p == preds[0] for p in preds):
        avg_proba = float(np.mean([c[2] for c in candidates]))
        if avg_proba >= L2_PROBA_THRESHOLD:
            return preds[0], avg_proba
        else:
            # return prediction but low confidence
            return preds[0], avg_proba

    # Otherwise pick model with highest probability
    best = max(candidates, key=lambda x: x[2])
    if best[2] >= L2_PROBA_THRESHOLD:
        return best[1], best[2]

    return None


# ========== LAYER 3: mBERT semantic search using sentence-transformers ===========

def l3_mbert_search(user_message: str) -> Optional[Tuple[str, float]]:
    global mbert_model, mbert_embeddings, dataset_df
    if mbert_model is None or mbert_embeddings is None:
        return None

    # Encode query
    query_emb = mbert_model.encode(user_message, convert_to_tensor=True)
    # Compute cosine similarity
    cos_scores = util.pytorch_cos_sim(query_emb, mbert_embeddings)[0]
    top_idx = int(torch_argmax(cos_scores)) if hasattr(cos_scores, 'argmax') else int(np.argmax(cos_scores.cpu().numpy()))
    top_score = float(cos_scores[top_idx])

    logger.debug("L3 mBERT top score: %s (idx=%s)", top_score, top_idx)

    if top_score >= L3_EMBED_THRESH:
        answer = dataset_df.loc[top_idx, "answer"]
        return str(answer), top_score

    return None


# Small helper to avoid torch import in top-level if not needed
def torch_argmax(tensor_obj):
    try:
        import torch
        return int(torch.argmax(tensor_obj))
    except Exception:
        try:
            # fallback for numpy
            return int(np.argmax(tensor_obj.cpu().numpy()))
        except Exception:
            return int(np.argmax(tensor_obj))


# ========== ENDPOINTS ===========
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "l1_ready": tfidf_matrix is not None,
        "l2_ready": (lr_model is not None) or (rf_model is not None),
        "l3_ready": mbert_model is not None
    })


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON provided"}), 400

    user_message = data.get("message") or data.get("query") or ""
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    logger.info("Incoming query: %s", user_message)

    # L1: dataset search
    l1 = l1_search(user_message)
    if l1:
        answer, score = l1
        logger.info("L1 matched (score=%.3f). Returning dataset answer.", score)
        return jsonify({"source": "dataset", "answer": answer, "score": score})

    # L2: pretrained models
    l2 = l2_predict(user_message)
    if l2:
        answer, conf = l2
        logger.info("L2 model provided answer (conf=%.3f).", conf)
        return jsonify({"source": "model", "answer": answer, "score": conf})

    # L3: mBERT semantic search
    l3 = l3_mbert_search(user_message)
    if l3:
        answer, score = l3
        logger.info("L3 mBERT matched (score=%.3f). Returning answer.", score)
        return jsonify({"source": "mbert", "answer": answer, "score": score})

    # Nothing found
    logger.info("No good match found in any layer.")
    return jsonify({"source": "none", "answer": None, "score": 0})


# ========== BOOTSTRAP ===========
if __name__ == "__main__":
    # Load dataset
    try:
        dataset_df = load_dataset(CSV_PATH)
        prepare_vectorizer_and_matrix(dataset_df)
    except Exception as e:
        logger.exception("Failed to load or prepare dataset: %s", e)
        dataset_df = None

    # Load LR / RF pickles
    try:
        load_models()
    except Exception as e:
        logger.exception("Failed to load models: %s", e)

    # Init mBERT embeddings (optional but recommended)
    try:
        if dataset_df is not None:
            init_mbert_embeddings(dataset_df)
    except Exception as e:
        logger.exception("Failed to init mBERT: %s", e)

    app.run(host="0.0.0.0", port=5000, debug=True)
