import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

import sys
import os

# Adding parent folder (backend/) to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app # app.py - to have access to its utility functions to be tested

# -------------------
# Utility Functions
# -------------------

def test_load_dataset_en(tmp_path):
    csv_file = tmp_path / "data.csv"
    df = pd.DataFrame({
        "Source text - Questions": ["Hi"],
        "Source Text - Answers": ["Hello"]
    })
    df.to_csv(csv_file, sep=";", index=False)

    app.LANG_MODE = "en"
    result = app.load_dataset(str(csv_file))
    assert list(result.columns) == ["question", "answer"]
    assert result.iloc[0].question == "Hi"


def test_load_dataset_bs(tmp_path):
    csv_file = tmp_path / "data_bs.csv"
    df = pd.DataFrame({
        "Bosnian translation - Questions": ["Kako si?"],
        "Bosnian translation - Answers": ["Dobro"]
    })
    df.to_csv(csv_file, sep=";", index=False)

    app.LANG_MODE = "bs"
    result = app.load_dataset(str(csv_file))
    assert result.iloc[0].answer == "Dobro"


def test_try_load_pickle_found(tmp_path):
    import pickle
    pkl_file = tmp_path / "vec.pkl"
    obj = {"test": 123}
    with open(pkl_file, "wb") as f:
        pickle.dump(obj, f)

    loaded = app.try_load_pickle(str(pkl_file))
    assert loaded["test"] == 123


def test_try_load_pickle_not_found(tmp_path):
    result = app.try_load_pickle(str(tmp_path / "missing.pkl"))
    assert result is None


@patch("app.try_load_pickle", return_value=None)
def test_prepare_vectorizer_and_matrix(mock_pickle):
    df = pd.DataFrame({"question": ["hello", "hi"]})
    app.prepare_vectorizer_and_matrix(df)
    assert app.vectorizer is not None
    assert app.tfidf_matrix.shape[0] == 2


@patch("app.try_load_pickle")
def test_load_models(mock_pickle):
    mock_pickle.return_value = "model"
    app.load_models()
    assert app.lr_model == "model" or app.rf_model == "model"


@patch("app.SentenceTransformer")
def test_init_mbert_embeddings(mock_st):
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.5, 0.2]])
    mock_st.return_value = mock_model

    df = pd.DataFrame({"question": ["hi"]})
    app.init_mbert_embeddings(df)
    assert app.mbert_model is not None
    assert app.mbert_embeddings is not None


@patch("app.get_translate_client")
def test_translate_text(mock_client):
    mock_client.return_value.translate.return_value = {"translatedText": "Bonjour"}
    result = app.translate_text("Hello", "fr")
    assert result == "Bonjour"


@patch("app.detect", return_value="en")
def test_detect_language(mock_detect):
    assert app.detect_language("hello") == "en"


# -------------------
# Layer 1 Tests
# -------------------

def test_l1_search_with_match():
    app.dataset_df = df = pd.DataFrame({"question": ["hi", "hello", "hey"], "answer": ["hello", "hi", "hey"]})
    app.prepare_vectorizer_and_matrix(app.dataset_df)
    result = app.l1_search("hi")
    assert result is not None
    answer, score = result
    assert answer == "hello"

# -------------------
# Layer 2 Tests
# -------------------

class DummyModel:
    def predict_proba(self, x):
        return np.array([[0.1, 0.9]])
    classes_ = np.array(["no", "yes"])

def test_l2_predict():
    app.vectorizer = app.TfidfVectorizer()
    app.vectorizer.fit(["hi", "bye"])
    app.lr_model = DummyModel()
    app.rf_model = DummyModel()
    result = app.l2_predict("hi")
    assert result[0] in ["yes", "no"]

# -------------------
# Layer 3 Tests
# -------------------

@patch("app.util.pytorch_cos_sim")
def test_l3_mbert_search(mock_cos):
    app.dataset_df = pd.DataFrame({"question": ["hi"], "answer": ["hello"]})
    app.mbert_model = MagicMock()
    app.mbert_model.encode.return_value = np.array([0.5, 0.5])
    app.mbert_embeddings = np.array([[0.5, 0.5]])
    mock_cos.return_value = np.array([[0.9]])

    result = app.l3_mbert_search("hi")
    assert result[0] == "hello"


# -------------------
# Endpoints
# -------------------

@patch("app.detect_language", return_value="en")
@patch("app.l1_search", return_value=("hello", 0.9))
def test_chat_endpoint(mock_l1, mock_detect):
    client = app.app.test_client()
    response = client.post("/chat", json={"message": "hi"})
    data = response.get_json()
    assert response.status_code == 200
    assert data["answer"] == "hello"


def test_health_endpoint():
    client = app.app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert "status" in data
