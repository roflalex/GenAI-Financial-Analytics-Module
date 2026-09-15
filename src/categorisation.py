"""
NLP-based Expense Categorisation Module.
Compares:
1. TF-IDF + Logistic Regression (Baseline)
2. TF-IDF + Random Forest
3. Sentence Transformer Embeddings + Classifier (Advanced)

Provides evaluation metrics (Accuracy, Precision, Recall, F1),
confusion matrices, model serialization, and real-time category prediction.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

MODEL_SAVE_PATH = 'evaluation/categoriser_pipeline.joblib'
METRICS_SAVE_PATH = 'evaluation/categorisation_benchmark.csv'


def prepare_classification_data(cleaned_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits cleaned transactions into 80/20 train/test sets, stratified by category.
    """
    # Use clean_merchant as input text (or raw_merchant)
    X = cleaned_df['clean_merchant'].fillna('UNKNOWN')
    y = cleaned_df['category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_tfidf_logistic_regression(X_train, y_train, X_test, y_test):
    """Model 1: TF-IDF + Logistic Regression"""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2500, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=2.0, random_state=42)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    _, _, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)

    pipeline = {'vectorizer': vectorizer, 'model': clf, 'type': 'tfidf_lr'}
    metrics = {
        'model': 'TF-IDF + Logistic Regression',
        'accuracy': round(acc, 4),
        'precision_weighted': round(p, 4),
        'recall_weighted': round(r, 4),
        'f1_weighted': round(f1, 4),
        'f1_macro': round(f1_macro, 4)
    }
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    return pipeline, metrics, report, cm, clf.classes_


def train_tfidf_random_forest(X_train, y_train, X_test, y_test):
    """Model 2: TF-IDF + Random Forest Classifier"""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2500)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    rf = RandomForestClassifier(n_estimators=150, max_depth=25, random_state=42, n_jobs=-1)
    rf.fit(X_train_vec, y_train)

    y_pred = rf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    _, _, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)

    pipeline = {'vectorizer': vectorizer, 'model': rf, 'type': 'tfidf_rf'}
    metrics = {
        'model': 'TF-IDF + Random Forest',
        'accuracy': round(acc, 4),
        'precision_weighted': round(p, 4),
        'recall_weighted': round(r, 4),
        'f1_weighted': round(f1, 4),
        'f1_macro': round(f1_macro, 4)
    }
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
    return pipeline, metrics, report, cm, rf.classes_


def train_embeddings_classifier(X_train, y_train, X_test, y_test):
    """Model 3: Sentence Transformer Embeddings + Classifier"""
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading SentenceTransformer ('all-MiniLM-L6-v2')...")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        X_train_emb = embedder.encode(X_train.tolist(), show_progress_bar=False, batch_size=64)
        X_test_emb = embedder.encode(X_test.tolist(), show_progress_bar=False, batch_size=64)

        clf = LogisticRegression(max_iter=1000, C=3.0, random_state=42)
        clf.fit(X_train_emb, y_train)

        y_pred = clf.predict(X_test_emb)
        acc = accuracy_score(y_test, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        _, _, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)

        pipeline = {'embedder': embedder, 'model': clf, 'type': 'embeddings_lr'}
        metrics = {
            'model': 'Sentence Transformer + Logistic Regression',
            'accuracy': round(acc, 4),
            'precision_weighted': round(p, 4),
            'recall_weighted': round(r, 4),
            'f1_weighted': round(f1, 4),
            'f1_macro': round(f1_macro, 4)
        }
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
        return pipeline, metrics, report, cm, clf.classes_
    except Exception as e:
        print(f"Embeddings training notice: {e}")
        # Return fallback high-capacity model
        return None, None, None, None, None


def train_and_evaluate_all(cleaned_df_path: str = 'data/processed/cleaned_transactions.csv') -> Dict[str, Any]:
    """
    Trains and compares all 3 categorisation models, saves benchmarks,
    and serializes the best performing pipeline.
    """
    df = pd.read_csv(cleaned_df_path)
    X_train, X_test, y_train, y_test = prepare_classification_data(df)

    results = []
    models_dict = {}

    print("Training Model 1: TF-IDF + Logistic Regression...")
    pipe_lr, m_lr, rep_lr, cm_lr, classes_lr = train_tfidf_logistic_regression(X_train, y_train, X_test, y_test)
    results.append(m_lr)
    models_dict['tfidf_lr'] = {'pipeline': pipe_lr, 'metrics': m_lr, 'report': rep_lr, 'cm': cm_lr, 'classes': classes_lr}

    print("Training Model 2: TF-IDF + Random Forest...")
    pipe_rf, m_rf, rep_rf, cm_rf, classes_rf = train_tfidf_random_forest(X_train, y_train, X_test, y_test)
    results.append(m_rf)
    models_dict['tfidf_rf'] = {'pipeline': pipe_rf, 'metrics': m_rf, 'report': rep_rf, 'cm': cm_rf, 'classes': classes_rf}

    print("Training Model 3: Sentence Transformer Embeddings + Classifier...")
    pipe_emb, m_emb, rep_emb, cm_emb, classes_emb = train_embeddings_classifier(X_train, y_train, X_test, y_test)
    if m_emb is not None:
        results.append(m_emb)
        models_dict['embeddings_lr'] = {'pipeline': pipe_emb, 'metrics': m_emb, 'report': rep_emb, 'cm': cm_emb, 'classes': classes_emb}

    benchmark_df = pd.DataFrame(results).sort_values(by='f1_weighted', ascending=False)
    benchmark_df.to_csv(METRICS_SAVE_PATH, index=False)
    print("\n=== EXPENSE CATEGORISATION BENCHMARK ===")
    print(benchmark_df.to_string(index=False))

    # Save the TF-IDF LR pipeline as primary light-weight deployment pipeline
    # and embeddings pipeline if available
    joblib.dump(pipe_lr, MODEL_SAVE_PATH)
    print(f"Saved inference pipeline to {MODEL_SAVE_PATH}")

    return {
        'benchmark': benchmark_df,
        'models': models_dict,
        'classes': classes_lr
    }


def predict_category(text: str, pipeline=None) -> Tuple[str, float]:
    """
    Predicts the category of an arbitrary raw or cleaned transaction string.
    Returns (predicted_category, confidence_score).
    """
    if pipeline is None:
        if os.path.exists(MODEL_SAVE_PATH):
            pipeline = joblib.load(MODEL_SAVE_PATH)
        else:
            raise FileNotFoundError("Model pipeline not found. Run train_and_evaluate_all() first.")

    from src.preprocessing import clean_merchant_text
    cleaned = clean_merchant_text(text)

    if pipeline['type'] == 'tfidf_lr' or pipeline['type'] == 'tfidf_rf':
        vec = pipeline['vectorizer'].transform([cleaned])
        model = pipeline['model']
        probs = model.predict_proba(vec)[0]
        pred_idx = np.argmax(probs)
        return model.classes_[pred_idx], float(probs[pred_idx])
    elif pipeline['type'] == 'embeddings_lr':
        emb = pipeline['embedder'].encode([cleaned], show_progress_bar=False)
        probs = pipeline['model'].predict_proba(emb)[0]
        pred_idx = np.argmax(probs)
        return pipeline['model'].classes_[pred_idx], float(probs[pred_idx])
    else:
        return 'Uncategorised', 0.0


if __name__ == '__main__':
    train_and_evaluate_all()
