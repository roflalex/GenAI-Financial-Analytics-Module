# ?? AI Personal Finance Forecasting & Grounded Insights

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-00ADB5.svg)](https://shap.readthedocs.io/)

An end-to-end Machine Learning and Grounded Generative AI system that digests raw, messy bank transactions, automatically categorises expenses with NLP, learns spending patterns, forecasts multi-horizon future expenditures without lookahead leakage, flags atypical financial anomalies, explains predictions with SHAP, and provides an interactive conversational AI analyst.

---

## ??? Core Philosophy: ML Predicts, GenAI Explains

A critical architecture distinction in this project:

> **Machine Learning does the quantitative analysis, prediction, classification, and anomaly detection.**  
> **Generative AI provides the natural language synthesis, explanation, and interactive querying layer.**

Instead of asking an LLM to hallucinate calculations on raw receipts, the application feeds structured numerical facts and Shapley attributions to the LLM under strict grounding constraints.

```
                      RAW TRANSACTIONS
                             ?
                             ?
                      DATA CLEANING
                (Deduplication, normalization,
                  multi-format date parser)
                             ?
                             ?
                    EXPENSE CLASSIFIER
                  (TF-IDF vs Embeddings)
                             ?
                             ?
                    FEATURE ENGINEERING
             (Lags, rolling means, behaviors)
                             ?
              ???????????????????????????????
              ?                             ?
       ANOMALY DETECTION            SPENDING FORECAST
     (Isolation Forest, LOF,      (Rolling-Window TimeSeries:
         Z-Score Bounds)             RF, XGBoost, Ridge)
              ?                             ?
              ???????????????????????????????
                             ?
                      SHAP EXPLAINER
                (Exact feature attributions)
                             ?
                             ?
                 STRUCTURED ANALYTICS JSON
                             ?
                             ?
                    AI FINANCIAL ANALYST
                 (Grounded GenAI / Local)
                             ?
                             ?
                    STREAMLIT DASHBOARD
```

---

## ?? Three Core Machine Learning Problems

### 1. Expense Categorisation (NLP Classification)
Automates classification of raw merchant strings (e.g. `"TESCO EXTRA BRAY -?48.72"` $\rightarrow$ `Groceries`, `"SPOTIFY AB 10.99 EUR"` $\rightarrow$ `Entertainment & Subscriptions`).

We compare three distinct paradigms:
- **Baseline**: TF-IDF (1?2 ngrams) + Multinomial Logistic Regression
- **Tree-based**: TF-IDF + Random Forest Classifier
- **Dense Semantic**: Sentence Transformer (`all-MiniLM-L6-v2`) Embeddings + Logistic Classifier

### 2. Multi-Horizon Spending Forecasting (Time-Series Regression)
Predicts total spending for the upcoming month alongside category breakdown and 80% prediction intervals.

**Rigorous Data Leakage Prevention:**
Traditional random train/test shuffling causes severe lookahead bias on financial data. We enforce a strict **Expanding Rolling Window (TimeSeriesSplit)**:
$$\text{Train on } [1 \dots t-1] \longrightarrow \text{Predict } t$$

Models benchmarked:
1. **Naive Baseline**: 3-Month moving average
2. **Linear Regression (Ridge)**: Capturing trend weights
3. **Random Forest Regressor**: Non-linear behavioral interactions
4. **XGBoost / Gradient Boosting Regressor**: Gradient boosted decision trees

### 3. Spending Anomaly Detection (Unsupervised Outlier Detection)
Flags unexpected financial spikes (e.g., normal dining of ?20??100 suddenly registering a ?446 transaction):
- **Isolation Forest**: Multivariate tree-partitioning outlier scoring
- **Local Outlier Factor (LOF)**: Density-based local neighborhood anomaly detection
- **Dynamic Statistical Bounds**: Category-specific Interquartile Range (IQR) and Z-score expected limits

---

## ?? Explainable AI (XAI) with SHAP

Why did the model predict next month's spending to rise to ?6,465? Rather than treating the tree model as a black box, we use `shap.TreeExplainer`:

```
Baseline Average Spending: ?5,620.53
Model Forecast:            ?6,465.94  (Net ?: +?845.41)

Top Contributing Factors:
  Recent Weekend Spending        ???  +?387.23
  Unique Merchant Diversity      ???  +?311.87
  Recent Transport Spending      ???  +?64.12
  Spending 3 Months Ago          ???  +?54.49
  Previous Month Spending        ???  +?31.47
  Recent Bills Spending          ???  -?28.48
```

The Grounded AI engine synthesizes these exact Shapley values into plain English for the user.

---

## ?? Benchmark Results

Evaluated across the pipeline:

| Problem Domain | Model | Primary Metric | Primary Score | Secondary Metric | Secondary Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Categorisation (NLP)** | Sentence Transformer + Logistic Regression | Weighted F1 | **0.9874** | Accuracy | **98.73%** |
| **Categorisation (NLP)** | TF-IDF + Logistic Regression | Weighted F1 | **0.9874** | Accuracy | **98.73%** |
| **Categorisation (NLP)** | TF-IDF + Random Forest | Weighted F1 | 0.9308 | Accuracy | 94.23% |
| **Forecasting (Time-Series)** | Random Forest Regressor | MAE | **?695.52** | MAPE | **13.46%** |
| **Forecasting (Time-Series)** | XGBoost Regressor | MAE | ?796.41 | MAPE | 15.57% |
| **Forecasting (Time-Series)** | Naive Baseline (3m Moving Avg) | MAE | ?801.83 | MAPE | 14.34% |
| **Forecasting (Time-Series)** | Linear Regression (Ridge) | MAE | ?886.79 | MAPE | 17.24% |
| **Anomaly Detection** | Consensus (Isolation Forest + LOF + IQR) | Outlier Rate | 4.99% | High Severity | 48 transactions |

---

## ?? Conversational Grounded AI & Data Q&A

Users can ask questions in natural language:

- **"What were my 3 biggest purchases?"**
  > *Queries top transactions:*  
  > 1. ?1,342.05 ? BOOKING.COM HOTEL (*Travel*) on 2023-10-09  
  > 2. ?1,132.39 ? RYANAIR AIRLINE (*Travel*) on 2025-03-21  
  > 3. ?938.71 ? PENNEYS PRIMARK BRAY (*Shopping*) on 2024-07-01  
- **"How much did I spend on weekends compared with weekdays?"**
  > *Calculates exact split:* Weekdays: ?4,023.03, Weekends: ?2,193.56 (Weekend Share: 35.3%).
- **"Why is my predicted spending higher next month?"**
  > *Translates SHAP drivers into clear narrative causality without hallucinations.*
- **"Am I on track to stay under ?6,000?"**
  > *Evaluates budget vs predicted point estimate and confidence boundaries.*

Dual Backend:
1. **Live LLM API**: Connects to OpenAI (`gpt-4o-mini`) when an API key is provided.
2. **Local Smart Analyst**: Fully deterministic rule-based grounded financial reasoning engine that works 100% offline out-of-the-box.

---

## ?? Repository Structure

```
financial-predictor/
??? data/
?   ??? raw/
?   ?   ??? messy_transactions.csv          # Synthetic raw dirty log (dates, casing, nulls)
?   ??? processed/
?       ??? cleaned_transactions.csv        # Cleaned, standardized transactions
?       ??? monthly_aggregated_features.csv # Aggregated time-series feature matrix
??? notebooks/
?   ??? 01_eda.ipynb                        # EDA, data hygiene, and distribution analysis
?   ??? 02_expense_classification.ipynb     # NLP categorisation comparisons & confusion matrices
?   ??? 03_spending_forecasting.ipynb       # Time-series rolling CV & forecast trajectories
?   ??? 04_model_evaluation.ipynb          # Anomaly radar, SHAP waterfall & LLM Q&A
??? src/
?   ??? __init__.py
?   ??? preprocessing.py                    # Messy data generator & cleaning pipeline
?   ??? categorisation.py                   # Multi-class NLP classifier training & prediction
?   ??? forecasting.py                      # Time-series feature engineering & rolling CV
?   ??? anomaly_detection.py                # Isolation Forest, LOF & IQR boundary detection
?   ??? explainability.py                   # SHAP TreeExplainer attribution calculations
?   ??? llm.py                             # Grounded context builder & natural language analyst
??? evaluation/
?   ??? model_results.csv                   # Master cross-model benchmark leaderboard
?   ??? categorisation_benchmark.csv        # Classification metrics
?   ??? forecasting_benchmark.csv           # Time-series error metrics
?   ??? detected_anomalies.csv              # Flagged outlier records
?   ??? categoriser_pipeline.joblib         # Serialized best NLP classifier
?   ??? best_forecaster.joblib              # Serialized production forecaster
??? app.py                                  # Full-featured Streamlit interactive dashboard
??? requirements.txt                        # Production dependencies
??? README.md                               # Project documentation
```

---

## ?? Quickstart Guide

### 1. Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/financial-predictor.git
cd financial-predictor
pip install -r requirements.txt
```

### 2. Run the Data & Modeling Pipeline
To generate raw data, clean it, train the models, and compute benchmarks:

```bash
# Generate and clean data
python src/preprocessing.py

# Train and benchmark categorisation models
python src/categorisation.py

# Feature engineering & rolling-window forecasting
python src/forecasting.py

# Run anomaly detection & SHAP explanation
python src/anomaly_detection.py
python src/explainability.py
```

### 3. Launch the Interactive Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ??? Tech Stack
- **Modeling & Machine Learning**: `scikit-learn`, `xgboost`, `sentence-transformers`, `shap`, `statsmodels`
- **Data Engineering**: `pandas`, `numpy`
- **Dashboard & Interactive UI**: `streamlit`, `plotly`
- **Natural Language & LLMs**: `openai` (optional live API), Local Deterministic Grounding Engine
