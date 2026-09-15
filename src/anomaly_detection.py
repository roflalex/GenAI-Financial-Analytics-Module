"""
Spending Anomaly Detection Module.
Identifies unusual financial transactions and category-level spending surges using:
1. Isolation Forest (Multivariate unsupervised outlier detection)
2. Local Outlier Factor (LOF - Density-based local outlier detection)
3. Statistical Category Z-Score / IQR Bounds (Expected spending ranges per category)

Computes normalized anomaly scores (0.0 to 1.0), expected typical ranges,
and human-readable explanations.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

ANOMALIES_SAVE_PATH = 'evaluation/detected_anomalies.csv'


def compute_category_baselines(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Computes statistical bounds for each category:
    median, mean, std, Q25, Q75, expected_min, expected_max.
    """
    baselines = {}
    for cat, grp in df.groupby('category'):
        amounts = grp['amount'].values
        mean_val = float(np.mean(amounts))
        std_val = float(np.std(amounts)) if len(amounts) > 1 else 10.0
        median_val = float(np.median(amounts))
        q25 = float(np.percentile(amounts, 25))
        q75 = float(np.percentile(amounts, 75))
        iqr = q75 - q25

        # Typical expected range: Q25 - 0.5*IQR to Q75 + 1.8*IQR (bounded by observed min/max)
        exp_min = max(float(grp['amount'].min()), round(max(2.0, q25 - 0.5 * iqr), 2))
        exp_max = round(min(float(grp['amount'].max()), q75 + 2.0 * iqr), 2)

        baselines[cat] = {
            'mean': round(mean_val, 2),
            'std': round(std_val, 2),
            'median': round(median_val, 2),
            'q25': round(q25, 2),
            'q75': round(q75, 2),
            'expected_min': exp_min,
            'expected_max': exp_max
        }
    return baselines


def detect_anomalies(df_path: str = 'data/processed/cleaned_transactions.csv',
                     contamination: float = 0.025) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs comprehensive anomaly detection:
    - Runs Isolation Forest on transaction amount and category relative deviation
    - Runs Local Outlier Factor (LOF)
    - Applies Category Z-Score bounds
    """
    df = pd.read_csv(df_path)
    df['date'] = pd.to_datetime(df['date'])

    baselines = compute_category_baselines(df)

    # Engineer relative deviation features
    df['category_median'] = df['category'].map(lambda c: baselines.get(c, {}).get('median', 30.0))
    df['category_std'] = df['category'].map(lambda c: max(1.0, baselines.get(c, {}).get('std', 15.0)))
    df['category_zscore'] = (df['amount'] - df['category_median']) / df['category_std']
    df['amount_ratio_to_median'] = df['amount'] / np.maximum(df['category_median'], 1.0)
    df['day_of_week_num'] = df['date'].dt.dayofweek

    # Feature matrix for unsupervised models
    feature_cols = ['amount', 'category_zscore', 'amount_ratio_to_median', 'is_weekend']
    X = df[feature_cols].copy()
    X['is_weekend'] = X['is_weekend'].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    iso_preds = iso_forest.fit_predict(X_scaled)  # -1 for anomaly, 1 for normal
    iso_scores = -iso_forest.decision_function(X_scaled)  # higher = more anomalous

    # Normalize iso_score to 0..1 range
    min_s, max_s = iso_scores.min(), iso_scores.max()
    norm_iso_scores = (iso_scores - min_s) / (max_s - min_s + 1e-8)

    # 2. Local Outlier Factor (LOF)
    lof = LocalOutlierFactor(n_neighbors=25, contamination=contamination, novelty=False)
    lof_preds = lof.fit_predict(X_scaled)
    lof_factors = -lof.negative_outlier_factor_  # ~1.0 is inlier, >1.5 is outlier

    df['iso_anomaly'] = (iso_preds == -1)
    df['lof_anomaly'] = (lof_preds == -1)
    df['zscore_anomaly'] = (df['category_zscore'] > 2.8)
    df['anomaly_score'] = np.round(norm_iso_scores, 4)

    # Multi-method consensus
    df['is_anomaly'] = df['iso_anomaly'] | df['zscore_anomaly'] | df['lof_anomaly']

    # Filter anomaly dataframe
    anomalies_df = df[df['is_anomaly']].copy().sort_values(by='anomaly_score', ascending=False)

    # Format output fields
    def build_explanation(row):
        cat = row['category']
        b = baselines.get(cat, {})
        med = b.get('median', 0.0)
        exp_max = b.get('expected_max', 0.0)
        mult = row['amount'] / med if med > 0 else 0
        return (f"Amount ?{row['amount']:.2f} is {mult:.1f}x typical {cat} median (?{med:.2f}). "
                f"Exceeds typical ceiling of ?{exp_max:.2f}.")

    anomalies_df['expected_range'] = anomalies_df['category'].map(
        lambda c: f"?{baselines.get(c, {}).get('expected_min', 0):.2f} ? ?{baselines.get(c, {}).get('expected_max', 0):.2f}"
    )
    anomalies_df['severity'] = np.where(anomalies_df['anomaly_score'] > 0.80, 'High', 'Medium')
    anomalies_df['explanation'] = anomalies_df.apply(build_explanation, axis=1)

    export_cols = [
        'date', 'raw_merchant', 'clean_merchant', 'category', 'amount',
        'expected_range', 'anomaly_score', 'severity', 'explanation'
    ]
    anomalies_export = anomalies_df[export_cols].copy()
    anomalies_export['date'] = anomalies_export['date'].dt.strftime('%Y-%m-%d')

    os.makedirs(os.path.dirname(ANOMALIES_SAVE_PATH), exist_ok=True)
    anomalies_export.to_csv(ANOMALIES_SAVE_PATH, index=False)

    stats = {
        'total_transactions': len(df),
        'anomalies_detected': len(anomalies_export),
        'high_severity_count': int((anomalies_export['severity'] == 'High').sum()),
        'anomaly_percentage': round((len(anomalies_export) / len(df)) * 100, 2),
        'baselines': baselines
    }

    return anomalies_export, stats


if __name__ == '__main__':
    anom_df, s = detect_anomalies()
    print(f"Total Transactions: {s['total_transactions']}")
    print(f"Detected Anomalies: {s['anomalies_detected']} ({s['anomaly_percentage']}%)")
    print("\nTop 5 Detected Anomalies:")
    print(anom_df[['date', 'clean_merchant', 'category', 'amount', 'anomaly_score', 'severity']].head(5).to_string(index=False))
