"""
Explainable AI (XAI) Module using SHAP (SHapley Additive exPlanations).
Computes exact Shapley attribution values for the production spending forecaster,
explaining why the model predicted next month's spending to rise or fall.
Generates structured driver summaries for the Grounded LLM and Plotly waterfall visualizations.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List

FEATURE_LABELS = {
    'previous_month_spending': 'Previous Month Spending',
    'spending_2_months_ago': 'Spending 2 Months Ago',
    'spending_3_months_ago': 'Spending 3 Months Ago',
    '3_month_moving_average': '3-Month Moving Average',
    '6_month_moving_average': '6-Month Moving Average',
    'month': 'Month of Year (Seasonality)',
    'prev_food_spending': 'Recent Food Spending',
    'prev_shopping_spending': 'Recent Shopping Spending',
    'prev_transport_spending': 'Recent Transport Spending',
    'prev_bills_spending': 'Recent Bills Spending',
    'number_of_unique_merchants': 'Unique Merchant Diversity',
    'weekend_spending': 'Recent Weekend Spending',
    'weekend_ratio': 'Weekend Spending Ratio'
}


def explain_next_month_forecast(forecaster_path: str = 'evaluation/best_forecaster.joblib') -> Dict[str, Any]:
    """
    Uses TreeExplainer to compute exact feature attributions for next month's prediction.
    """
    from src.forecasting import forecast_next_month

    fc = forecast_next_month()
    saved = joblib.load(forecaster_path)
    model = saved['model']
    feat_cols = saved['feature_cols']
    valid_df = saved['df_monthly']

    X_train = valid_df[feat_cols].astype(float)
    X_next = pd.DataFrame([fc['features_used']])[feat_cols].astype(float)

    # Initialize TreeExplainer
    explainer = shap.TreeExplainer(model, data=X_train)
    shap_values = explainer(X_next)

    # Base value (mean prediction over training dataset)
    base_val = float(shap_values.base_values[0]) if hasattr(shap_values.base_values, '__len__') else float(shap_values.base_values)
    pred_val = float(fc['predicted_total'])

    # Feature contributions
    vals = shap_values.values[0]
    feature_impacts = []

    for col, val, shap_val in zip(feat_cols, X_next.iloc[0], vals):
        lbl = FEATURE_LABELS.get(col, col)
        feature_impacts.append({
            'feature': col,
            'label': lbl,
            'value': round(float(val), 2),
            'shap_impact_eur': round(float(shap_val), 2),
            'abs_impact': abs(round(float(shap_val), 2)),
            'direction': 'increases_spending' if shap_val > 0 else 'decreases_spending'
        })

    # Sort by absolute impact
    feature_impacts = sorted(feature_impacts, key=lambda x: x['abs_impact'], reverse=True)

    # Top upward and downward drivers
    top_positive = [f for f in feature_impacts if f['shap_impact_eur'] > 0][:4]
    top_negative = [f for f in feature_impacts if f['shap_impact_eur'] < 0][:4]

    # Format text explanation
    pos_desc = [f"{f['label']} (+?{f['shap_impact_eur']:.2f})" for f in top_positive]
    neg_desc = [f"{f['label']} (-?{abs(f['shap_impact_eur']):.2f})" for f in top_negative]

    summary_text = (
        f"Baseline historical average spending is ?{base_val:.2f}. "
        f"Next month is forecasted at ?{pred_val:.2f} (a net difference of ?{pred_val - base_val:+.2f}). "
    )
    if pos_desc:
        summary_text += f"Primary factors driving predicted spending upward: {', '.join(pos_desc)}. "
    if neg_desc:
        summary_text += f"Factors pulling predicted spending downward: {', '.join(neg_desc)}."

    return {
        'base_value': round(base_val, 2),
        'predicted_value': round(pred_val, 2),
        'total_shap_adjustment': round(pred_val - base_val, 2),
        'feature_impacts': feature_impacts,
        'top_positive_drivers': top_positive,
        'top_negative_drivers': top_negative,
        'summary_text': summary_text,
        'shap_object': shap_values,
        'X_next': X_next
    }


if __name__ == '__main__':
    explanation = explain_next_month_forecast()
    print("=== SHAP FORECAST EXPLANATION ===")
    print(f"Base Value: EUR {explanation['base_value']}")
    print(f"Predicted Total: EUR {explanation['predicted_value']}")
    print("\nTop Contributing Features:")
    for f in explanation['feature_impacts'][:6]:
        sign = '+' if f['shap_impact_eur'] > 0 else ''
        print(f"  {f['label']:30} -> {sign}EUR {f['shap_impact_eur']:.2f} (input: {f['value']})")
    print(f"\nSummary:\n{explanation['summary_text']}")
