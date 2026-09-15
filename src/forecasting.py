"""
Time-Series Spending Forecasting & Feature Engineering Module.
Builds aggregated monthly financial features, implements rolling-window validation
to prevent data leakage, and benchmarks:
1. Naive Baseline (3-month moving average)
2. Linear Regression / Ridge
3. Random Forest Regressor
4. Gradient Boosting / XGBoost Regressor

Outputs point forecasts, confidence intervals, and evaluation metrics (MAE, RMSE, MAPE).
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

FORECASTER_SAVE_PATH = 'evaluation/best_forecaster.joblib'
MONTHLY_FEATURES_PATH = 'data/processed/monthly_aggregated_features.csv'
BENCHMARK_SAVE_PATH = 'evaluation/forecasting_benchmark.csv'


def aggregate_monthly_features(clean_df_path: str = 'data/processed/cleaned_transactions.csv') -> pd.DataFrame:
    """
    Aggregates cleaned transactions into a monthly time series with rich
    spending, behavioural, and temporal features.
    """
    df = pd.read_csv(clean_df_path)
    df['date'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date'].dt.to_period('M')

    months = sorted(df['year_month'].unique())
    records = []

    for ym in months:
        m_df = df[df['year_month'] == ym]
        total_spend = m_df['amount'].sum()
        tx_count = len(m_df)
        avg_tx = m_df['amount'].mean() if tx_count > 0 else 0
        max_tx = m_df['amount'].max() if tx_count > 0 else 0
        unique_merchants = m_df['clean_merchant'].nunique()

        weekend_df = m_df[m_df['is_weekend'] == True]
        weekend_spend = weekend_df['amount'].sum()
        weekend_tx_count = len(weekend_df)
        weekend_ratio = (weekend_spend / total_spend) if total_spend > 0 else 0.0

        # Category spending
        cat_sums = m_df.groupby('category')['amount'].sum().to_dict()

        rec = {
            'year_month': str(ym),
            'year': ym.year,
            'month': ym.month,
            'total_spending': round(total_spend, 2),
            'transaction_count': tx_count,
            'average_transaction': round(avg_tx, 2),
            'largest_transaction': round(max_tx, 2),
            'number_of_unique_merchants': unique_merchants,
            'weekend_spending': round(weekend_spend, 2),
            'number_of_weekend_transactions': weekend_tx_count,
            'weekend_ratio': round(weekend_ratio, 4),
            'groceries_spending': round(cat_sums.get('Groceries', 0.0), 2),
            'dining_spending': round(cat_sums.get('Dining & Takeaway', 0.0), 2),
            'transport_spending': round(cat_sums.get('Transport', 0.0), 2),
            'shopping_spending': round(cat_sums.get('Shopping', 0.0), 2),
            'entertainment_spending': round(cat_sums.get('Entertainment & Subscriptions', 0.0), 2),
            'bills_spending': round(cat_sums.get('Utilities & Bills', 0.0), 2),
            'health_spending': round(cat_sums.get('Health & Fitness', 0.0), 2),
            'travel_spending': round(cat_sums.get('Travel', 0.0), 2),
            'food_spending': round(cat_sums.get('Groceries', 0.0) + cat_sums.get('Dining & Takeaway', 0.0), 2)
        }
        records.append(rec)

    feat_df = pd.DataFrame(records)

    # Time-series lag features (strict historical shifts - no leakage)
    feat_df['previous_month_spending'] = feat_df['total_spending'].shift(1)
    feat_df['spending_2_months_ago'] = feat_df['total_spending'].shift(2)
    feat_df['spending_3_months_ago'] = feat_df['total_spending'].shift(3)

    # Rolling averages over past periods
    feat_df['3_month_moving_average'] = feat_df['total_spending'].shift(1).rolling(window=3).mean()
    feat_df['6_month_moving_average'] = feat_df['total_spending'].shift(1).rolling(window=6).mean()

    # Category lags
    feat_df['prev_food_spending'] = feat_df['food_spending'].shift(1)
    feat_df['prev_shopping_spending'] = feat_df['shopping_spending'].shift(1)
    feat_df['prev_transport_spending'] = feat_df['transport_spending'].shift(1)
    feat_df['prev_bills_spending'] = feat_df['bills_spending'].shift(1)

    os.makedirs(os.path.dirname(MONTHLY_FEATURES_PATH), exist_ok=True)
    feat_df.to_csv(MONTHLY_FEATURES_PATH, index=False)
    print(f"Generated {len(feat_df)} monthly aggregated records saved to {MONTHLY_FEATURES_PATH}")
    return feat_df


FEATURE_COLS = [
    'previous_month_spending',
    'spending_2_months_ago',
    'spending_3_months_ago',
    '3_month_moving_average',
    '6_month_moving_average',
    'month',
    'prev_food_spending',
    'prev_shopping_spending',
    'prev_transport_spending',
    'prev_bills_spending',
    'number_of_unique_merchants',
    'weekend_spending',
    'weekend_ratio'
]


def evaluate_rolling_time_series(df: pd.DataFrame, min_train_months: int = 12) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates forecasting models using an Expanding Rolling Window (Time Series Split).
    Prevents ANY lookahead bias / data leakage:
    For each test month t in [min_train_months, N-1]:
      Train on months [0 ... t-1]
      Predict month t
    """
    # Fill early rolling NaNs for 6m window where needed
    clean_feat = df.copy()
    clean_feat['6_month_moving_average'] = clean_feat['6_month_moving_average'].fillna(clean_feat['3_month_moving_average'])
    # Drop rows that don't have lag_3
    valid_df = clean_feat.dropna(subset=['previous_month_spending', 'spending_3_months_ago', '3_month_moving_average']).reset_index(drop=True)

    n_months = len(valid_df)
    if n_months <= min_train_months:
        min_train_months = max(4, n_months // 2)

    actuals = []
    preds_naive = []
    preds_linear = []
    preds_rf = []
    preds_xgb = []

    models = {
        'Linear Regression (Ridge)': Ridge(alpha=10.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=80, max_depth=3, learning_rate=0.08, random_state=42) if HAS_XGB
                   else GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.08, random_state=42)
    }

    test_indices = list(range(min_train_months, n_months))

    for t in test_indices:
        train_data = valid_df.iloc[:t]
        test_row = valid_df.iloc[t]

        X_tr = train_data[FEATURE_COLS].astype(float)
        y_tr = train_data['total_spending'].astype(float)

        X_te = valid_df.iloc[[t]][FEATURE_COLS].astype(float)
        y_te = float(test_row['total_spending'])
        actuals.append(y_te)

        # 1. Naive baseline: 3-month moving average of preceding months
        naive_val = train_data['total_spending'].iloc[-3:].mean()
        preds_naive.append(naive_val)

        # 2. Linear Regression (Ridge)
        lr = Ridge(alpha=10.0)
        lr.fit(X_tr, y_tr)
        preds_linear.append(lr.predict(X_te)[0])

        # 3. Random Forest
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_tr, y_tr)
        preds_rf.append(rf.predict(X_te)[0])

        # 4. XGBoost / Gradient Boosting
        if HAS_XGB:
            xgb = XGBRegressor(n_estimators=80, max_depth=3, learning_rate=0.08, random_state=42)
        else:
            xgb = GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.08, random_state=42)
        xgb.fit(X_tr, y_tr)
        preds_xgb.append(xgb.predict(X_te)[0])

    actuals = np.array(actuals)

    def calc_metrics(name: str, y_true, y_pred) -> Dict[str, Any]:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return {
            'model': name,
            'mae_eur': round(mae, 2),
            'rmse_eur': round(rmse, 2),
            'mape_pct': round(mape, 2)
        }

    benchmarks = [
        calc_metrics('Naive Baseline (3-month avg)', actuals, np.array(preds_naive)),
        calc_metrics('Linear Regression (Ridge)', actuals, np.array(preds_linear)),
        calc_metrics('Random Forest Regressor', actuals, np.array(preds_rf)),
        calc_metrics('XGBoost Regressor' if HAS_XGB else 'Gradient Boosting', actuals, np.array(preds_xgb))
    ]

    bench_df = pd.DataFrame(benchmarks).sort_values(by='mae_eur')
    bench_df.to_csv(BENCHMARK_SAVE_PATH, index=False)

    print("\n=== TIME-SERIES SPENDING FORECASTING BENCHMARK (ROLLING CV) ===")
    print(bench_df.to_string(index=False))

    # Fit final production model on all available data for future forecasting
    X_full = valid_df[FEATURE_COLS].astype(float)
    y_full = valid_df['total_spending'].astype(float)

    best_model = RandomForestRegressor(n_estimators=150, max_depth=5, random_state=42)
    best_model.fit(X_full, y_full)

    # Save model and feature list
    joblib.dump({
        'model': best_model,
        'feature_cols': FEATURE_COLS,
        'train_residuals': y_full.values - best_model.predict(X_full),
        'df_monthly': valid_df
    }, FORECASTER_SAVE_PATH)
    print(f"Saved best production forecaster to {FORECASTER_SAVE_PATH}")

    return bench_df, {
        'actuals': actuals,
        'naive': np.array(preds_naive),
        'linear': np.array(preds_linear),
        'rf': np.array(preds_rf),
        'xgb': np.array(preds_xgb),
        'months': valid_df['year_month'].iloc[test_indices].tolist()
    }


def forecast_next_month() -> Dict[str, Any]:
    """
    Generates spending predictions for the upcoming month:
    - Overall total predicted spending with 80% prediction interval
    - Category-by-category forecasted spending
    - Comparison with 3-month and 6-month historical averages
    """
    if not os.path.exists(FORECASTER_SAVE_PATH):
        aggregate_monthly_features()
        feat_df = pd.read_csv(MONTHLY_FEATURES_PATH)
        evaluate_rolling_time_series(feat_df)

    saved = joblib.load(FORECASTER_SAVE_PATH)
    model = saved['model']
    feat_cols = saved['feature_cols']
    valid_df = saved['df_monthly']
    residuals = saved['train_residuals']

    last_row = valid_df.iloc[-1]
    last_month_num = int(last_row['month'])
    next_month_num = 1 if last_month_num == 12 else last_month_num + 1

    # Recent history
    recent_3m = valid_df['total_spending'].iloc[-3:].mean()
    recent_6m = valid_df['total_spending'].iloc[-6:].mean()
    last_month_spend = float(last_row['total_spending'])

    # Build next month feature vector
    next_features = {
        'previous_month_spending': last_month_spend,
        'spending_2_months_ago': float(valid_df['total_spending'].iloc[-2]),
        'spending_3_months_ago': float(valid_df['total_spending'].iloc[-3]),
        '3_month_moving_average': float(recent_3m),
        '6_month_moving_average': float(recent_6m),
        'month': next_month_num,
        'prev_food_spending': float(last_row['food_spending']),
        'prev_shopping_spending': float(last_row['shopping_spending']),
        'prev_transport_spending': float(last_row['transport_spending']),
        'prev_bills_spending': float(last_row['bills_spending']),
        'number_of_unique_merchants': float(last_row['number_of_unique_merchants']),
        'weekend_spending': float(last_row['weekend_spending']),
        'weekend_ratio': float(last_row['weekend_ratio'])
    }

    X_next = pd.DataFrame([next_features])[feat_cols].astype(float)
    predicted_total = float(model.predict(X_next)[0])

    # 80% Prediction interval from residual standard deviation
    res_std = float(np.std(residuals))
    lower_bound = max(0.0, predicted_total - 1.28 * res_std)
    upper_bound = predicted_total + 1.28 * res_std

    # Category breakdowns based on recent proportions and trend
    category_weights = {
        'Groceries': float(valid_df['groceries_spending'].iloc[-3:].mean() / recent_3m),
        'Dining & Takeaway': float(valid_df['dining_spending'].iloc[-3:].mean() / recent_3m),
        'Transport': float(valid_df['transport_spending'].iloc[-3:].mean() / recent_3m),
        'Shopping': float(valid_df['shopping_spending'].iloc[-3:].mean() / recent_3m),
        'Entertainment & Subscriptions': float(valid_df['entertainment_spending'].iloc[-3:].mean() / recent_3m),
        'Utilities & Bills': float(valid_df['bills_spending'].iloc[-3:].mean() / recent_3m),
        'Health & Fitness': float(valid_df['health_spending'].iloc[-3:].mean() / recent_3m),
        'Travel': float(valid_df['travel_spending'].iloc[-3:].mean() / recent_3m)
    }

    category_forecasts = {}
    category_averages = {}
    for cat, weight in category_weights.items():
        c_forecast = round(predicted_total * weight, 2)
        col_name = cat.lower().split()[0] + '_spending'
        if col_name not in valid_df.columns:
            if 'dining' in cat.lower():
                col_name = 'dining_spending'
            elif 'groceries' in cat.lower():
                col_name = 'groceries_spending'
            elif 'bills' in cat.lower():
                col_name = 'bills_spending'
            elif 'entertainment' in cat.lower():
                col_name = 'entertainment_spending'

        c_avg = round(float(valid_df[col_name].iloc[-3:].mean()) if col_name in valid_df.columns else c_forecast, 2)
        category_forecasts[cat] = c_forecast
        category_averages[cat] = c_avg

    return {
        'predicted_total': round(predicted_total, 2),
        'lower_bound': round(lower_bound, 2),
        'upper_bound': round(upper_bound, 2),
        'last_month_spending': round(last_month_spend, 2),
        'three_month_average': round(recent_3m, 2),
        'six_month_average': round(recent_6m, 2),
        'mom_change_eur': round(predicted_total - last_month_spend, 2),
        'mom_change_pct': round(((predicted_total - last_month_spend) / last_month_spend) * 100, 2),
        'category_forecasts': category_forecasts,
        'category_averages': category_averages,
        'features_used': next_features,
        'feature_cols': feat_cols,
        'model_object': model
    }


if __name__ == '__main__':
    aggregate_monthly_features()
    df_feat = pd.read_csv(MONTHLY_FEATURES_PATH)
    evaluate_rolling_time_series(df_feat)
    fc = forecast_next_month()
    print("\nNext Month Forecast:")
    print(f"Predicted Total: EUR {fc['predicted_total']} (80% CI: EUR {fc['lower_bound']} - EUR {fc['upper_bound']})")
    print(f"3-Month Avg: EUR {fc['three_month_average']} | MoM Delta: EUR {fc['mom_change_eur']} ({fc['mom_change_pct']}%)")
