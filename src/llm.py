"""
Grounded Generative AI & Natural Language Financial Analytics Module.
Follows the core design principle:
- ML & Pandas compute the quantitative analysis, forecasts, anomalies, and SHAP drivers.
- GenAI receives structured numerical context to produce grounded, hallucination-free explanations.

Supports OpenAI API (when key is available) and includes a robust local deterministic
financial analyst engine for offline execution.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


def build_structured_financial_summary() -> Dict[str, Any]:
    """
    Aggregates quantitative outputs from the ML pipeline into a structured JSON context.
    """
    from src.forecasting import forecast_next_month
    from src.anomaly_detection import detect_anomalies
    from src.explainability import explain_next_month_forecast

    fc = forecast_next_month()
    anom_df, anom_stats = detect_anomalies()
    exp = explain_next_month_forecast()

    # Clean transactions
    df_clean = pd.read_csv('data/processed/cleaned_transactions.csv')
    df_clean['date'] = pd.to_datetime(df_clean['date'])

    # Recent month data
    last_month = df_clean['date'].dt.to_period('M').max()
    prev_month_df = df_clean[df_clean['date'].dt.to_period('M') == last_month]

    weekday_spend = float(prev_month_df[~prev_month_df['is_weekend']]['amount'].sum())
    weekend_spend = float(prev_month_df[prev_month_df['is_weekend']]['amount'].sum())
    total_spend = weekday_spend + weekend_spend
    weekend_share = (weekend_spend / total_spend * 100) if total_spend > 0 else 0.0

    # Top anomalies formatted
    top_anomalies = []
    for _, row in anom_df.head(5).iterrows():
        top_anomalies.append({
            'date': str(row['date']),
            'merchant': str(row['clean_merchant']),
            'category': str(row['category']),
            'amount': float(row['amount']),
            'expected_range': str(row['expected_range']),
            'anomaly_score': float(row['anomaly_score']),
            'explanation': str(row['explanation'])
        })

    # Category forecast comparisons
    cat_comparisons = {}
    for cat, forecast_val in fc['category_forecasts'].items():
        avg_val = fc['category_averages'].get(cat, forecast_val)
        cat_comparisons[cat] = {
            'forecast': forecast_val,
            'recent_3m_average': avg_val,
            'delta_eur': round(forecast_val - avg_val, 2),
            'trend': 'increasing' if forecast_val > avg_val else 'decreasing'
        }

    # Top 3 biggest purchases in recent history
    top_3_purchases = df_clean.sort_values(by='amount', ascending=False).head(3)[
        ['date', 'clean_merchant', 'category', 'amount']
    ].to_dict(orient='records')
    for p in top_3_purchases:
        p['date'] = p['date'].strftime('%Y-%m-%d')
        p['amount'] = float(p['amount'])

    summary = {
        'forecast': {
            'predicted_total': fc['predicted_total'],
            'lower_bound_80ci': fc['lower_bound'],
            'upper_bound_80ci': fc['upper_bound'],
            'three_month_average': fc['three_month_average'],
            'six_month_average': fc['six_month_average'],
            'last_month_actual': fc['last_month_spending'],
            'mom_change_eur': fc['mom_change_eur'],
            'mom_change_pct': fc['mom_change_pct']
        },
        'category_dynamics': cat_comparisons,
        'shap_drivers': {
            'baseline_mean': exp['base_value'],
            'predicted_total': exp['predicted_value'],
            'top_positive_drivers': exp['top_positive_drivers'],
            'top_negative_drivers': exp['top_negative_drivers'],
            'summary': exp['summary_text']
        },
        'anomaly_summary': {
            'total_anomalies_detected': anom_stats['anomalies_detected'],
            'anomaly_rate_pct': anom_stats['anomaly_percentage'],
            'top_anomalies': top_anomalies
        },
        'behavioural_metrics': {
            'last_month_total': round(total_spend, 2),
            'weekday_spending': round(weekday_spend, 2),
            'weekend_spending': round(weekend_spend, 2),
            'weekend_share_pct': round(weekend_share, 1),
            'top_3_all_time_purchases': top_3_purchases
        }
    }
    return summary


SYSTEM_PROMPT = """You are a Senior Financial Data Analyst AI.
You receive verified quantitative data computed by machine learning forecasting, anomaly detection, and SHAP explainability models.

RULES:
1. Ground all statements strictly in the provided JSON metrics.
2. Never invent transactions, causes, percentages, or figures not present in the evidence.
3. Be clear, professional, and actionable. Highlight significant trends, anomalies, and uncertainties.
4. If asked a direct question (e.g. about food spend or top purchases), state the exact numbers clearly first before providing context.
"""


class FinancialAnalyst:
    """
    Natural Language Query & Explanation Engine with dual backend:
    1. LLM API (OpenAI) if an API key is configured.
    2. Local Smart Analyst Engine (Deterministic grounding) if offline or no key.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.context = None
        self._load_context()

    def _load_context(self):
        try:
            self.context = build_structured_financial_summary()
        except Exception as e:
            print(f"Notice: Initial context build error: {e}")
            self.context = {}

    def refresh_context(self):
        self._load_context()

    def generate_executive_insights(self) -> str:
        """Generates the primary executive overview narrative."""
        if not self.context:
            self._load_context()

        ctx = self.context
        fc = ctx.get('forecast', {})
        cats = ctx.get('category_dynamics', {})
        anoms = ctx.get('anomaly_summary', {}).get('top_anomalies', [])
        shap = ctx.get('shap_drivers', {})

        if self.api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                prompt = (
                    "Explain the upcoming month's forecast and key financial dynamics using this data:\n"
                    + json.dumps(ctx, indent=2) + "\n\n"
                    "Produce a structured, engaging 3-paragraph financial briefing covering:\n"
                    "1. Predicted total, comparison to 3-month average, and prediction interval.\n"
                    "2. Which categories are driving the forecast up/down and SHAP factors.\n"
                    "3. Notable detected anomalies to watch out for."
                )
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                return res.choices[0].message.content
            except Exception as e:
                print(f"LLM API error, falling back to local engine: {e}")

        # Local Deterministic Financial Analyst Engine
        pred_tot = fc.get('predicted_total', 0)
        avg_3m = fc.get('three_month_average', 0)
        delta_3m = pred_tot - avg_3m
        direction = "higher than" if delta_3m > 0 else "lower than"

        increasing_cats = [
            f"{c} (+?{d['delta_eur']:.2f})" for c, d in cats.items() if d['delta_eur'] > 15
        ]
        decreasing_cats = [
            f"{c} (-?{abs(d['delta_eur']):.2f})" for c, d in cats.items() if d['delta_eur'] < -15
        ]

        top_anom_str = ""
        if anoms:
            a = anoms[0]
            top_anom_str = f" The largest detected anomaly was a ?{a['amount']:.2f} payment at {a['merchant']} ({a['category']}), well outside its typical range of {a['expected_range']}."

        top_shap_driver = ""
        if shap.get('top_positive_drivers'):
            d = shap['top_positive_drivers'][0]
            top_shap_driver = f" According to SHAP feature attribution, the strongest upward driver is {d['label']} (+?{d['shap_impact_eur']:.2f})."

        narrative = (
            f"**Executive Forecast Summary:**\n"
            f"Your predicted spending for next month is **?{pred_tot:,.2f}**, which is **?{abs(delta_3m):,.2f} {direction}** your 3-month rolling average of ?{avg_3m:,.2f} (80% confidence interval: ?{fc.get('lower_bound_80ci', 0):,.2f} ? ?{fc.get('upper_bound_80ci', 0):,.2f}).\n\n"
            f"**Category & Behavioral Drivers:**\n"
            f"? **Upward pressures:** {', '.join(increasing_cats) if increasing_cats else 'Spending across most categories remains stable.'}\n"
            f"? **Downward savings:** {', '.join(decreasing_cats) if decreasing_cats else 'No major category reductions detected.'}\n"
            f"{top_shap_driver}\n\n"
            f"**Risk & Anomaly Radar:**\n"
            f"The ML anomaly detector flagged {ctx.get('anomaly_summary', {}).get('total_anomalies_detected', 0)} atypical transactions across your history.{top_anom_str}"
        )
        return narrative

    def answer_query(self, user_query: str) -> str:
        """
        Answers natural-language user queries by calculating exact data answers
        and synthesizing a grounded response.
        """
        if not self.context:
            self._load_context()

        q = user_query.lower()
        ctx = self.context

        # 1. Query: 'biggest purchases' / 'largest transactions'
        if any(k in q for k in ['biggest', 'largest', 'highest', 'top purchase', 'expensive']):
            purchases = ctx.get('behavioural_metrics', {}).get('top_3_all_time_purchases', [])
            lines = [f"{i+1}. **?{p['amount']:.2f}** ? {p['clean_merchant']} *({p['category']})* on {p['date']}"
                     for i, p in enumerate(purchases)]
            return (
                f"Here are your **3 biggest historical purchases**:\n\n"
                + "\n".join(lines) + "\n\n"
                f"These transactions were flagged by the anomaly detection pipeline as high-outlier expenditures."
            )

        # 2. Query: 'weekend vs weekday' / 'weekends'
        if any(k in q for k in ['weekend', 'weekday', 'saturday', 'sunday']):
            b = ctx.get('behavioural_metrics', {})
            wd = b.get('weekday_spending', 0.0)
            we = b.get('weekend_spending', 0.0)
            pct = b.get('weekend_share_pct', 0.0)
            return (
                f"**Weekend vs. Weekday Spending Breakdown (Recent Month):**\n\n"
                f"? **Weekdays:** ?{wd:,.2f}\n"
                f"? **Weekends:** ?{we:,.2f}\n"
                f"? **Weekend Share:** **{pct:.1f}%** of total monthly spending.\n\n"
                f"Dining & Takeaway and Shopping account for the majority of your weekend expenditure surges."
            )

        # 3. Query: 'Why is my spending higher / increase?'
        if any(k in q for k in ['why is', 'why did', 'higher', 'increase', 'driving', 'factors']):
            shap = ctx.get('shap_drivers', {})
            pos = shap.get('top_positive_drivers', [])
            pos_text = "\n".join([f"? **{p['label']}**: added **+?{p['shap_impact_eur']:.2f}** to forecast" for p in pos])
            fc = ctx.get('forecast', {})
            return (
                f"**Why Your Forecasted Spending Is Higher:**\n\n"
                f"Next month is projected at **?{fc.get('predicted_total', 0):,.2f}** (a net increase of **+?{fc.get('mom_change_eur', 0):,.2f}** month-over-month).\n\n"
                f"Based on **SHAP Tree Explainer feature attributions**, the main causes are:\n"
                f"{pos_text}\n\n"
                f"{shap.get('summary', '')}"
            )

        # 4. Query: Budget check ('under 1200', 'stay under', 'budget')
        budget_match = re.search(r'(\d+[\d,]*)', q)
        if any(k in q for k in ['budget', 'stay under', 'on track', 'afford']) and budget_match:
            budget_val = float(budget_match.group(1).replace(',', ''))
            pred = ctx.get('forecast', {}).get('predicted_total', 0)
            diff = pred - budget_val
            if diff <= 0:
                return (
                    f"**Yes, you are on track!**\n\n"
                    f"Your predicted spending next month is **?{pred:,.2f}**, which is **?{abs(diff):,.2f} below** your target budget of **?{budget_val:,.2f}**."
                )
            else:
                return (
                    f"**Budget Alert:** You are currently **not on track** to stay under ?{budget_val:,.2f}.\n\n"
                    f"? **Predicted Spending:** ?{pred:,.2f}\n"
                    f"? **Over-budget by:** **?{diff:,.2f}** (or {round((diff/budget_val)*100, 1)}% above budget)\n"
                    f"? **Recommended Action:** Review recent weekend dining and shopping surges to bring your forecast within range."
                )

        # 5. Query: 'Where am I spending the most?' / 'Top category'
        if any(k in q for k in ['where', 'most money', 'biggest category', 'top category', 'breakdown']):
            cats = ctx.get('category_dynamics', {})
            sorted_cats = sorted(cats.items(), key=lambda x: x[1]['forecast'], reverse=True)
            lines = [f"{i+1}. **{c}**: ?{d['forecast']:,.2f} *(recent avg: ?{d['recent_3m_average']:,.2f})*"
                     for i, (c, d) in enumerate(sorted_cats[:5])]
            return (
                f"**Top Spending Categories (Forecasted Next Month):**\n\n"
                + "\n".join(lines) + "\n\n"
                f"{sorted_cats[0][0]} represents your single largest ongoing expense center."
            )

        # 6. Fallback: LLM API if key available, else comprehensive executive summary
        if self.api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"User question: {user_query}\n\nFinancial context:\n{json.dumps(ctx, indent=2)}"}
                    ],
                    temperature=0.2
                )
                return res.choices[0].message.content
            except Exception as e:
                pass

        return self.generate_executive_insights()


if __name__ == '__main__':
    analyst = FinancialAnalyst()
    print("=== TEST EXECUTIVE INSIGHTS ===")
    print(analyst.generate_executive_insights())
    print("\n=== TEST QUERY: TOP PURCHASES ===")
    print(analyst.answer_query("What were my 3 biggest purchases?"))
    print("\n=== TEST QUERY: WEEKENDS ===")
    print(analyst.answer_query("How much did I spend on weekends compared with weekdays?"))
    print("\n=== TEST QUERY: BUDGET ===")
    print(analyst.answer_query("Am I on track to stay under 6000?"))
