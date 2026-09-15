import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Ensure local imports work cleanly
sys.path.insert(0, os.path.abspath('.'))

from src.preprocessing import clean_transactions, parse_inconsistent_amount
from src.categorisation import predict_category
from src.forecasting import forecast_next_month
from src.anomaly_detection import detect_anomalies
from src.explainability import explain_next_month_forecast
from src.llm import FinancialAnalyst

st.set_page_config(
    page_title="AI Personal Finance Forecasting & Insights",
    page_icon="€",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, polished UI
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 4px;
    }
    .metric-delta {
        font-size: 0.88rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .delta-pos { color: #dc2626; } /* Spending increase is red */
    .delta-neg { color: #16a34a; } /* Spending decrease is green */
    .badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-med {
        background-color: #fef3c7;
        color: #92400e;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    clean_df = pd.read_csv('data/processed/cleaned_transactions.csv')
    clean_df['date'] = pd.to_datetime(clean_df['date'])
    monthly_df = pd.read_csv('data/processed/monthly_aggregated_features.csv')
    anom_df = pd.read_csv('evaluation/detected_anomalies.csv')
    model_bench_df = pd.read_csv('evaluation/model_results.csv')
    return clean_df, monthly_df, anom_df, model_bench_df


@st.cache_data
def load_ml_insights():
    fc = forecast_next_month()
    exp = explain_next_month_forecast()
    return fc, exp


# Load cached data
try:
    clean_df, monthly_df, anom_df, model_bench_df = load_data()
    fc, exp = load_ml_insights()
except Exception as e:
    st.error(f"Please generate datasets first by running the training pipeline: {e}")
    st.stop()


# Sidebar Navigation & Settings
with st.sidebar:
    st.title("AI Finance Studio")
    st.markdown("*ML-driven analysis & Grounded GenAI insights*")
    st.markdown("---")

    nav_choice = st.radio(
        "Navigation",
        [
            "Executive Overview",
            "Category Deep-Dive",
            "Anomaly Radar",
            "Model Arena & Explainability",
            "AI Financial Analyst (Chat)",
            "Data Management"
        ]
    )

    st.markdown("---")
    st.markdown("### GenAI Engine Settings")
    api_key_input = st.text_input(
        "OpenAI API Key (Optional)",
        type="password",
        value=os.environ.get('OPENAI_API_KEY', ''),
        help="Leave blank to use the deterministic local financial analyst engine."
    )
    if api_key_input:
        st.success("API key registered. Live LLM reasoning enabled.")
    else:
        st.info("Operating in Offline Mode (Grounded Local Analyst).")

    st.markdown("---")
    st.caption("AI Personal Finance Studio v1.0 - Built with Streamlit, Scikit-Learn, XGBoost & SHAP")


# Initialize Analyst
analyst = FinancialAnalyst(api_key=api_key_input if api_key_input else None)


# ----------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW
# ----------------------------------------------------
if nav_choice == "Executive Overview":
    st.header("Executive Financial Overview & Forecast")
    st.markdown("Real-time spending intelligence, machine learning forecasting, and AI synthesis.")

    # Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    last_spend = fc['last_month_spending']
    pred_spend = fc['predicted_total']
    mom_delta = fc['mom_change_eur']
    mom_pct = fc['mom_change_pct']
    avg_3m = fc['three_month_average']

    delta_class = "delta-pos" if mom_delta > 0 else "delta-neg"
    sign = "+" if mom_delta > 0 else ""

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Previous Month Spending</div>
            <div class="metric-value">€{last_spend:,.2f}</div>
            <div class="metric-delta">Actual recorded total</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Predicted Next Month</div>
            <div class="metric-value">€{pred_spend:,.2f}</div>
            <div class="metric-delta {delta_class}">{sign}€{mom_delta:,.2f} ({sign}{mom_pct:.1f}% MoM)</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">80% Confidence Interval</div>
            <div class="metric-value" style="font-size: 1.35rem;">€{fc['lower_bound']:,.0f} - €{fc['upper_bound']:,.0f}</div>
            <div class="metric-delta">Residual error band</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">3-Month Moving Average</div>
            <div class="metric-value">€{avg_3m:,.2f}</div>
            <div class="metric-delta">Baseline trend</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Chart & Forecast Row
    chart_col, cat_col = st.columns([2, 1])

    with chart_col:
        st.subheader("Spending Trajectory & Upcoming Forecast")

        hist_months = monthly_df['year_month'].tolist()
        hist_totals = monthly_df['total_spending'].tolist()

        # Build trajectory
        next_month_label = "2025-07 (Forecast)"
        plot_months = hist_months + [next_month_label]
        plot_actuals = hist_totals + [None]
        plot_forecast = [None] * (len(hist_months) - 1) + [hist_totals[-1], pred_spend]

        fig = go.Figure()

        # Historical line
        fig.add_trace(go.Scatter(
            x=hist_months,
            y=hist_totals,
            mode='lines+markers',
            name='Historical Actual Spending',
            line=dict(color='#2563eb', width=2.5),
            marker=dict(size=6)
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=[hist_months[-1], next_month_label],
            y=[hist_totals[-1], pred_spend],
            mode='lines+markers',
            name='ML Forecast',
            line=dict(color='#ef4444', width=3, dash='dash'),
            marker=dict(size=8, color='#ef4444')
        ))

        # Confidence Interval error bar on forecast
        fig.add_trace(go.Scatter(
            x=[next_month_label, next_month_label],
            y=[fc['lower_bound'], fc['upper_bound']],
            mode='lines',
            line=dict(color='#f87171', width=4),
            name='80% Prediction Interval'
        ))

        fig.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode='x unified',
            xaxis_title="Month",
            yaxis_title="Monthly Spending (€)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with cat_col:
        st.subheader("Forecasted Category Split")
        cat_fc_df = pd.DataFrame([
            {'Category': cat, 'Amount': val} for cat, val in fc['category_forecasts'].items()
        ]).sort_values(by='Amount', ascending=False)

        donut_fig = px.pie(
            cat_fc_df,
            values='Amount',
            names='Category',
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        donut_fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=20, b=10),
            showlegend=False
        )
        donut_fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(donut_fig, use_container_width=True)

    # Grounded AI Briefing Card
    st.subheader("Grounded AI Financial Briefing")
    with st.container():
        st.info(analyst.generate_executive_insights())


# ----------------------------------------------------
# TAB 2: CATEGORY DEEP DIVE
# ----------------------------------------------------
elif nav_choice == "Category Deep-Dive":
    st.header("Category Dynamics & Spending Trends")

    cat_totals = clean_df.groupby('category')['amount'].agg(['sum', 'count', 'mean']).reset_index()
    cat_totals.columns = ['Category', 'Total Spent (€)', 'Transaction Count', 'Average Transaction (€)']
    cat_totals = cat_totals.sort_values(by='Total Spent (€)', ascending=False)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Monthly Category Spending Trends")
        cat_month_pivot = clean_df.pivot_table(index='year_month', columns='category', values='amount', aggfunc='sum', fill_value=0)

        fig = px.bar(
            cat_month_pivot.reset_index(),
            x='year_month',
            y=cat_month_pivot.columns.tolist(),
            labels={'value': 'Spending (€)', 'year_month': 'Month'},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig.update_layout(height=420, barmode='stack', margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Weekend vs. Weekday Split")
        weekend_comp = clean_df.groupby(['category', 'is_weekend'])['amount'].sum().unstack(fill_value=0)
        weekend_comp.columns = ['Weekday', 'Weekend']
        weekend_comp['Weekend Share %'] = (weekend_comp['Weekend'] / (weekend_comp['Weekday'] + weekend_comp['Weekend']) * 100).round(1)

        we_fig = px.bar(
            weekend_comp.reset_index(),
            x='Category',
            y=['Weekday', 'Weekend'],
            barmode='group',
            color_discrete_map={'Weekday': '#3b82f6', 'Weekend': '#f59e0b'}
        )
        we_fig.update_layout(height=420, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(we_fig, use_container_width=True)

    st.subheader("Category Summary Metrics")
    st.dataframe(
        cat_totals.style.format({
            'Total Spent (€)': '€{:,.2f}',
            'Transaction Count': '{:,}',
            'Average Transaction (€)': '€{:,.2f}'
        }),
        use_container_width=True
    )


# ----------------------------------------------------
# TAB 3: ANOMALY RADAR
# ----------------------------------------------------
elif nav_choice == "Anomaly Radar":
    st.header("Spending Anomaly Detection Radar")
    st.markdown("Identifies unusual spikes and transactions using **Isolation Forest**, **Local Outlier Factor (LOF)**, and dynamic statistical IQR bounds.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Anomalies Detected", f"{len(anom_df):,}", f"{(len(anom_df)/len(clean_df)*100):.1f}% of transactions")
    with col2:
        high_sev = len(anom_df[anom_df['severity'] == 'High'])
        st.metric("High Severity Outliers", f"{high_sev:,}", "Score > 0.80")
    with col3:
        max_outlier = anom_df['amount'].max()
        st.metric("Highest Outlier Amount", f"€{max_outlier:,.2f}")

    st.markdown("---")

    # Filters
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        score_thresh = st.slider("Minimum Anomaly Score", 0.0, 1.0, 0.60, 0.05)
    with f_col2:
        cat_filter = st.multiselect("Filter by Category", options=clean_df['category'].unique(), default=clean_df['category'].unique()[:4])
    with f_col3:
        sev_filter = st.multiselect("Severity", options=['High', 'Medium'], default=['High', 'Medium'])

    filtered_anom = anom_df[
        (anom_df['anomaly_score'] >= score_thresh) &
        (anom_df['category'].isin(cat_filter)) &
        (anom_df['severity'].isin(sev_filter))
    ]

    st.subheader(f"Flagged Atypical Transactions ({len(filtered_anom)} matching)")

    # Scatter plot highlighting anomalies against normal distribution
    clean_sample = clean_df[clean_df['category'].isin(cat_filter)].copy()
    clean_sample['is_anomaly_point'] = clean_sample.index.isin(filtered_anom.index)

    scatter_fig = px.scatter(
        clean_sample,
        x='date',
        y='amount',
        color='category',
        size='amount',
        hover_data=['clean_merchant', 'category'],
        title="Transaction Scatter Highlighting Outlier Spikes"
    )
    scatter_fig.update_layout(height=400, margin=dict(l=20, r=20, t=35, b=20))
    st.plotly_chart(scatter_fig, use_container_width=True)

    # Anomaly table
    st.dataframe(
        filtered_anom[['date', 'clean_merchant', 'category', 'amount', 'expected_range', 'anomaly_score', 'severity', 'explanation']]
        .rename(columns={
            'clean_merchant': 'Merchant',
            'category': 'Category',
            'amount': 'Amount (€)',
            'expected_range': 'Typical Range',
            'anomaly_score': 'Score (0-1)',
            'severity': 'Severity',
            'explanation': 'AI Diagnosis'
        }),
        use_container_width=True
    )


# ----------------------------------------------------
# TAB 4: MODEL ARENA & EXPLAINABILITY
# ----------------------------------------------------
elif nav_choice == "Model Arena & Explainability":
    st.header("Machine Learning Benchmarks & Explainable AI (SHAP)")
    st.markdown("Inspect performance across the three core ML problems and explore exact Shapley feature attributions.")

    st.subheader("Model Leaderboard")
    st.dataframe(model_bench_df, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Real-Time Transaction Categoriser")
        st.markdown("Test the trained NLP classification pipeline on raw text:")
        user_tx = st.text_input("Enter a raw transaction string:", value="TESCO EXTRA BRAY -€48.72")
        if user_tx:
            pred_cat, conf = predict_category(user_tx)
            st.success(f"**Predicted Category:** {pred_cat} (Confidence: **{conf:.1%}**)")
            st.caption("Cleaned tokens parsed by pipeline vectorizer and mapped to category probabilities.")

    with col2:
        st.subheader("Time-Series Leakage Prevention")
        st.info("""
        **Why Traditional K-Fold Fails on Financial Time-Series:**
        Randomly shuffling transactions causes future information to leak into past predictions (lookahead bias).

        **Our Solution:**
        We enforce an **Expanding Rolling Window (TimeSeriesSplit)**:
        - Train strictly on months $t \\in [1 \\dots k-1]$
        - Evaluate strictly on unseen month $k$
        - Increment $k$ forward chronologically.
        """)

    st.markdown("---")
    st.subheader("Explainable AI: SHAP Waterfall Attribution")
    st.markdown(f"**Baseline Average Spending:** €{exp['base_value']:,.2f} - **Model Forecast:** €{exp['predicted_value']:,.2f} *(Net: €{exp['total_shap_adjustment']:+,.2f})*")

    shap_df = pd.DataFrame(exp['feature_impacts']).head(8)
    shap_df['color'] = np.where(shap_df['shap_impact_eur'] > 0, '#ef4444', '#10b981')

    shap_fig = go.Figure(go.Bar(
        x=shap_df['shap_impact_eur'],
        y=shap_df['label'],
        orientation='h',
        marker=dict(color=shap_df['color']),
        text=[f"€{val:+,.2f}" for val in shap_df['shap_impact_eur']],
        textposition='auto'
    ))
    shap_fig.update_layout(
        title="SHAP Feature Attributions on Next Month's Spending",
        xaxis_title="Contribution to Predicted Spending (€)",
        yaxis=dict(autorange="reversed"),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(shap_fig, use_container_width=True)
    st.caption(exp['summary_text'])


# ----------------------------------------------------
# TAB 5: AI FINANCIAL ANALYST (CHAT)
# ----------------------------------------------------
elif nav_choice == "AI Financial Analyst (Chat)":
    st.header("Conversational Financial Analytics")
    st.markdown("""
    Ask questions in plain English. The system calculates exact numerical evidence from your data,
    and the AI presents clear, grounded answers without hallucinations.
    """)

    # Quick prompt buttons
    st.markdown("**Quick Inquiry Prompts:**")
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

    prompt_to_run = None
    with btn_col1:
        if st.button("Why is spending higher next month?"):
            prompt_to_run = "Why is my predicted spending higher next month?"
    with btn_col2:
        if st.button("What were my 3 biggest purchases?"):
            prompt_to_run = "What were my 3 biggest purchases?"
    with btn_col3:
        if st.button("Weekend vs weekday spending?"):
            prompt_to_run = "How much did I spend on weekends compared with weekdays?"
    with btn_col4:
        if st.button("Am I on track for €6,000 budget?"):
            prompt_to_run = "Am I on track to stay under 6000?"

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your AI Financial Analyst. Ask me about your spending forecasts, biggest purchases, weekend trends, or budget status."}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_query = st.chat_input("Ask a financial question...")

    if prompt_to_run:
        user_query = prompt_to_run

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing financial data & generating grounded response..."):
                response = analyst.answer_query(user_query)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


# ----------------------------------------------------
# TAB 6: DATA MANAGEMENT
# ----------------------------------------------------
elif nav_choice == "Data Management":
    st.header("Data Engineering & Cleaning Inspector")
    st.markdown("Compare the raw, dirty transactional log with the cleaned and standardized data.")

    raw_df = pd.read_csv('data/raw/messy_transactions.csv')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Messy Export (Input)")
        st.dataframe(raw_df.head(15), use_container_width=True)
        st.caption(f"Total Raw Records: {len(raw_df):,}")

    with col2:
        st.subheader("Cleaned & Standardized Data (Output)")
        st.dataframe(clean_df.head(15), use_container_width=True)
        st.caption(f"Total Clean Records: {len(clean_df):,}")

    st.markdown("---")
    st.subheader("Export Processed Dataset")
    csv_bytes = clean_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Cleaned Transactions CSV",
        data=csv_bytes,
        file_name="cleaned_transactions.csv",
        mime="text/csv"
    )
