# ============================================================
# FinTwinAI DASHBOARD
# Part 1: Setup, Styling, Data Loading and Common Functions
# ============================================================

from pathlib import Path
import warnings

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FinTwinAI | Financial Digital Twin",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        font-size: 1.1rem;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        border-bottom: 2px solid #4b6cb7;
        padding-bottom: 5px;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(75, 108, 183, 0.10);
        border-left: 5px solid #4b6cb7;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "fintwinai_outputs"


# ============================================================
# DATA LOADING FUNCTION
# ============================================================

@st.cache_data
def load_csv(filename):

    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(file_path)

    except Exception as error:
        st.warning(f"Error reading {filename}: {error}")
        return pd.DataFrame()


# ============================================================
# COLUMN SEARCH FUNCTION
# ============================================================

def find_column(dataframe, possible_names):

    if dataframe.empty:
        return None

    column_dictionary = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    # Exact matching
    for name in possible_names:

        name_lower = name.strip().lower()

        if name_lower in column_dictionary:
            return column_dictionary[name_lower]

    # Partial matching
    for column in dataframe.columns:

        column_lower = str(column).strip().lower()

        for name in possible_names:

            if name.strip().lower() in column_lower:
                return column

    return None


# ============================================================
# NUMERICAL COLUMN FUNCTION
# ============================================================

def get_numeric_column(dataframe, possible_names):

    selected_column = find_column(
        dataframe,
        possible_names
    )

    if selected_column is None:

        return pd.Series(
            0.0,
            index=dataframe.index
        )

    return pd.to_numeric(
        dataframe[selected_column],
        errors="coerce"
    ).fillna(0)


# ============================================================
# FORMATTING FUNCTIONS
# ============================================================

def format_inr(value):

    try:
        return f"₹{float(value):,.2f}"

    except Exception:
        return "₹0.00"


def format_percentage(value):

    try:
        return f"{float(value):.2f}%"

    except Exception:
        return "0.00%"


# ============================================================
# RISK CATEGORY FUNCTION
# ============================================================

def get_risk_category(default_probability):

    probability = float(default_probability)

    if probability < 0.20:
        return "Low"

    elif probability < 0.40:
        return "Moderate"

    elif probability < 0.60:
        return "High"

    else:
        return "Very High"


# ============================================================
# HEALTH SCORE FUNCTION
# ============================================================

def calculate_health_score(default_probability):

    probability = np.clip(
        float(default_probability),
        0,
        1
    )

    score = (1 - probability) * 100

    return round(score, 1)


# ============================================================
# SCORE INTERPRETATION
# ============================================================

def interpret_health_score(score):

    if score >= 75:

        return (
            "Your financial profile appears relatively stable "
            "according to the entered information."
        )

    elif score >= 55:

        return (
            "Your profile shows some financial pressure points "
            "that should be monitored."
        )

    elif score >= 35:

        return (
            "Your profile contains several warning signals. "
            "Review your debt and repayment capacity."
        )

    else:

        return (
            "Your profile shows substantial warning signals. "
            "Consider professional financial guidance."
        )


# ============================================================
# EMI CALCULATION FUNCTION
# ============================================================

def calculate_emi(
    principal,
    annual_interest_rate,
    loan_term_months
):

    principal = max(float(principal), 0)

    annual_interest_rate = max(
        float(annual_interest_rate),
        0
    )

    loan_term_months = max(
        int(loan_term_months),
        1
    )

    monthly_rate = annual_interest_rate / 1200

    # Zero-interest loan
    if monthly_rate == 0:

        return principal / loan_term_months

    factor = (
        1 + monthly_rate
    ) ** loan_term_months

    monthly_emi = (
        principal
        * monthly_rate
        * factor
        / (factor - 1)
    )

    return monthly_emi


# ============================================================
# HEALTH SCORE GAUGE
# ============================================================

def create_health_gauge(score, title):

    score = float(
        np.clip(score, 0, 100)
    )

    if score >= 75:
        gauge_color = "#2ca02c"

    elif score >= 55:
        gauge_color = "#f1c40f"

    elif score >= 35:
        gauge_color = "#e67e22"

    else:
        gauge_color = "#d62728"

    figure = go.Figure(
        go.Indicator(

            mode="gauge+number",

            value=score,

            number={
                "suffix": "/100"
            },

            title={
                "text": title
            },

            gauge={

                "axis": {
                    "range": [0, 100]
                },

                "bar": {
                    "color": gauge_color
                },

                "steps": [

                    {
                        "range": [0, 35],
                        "color": "#f8d7da"
                    },

                    {
                        "range": [35, 55],
                        "color": "#ffe5cc"
                    },

                    {
                        "range": [55, 75],
                        "color": "#fff3cd"
                    },

                    {
                        "range": [75, 100],
                        "color": "#d1e7dd"
                    }

                ]

            }

        )
    )

    figure.update_layout(
        height=280,
        margin=dict(
            t=55,
            b=10,
            l=20,
            r=20
        )
    )

    return figure


# ============================================================
# LOAD PROJECT OUTPUT FILES
# ============================================================

risk_profile = load_csv(
    "dashboard_risk_profile.csv"
)

model_comparison = load_csv(
    "final_model_comparison.csv"
)

scenario_analysis = load_csv(
    "final_scenario_analysis.csv"
)

shap_importance = load_csv(
    "shap_feature_importance_final.csv"
)

fred_data = load_csv(
    "fred_macro_data.csv"
)

digital_twin_profiles = load_csv(
    "digital_twin_profiles_final.csv"
)

monthly_risk = load_csv(
    "monthly_portfolio_risk_summary.csv"
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown(
    """
    <div style="text-align:center">

    <h2>💳 FinTwinAI</h2>

    <p>Financial Digital Twin</p>

    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")


page = st.sidebar.radio(
    "📌 Navigate",

    [

        "🏠 Executive Dashboard",

        "👤 Personal Financial Health",

        "📊 Portfolio Analytics",

        "🌐 GraphSAGE Network Risk",

        "🔎 SHAP Explainability",

        "🤖 Model Performance",

        "📉 Macro Stress Testing",

        "⏳ Survival Analysis",

        "ℹ️ About Project"

    ]
)


# ============================================================
# SIDEBAR PROJECT INFORMATION
# ============================================================

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **Project:** FinTwinAI

    **Student:** Priti Prakash Mohanta

    **Supervisor:** Dr. Bhavna Saini

    **Department:** Data Science and Analytics

    **University:** Central University of Rajasthan

    **Programme:** M.Sc. CS – Big Data Analytics
    """
)

st.sidebar.info(
    "This is an academic prototype. "
    "It is not financial advice or a credit approval system."
)


# ============================================================
# PART 1 TEST MESSAGE
# ============================================================

if page == "🏠 Executive Dashboard":

    st.markdown(
        '<div class="main-title">💳 FinTwinAI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">'
        'Explainable AI-Driven Personal Financial Digital Twin'
        '</div>',
        unsafe_allow_html=True
    )

    
    st.write("Loaded files:")

    file_status = pd.DataFrame(
        {
            "File": [
                "dashboard_risk_profile.csv",
                "final_model_comparison.csv",
                "final_scenario_analysis.csv",
                "shap_feature_importance_final.csv",
                "fred_macro_data.csv",
                "digital_twin_profiles_final.csv",
                "monthly_portfolio_risk_summary.csv"
            ],

            "Status": [

                "Available"
                if not (OUTPUT_DIR / filename).exists()
                else "Available"

                for filename in [

                    "dashboard_risk_profile.csv",
                    "final_model_comparison.csv",
                    "final_scenario_analysis.csv",
                    "shap_feature_importance_final.csv",
                    "fred_macro_data.csv",
                    "digital_twin_profiles_final.csv",
                    "monthly_portfolio_risk_summary.csv"

                ]

            ]
        }
    )

    st.dataframe(
        file_status,
        use_container_width=True,
        hide_index=True
    )
# ============================================================
# PAGE 1 — EXECUTIVE DASHBOARD
# ============================================================

if page == "🏠 Executive Dashboard":

    st.markdown(
        '<p class="main-title">💳 FinTwinAI</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="sub-title">'
        'Explainable AI-Driven Personal Financial Digital Twin'
        '</p>',
        unsafe_allow_html=True
    )

    st.info(
        """
        **FinTwinAI** combines credit-risk analysis, Graph Neural Networks,
        explainable AI, macroeconomic stress testing and financial-health
        indicators in one dashboard.
        """
    )

    st.markdown("---")

    # --------------------------------------------------------
    # DATASET STATUS
    # --------------------------------------------------------

    st.subheader("📁 Project Data Status")

    file_list = {
        "Risk Profile": "risk_profile.csv",
        "Model Comparison": "model_comparison.csv",
        "Scenario Analysis": "scenario_analysis.csv",
        "SHAP Importance": "shap_importance.csv",
        "FRED Data": "fred_data.csv",
        "Digital Twin Profiles": "digital_twin_profiles.csv",
        "Monthly Risk": "monthly_risk.csv"
    }

    status_rows = []

    for label, filename in file_list.items():

        file_path = OUTPUT_DIR / filename
        exists = file_path.exists()

        status_rows.append({
            "Dataset": label,
            "File": filename,
            "Status": "✅ Available" if exists else "❌ Missing"
        })

    status_df = pd.DataFrame(status_rows)

    st.dataframe(
        status_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # KEY PROJECT METRICS
    # --------------------------------------------------------

    st.subheader("📊 Project Overview")

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Dataset Records",
        "1,345,350"
    )

    metric2.metric(
        "Engineered Features",
        "26"
    )

    metric3.metric(
        "GNN Nodes",
        "30,000"
    )

    metric4.metric(
        "Graph Edges",
        "369,098"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # MODEL PERFORMANCE SUMMARY
    # --------------------------------------------------------

    st.subheader("🤖 Model Performance")

    st.markdown(
        """
        The project compares a traditional neural-network baseline
        with a GraphSAGE model.
        """
    )

    model_metrics = pd.DataFrame({
        "Model": [
            "MLP Baseline",
            "GraphSAGE GNN"
        ],
        "Accuracy": [
            0.7430,
            0.6432
        ],
        "Precision": [
            0.2430,
            0.3221
        ],
        "Recall": [
            0.0977,
            0.6113
        ],
        "F1 Score": [
            0.1393,
            0.4219
        ],
        "ROC-AUC": [
            0.5533,
            0.6858
        ],
        "Average Precision": [
            0.2417,
            0.3560
        ]
    })

    st.dataframe(
        model_metrics.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}",
            "ROC-AUC": "{:.4f}",
            "Average Precision": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The MLP has higher accuracy, while GraphSAGE identifies "
        "more positive-risk cases through higher recall."
    )

    # --------------------------------------------------------
    # MODEL COMPARISON CHART
    # --------------------------------------------------------

    chart_df = model_metrics.melt(
        id_vars="Model",
        value_vars=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ],
        var_name="Metric",
        value_name="Value"
    )

    fig_model = px.bar(
        chart_df,
        x="Metric",
        y="Value",
        color="Model",
        barmode="group",
        title="Model Performance Comparison",
        range_y=[0, 1]
    )

    fig_model.update_layout(
        height=450,
        legend_title_text="Model"
    )

    st.plotly_chart(
        fig_model,
        use_container_width=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # CALIBRATION RESULTS
    # --------------------------------------------------------

    st.subheader("🎯 Probability Calibration")

    calibration_df = pd.DataFrame({
        "Model": [
            "MLP",
            "GraphSAGE"
        ],
        "Brier Score Before Calibration": [
            0.228459,
            0.222188
        ],
        "Brier Score After Calibration": [
            0.167802,
            0.158626
        ]
    })

    st.dataframe(
        calibration_df.style.format({
            "Brier Score Before Calibration": "{:.6f}",
            "Brier Score After Calibration": "{:.6f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "Platt scaling reduced the Brier score for both models, "
        "indicating improved probability calibration on the validation data."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # PROJECT PIPELINE
    # --------------------------------------------------------

    st.subheader("🔄 FinTwinAI Pipeline")

    st.code(
        """
LendingClub Dataset
        ↓
Data Cleaning and Preprocessing
        ↓
Feature Engineering
        ↓
MLP Baseline Model
        ↓
Graph Construction using KNN
        ↓
GraphSAGE GNN Model
        ↓
Probability Calibration
        ↓
SHAP Explainability
        ↓
FRED Macroeconomic Indicators
        ↓
Stress Testing and Digital Twin
        ↓
Streamlit Dashboard
        """,
        language="text"
    )

    st.warning(
        """
        **Research limitation:** The constructed graph represents
        feature similarity between borrowers. It is not a confirmed
        real-world borrower relationship or financial contagion network.
        """
    )

    st.markdown("---")

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.caption(
        """
        FinTwinAI is an academic research prototype. Its outputs are
        educational estimates and should not be treated as official
        credit approval, investment advice or financial advice.
        """
    )
# ============================================================
# PAGE 2 — INDIVIDUAL FINANCIAL HEALTH ANALYSIS
# ============================================================

elif page == "👤 Personal Financial Health":

    st.title("👤 Personal Financial Health Analysis")

    st.markdown(
        """
        Enter your financial information to estimate your monthly EMI,
        repayment burden and financial-health indicators.
        """
    )

    st.info(
        "This is an educational assessment, not an official bank "
        "credit score or loan approval decision."
    )

    # --------------------------------------------------------
    # USER INPUT FORM
    # --------------------------------------------------------

    with st.form("financial_health_form"):

        st.subheader("💰 Income and Loan Details")

        col1, col2, col3 = st.columns(3)

        with col1:

            monthly_income = st.number_input(
                "Monthly Income (₹)",
                min_value=1_000.0,
                max_value=10_000_000.0,
                value=50_000.0,
                step=1_000.0
            )

            loan_amount = st.number_input(
                "Loan Amount (₹)",
                min_value=1_000.0,
                max_value=10_000_000.0,
                value=300_000.0,
                step=5_000.0
            )

            interest_rate = st.number_input(
                "Annual Interest Rate (%)",
                min_value=0.0,
                max_value=40.0,
                value=12.0,
                step=0.5
            )

            loan_term = st.selectbox(
                "Loan Term (Months)",
                [12, 24, 36, 48, 60, 72, 84],
                index=2
            )

        with col2:

            existing_emi = st.number_input(
                "Existing Monthly EMIs (₹)",
                min_value=0.0,
                max_value=500_000.0,
                value=5_000.0,
                step=500.0
            )

            monthly_expenses = st.number_input(
                "Monthly Living Expenses (₹)",
                min_value=0.0,
                max_value=1_000_000.0,
                value=20_000.0,
                step=1_000.0
            )

            credit_score = st.number_input(
                "Credit Score / FICO",
                min_value=300,
                max_value=900,
                value=700,
                step=5
            )

        with col3:

            revolving_utilization = st.slider(
                "Credit Utilization (%)",
                min_value=0.0,
                max_value=100.0,
                value=30.0,
                step=1.0
            )

            delinquencies = st.number_input(
                "Delinquencies in Last 2 Years",
                min_value=0,
                max_value=20,
                value=0,
                step=1
            )

            employment_years = st.number_input(
                "Employment Experience (Years)",
                min_value=0.0,
                max_value=50.0,
                value=3.0,
                step=0.5
            )

        submitted = st.form_submit_button(
            "🔍 Generate Financial Health Report",
            use_container_width=True
        )

    # --------------------------------------------------------
    # CALCULATIONS
    # --------------------------------------------------------

    if submitted:

        if monthly_income <= 0 or loan_amount <= 0:

            st.error(
                "Income and loan amount must be greater than zero."
            )

        else:

            # Monthly interest rate
            monthly_rate = interest_rate / 100 / 12

            # EMI calculation
            if monthly_rate == 0:

                emi = loan_amount / loan_term

            else:

                emi = (
                    loan_amount
                    * monthly_rate
                    * (1 + monthly_rate) ** loan_term
                    / (
                        (1 + monthly_rate) ** loan_term - 1
                    )
                )

            total_repayment = emi * loan_term
            total_interest = total_repayment - loan_amount

            # Financial ratios
            total_monthly_debt = existing_emi + emi

            debt_to_income = (
                total_monthly_debt / monthly_income
            ) * 100

            emi_to_income = (
                emi / monthly_income
            ) * 100

            loan_to_annual_income = (
                loan_amount / (monthly_income * 12)
            )

            remaining_cash = (
                monthly_income
                - monthly_expenses
                - total_monthly_debt
            )

            # ------------------------------------------------
            # EDUCATIONAL RISK ESTIMATION
            # ------------------------------------------------

            risk_points = 0

            if debt_to_income > 40:
                risk_points += 25
            elif debt_to_income > 30:
                risk_points += 15

            if emi_to_income > 30:
                risk_points += 20
            elif emi_to_income > 20:
                risk_points += 10

            if credit_score < 600:
                risk_points += 25
            elif credit_score < 680:
                risk_points += 15

            if revolving_utilization > 70:
                risk_points += 15
            elif revolving_utilization > 30:
                risk_points += 8

            if delinquencies > 0:
                risk_points += 15

            if interest_rate > 18:
                risk_points += 10

            educational_risk = min(
                risk_points / 100,
                0.95
            )

            health_score = round(
                (1 - educational_risk) * 100,
                1
            )

            if health_score >= 75:
                risk_category = "Low Risk"
            elif health_score >= 55:
                risk_category = "Moderate Risk"
            elif health_score >= 35:
                risk_category = "High Risk"
            else:
                risk_category = "Very High Risk"

            # ------------------------------------------------
            # REPORT HEADER
            # ------------------------------------------------

            st.markdown("---")
            st.subheader("📊 Your Financial Health Report")

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Monthly EMI",
                format_inr(emi)
            )

            m2.metric(
                "Health Score",
                f"{health_score}/100"
            )

            m3.metric(
                "Debt-to-Income",
                f"{debt_to_income:.1f}%"
            )

            m4.metric(
                "Risk Category",
                risk_category
            )

            # ------------------------------------------------
            # EMI SUMMARY
            # ------------------------------------------------

            st.markdown("---")
            st.subheader("💳 Loan Repayment Summary")

            emi_col1, emi_col2, emi_col3 = st.columns(3)

            emi_col1.metric(
                "Loan Amount",
                format_inr(loan_amount)
            )

            emi_col2.metric(
                "Total Repayment",
                format_inr(total_repayment)
            )

            emi_col3.metric(
                "Total Interest",
                format_inr(total_interest)
            )

            repayment_df = pd.DataFrame({
                "Component": [
                    "Principal Amount",
                    "Total Interest",
                    "Total Repayment"
                ],
                "Amount (₹)": [
                    loan_amount,
                    total_interest,
                    total_repayment
                ]
            })

            fig_repayment = px.bar(
                repayment_df,
                x="Component",
                y="Amount (₹)",
                title="Loan Repayment Breakdown",
                text_auto=".2s"
            )

            st.plotly_chart(
                fig_repayment,
                use_container_width=True
            )

            # ------------------------------------------------
            # CASH FLOW ANALYSIS
            # ------------------------------------------------

            st.markdown("---")
            st.subheader("💰 Monthly Cash Flow")

            cashflow_df = pd.DataFrame({
                "Category": [
                    "Monthly Income",
                    "Living Expenses",
                    "Existing EMIs",
                    "New EMI",
                    "Remaining Cash"
                ],
                "Amount (₹)": [
                    monthly_income,
                    monthly_expenses,
                    existing_emi,
                    emi,
                    max(remaining_cash, 0)
                ]
            })

            st.dataframe(
                cashflow_df.style.format({
                    "Amount (₹)": "₹{:,.2f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            if remaining_cash < 0:

                st.error(
                    "⚠️ Your calculated monthly expenses and debt "
                    "payments are greater than your income."
                )

            elif remaining_cash < monthly_income * 0.10:

                st.warning(
                    "⚠️ Your remaining monthly cash is relatively low. "
                    "Keep an emergency fund before taking additional debt."
                )

            else:

                st.success(
                    "✅ Your calculated monthly cash flow is positive."
                )

            # ------------------------------------------------
            # FINANCIAL INDICATORS
            # ------------------------------------------------

            st.markdown("---")
            st.subheader("📈 Important Financial Indicators")

            indicator_df = pd.DataFrame({
                "Indicator": [
                    "EMI-to-Income Ratio",
                    "Debt-to-Income Ratio",
                    "Loan-to-Annual-Income Ratio",
                    "Credit Utilization",
                    "Credit Score",
                    "Delinquencies"
                ],
                "Value": [
                    f"{emi_to_income:.2f}%",
                    f"{debt_to_income:.2f}%",
                    f"{loan_to_annual_income:.2f}",
                    f"{revolving_utilization:.2f}%",
                    f"{credit_score}",
                    f"{delinquencies}"
                ]
            })

            st.dataframe(
                indicator_df,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # PERSONALIZED RECOMMENDATIONS
            # ------------------------------------------------

            st.markdown("---")
            st.subheader("📝 Personalized Recommendations")

            recommendations = []

            if debt_to_income > 40:

                recommendations.append(
                    "🔴 Your debt-to-income ratio is high. "
                    "Avoid taking unnecessary additional loans."
                )

            elif debt_to_income > 30:

                recommendations.append(
                    "🟠 Your debt burden is moderate to high. "
                    "Prepare a monthly repayment budget."
                )

            else:

                recommendations.append(
                    "🟢 Your calculated debt-to-income ratio is "
                    "within the selected educational range."
                )

            if emi_to_income > 30:

                recommendations.append(
                    "⚠️ The new EMI uses a large portion of your income. "
                    "Consider a smaller loan amount or longer repayment period."
                )

            if credit_score < 680:

                recommendations.append(
                    "📌 Review your credit report and pay all bills "
                    "and EMIs on time."
                )

            if revolving_utilization > 70:

                recommendations.append(
                    "💳 Credit utilization is high. "
                    "Try to reduce outstanding revolving balances."
                )

            elif revolving_utilization > 30:

                recommendations.append(
                    "💳 Consider reducing credit utilization gradually."
                )

            if delinquencies > 0:

                recommendations.append(
                    "⏰ Previous delinquencies were reported. "
                    "Use reminders or automatic payments to avoid missed dues."
                )

            if interest_rate > 18:

                recommendations.append(
                    "💡 Your interest rate is high in this assessment. "
                    "Compare eligible alternatives carefully before refinancing."
                )

            if remaining_cash < 0:

                recommendations.append(
                    "🚨 Your projected monthly cash flow is negative. "
                    "Review expenses and debt obligations immediately."
                )

            elif remaining_cash < monthly_income * 0.10:

                recommendations.append(
                    "🏦 Build an emergency fund before increasing your debt."
                )

            if not recommendations:

                recommendations.append(
                    "✅ Continue monitoring income, expenses and repayments."
                )

            for recommendation in recommendations:

                st.markdown(
                    f"- {recommendation}"
                )

            # ------------------------------------------------
            # SIMPLE EXPLANATION
            # ------------------------------------------------

            st.markdown("---")
            st.subheader("📚 Explanation in Simple Language")

            st.markdown(
                f"""
                **Your estimated monthly EMI is {format_inr(emi)}.**

                This means you may need to pay approximately this amount
                every month for {loan_term} months.

                Your calculated debt-to-income ratio is
                **{debt_to_income:.1f}%**. This compares your monthly debt
                payments with your monthly income.

                Your estimated remaining cash after expenses and debt
                payments is **{format_inr(remaining_cash)} per month**.

                The displayed health score is an educational,
                rule-based indicator. It is **not the trained GraphSAGE
                model's prediction** and should not be treated as a
                certified credit rating.
                """
            )

            # ------------------------------------------------
            # DISCLAIMER
            # ------------------------------------------------

            st.caption(
                """
                Disclaimer: This assessment is for academic and
                educational purposes only. Actual loan eligibility,
                interest rates and credit decisions depend on financial
                institutions and verified borrower information.
                """
            )
# ============================================================
# PART 4: GNN, SURVIVAL, MACRO, PORTFOLIO & METRICS PAGES
# ============================================================


# ============================================================
# PAGE 3: NETWORK RISK (GNN)
# ============================================================

elif page == "🌐 GraphSAGE Network Risk":

    st.title("🌐 Network Risk Analysis")
    st.markdown(
        "Graph-based analysis using a feature-similarity network."
    )

    st.warning(
        "Important: The graph is constructed using feature similarity "
        "through KNN. It is not a confirmed real-world borrower network "
        "and does not directly prove financial contagion."
    )

    st.subheader("📌 Graph Statistics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Graph Nodes", "30,000")
    col2.metric("Graph Edges", "369,098")
    col3.metric("Average Degree", "12.30")
    col4.metric("GraphSAGE ROC-AUC", "0.6858")

    st.divider()

    st.subheader("📊 GraphSAGE Confusion Matrix")

    confusion_matrix = np.array([
        [2565, 1370],
        [414, 651]
    ])

    fig = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix,
            x=["Predicted Non-Default", "Predicted Default"],
            y=["Actual Non-Default", "Actual Default"],
            text=confusion_matrix,
            texttemplate="%{text}",
            colorscale="Blues"
        )
    )

    fig.update_layout(
        title="GraphSAGE Test Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧠 GraphSAGE Interpretation")

    st.markdown("""
    **GraphSAGE Test Results**

    - Accuracy: **64.32%**
    - Precision: **32.21%**
    - Recall: **61.13%**
    - F1-Score: **42.19%**
    - ROC-AUC: **68.58%**
    - Average Precision: **35.60%**

    The model identifies a larger proportion of default cases than the
    MLP baseline, but it also generates more false positives.

    Therefore, the model should be used as a risk-screening tool rather
    than an automatic loan approval or rejection system.
    """)

    with st.expander("⚠️ Network Limitations"):
        st.markdown("""
        1. The graph is based on feature similarity.
        2. It is not a verified borrower-to-borrower relationship network.
        3. KNN graph construction used the complete sampled feature space.
        4. The graph should not be interpreted as proof of contagion.
        5. External validation is required before real-world deployment.
        """)


# ============================================================
# PAGE 4: SURVIVAL TIMELINE
# ============================================================

elif page == "⏳ Survival Analysis":

    st.title("⏳ Survival Timeline Analysis")

    st.info(
        "This page presents an illustrative stress curve. "
        "It is not a validated DeepSurv prediction because actual "
        "event dates and follow-up durations were not established."
    )

    st.subheader("🎛️ Scenario Controls")

    health_score = st.slider(
        "Borrower Health Score",
        min_value=0,
        max_value=100,
        value=60,
        step=1
    )

    stress_level = st.slider(
        "Network/Macro Stress Level",
        min_value=0,
        max_value=100,
        value=30,
        step=5
    )

    months = np.arange(1, 37)

    # Illustrative hazard calculation
    base_hazard = 0.015

    health_effect = (100 - health_score) / 100
    stress_effect = stress_level / 100

    hazard = base_hazard * (
        1 + health_effect + stress_effect
    )

    survival_probability = np.exp(-hazard * months)
    illustrative_event_probability = 1 - survival_probability

    timeline_df = pd.DataFrame({
        "Month": months,
        "Illustrative Survival Probability": survival_probability,
        "Illustrative Event Probability": illustrative_event_probability
    })

    st.subheader("📈 Illustrative Survival Curve")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=timeline_df["Month"],
            y=timeline_df["Illustrative Survival Probability"],
            mode="lines",
            name="Survival Probability"
        )
    )

    fig.update_layout(
        title="Illustrative Survival Curve",
        xaxis_title="Months",
        yaxis_title="Survival Probability",
        yaxis=dict(range=[0, 1])
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Scenario Summary")

    selected_months = [6, 12, 24, 36]

    summary_data = []

    for month in selected_months:
        survival = survival_probability[month - 1]
        event_probability = 1 - survival

        summary_data.append({
            "Month": month,
            "Survival Probability": survival,
            "Event Probability": event_probability
        })

    summary_df = pd.DataFrame(summary_data)

    summary_df["Survival Probability"] = (
        summary_df["Survival Probability"].map(
            lambda x: f"{x:.2%}"
        )
    )

    summary_df["Event Probability"] = (
        summary_df["Event Probability"].map(
            lambda x: f"{x:.2%}"
        )
    )

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )

    with st.expander("📌 Important Methodological Note"):
        st.markdown("""
        A proper survival model requires:

        - Time-to-event information
        - Default or non-default event indicator
        - Last observed date
        - Loan origination date
        - Censoring information

        The current graph is only an illustrative mathematical curve.
        It must not be reported as an actual trained DeepSurv result.
        """)


# ============================================================
# PAGE 5: MACRO STRESS TEST
# ============================================================

elif page == "📉 Macro Stress Testing":

    st.title("📊 Macroeconomic Stress Testing")

    st.markdown(
        "Scenario-based analysis of expected loss under increasing "
        "probability-of-default assumptions."
    )

    st.warning(
        "Expected loss values are shown in the original dataset units. "
        "The dataset currency is not specified."
    )

    stress_data = pd.DataFrame({
        "Scenario": [
            "Baseline",
            "Moderate Stress",
            "Severe Stress"
        ],
        "PD Multiplier": [
            1.00,
            1.20,
            1.50
        ],
        "Expected Loss": [
            35861090,
            42763120,
            51566810
        ],
        "Increase from Baseline": [
            0,
            6902027,
            15705720
        ],
        "Percentage Increase": [
            0,
            19.2466,
            43.7960
        ]
    })

    st.subheader("📌 Stress Scenario Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Baseline Expected Loss",
        f"{stress_data.loc[0, 'Expected Loss']:,.0f}"
    )

    col2.metric(
        "Moderate Stress",
        f"{stress_data.loc[1, 'Expected Loss']:,.0f}",
        "+19.25%"
    )

    col3.metric(
        "Severe Stress",
        f"{stress_data.loc[2, 'Expected Loss']:,.0f}",
        "+43.80%"
    )

    st.divider()

    st.subheader("📊 Expected Loss by Scenario")

    fig = px.bar(
        stress_data,
        x="Scenario",
        y="Expected Loss",
        text="Expected Loss",
        title="Expected Loss under Stress Scenarios"
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Detailed Stress Table")

    display_stress = stress_data.copy()

    display_stress["Expected Loss"] = (
        display_stress["Expected Loss"].map(
            lambda x: f"{x:,.0f}"
        )
    )

    display_stress["Increase from Baseline"] = (
        display_stress["Increase from Baseline"].map(
            lambda x: f"{x:,.0f}"
        )
    )

    display_stress["Percentage Increase"] = (
        display_stress["Percentage Increase"].map(
            lambda x: f"{x:.2f}%"
        )
    )

    st.dataframe(
        display_stress,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("🌍 Macro Indicators")

    if fred_data is not None and not fred_data.empty:

        st.success("Macroeconomic data loaded successfully.")

        st.dataframe(
            fred_data.head(10),
            use_container_width=True,
            hide_index=True
        )

        numeric_columns = fred_data.select_dtypes(
            include=np.number
        ).columns.tolist()

        if len(numeric_columns) > 0:

            selected_macro = st.selectbox(
                "Select Macro Indicator",
                numeric_columns
            )

            st.line_chart(
                fred_data[selected_macro]
            )

    else:
        st.warning(
            "FRED macroeconomic data is not available."
        )

    with st.expander("⚠️ Stress Test Limitations"):
        st.markdown("""
        - The scenarios apply assumed PD multipliers.
        - They are not complete macroeconomic forecasts.
        - The Macro Stress Index is a constructed indicator.
        - Additional validation is required before financial deployment.
        """)


# ============================================================
# PAGE 6: PORTFOLIO DASHBOARD
# ============================================================

elif page == "📊 Portfolio Analytics":

    st.title("📈 Portfolio Risk Dashboard")

    st.markdown(
        "Aggregated portfolio exposure and expected loss across "
        "risk categories."
    )

    st.warning(
        "Exposure and expected loss are presented in the original "
        "dataset units because the currency was not specified."
    )

    portfolio_df = pd.DataFrame({
        "Risk Category": [
            "Very High",
            "High",
            "Medium",
            "Low"
        ],
        "Borrower Count": [
            1196,
            1791,
            1613,
            400
        ],
        "Average Health Score": [
            70.9448,
            49.5733,
            30.9351,
            15.2788
        ],
        "Exposure": [
            22637825,
            25201775,
            20099325,
            4919925
        ],
        "Expected Loss": [
            16311740,
            12595090,
            6234577,
            719687
        ]
    })

    st.subheader("📌 Portfolio Summary")

    total_borrowers = portfolio_df["Borrower Count"].sum()
    total_exposure = portfolio_df["Exposure"].sum()
    total_expected_loss = portfolio_df["Expected Loss"].sum()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Borrowers",
        f"{total_borrowers:,}"
    )

    col2.metric(
        "Total Exposure",
        f"{total_exposure:,.0f}"
    )

    col3.metric(
        "Total Expected Loss",
        f"{total_expected_loss:,.0f}"
    )

    st.divider()

    st.subheader("📊 Borrowers by Risk Category")

    fig = px.bar(
        portfolio_df,
        x="Risk Category",
        y="Borrower Count",
        text="Borrower Count",
        title="Risk Category Distribution"
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("💰 Exposure by Risk Category")

    fig = px.bar(
        portfolio_df,
        x="Risk Category",
        y="Exposure",
        text="Exposure",
        title="Portfolio Exposure"
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("⚠️ Expected Loss by Risk Category")

    fig = px.bar(
        portfolio_df,
        x="Risk Category",
        y="Expected Loss",
        text="Expected Loss",
        title="Expected Loss Distribution"
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Portfolio Data")

    display_portfolio = portfolio_df.copy()

    display_portfolio["Average Health Score"] = (
        display_portfolio["Average Health Score"].map(
            lambda x: f"{x:.2f}"
        )
    )

    display_portfolio["Exposure"] = (
        display_portfolio["Exposure"].map(
            lambda x: f"{x:,.0f}"
        )
    )

    display_portfolio["Expected Loss"] = (
        display_portfolio["Expected Loss"].map(
            lambda x: f"{x:,.0f}"
        )
    )

    st.dataframe(
        display_portfolio,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 7: PERFORMANCE METRICS
# ============================================================

elif page == "🤖 Model Performance":

    st.title("📋 Model Performance Metrics")

    st.markdown(
        "Comparison of the baseline MLP model and the GraphSAGE model."
    )

    st.subheader("📊 Model Comparison")

    metrics_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC",
            "Average Precision"
        ],
        "MLP": [
            0.7430,
            0.2430,
            0.0977,
            0.1393,
            0.5533,
            0.2417
        ],
        "GraphSAGE": [
            0.6432,
            0.3221,
            0.6113,
            0.4219,
            0.6858,
            0.3560
        ]
    })

    display_metrics = metrics_df.copy()

    display_metrics["MLP"] = display_metrics["MLP"].map(
        lambda x: f"{x:.4f}"
    )

    display_metrics["GraphSAGE"] = display_metrics["GraphSAGE"].map(
        lambda x: f"{x:.4f}"
    )

    st.dataframe(
        display_metrics,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📈 ROC-AUC Comparison")

    auc_df = pd.DataFrame({
        "Model": ["MLP", "GraphSAGE"],
        "ROC-AUC": [0.5533, 0.6858]
    })

    fig = px.bar(
        auc_df,
        x="Model",
        y="ROC-AUC",
        text="ROC-AUC",
        title="ROC-AUC Comparison"
    )

    fig.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside"
    )

    fig.update_yaxes(range=[0, 1])

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🎯 Calibration Results")

    calibration_df = pd.DataFrame({
        "Model": ["MLP", "MLP", "GraphSAGE", "GraphSAGE"],
        "Stage": [
            "Before Calibration",
            "After Calibration",
            "Before Calibration",
            "After Calibration"
        ],
        "Brier Score": [
            0.228459,
            0.167802,
            0.222188,
            0.158626
        ]
    })

    st.dataframe(
        calibration_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("⚙️ Training Configuration")

    training_info = {
        "Feature Count": "26",
        "Graph Nodes": "30,000",
        "Graph Edges": "369,098",
        "Hidden Dimensions": "64",
        "GraphSAGE Layers": "3",
        "Dropout": "0.30",
        "Learning Rate": "0.001",
        "Weight Decay": "0.0001",
        "Optimizer": "Adam",
        "Loss Function": "Weighted Cross Entropy"
    }

    training_df = pd.DataFrame(
        list(training_info.items()),
        columns=["Parameter", "Value"]
    )

    st.dataframe(
        training_df,
        use_container_width=True,
        hide_index=True
    )

    with st.expander("⚠️ Important Evaluation Notes"):
        st.markdown("""
        **Temporal split used:**

        - Training data: Issue year ≤ 2015
        - Validation data: 2016
        - Test data: 2017 and later

        **Model limitations:**

        - The GraphSAGE model has lower accuracy than the MLP.
        - GraphSAGE has higher recall and F1-score.
        - The graph represents feature similarity rather than verified
          borrower relationships.
        - Threshold selection should be validated independently.
        - Results should not be treated as production-ready credit decisions.
        """)

    st.success(
        "The dashboard combines predictive performance, portfolio exposure, "
        "stress testing and borrower-level educational analysis."
    )
# ============================================================
# SHAP EXPLAINABILITY PAGE
# ============================================================

elif "SHAP Explainability" in page:

    st.title("🔍 SHAP Explainability")

    st.markdown("""
    SHAP (SHapley Additive exPlanations) helps us understand
    how individual features influence model predictions.
    """)

    st.subheader("🧠 Why SHAP?")

    st.markdown("""
    SHAP is used to:

    - Explain individual predictions
    - Identify important risk factors
    - Improve model transparency
    - Support responsible financial decision-making
    - Understand positive and negative feature contributions
    """)

    st.divider()

    st.subheader("📊 Feature Importance")

    if shap_importance is not None and not shap_importance.empty:

        st.success("SHAP importance data loaded.")

        st.dataframe(
            shap_importance,
            use_container_width=True,
            hide_index=True
        )

        numeric_columns = shap_importance.select_dtypes(
            include=np.number
        ).columns.tolist()

        if len(numeric_columns) > 0:

            selected_column = st.selectbox(
                "Select SHAP Importance Column",
                numeric_columns
            )

            st.bar_chart(
                shap_importance[selected_column]
            )

    else:

        st.warning(
            "SHAP importance CSV is unavailable."
        )

        st.info("""
        The SHAP analysis requires the saved SHAP values
        generated during model training.
        """)

    st.subheader("📌 Example Interpretation")

    st.markdown("""
    **Possible financial risk factors include:**

    - Interest rate
    - Debt-to-income ratio
    - Credit score
    - Revolving utilization
    - Annual income
    - Delinquencies
    - Loan installment

    A positive SHAP value increases the model's output for
    the selected class, while a negative value decreases it.
    The interpretation depends on the selected model and class.
    """)

    with st.expander("⚠️ Important Limitation"):
        st.write("""
        SHAP explains model behavior. It does not prove causality.
        A feature being important does not necessarily mean that
        changing the feature will cause the predicted risk to change.
        """)


# ============================================================
# ABOUT PROJECT PAGE
# ============================================================

elif "About Project" in page:

    st.title("ℹ️ About FinTwinAI")

    st.subheader(
        "Explainable AI-Driven Personal Financial Digital Twin"
    )

    st.markdown("""
    **FinTwinAI** is an academic financial risk-analysis project
    designed to combine machine learning, graph learning,
    explainable AI and scenario-based analysis.
    """)

    st.divider()

    st.subheader("🎯 Project Objectives")

    st.markdown("""
    1. Analyse borrower financial risk.
    2. Compare traditional machine learning with GraphSAGE.
    3. Construct a feature-similarity graph.
    4. Apply explainable AI using SHAP.
    5. Perform macroeconomic stress testing.
    6. Develop an educational personal financial health tool.
    7. Present portfolio-level risk insights.
    """)

    st.subheader("🧠 Technologies Used")

    technology_df = pd.DataFrame({
        "Area": [
            "Programming",
            "Data Processing",
            "Machine Learning",
            "Graph Learning",
            "Explainable AI",
            "Dashboard"
        ],
        "Technology": [
            "Python",
            "Pandas, NumPy",
            "MLP / Traditional ML",
            "GraphSAGE",
            "SHAP",
            "Streamlit"
        ]
    })

    st.dataframe(
        technology_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📊 Main Components")

    components = [
        "Executive Dashboard",
        "Personal Financial Health",
        "Portfolio Analytics",
        "GraphSAGE Network Risk",
        "SHAP Explainability",
        "Model Performance",
        "Macro Stress Testing",
        "Survival Analysis"
    ]

    for component in components:
        st.write(f"✅ {component}")

    st.divider()

    st.subheader("👨‍🎓 Project Information")

    project_info = {
        "Project Name": "FinTwinAI",
        "Student": "Priti Prakash Mohanta",
        "Supervisor": "Dr. Bhavna Saini",
        "Department": "Data Science and Analytics",
        "University": "Central University of Rajasthan"
    }

    for key, value in project_info.items():
        st.write(f"**{key}:** {value}")

    st.divider()

    st.warning("""
    This dashboard is intended for academic and analytical purposes.
    It should not be used as an autonomous loan approval,
    rejection or financial advisory system.
    """)