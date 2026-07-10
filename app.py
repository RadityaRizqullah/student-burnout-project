"""
🎓 Student Burnout & Dropout Risk Dashboard
=============================================
An interactive Streamlit dashboard for analyzing student burnout levels
and dropout risk factors. Built with Plotly, Pandas, and scikit-learn.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Page config & theme
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Burnout & Dropout Risk Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS for dark theme styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSelectbox label {
        color: #c9d1d9 !important;
        font-weight: 500;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1c2333 0%, #161b22 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-card h3 {
        color: #8b949e;
        font-size: 0.85rem;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        color: #58a6ff;
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-card .subtext {
        color: #8b949e;
        font-size: 0.8rem;
        margin-top: 4px;
    }
    
    /* Risk high */
    .risk-high {
        background: linear-gradient(135deg, #2d1117 0%, #161b22 100%);
        border-color: #f85149;
    }
    .risk-high .value { color: #f85149; }
    
    /* Risk medium */
    .risk-medium {
        background: linear-gradient(135deg, #2d2200 0%, #161b22 100%);
        border-color: #d29922;
    }
    .risk-medium .value { color: #d29922; }
    
    /* Risk low */
    .risk-low {
        background: linear-gradient(135deg, #0d2217 0%, #161b22 100%);
        border-color: #3fb950;
    }
    .risk-low .value { color: #3fb950; }
    
    /* Section headers */
    .section-header {
        color: #f0f6fc;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #21262d;
    }
    
    /* Info boxes */
    .info-box {
        background: #161b22;
        border-left: 3px solid #58a6ff;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 12px 0;
        color: #c9d1d9;
    }
    .info-box.warning {
        border-left-color: #d29922;
    }
    .info-box.success {
        border-left-color: #3fb950;
    }
    
    /* Risk factor chips */
    .risk-chip {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 3px 4px;
    }
    .chip-increase {
        background: rgba(248, 81, 73, 0.15);
        color: #f85149;
        border: 1px solid rgba(248, 81, 73, 0.3);
    }
    .chip-decrease {
        background: rgba(63, 185, 80, 0.15);
        color: #3fb950;
        border: 1px solid rgba(63, 185, 80, 0.3);
    }
    .chip-neutral {
        background: rgba(139, 148, 158, 0.15);
        color: #8b949e;
        border: 1px solid rgba(139, 148, 158, 0.3);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        color: #8b949e;
        border-radius: 8px 8px 0 0;
        border: 1px solid #30363d;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1c2333 !important;
        color: #58a6ff !important;
        border-bottom-color: #58a6ff !important;
    }
    
    /* DataFrame styling */
    [data-testid="stDataFrame"] {
        border: 1px solid #30363d;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Plotly color palette for dark theme
COLORS = {
    "primary": "#58a6ff",
    "secondary": "#f0883e",
    "accent": "#bc8cff",
    "success": "#3fb950",
    "warning": "#d29922",
    "danger": "#f85149",
    "bg": "#0d1117",
    "card": "#161b22",
    "border": "#30363d",
    "text": "#c9d1d9",
    "text_muted": "#8b949e",
}

BURNOUT_COLORS = {"Low": "#3fb950", "Medium": "#d29922", "High": "#f85149"}
BURNOUT_ORDER = ["Low", "Medium", "High"]
DROPOUT_COLORS = {"No": "#3fb950", "Yes": "#f85149"}

DEPT_ORDER = ["Engineering", "Business", "Science", "Arts", "Medicine", "Law"]
INCOME_ORDER = ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"]

PLOTLY_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#c9d1d9", family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="#30363d",
        font=dict(color="#c9d1d9"),
    ),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
)

# Feature definitions for the early warning form
CATEGORICAL_FEATURES = {
    "Gender": ["Female", "Male", "Other"],
    "Department": ["Engineering", "Business", "Science", "Arts", "Medicine", "Law"],
    "Residence_Type": ["Hostel", "Day Scholar", "PG/Rented"],
    "Part_Time_Job": ["Yes", "No"],
    "Family_Income_Bracket": ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"],
    "Counseling_Access": ["Yes", "No"],
}

CONTINUOUS_FEATURES = {
    "Age": {"min_value": 17, "max_value": 25, "value": 21, "step": 1},
    "Year_of_Study": {"min_value": 1, "max_value": 4, "value": 2, "step": 1},
    "Attendance_Percent": {"min_value": 30.0, "max_value": 100.0, "value": 85.0, "step": 0.1},
    "Study_Hours_Per_Day": {"min_value": 0.0, "max_value": 12.0, "value": 4.0, "step": 0.1},
    "Previous_GPA": {"min_value": 2.0, "max_value": 10.0, "value": 7.0, "step": 0.01},
    "Backlogs": {"min_value": 0, "max_value": 6, "value": 0, "step": 1},
    "Sleep_Hours": {"min_value": 2.0, "max_value": 12.0, "value": 7.0, "step": 0.5},
    "Screen_Time_Hours": {"min_value": 0.0, "max_value": 16.0, "value": 6.0, "step": 0.5},
    "Exercise_Freq_Per_Week": {"min_value": 0, "max_value": 7, "value": 3, "step": 1},
    "Social_Activity_Score": {"min_value": 0.0, "max_value": 10.0, "value": 5.0, "step": 0.1},
    "Financial_Stress_Score": {"min_value": 0.0, "max_value": 10.0, "value": 5.0, "step": 0.1},
    "Family_Support_Score": {"min_value": 0.0, "max_value": 10.0, "value": 6.0, "step": 0.1},
    "Stress_Level": {"min_value": 0.0, "max_value": 10.0, "value": 5.0, "step": 0.1},
    "Anxiety_Score": {"min_value": 0.0, "max_value": 10.0, "value": 5.0, "step": 0.1},
    "Motivation_Score": {"min_value": 0.0, "max_value": 10.0, "value": 5.0, "step": 0.1},
    "Peer_Pressure_Score": {"min_value": 0.0, "max_value": 10.0, "value": 5.0, "step": 0.1},
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load and return the student dataset."""
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_data
def load_data_cleaned():
    """Load data with basic imputation for analysis."""
    df = pd.read_csv(DATA_PATH)
    # Fill numeric NaNs with medians
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != "Student_ID":
            df[col] = df[col].fillna(df[col].median())
    # Fill categorical NaNs with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if col not in ("Student_ID", "Burnout_Level", "Dropout_Risk"):
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])
    return df


def load_model_files():
    """Load trained models and metadata if available."""
    models = {}
    files = {
        "dropout_model": "dropout_model.joblib",
        "burnout_model": "burnout_model.joblib",
        "feature_metadata": "feature_metadata.joblib",
    }
    try:
        import joblib
        for key, fname in files.items():
            fpath = os.path.join(MODELS_DIR, fname)
            if os.path.exists(fpath):
                models[key] = joblib.load(fpath)
            else:
                models[key] = None
    except ImportError:
        for key in files:
            models[key] = None
    return models


@st.cache_data
def load_shap_data():
    """Load SHAP importance data if available."""
    shap_path = os.path.join(MODELS_DIR, "shap_importance.csv")
    if os.path.exists(shap_path):
        return pd.read_csv(shap_path)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def metric_card(title, value, subtext="", css_class=""):
    """Render a styled metric card."""
    cls = f"metric-card {css_class}" if css_class else "metric-card"
    sub_html = f'<div class="subtext">{subtext}</div>' if subtext else ""
    st.markdown(f"""
    <div class="{cls}">
        <h3>{title}</h3>
        <div class="value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def risk_chip(label, direction):
    """Render a risk factor chip."""
    css = f"chip-{direction}"
    return f'<span class="risk-chip {css}">{label}</span>'


def format_percent(val):
    """Format a probability as percentage."""
    return f"{val * 100:.1f}%"


def create_gauge_chart(value, title, max_val=1.0, color="#58a6ff"):
    """Create a Plotly gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title=dict(text=title, font=dict(size=16, color="#c9d1d9")),
        number=dict(suffix="%", font=dict(size=28, color=color)),
        gauge=dict(
            axis=dict(range=[0, max_val * 100], tickcolor="#8b949e",
                      tickfont=dict(color="#8b949e")),
            bar=dict(color=color),
            bgcolor="#161b22",
            borderwidth=1,
            bordercolor="#30363d",
            steps=[
                dict(range=[0, 33], color="rgba(63, 185, 80, 0.15)"),
                dict(range=[33, 66], color="rgba(210, 153, 34, 0.15)"),
                dict(range=[66, 100], color="rgba(248, 81, 73, 0.15)"),
            ],
            threshold=dict(
                line=dict(color=color, width=3),
                thickness=0.75,
                value=value * 100,
            ),
        ),
    ))
    fig.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"),
        margin=dict(l=30, r=30, t=40, b=20),
    )
    return fig


def create_donut_chart(labels, values, colors, title=""):
    """Create a Plotly donut/pie chart."""
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#0d1117", width=2)),
        textfont=dict(size=13, color="#c9d1d9"),
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#c9d1d9"), x=0.5),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


def explain_risk_factors(input_data, df):
    """Generate risk factor explanations based on input vs dataset averages."""
    risk_up = []
    risk_down = []

    numeric_means = df.select_dtypes(include=[np.number]).mean()

    # Attendance
    if input_data.get("Attendance_Percent", 85) < numeric_means.get("Attendance_Percent", 85):
        risk_up.append(("Low Attendance", f"Your attendance ({input_data.get('Attendance_Percent', 85):.0f}%) is below average ({numeric_means.get('Attendance_Percent', 85):.0f}%)"))
    else:
        risk_down.append(("Good Attendance", f"Your attendance ({input_data.get('Attendance_Percent', 85):.0f}%) is above average"))

    # GPA
    if input_data.get("Previous_GPA", 7) < numeric_means.get("Previous_GPA", 7):
        risk_up.append(("Low GPA", f"GPA ({input_data.get('Previous_GPA', 7):.1f}) is below average ({numeric_means.get('Previous_GPA', 7):.1f})"))
    else:
        risk_down.append(("Strong GPA", f"GPA ({input_data.get('Previous_GPA', 7):.1f}) is above average"))

    # Backlogs
    if input_data.get("Backlogs", 0) > 1:
        risk_up.append(("Academic Backlogs", f"{int(input_data.get('Backlogs', 0))} backlogs increase risk"))
    else:
        risk_down.append(("Few/No Backlogs", "Minimal academic backlogs"))

    # Sleep
    sleep = input_data.get("Sleep_Hours", 7)
    if sleep < 6:
        risk_up.append(("Insufficient Sleep", f"Only {sleep:.1f} hrs sleep (recommended: 7-8 hrs)"))
    elif sleep > 8:
        risk_down.append(("Good Sleep", f"{sleep:.1f} hrs sleep — adequate rest"))

    # Screen time
    screen = input_data.get("Screen_Time_Hours", 6)
    if screen > 8:
        risk_up.append(("Excessive Screen Time", f"{screen:.1f} hrs/day — linked to higher stress"))
    elif screen < 4:
        risk_down.append(("Moderate Screen Time", f"Healthy screen usage at {screen:.1f} hrs/day"))

    # Exercise
    exercise = input_data.get("Exercise_Freq_Per_Week", 3)
    if exercise >= 4:
        risk_down.append(("Active Lifestyle", f"Exercises {exercise}x/week — protective factor"))
    elif exercise <= 1:
        risk_up.append(("Sedentary Lifestyle", f"Only {exercise}x/week exercise — increases burnout risk"))

    # Stress
    stress = input_data.get("Stress_Level", 5)
    if stress > 7:
        risk_up.append(("High Stress", f"Stress level {stress:.1f}/10 is very elevated"))
    elif stress < 4:
        risk_down.append(("Low Stress", f"Stress level {stress:.1f}/10 — well managed"))

    # Anxiety
    anxiety = input_data.get("Anxiety_Score", 5)
    if anxiety > 7:
        risk_up.append(("High Anxiety", f"Anxiety score {anxiety:.1f}/10 is concerning"))
    elif anxiety < 3:
        risk_down.append(("Low Anxiety", "Anxiety levels are well within normal range"))

    # Motivation
    motivation = input_data.get("Motivation_Score", 5)
    if motivation > 7:
        risk_down.append(("High Motivation", f"Motivation at {motivation:.1f}/10 — strong drive"))
    elif motivation < 4:
        risk_up.append(("Low Motivation", f"Motivation at {motivation:.1f}/10 — needs support"))

    # Financial stress
    fin_stress = input_data.get("Financial_Stress_Score", 5)
    if fin_stress > 7:
        risk_up.append(("Financial Stress", f"Financial stress at {fin_stress:.1f}/10"))
    elif fin_stress < 3:
        risk_down.append(("Financial Stability", "Low financial stress — positive factor"))

    # Family support
    family_support = input_data.get("Family_Support_Score", 6)
    if family_support > 7:
        risk_down.append(("Strong Family Support", f"Support score {family_support:.1f}/10 — protective"))
    elif family_support < 4:
        risk_up.append(("Low Family Support", f"Support score {family_support:.1f}/10 — risk factor"))

    # Study hours
    study = input_data.get("Study_Hours_Per_Day", 4)
    if study > 7:
        risk_up.append(("Over-Studying", f"{study:.1f} hrs/day — may indicate unhealthy study habits"))
    elif study >= 3:
        risk_down.append(("Adequate Study Time", f"{study:.1f} hrs/day — balanced approach"))

    return risk_up, risk_down


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Student Dashboard")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📊 Overview", "🔍 EDA Explorer", "⚠️ Early Warning System",
         "🤖 Model Performance", "💡 Insights & Recommendations"],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style='padding: 10px; border-radius: 8px; background: #1c2333; border: 1px solid #30363d;'>
        <p style='color: #8b949e; font-size: 0.8rem; margin: 0;'>
            📁 Dataset: 800 students<br>
            📋 Features: 25 columns<br>
            🏷️ Targets: Dropout Risk, Burnout Level
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# 📊 Dataset Overview")
    st.markdown("*Student Burnout & Dropout Risk — Exploring patterns across 800 students and 25 features*")

    df = load_data()

    # ── Key Metrics Row ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)

    total_students = len(df)
    dropout_rate = (df["Dropout_Risk"] == "Yes").sum() / total_students
    high_burnout_pct = (df["Burnout_Level"] == "High").sum() / total_students
    avg_gpa = df["Previous_GPA"].mean()
    avg_attendance = df["Attendance_Percent"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        metric_card("Total Students", f"{total_students:,}", "In dataset")
    with col2:
        metric_card("Dropout Rate", f"{dropout_rate:.1%}", f"{(df['Dropout_Risk']=='Yes').sum()} flagged")
    with col3:
        metric_card("High Burnout", f"{high_burnout_pct:.1%}", f"{(df['Burnout_Level']=='High').sum()} students")
    with col4:
        metric_card("Avg GPA", f"{avg_gpa:.2f}", "out of 10.0")
    with col5:
        metric_card("Avg Attendance", f"{avg_attendance:.1f}%", "across all students")

    st.markdown("")

    # ── Distribution Charts Row ──────────────────────────────────────────
    st.markdown('<div class="section-header">Target Distributions</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        # Dropout Risk distribution
        dr_counts = df["Dropout_Risk"].value_counts()
        fig_dr = create_donut_chart(
            labels=dr_counts.index.tolist(),
            values=dr_counts.values.tolist(),
            colors=[DROPOUT_COLORS[l] for l in dr_counts.index],
            title="Dropout Risk Distribution",
        )
        st.plotly_chart(fig_dr, use_container_width=True)

    with col_right:
        # Burnout Level distribution
        bl_counts = df["Burnout_Level"].value_counts()
        fig_bl = create_donut_chart(
            labels=bl_counts.index.tolist(),
            values=bl_counts.values.tolist(),
            colors=[BURNOUT_COLORS[l] for l in bl_counts.index],
            title="Burnout Level Distribution",
        )
        st.plotly_chart(fig_bl, use_container_width=True)

    # ── Feature Distributions Row ────────────────────────────────────────
    st.markdown('<div class="section-header">Key Feature Distributions</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Demographics", "📈 Academic", "🧠 Well-being", "💰 Socioeconomic"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="Age", color="Dropout_Risk",
                               color_discrete_map=DROPOUT_COLORS, barmode="overlay",
                               title="Age Distribution by Dropout Risk",
                               opacity=0.75)
            fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            gender_data = df.groupby(["Gender", "Dropout_Risk"]).size().reset_index(name="count")
            fig = px.bar(gender_data, x="Gender", y="count", color="Dropout_Risk",
                         color_discrete_map=DROPOUT_COLORS, barmode="group",
                         title="Gender × Dropout Risk",
                         labels={"count": "Students"})
            fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=350)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="Previous_GPA", color="Burnout_Level",
                               color_discrete_map=BURNOUT_COLORS,
                               barmode="overlay", title="GPA Distribution by Burnout Level",
                               opacity=0.75, nbins=30)
            fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(df, x="Burnout_Level", y="Attendance_Percent",
                         color="Burnout_Level",
                         color_discrete_map=BURNOUT_COLORS,
                         title="Attendance by Burnout Level",
                         category_orders={"Burnout_Level": BURNOUT_ORDER})
            fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.violin(df, x="Burnout_Level", y="Stress_Level",
                            color="Burnout_Level",
                            color_discrete_map=BURNOUT_COLORS,
                            box=True, points="outliers",
                            title="Stress Level by Burnout",
                            category_orders={"Burnout_Level": BURNOUT_ORDER})
            fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.violin(df, x="Burnout_Level", y="Sleep_Hours",
                            color="Burnout_Level",
                            color_discrete_map=BURNOUT_COLORS,
                            box=True, points="outliers",
                            title="Sleep Hours by Burnout Level",
                            category_orders={"Burnout_Level": BURNOUT_ORDER})
            fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            inc_data = df.groupby(["Family_Income_Bracket", "Dropout_Risk"]).size().reset_index(name="count")
            fig = px.bar(inc_data, x="Family_Income_Bracket", y="count", color="Dropout_Risk",
                         color_discrete_map=DROPOUT_COLORS, barmode="group",
                         title="Dropout Risk by Income Bracket",
                         category_orders={"Family_Income_Bracket": INCOME_ORDER},
                         labels={"count": "Students"})
            fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(df, x="Family_Income_Bracket", y="Financial_Stress_Score",
                         color="Family_Income_Bracket",
                         title="Financial Stress by Income",
                         category_orders={"Family_Income_Bracket": INCOME_ORDER})
            fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Interactive Data Table ───────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Interactive Data Explorer</div>', unsafe_allow_html=True)

    with st.expander("Click to explore the full dataset", expanded=False):
        # Filter controls
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            dept_filter = st.multiselect(
                "Filter by Department",
                options=df["Department"].dropna().unique().tolist(),
                default=df["Department"].dropna().unique().tolist(),
            )
        with fc2:
            dropout_filter = st.multiselect(
                "Filter by Dropout Risk",
                options=["Yes", "No"],
                default=["Yes", "No"],
            )
        with fc3:
            burnout_filter = st.multiselect(
                "Filter by Burnout Level",
                options=BURNOUT_ORDER,
                default=BURNOUT_ORDER,
            )

        filtered = df[
            df["Department"].isin(dept_filter)
            & df["Dropout_Risk"].isin(dropout_filter)
            & df["Burnout_Level"].isin(burnout_filter)
        ]
        st.caption(f"Showing {len(filtered)} of {len(df)} students")
        st.dataframe(filtered, use_container_width=True, height=400)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2: EDA EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA Explorer":
    st.markdown("# 🔍 Exploratory Data Analysis")
    st.markdown("*Deep-dive into feature distributions and correlations*")

    df = load_data_cleaned()

    # ── Feature Selector ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Feature Distribution Explorer</div>', unsafe_allow_html=True)

    all_features = [c for c in df.columns if c not in ("Student_ID", "Burnout_Level", "Dropout_Risk")]

    selected_feature = st.selectbox(
        "Select a feature to explore:",
        all_features,
        index=all_features.index("Stress_Level") if "Stress_Level" in all_features else 0,
    )

    is_numeric = pd.api.types.is_numeric_dtype(df[selected_feature])

    col_left, col_right = st.columns([3, 2])

    with col_left:
        if is_numeric:
            fig = px.histogram(
                df, x=selected_feature, color="Burnout_Level",
                color_discrete_map=BURNOUT_COLORS,
                barmode="overlay", opacity=0.7,
                title=f"{selected_feature} — Colored by Burnout Level",
                category_orders={"Burnout_Level": BURNOUT_ORDER},
                marginal="box",
            )
        else:
            ct = df.groupby([selected_feature, "Burnout_Level"]).size().reset_index(name="count")
            fig = px.bar(
                ct, x=selected_feature, y="count", color="Burnout_Level",
                color_discrete_map=BURNOUT_COLORS,
                barmode="group",
                title=f"{selected_feature} — By Burnout Level",
                category_orders={"Burnout_Level": BURNOUT_ORDER},
            )
        fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        if is_numeric:
            fig2 = px.histogram(
                df, x=selected_feature, color="Dropout_Risk",
                color_discrete_map=DROPOUT_COLORS,
                barmode="overlay", opacity=0.7,
                title=f"{selected_feature} — Colored by Dropout Risk",
            )
        else:
            ct2 = df.groupby([selected_feature, "Dropout_Risk"]).size().reset_index(name="count")
            fig2 = px.bar(
                ct2, x=selected_feature, y="count", color="Dropout_Risk",
                color_discrete_map=DROPOUT_COLORS,
                barmode="group",
                title=f"{selected_feature} — By Dropout Risk",
            )
        fig2.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=420)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Statistical Summary ──────────────────────────────────────────────
    st.markdown(f'<div class="section-header">Statistical Summary — {selected_feature}</div>', unsafe_allow_html=True)

    if is_numeric:
        tab_stat1, tab_stat2 = st.tabs(["By Burnout Level", "By Dropout Risk"])

        with tab_stat1:
            stats_bl = df.groupby("Burnout_Level")[selected_feature].agg(
                ["count", "mean", "median", "std", "min", "max"]
            ).round(3)
            stats_bl = stats_bl.reindex(BURNOUT_ORDER)
            st.dataframe(stats_bl, use_container_width=True)

        with tab_stat2:
            stats_dr = df.groupby("Dropout_Risk")[selected_feature].agg(
                ["count", "mean", "median", "std", "min", "max"]
            ).round(3)
            st.dataframe(stats_dr, use_container_width=True)
    else:
        ct = pd.crosstab(df[selected_feature], df["Burnout_Level"], normalize="index").round(3) * 100
        ct = ct.reindex(columns=BURNOUT_ORDER, fill_value=0)
        st.markdown(f"**Burnout Level distribution (%) within each {selected_feature}:**")
        st.dataframe(ct.style.format("{:.1f}%").background_gradient(
            cmap="RdYlGn_r", axis=1
        ), use_container_width=True)

    # ── Correlation Matrix ───────────────────────────────────────────────
    st.markdown('<div class="section-header">🔗 Correlation Matrix</div>', unsafe_allow_html=True)

    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        corr = numeric_df.corr()

        fig_corr = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            title="Feature Correlation Matrix",
            labels=dict(color="Correlation"),
            text_auto=".2f",
            aspect="auto",
        )
        _layout = {**PLOTLY_LAYOUT_DEFAULTS, "height": max(500, len(corr.columns) * 35), "font": dict(size=10)}
        fig_corr.update_layout(**_layout)
        fig_corr.update_traces(textfont=dict(size=8))
        st.plotly_chart(fig_corr, use_container_width=True)

        # Top correlations
        st.markdown("**Top Feature Correlations:**")
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                corr_pairs.append({
                    "Feature 1": corr.columns[i],
                    "Feature 2": corr.columns[j],
                    "Correlation": corr.iloc[i, j],
                })
        corr_df = pd.DataFrame(corr_pairs).sort_values("Correlation", key=abs, ascending=False)
        top_corr = pd.concat([
            corr_df.head(10),
            corr_df.tail(10),
        ]).drop_duplicates().head(15)
        fig_top = px.bar(
            top_corr, x="Correlation",
            y=top_corr.apply(lambda r: f"{r['Feature 1']} ↔ {r['Feature 2']}", axis=1),
            orientation="h",
            color="Correlation",
            color_continuous_scale=["#f85149", "#30363d", "#58a6ff"],
            color_continuous_midpoint=0,
            title="Strongest Correlations",
        )
        _layout = {**PLOTLY_LAYOUT_DEFAULTS, "height": 400, "yaxis": dict(autorange="reversed")}
        fig_top.update_layout(**_layout)
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation analysis.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3: EARLY WARNING SYSTEM
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Early Warning System":
    st.markdown("# ⚠️ Early Warning System")
    st.markdown("*Input a student's profile to get instant dropout and burnout risk predictions*")

    df = load_data_cleaned()

    # Check if models exist
    model_files = load_model_files()
    models_available = all(v is not None for v in model_files.values())

    if not models_available:
        st.markdown("""
        <div class="info-box warning">
            ⚠️ <strong>Models not yet trained.</strong> The prediction system requires trained models.<br>
            Run the training pipeline first to generate:<br>
            • <code>models/dropout_model.joblib</code><br>
            • <code>models/burnout_model.joblib</code><br>
            • <code>models/feature_metadata.joblib</code><br><br>
            The form below will still let you explore inputs, but predictions use rule-based estimation.
        </div>
        """, unsafe_allow_html=True)

    # ── Input Form ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📝 Student Profile Input</div>', unsafe_allow_html=True)

    # Store predictions in session state
    if "prediction" not in st.session_state:
        st.session_state.prediction = None

    with st.form("student_form"):
        st.markdown("##### 👤 Demographics & Background")
        dc1, dc2, dc3, dc4 = st.columns(4)
        with dc1:
            input_age = st.slider("Age", **CONTINUOUS_FEATURES["Age"])
        with dc2:
            input_gender = st.selectbox("Gender", CATEGORICAL_FEATURES["Gender"], index=1)
        with dc3:
            input_year = st.slider("Year of Study", **CONTINUOUS_FEATURES["Year_of_Study"])
        with dc4:
            input_dept = st.selectbox("Department", CATEGORICAL_FEATURES["Department"])

        st.markdown("##### 🏠 Living Situation")
        lc1, lc2, lc3, lc4 = st.columns(4)
        with lc1:
            input_residence = st.selectbox("Residence Type", CATEGORICAL_FEATURES["Residence_Type"])
        with lc2:
            input_income = st.selectbox("Family Income", CATEGORICAL_FEATURES["Family_Income_Bracket"],
                                         index=2)
        with lc3:
            input_job = st.selectbox("Part-Time Job", CATEGORICAL_FEATURES["Part_Time_Job"])
        with lc4:
            input_counseling = st.selectbox("Counseling Access", CATEGORICAL_FEATURES["Counseling_Access"])

        st.markdown("##### 📚 Academic Performance")
        ac1, ac2, ac3, ac4 = st.columns(4)
        with ac1:
            input_attendance = st.slider("Attendance %", **CONTINUOUS_FEATURES["Attendance_Percent"])
        with ac2:
            input_study_hrs = st.slider("Study Hours/Day", **CONTINUOUS_FEATURES["Study_Hours_Per_Day"])
        with ac3:
            input_gpa = st.slider("Previous GPA", **CONTINUOUS_FEATURES["Previous_GPA"])
        with ac4:
            input_backlogs = st.slider("Backlogs", **CONTINUOUS_FEATURES["Backlogs"])

        st.markdown("##### 🧠 Well-being & Mental Health")
        wc1, wc2, wc3, wc4 = st.columns(4)
        with wc1:
            input_sleep = st.slider("Sleep Hours", **CONTINUOUS_FEATURES["Sleep_Hours"])
        with wc2:
            input_screen = st.slider("Screen Time (hrs/day)", **CONTINUOUS_FEATURES["Screen_Time_Hours"])
        with wc3:
            input_exercise = st.slider("Exercise (times/week)", **CONTINUOUS_FEATURES["Exercise_Freq_Per_Week"])
        with wc4:
            input_social = st.slider("Social Activity Score", **CONTINUOUS_FEATURES["Social_Activity_Score"])

        st.markdown("##### 💭 Stress & Motivation")
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            input_fin_stress = st.slider("Financial Stress", **CONTINUOUS_FEATURES["Financial_Stress_Score"])
        with sc2:
            input_family_support = st.slider("Family Support", **CONTINUOUS_FEATURES["Family_Support_Score"])
        with sc3:
            input_stress = st.slider("Stress Level", **CONTINUOUS_FEATURES["Stress_Level"])
        with sc4:
            input_anxiety = st.slider("Anxiety Score", **CONTINUOUS_FEATURES["Anxiety_Score"])

        sm1, sm2 = st.columns(2)
        with sm1:
            input_motivation = st.slider("Motivation Score", **CONTINUOUS_FEATURES["Motivation_Score"])
        with sm2:
            input_peer = st.slider("Peer Pressure Score", **CONTINUOUS_FEATURES["Peer_Pressure_Score"])

        submitted = st.form_submit_button(
            "🔮 Predict Risk",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        # Build input dict
        input_data = {
            "Age": input_age,
            "Gender": input_gender,
            "Year_of_Study": input_year,
            "Department": input_dept,
            "Residence_Type": input_residence,
            "Attendance_Percent": input_attendance,
            "Study_Hours_Per_Day": input_study_hrs,
            "Previous_GPA": input_gpa,
            "Backlogs": input_backlogs,
            "Sleep_Hours": input_sleep,
            "Screen_Time_Hours": input_screen,
            "Exercise_Freq_Per_Week": input_exercise,
            "Social_Activity_Score": input_social,
            "Part_Time_Job": input_job,
            "Family_Income_Bracket": input_income,
            "Financial_Stress_Score": input_fin_stress,
            "Family_Support_Score": input_family_support,
            "Stress_Level": input_stress,
            "Anxiety_Score": input_anxiety,
            "Motivation_Score": input_motivation,
            "Peer_Pressure_Score": input_peer,
            "Counseling_Access": input_counseling,
        }

        dropout_prob = None
        burnout_pred = None

        if models_available and model_files.get("dropout_model") is not None and model_files.get("burnout_model") is not None:
            # Use trained models
            try:
                input_df = pd.DataFrame([input_data])

                # Apply same preprocessing as training
                meta = model_files.get("feature_metadata")
                if meta is not None and isinstance(meta, dict):
                    feature_cols = meta.get("feature_columns", None)
                    if feature_cols:
                        # One-hot encode
                        input_encoded = pd.get_dummies(input_df, drop_first=True)
                        # Ensure all columns
                        for col in feature_cols:
                            if col not in input_encoded.columns:
                                input_encoded[col] = 0
                        input_encoded = input_encoded[feature_cols]

                        dropout_prob = model_files["dropout_model"].predict_proba(input_encoded)[0][1]
                        burnout_pred = model_files["burnout_model"].predict(input_encoded)[0]
                else:
                    input_encoded = pd.get_dummies(input_df, drop_first=True)
                    dropout_prob = model_files["dropout_model"].predict_proba(input_encoded)[0][1]
                    burnout_pred = model_files["burnout_model"].predict(input_encoded)[0]
            except Exception as e:
                st.error(f"Model prediction error: {e}")

        # Fallback: rule-based estimation
        if dropout_prob is None:
            # Simple weighted scoring based on feature analysis
            score = 0.0

            # Academic factors (negative GPA impact)
            score += max(0, (7.0 - input_gpa) / 7.0) * 0.20
            score += max(0, (85.0 - input_attendance) / 55.0) * 0.15
            score += min(input_backlogs / 5.0, 1.0) * 0.12

            # Well-being factors
            score += max(0, (10 - input_sleep) / 6.0) * 0.08 if input_sleep < 7 else 0
            score += min(input_screen / 12.0, 1.0) * 0.05
            score += max(0, (4 - input_exercise) / 4.0) * 0.06

            # Psychological factors
            score += min(input_stress / 10.0, 1.0) * 0.15
            score += min(input_anxiety / 10.0, 1.0) * 0.10
            score += max(0, (10 - input_motivation) / 10.0) * 0.09

            # Support factors
            score += max(0, (10 - input_family_support) / 10.0) * 0.05
            score += min(input_fin_stress / 10.0, 1.0) * 0.05

            # Reduce for protective factors
            if input_counseling == "Yes":
                score *= 0.85
            if input_job == "Yes":
                score *= 1.05

            dropout_prob = np.clip(score, 0.05, 0.95)

        if burnout_pred is None:
            # Rule-based burnout estimation
            stress_comp = (input_stress * 0.3 + input_anxiety * 0.25 + input_fin_stress * 0.15 + input_peer * 0.1)
            protective = (input_family_support * 0.1 + input_motivation * 0.05 + input_sleep * 0.03)
            burnout_score = (stress_comp - protective + 3) / 10.0

            if burnout_score > 0.6:
                burnout_pred = "High"
            elif burnout_score > 0.4:
                burnout_pred = "Medium"
            else:
                burnout_pred = "Low"

        # Store in session state
        st.session_state.prediction = {
            "dropout_prob": dropout_prob,
            "burnout_pred": burnout_pred,
            "input_data": input_data,
        }

    # ── Display Predictions ──────────────────────────────────────────────
    if st.session_state.prediction is not None:
        pred = st.session_state.prediction
        dropout_prob = pred["dropout_prob"]
        burnout_pred = pred["burnout_pred"]
        input_data = pred["input_data"]

        st.markdown("---")
        st.markdown('<div class="section-header">🎯 Risk Assessment Results</div>', unsafe_allow_html=True)

        # Determine risk level
        if dropout_prob > 0.6:
            risk_css = "risk-high"
            risk_label = "🔴 HIGH RISK"
            risk_color = "#f85149"
        elif dropout_prob > 0.35:
            risk_css = "risk-medium"
            risk_label = "🟡 MODERATE RISK"
            risk_color = "#d29922"
        else:
            risk_css = "risk-low"
            risk_label = "🟢 LOW RISK"
            risk_color = "#3fb950"

        # Metrics row
        m1, m2, m3 = st.columns(3)
        with m1:
            metric_card("Dropout Probability", f"{dropout_prob:.1%}", risk_label, risk_css)
        with m2:
            burnout_color = {"Low": "#3fb950", "Medium": "#d29922", "High": "#f85149"}.get(burnout_pred, "#58a6ff")
            burnout_css = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}.get(burnout_pred, "")
            metric_card("Predicted Burnout", burnout_pred, f"{'Needs attention' if burnout_pred != 'Low' else 'Well managed'}", burnout_css)
        with m3:
            # Composite risk score
            composite = dropout_prob * 0.6 + {"Low": 0.15, "Medium": 0.5, "High": 0.85}.get(burnout_pred, 0.5) * 0.4
            composite_css = "risk-high" if composite > 0.6 else ("risk-medium" if composite > 0.35 else "risk-low")
            metric_card("Composite Risk", f"{composite:.1%}", "Combined assessment", composite_css)

        st.markdown("")

        # Gauge charts
        g1, g2 = st.columns(2)
        with g1:
            fig_gauge = create_gauge_chart(dropout_prob, "Dropout Risk", color=risk_color)
            st.plotly_chart(fig_gauge, use_container_width=True)
        with g2:
            burnout_val = {"Low": 0.2, "Medium": 0.55, "High": 0.85}.get(burnout_pred, 0.5)
            fig_gauge2 = create_gauge_chart(burnout_val, f"Burnout Level: {burnout_pred}", color=burnout_color)
            st.plotly_chart(fig_gauge2, use_container_width=True)

        # ── Risk Factors Explanation ─────────────────────────────────────
        st.markdown('<div class="section-header">🔍 Risk Factor Analysis</div>', unsafe_allow_html=True)

        risk_up, risk_down = explain_risk_factors(input_data, df)

        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("**🔺 Factors Increasing Risk:**")
            if risk_up:
                for label, explanation in risk_up:
                    st.markdown(
                        f'<div style="padding: 8px 12px; margin: 6px 0; border-radius: 8px; '
                        f'background: rgba(248,81,73,0.08); border-left: 3px solid #f85149;">'
                        f'<strong style="color: #f85149;">{label}</strong><br>'
                        f'<span style="color: #8b949e; font-size: 0.85rem;">{explanation}</span></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div class="info-box success">No major risk factors identified — well done! ✅</div>',
                    unsafe_allow_html=True,
                )

        with rc2:
            st.markdown("**🔻 Protective Factors (Lowering Risk):**")
            if risk_down:
                for label, explanation in risk_down:
                    st.markdown(
                        f'<div style="padding: 8px 12px; margin: 6px 0; border-radius: 8px; '
                        f'background: rgba(63,185,80,0.08); border-left: 3px solid #3fb950;">'
                        f'<strong style="color: #3fb950;">{label}</strong><br>'
                        f'<span style="color: #8b949e; font-size: 0.85rem;">{explanation}</span></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("No strong protective factors detected. Consider addressing the risk factors above.")

        # ── Recommendations ──────────────────────────────────────────────
        st.markdown('<div class="section-header">📋 Personalized Recommendations</div>', unsafe_allow_html=True)

        recs = []
        if input_data["Stress_Level"] > 6:
            recs.append(("🧠 Stress Management", "Consider stress management workshops, mindfulness practice, or counseling sessions."))
        if input_data["Sleep_Hours"] < 6.5:
            recs.append(("😴 Sleep Hygiene", "Aim for 7-8 hours. Maintain a consistent sleep schedule and limit screen time before bed."))
        if input_data["Exercise_Freq_Per_Week"] < 3:
            recs.append(("🏃 Physical Activity", "Aim for 3-4 exercise sessions per week. Even walking can significantly reduce stress."))
        if input_data["Attendance_Percent"] < 80:
            recs.append(("📅 Improve Attendance", "Attendance below 80% is a strong dropout predictor. Set daily reminders and attend consistently."))
        if input_data["Previous_GPA"] < 6:
            recs.append(("📚 Academic Support", "Seek tutoring services, study groups, or academic mentoring programs."))
        if input_data["Anxiety_Score"] > 6:
            recs.append(("💬 Mental Health", "High anxiety detected. Please consider professional counseling support."))
        if input_data["Family_Support_Score"] < 4:
            recs.append(("👨‍👩‍👧 Family Engagement", "Low family support. Consider peer mentoring or campus support networks."))
        if input_data["Motivation_Score"] < 4:
            recs.append(("🔥 Motivation Boost", "Low motivation. Explore career counseling, goal-setting workshops, or peer mentorship."))
        if input_data["Counseling_Access"] == "No":
            recs.append(("🎓 Access Counseling", "Counseling is not currently accessible. Contact student services to improve access."))
        if input_data["Financial_Stress_Score"] > 6:
            recs.append(("💰 Financial Aid", "High financial stress. Explore scholarships, grants, or part-time campus work-study programs."))

        if not recs:
            recs.append(("✅ All Clear", "This student profile shows low risk across all factors. Continue monitoring."))

        for title, desc in recs:
            st.markdown(f"**{title}**: {desc}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4: MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown("# 🤖 Model Performance")
    st.markdown("*Evaluation metrics, confusion matrices, and feature importance for trained models*")

    models = load_model_files()
    shap_data = load_shap_data()
    models_available = all(v is not None for v in models.values())

    if not models_available:
        st.markdown("""
        <div class="info-box warning">
            ⚠️ <strong>Models not yet trained.</strong> Train the models first to view performance metrics.<br>
            Expected files in <code>models/</code>:<br>
            • <code>dropout_model.joblib</code> — Dropout risk classifier<br>
            • <code>burnout_model.joblib</code> — Burnout level classifier<br>
            • <code>feature_metadata.joblib</code> — Feature encoders & metadata<br>
            • <code>model_metrics.json</code> — Stored evaluation metrics<br>
            • <code>shap_importance.csv</code> — SHAP feature importance data
        </div>
        """, unsafe_allow_html=True)

        # Show placeholder metrics for demo
        st.markdown('<div class="section-header">📊 Placeholder Performance Comparison</div>', unsafe_allow_html=True)

        demo_metrics = pd.DataFrame({
            "Model": ["Random Forest", "XGBoost", "Logistic Regression", "SVM"],
            "Accuracy": [0.85, 0.87, 0.78, 0.80],
            "F1-Score": [0.83, 0.86, 0.76, 0.78],
            "Precision": [0.84, 0.85, 0.77, 0.79],
            "Recall": [0.82, 0.87, 0.75, 0.77],
            "AUC-ROC": [0.91, 0.93, 0.84, 0.86],
        })

        fig = px.bar(
            demo_metrics.melt(id_vars="Model", var_name="Metric", value_name="Score"),
            x="Model", y="Score", color="Metric",
            barmode="group",
            title="Placeholder Model Comparison (Train models to see actual metrics)",
            labels={"Score": "Score"},
            color_discrete_sequence=["#58a6ff", "#bc8cff", "#3fb950", "#f0883e", "#d29922"],
        )
        fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=400)
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        # Placeholder confusion matrices
        st.markdown('<div class="section-header">🎯 Confusion Matrices (Placeholder)</div>', unsafe_allow_html=True)

        cols = st.columns(2)
        for idx, (model_name, acc) in enumerate([("Dropout Classifier", 0.87), ("Burnout Classifier", 0.83)]):
            with cols[idx]:
                # Generate placeholder confusion matrix
                n = 100
                tp = int(acc * n * 0.5)
                tn = int(acc * n * 0.5)
                fn = n - tp
                fp = n - tn

                if "Burnout" in model_name:
                    labels = ["Low/Med", "High"]
                    cm = np.array([[tp + 20, fn - 20], [fp + 5, tn - 5]])
                    cm = np.clip(cm, 0, None)
                else:
                    labels = ["No", "Yes"]
                    cm = np.array([[tp, fn], [fp, tn]])

                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    x=labels, y=labels,
                    title=f"{model_name} — Confusion Matrix",
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                )
                fig_cm.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=320)
                st.plotly_chart(fig_cm, use_container_width=True)

    else:
        # Load and display actual metrics
        import json

        metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")

        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                all_metrics = json.load(f)

            st.markdown('<div class="section-header">📊 Model Comparison</div>', unsafe_allow_html=True)

            # Build a flat DataFrame from the nested JSON
            rows = []
            for task_key, task_label in [("binary", "Dropout Risk"), ("multiclass", "Burnout Level")]:
                if task_key in all_metrics and "models" in all_metrics[task_key]:
                    for model_name, scores in all_metrics[task_key]["models"].items():
                        row = {"Model": model_name, "Target": task_label}
                        row.update(scores)
                        rows.append(row)

            if rows:
                metrics_df = pd.DataFrame(rows)
                metric_cols = [c for c in metrics_df.columns if c not in ("Model", "Target")]

                # ── Charts: one grouped bar per task ──
                palette = ["#58a6ff", "#bc8cff", "#3fb950", "#f0883e"]
                for task_label, group in metrics_df.groupby("Target"):
                    fig = px.bar(
                        group.melt(id_vars=["Model", "Target"], value_vars=metric_cols,
                                   var_name="Metric", value_name="Score"),
                        x="Model", y="Score", color="Metric",
                        barmode="group",
                        title=f"Model Performance — {task_label}",
                        color_discrete_sequence=palette,
                    )
                    fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=400)
                    fig.update_yaxes(range=[0, 1.05])
                    st.plotly_chart(fig, use_container_width=True)

                # ── Summary table ──
                fmt = {c: "{:.4f}" for c in metric_cols}
                st.dataframe(
                    metrics_df.style.format(fmt).background_gradient(
                        subset=metric_cols, cmap="RdYlGn", vmin=0, vmax=1
                    ),
                    use_container_width=True,
                )

                # ── Best-model callouts ──
                best_bin = all_metrics.get("best_binary_model", "")
                best_mc  = all_metrics.get("best_multiclass_model", "")
                if best_bin or best_mc:
                    cols = st.columns(2)
                    if best_bin:
                        cols[0].success(f"🏆 Best Dropout Risk model: **{best_bin}**")
                    if best_mc:
                        cols[1].success(f"🏆 Best Burnout Level model: **{best_mc}**")
            else:
                st.warning("model_metrics.json is empty or has unexpected structure.")
        else:
            st.info("No model_metrics.json found. Showing trained model summary...")

            # Try to extract basic info from the model objects
            for name, model in models.items():
                if model is not None:
                    st.markdown(f"**{name}**: `{type(model).__name__}` loaded successfully ✅")

    # ── SHAP Feature Importance ──────────────────────────────────────────
    st.markdown('<div class="section-header">🔬 Feature Importance</div>', unsafe_allow_html=True)

    if shap_data is not None:
        # Real SHAP data
        st.markdown("*SHAP feature importance — values show impact on model predictions*")

        if "feature" in shap_data.columns and "importance" in shap_data.columns:
            shap_sorted = shap_data.sort_values("importance", ascending=True).tail(20)
            fig_shap = px.bar(
                shap_sorted, x="importance", y="feature",
                orientation="h",
                title="Top 20 Features by SHAP Importance",
                color="importance",
                color_continuous_scale="viridis",
            )
            fig_shap.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=max(400, len(shap_sorted) * 25))
            st.plotly_chart(fig_shap, use_container_width=True)

            # Feature importance table
            st.markdown("**Feature Importance Ranking:**")
            ranking = shap_data.sort_values("importance", ascending=False).reset_index(drop=True)
            ranking.index += 1
            ranking.columns = ["Feature", "Importance"] if len(ranking.columns) == 2 else ranking.columns
            st.dataframe(ranking, use_container_width=True)
        else:
            st.dataframe(shap_data, use_container_width=True)
    else:
        # Placeholder importance based on domain knowledge
        st.markdown("*Based on domain analysis — train SHAP explainer for model-specific values*")

        placeholder_importance = pd.DataFrame({
            "Feature": [
                "Stress_Level", "GPA", "Anxiety_Score", "Attendance",
                "Sleep_Hours", "Motivation_Score", "Financial_Stress",
                "Study_Hours", "Backlogs", "Screen_Time",
                "Exercise_Freq", "Family_Support", "Peer_Pressure",
                "Social_Activity", "Age",
            ],
            "Importance": [0.18, 0.15, 0.13, 0.12, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02, 0.01, 0.01],
        })

        fig_imp = px.bar(
            placeholder_importance.sort_values("Importance", ascending=True),
            x="Importance", y="Feature",
            orientation="h",
            title="Estimated Feature Importance (Domain-based)",
            color="Importance",
            color_continuous_scale="viridis",
        )
        fig_imp.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=500)
        st.plotly_chart(fig_imp, use_container_width=True)

        st.caption("⚠️ This is a placeholder. Train models with SHAP explainer for accurate importance scores.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5: INSIGHTS & RECOMMENDATIONS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "💡 Insights & Recommendations":
    st.markdown("# 💡 Insights & Recommendations")
    st.markdown("*Data-driven findings about student burnout and dropout risk*")

    df = load_data_cleaned()

    # ── Key Findings ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔑 Key Findings</div>', unsafe_allow_html=True)

    dropout_yes = df[df["Dropout_Risk"] == "Yes"]
    dropout_no = df[df["Dropout_Risk"] == "No"]
    high_burnout = df[df["Burnout_Level"] == "High"]
    low_burnout = df[df["Burnout_Level"] == "Low"]

    # Compute key stat differences
    stats_comparisons = pd.DataFrame({
        "Metric": ["Avg GPA", "Avg Attendance %", "Avg Stress", "Avg Anxiety",
                    "Avg Sleep Hours", "Avg Motivation", "Avg Backlogs"],
        "High Risk (Dropout=Yes)": [
            dropout_yes["Previous_GPA"].mean(),
            dropout_yes["Attendance_Percent"].mean(),
            dropout_yes["Stress_Level"].mean(),
            dropout_yes["Anxiety_Score"].mean(),
            dropout_yes["Sleep_Hours"].mean(),
            dropout_yes["Motivation_Score"].mean(),
            dropout_yes["Backlogs"].mean(),
        ],
        "Low Risk (Dropout=No)": [
            dropout_no["Previous_GPA"].mean(),
            dropout_no["Attendance_Percent"].mean(),
            dropout_no["Stress_Level"].mean(),
            dropout_no["Anxiety_Score"].mean(),
            dropout_no["Sleep_Hours"].mean(),
            dropout_no["Motivation_Score"].mean(),
            dropout_no["Backlogs"].mean(),
        ],
    }).round(2)

    stats_comparisons["Difference"] = (
        stats_comparisons["High Risk (Dropout=Yes)"] - stats_comparisons["Low Risk (Dropout=No)"]
    ).round(2)

    # Key insight cards
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        dropout_rate = len(dropout_yes) / len(df)
        metric_card("Dropout Rate", f"{dropout_rate:.1%}", f"{len(dropout_yes)} of {len(df)} students", "risk-high" if dropout_rate > 0.5 else "risk-medium")
    with ic2:
        high_bo = len(high_burnout) / len(df)
        metric_card("High Burnout Rate", f"{high_bo:.1%}", f"{len(high_burnout)} students", "risk-high" if high_bo > 0.3 else "risk-medium")
    with ic3:
        gpa_diff = stats_comparisons.loc[0, "Difference"]
        metric_card("GPA Gap", f"{abs(gpa_diff):.2f}", "Between high/low risk students")

    st.markdown("")

    st.markdown("""
    <div class="info-box">
        <strong>📌 Top Insight:</strong> Students at dropout risk show significantly lower GPA, attendance,
        and motivation, alongside higher stress, anxiety, and financial stress.
        These factors compound — the strongest predictor combinations involve multiple risk factors occurring together.
    </div>
    """, unsafe_allow_html=True)

    # ── Comparison Table ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Risk Group Comparison</div>', unsafe_allow_html=True)
    st.dataframe(stats_comparisons, use_container_width=True)

    # ── Department Comparison ─────────────────────────────────────────────
    st.markdown('<div class="section-header">🏛️ Department Comparison</div>', unsafe_allow_html=True)

    dept_stats = df.groupby("Department").agg(
        Students=("Student_ID", "count"),
        Dropout_Rate=("Dropout_Risk", lambda x: (x == "Yes").mean()),
        High_Burnout_Rate=("Burnout_Level", lambda x: (x == "High").mean()),
        Avg_GPA=("Previous_GPA", "mean"),
        Avg_Stress=("Stress_Level", "mean"),
        Avg_Attendance=("Attendance_Percent", "mean"),
        Avg_Motivation=("Motivation_Score", "mean"),
    ).round(3).sort_values("Dropout_Rate", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            dept_stats.reset_index(),
            x="Department", y=["Dropout_Rate", "High_Burnout_Rate"],
            barmode="group",
            title="Dropout & High Burnout Rate by Department",
            labels={"value": "Rate", "variable": "Metric"},
            color_discrete_map={"Dropout_Rate": "#f85149", "High_Burnout_Rate": "#d29922"},
            category_orders={"Department": DEPT_ORDER},
        )
        fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=400, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            dept_stats.reset_index(),
            x="Department", y=["Avg_GPA", "Avg_Stress", "Avg_Motivation"],
            barmode="group",
            title="GPA, Stress & Motivation by Department",
            labels={"value": "Score", "variable": "Metric"},
            color_discrete_map={"Avg_GPA": "#58a6ff", "Avg_Stress": "#f85149", "Avg_Motivation": "#3fb950"},
            category_orders={"Department": DEPT_ORDER},
        )
        fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        dept_stats.style.format({
            "Dropout_Rate": "{:.1%}",
            "High_Burnout_Rate": "{:.1%}",
            "Avg_GPA": "{:.2f}",
            "Avg_Stress": "{:.2f}",
            "Avg_Attendance": "{:.1f}%",
            "Avg_Motivation": "{:.2f}",
        }),
        use_container_width=True,
    )

    # ── Risk Factor Breakdown by Income ──────────────────────────────────
    st.markdown('<div class="section-header">💰 Risk Factor Breakdown by Income Bracket</div>', unsafe_allow_html=True)

    income_stats = df.groupby("Family_Income_Bracket").agg(
        Students=("Student_ID", "count"),
        Dropout_Rate=("Dropout_Risk", lambda x: (x == "Yes").mean()),
        Avg_Financial_Stress=("Financial_Stress_Score", "mean"),
        Avg_Stress=("Stress_Level", "mean"),
        Avg_GPA=("Previous_GPA", "mean"),
        High_Burnout=("Burnout_Level", lambda x: (x == "High").mean()),
    ).round(3)
    income_stats = income_stats.reindex(INCOME_ORDER).dropna(how="all")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            income_stats.reset_index(),
            x="Family_Income_Bracket", y=["Dropout_Rate", "High_Burnout"],
            barmode="group",
            title="Dropout & Burnout Risk by Income",
            labels={"value": "Rate", "variable": "Risk Type"},
            color_discrete_map={"Dropout_Rate": "#f85149", "High_Burnout": "#d29922"},
            category_orders={"Family_Income_Bracket": INCOME_ORDER},
        )
        fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=380, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(
            income_stats.reset_index(),
            x="Family_Income_Bracket", y=["Avg_Financial_Stress", "Avg_Stress"],
            markers=True,
            title="Stress Levels by Income Bracket",
            labels={"value": "Score", "variable": "Metric"},
            color_discrete_map={"Avg_Financial_Stress": "#f0883e", "Avg_Stress": "#bc8cff"},
            category_orders={"Family_Income_Bracket": INCOME_ORDER},
        )
        fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=380)
        st.plotly_chart(fig, use_container_width=True)

    # ── Burnout-Dropout Cross Analysis ───────────────────────────────────
    st.markdown('<div class="section-header">🔗 Burnout ↔ Dropout Cross Analysis</div>', unsafe_allow_html=True)

    cross = pd.crosstab(df["Burnout_Level"], df["Dropout_Risk"], normalize="index") * 100
    cross = cross.reindex(BURNOUT_ORDER)

    fig_cross = px.bar(
        cross.reset_index().melt(id_vars="Burnout_Level", var_name="Dropout_Risk", value_name="Percentage"),
        x="Burnout_Level", y="Percentage", color="Dropout_Risk",
        color_discrete_map=DROPOUT_COLORS,
        barmode="stack",
        title="Dropout Risk Within Each Burnout Level (%)",
        category_orders={"Burnout_Level": BURNOUT_ORDER},
    )
    _layout = {**PLOTLY_LAYOUT_DEFAULTS, "height": 380, "yaxis": dict(title="Percentage", ticksuffix="%")}
    fig_cross.update_layout(**_layout)
    st.plotly_chart(fig_cross, use_container_width=True)

    # ── Recommended Interventions ────────────────────────────────────────
    st.markdown('<div class="section-header">🎓 Recommended Interventions</div>', unsafe_allow_html=True)

    interventions = [
        {
            "title": "🎯 Targeted Academic Support",
            "description": "Identify students with GPA < 6 and backlogs > 2 early. Provide tutoring, study skills workshops, and peer mentoring programs.",
            "impact": "High",
            "effort": "Medium",
            "target_group": f"~{len(df[(df['Previous_GPA'] < 6) | (df['Backlogs'] > 2)])} students at risk",
        },
        {
            "title": "🧠 Mental Health & Counseling Expansion",
            "description": f"Students with high stress (>7) and anxiety (>6) need proactive outreach. Only {df['Counseling_Access'].value_counts().get('Yes', 0)/len(df)*100:.0f}% currently have counseling access.",
            "impact": "High",
            "effort": "High",
            "target_group": f"~{len(df[(df['Stress_Level'] > 7) | (df['Anxiety_Score'] > 6)])} students with elevated stress/anxiety",
        },
        {
            "title": "😴 Sleep & Wellness Programs",
            "description": "Students sleeping < 6 hours are significantly more likely to experience burnout. Launch campus sleep hygiene campaigns and wellness challenges.",
            "impact": "Medium",
            "effort": "Low",
            "target_group": f"~{len(df[df['Sleep_Hours'] < 6])} students sleeping < 6 hrs",
        },
        {
            "title": "💰 Financial Aid Enhancement",
            "description": "Students in Low/Lower-Middle income brackets show higher financial stress and dropout rates. Expand scholarship and work-study programs.",
            "impact": "High",
            "effort": "High",
            "target_group": f"~{len(df[df['Family_Income_Bracket'].isin(['Low', 'Lower-Middle'])])} students in lower income brackets",
        },
        {
            "title": "📅 Attendance Monitoring System",
            "description": "Attendance below 80% is one of the strongest dropout predictors. Implement early-warning alerts when attendance drops below thresholds.",
            "impact": "High",
            "effort": "Low",
            "target_group": f"~{len(df[df['Attendance_Percent'] < 80])} students with < 80% attendance",
        },
        {
            "title": "💪 Motivation & Engagement Programs",
            "description": "Low motivation scores correlate strongly with both burnout and dropout. Career counseling, goal-setting workshops, and peer communities can help.",
            "impact": "Medium",
            "effort": "Medium",
            "target_group": f"~{len(df[df['Motivation_Score'] < 4])} students with low motivation",
        },
    ]

    for interv in interventions:
        impact_color = "#f85149" if interv["impact"] == "High" else ("#d29922" if interv["impact"] == "Medium" else "#3fb950")
        effort_color = "#3fb950" if interv["effort"] == "Low" else ("#d29922" if interv["effort"] == "Medium" else "#f85149")

        st.markdown(f"""
        <div style="padding: 16px 20px; margin: 10px 0; border-radius: 10px;
                    background: #161b22; border: 1px solid #30363d;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <strong style="color: #f0f6fc; font-size: 1.05rem;">{interv['title']}</strong>
                <div>
                    <span class="risk-chip" style="background: {impact_color}22; color: {impact_color}; border-color: {impact_color}44;">
                        Impact: {interv['impact']}
                    </span>
                    <span class="risk-chip" style="background: {effort_color}22; color: {effort_color}; border-color: {effort_color}44;">
                        Effort: {interv['effort']}
                    </span>
                </div>
            </div>
            <p style="color: #c9d1d9; margin: 6px 0; font-size: 0.9rem;">{interv['description']}</p>
            <p style="color: #8b949e; margin: 0; font-size: 0.8rem;">📊 Target: {interv['target_group']}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Closing Summary ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 24px; border-radius: 12px; background: linear-gradient(135deg, #1c2333, #161b22); border: 1px solid #30363d;">
        <h3 style="color: #f0f6fc; margin-bottom: 12px;">📌 Summary</h3>
        <p style="color: #c9d1d9; max-width: 700px; margin: 0 auto; line-height: 1.6;">
            This analysis of <strong>800 students</strong> reveals that burnout and dropout risk are deeply
            interconnected. The most impactful interventions combine <strong>academic support</strong>,
            <strong>mental health resources</strong>, and <strong>financial aid</strong> — addressing
            the root causes rather than symptoms alone. Early identification through the
            <strong>Early Warning System</strong> (Page 3) can help institutions intervene before
            crisis points are reached.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#8b949e; font-size:0.75rem; padding: 8px;'>"
    "🎓 Student Burnout & Dropout Risk Dashboard | Built with Streamlit & Plotly | "
    "800 Students × 25 Features"
    "</div>",
    unsafe_allow_html=True,
)
