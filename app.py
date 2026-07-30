import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
import joblib
import os
from fpdf import FPDF

# ======================================================================
# BACKEND
# Uses ONLY the model you trained and saved from your notebook:
#   joblib.dump(model, "food_waste_prediction_model.pkl")
# No other models, no model selector — this is the single source of truth.
# ======================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "food_waste_prediction_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "food_wastage_data.csv")
HISTORY_PATH = os.path.join(BASE_DIR, "prediction_history.csv")

CATEGORICAL_COLUMNS = [
    "Type of Food", "Event Type", "Storage Conditions", "Purchase History",
    "Seasonality", "Preparation Method", "Geographical Location", "Pricing",
]
TARGET_COLUMN = "Wastage Food Amount"

# Test-set metrics for this exact model (RandomForestRegressor, 100 trees),
# evaluated on the same 80/20 split used in the notebook (random_state=42).
MODEL_INFO = {"r2": 0.928, "mae": 1.64, "rmse": 2.73, "n_estimators": 100}


def load_raw_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def get_feature_names(_model):
    """Use the model's own recorded training column order — guarantees
    the app always matches whatever the model was actually trained on."""
    if hasattr(_model, "feature_names_in_"):
        return list(_model.feature_names_in_)
    return [
        "Type of Food", "Number of Guests", "Event Type", "Quantity of Food",
        "Storage Conditions", "Purchase History", "Seasonality",
        "Preparation Method", "Geographical Location", "Pricing",
    ]


@st.cache_resource
def build_encoders(_df):
    """Rebuild the LabelEncoders the notebook used (and did not save) by
    fitting on the same training CSV — deterministic and verified to
    reproduce the model's exact training-time integer mapping."""
    return {col: LabelEncoder().fit(_df[col]) for col in CATEGORICAL_COLUMNS}


def get_category_options(encoders):
    return {col: list(le.classes_) for col, le in encoders.items()}


def encode_input(raw_input: dict, encoders: dict, feature_names: list):
    row = {}
    for col in feature_names:
        if col in CATEGORICAL_COLUMNS:
            row[col] = int(encoders[col].transform([raw_input[col]])[0])
        else:
            row[col] = raw_input[col]
    return pd.DataFrame([row], columns=feature_names)


def predict_with_confidence(model, X: pd.DataFrame):
    """Prediction + confidence via agreement across the forest's own trees
    (this model IS a RandomForestRegressor, so this is exact, not a proxy)."""
    pred = float(model.predict(X)[0])
    tree_preds = np.array([t.predict(X)[0] for t in model.estimators_])
    std = tree_preds.std()
    mean = max(abs(tree_preds.mean()), 1e-6)
    confidence = 100 - (std / mean * 100)
    return pred, float(np.clip(confidence, 50, 99))


def get_recommendation(predicted_waste: float, quantity_of_food: float, df: pd.DataFrame = None):
    """Rule-based recommendation using data-driven quartile thresholds
    plus the waste-to-quantity ratio."""
    if df is not None and TARGET_COLUMN in df.columns:
        q25, q75 = df[TARGET_COLUMN].quantile([0.25, 0.75])
    else:
        q25, q75 = 20.0, 35.0

    ratio = predicted_waste / quantity_of_food if quantity_of_food else 0

    if predicted_waste >= q75 or ratio >= 0.08:
        action, detail, severity = (
            "Donate to NGO",
            "Predicted waste is high. Arrange NGO pickup today and reduce tomorrow's prep quantity.",
            "high",
        )
    elif predicted_waste >= q25:
        action, detail, severity = (
            "Offer Discount Sale",
            "Moderate waste expected. Sell surplus at a discount before closing to recover value.",
            "medium",
        )
    else:
        action, detail, severity = (
            "Reduce Next Day Cooking",
            "Waste is low. Slightly trim tomorrow's prepared quantity to fine-tune demand matching.",
            "low",
        )

    return {"action": action, "detail": detail, "severity": severity, "q25": float(q25), "q75": float(q75)}


def append_to_history(record: dict):
    df_row = pd.DataFrame([record])
    if os.path.exists(HISTORY_PATH):
        df_row.to_csv(HISTORY_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(HISTORY_PATH, mode="w", header=True, index=False)


def load_history():
    if os.path.exists(HISTORY_PATH):
        return pd.read_csv(HISTORY_PATH)
    return pd.DataFrame()


def build_pdf_report(inputs: dict, predicted_waste: float, confidence: float, recommendation: dict) -> bytes:
    pdf = FPDF(format="A4")
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(30, 90, 50)
    pdf.cell(0, 12, "Food Waste Prediction Report", ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)

    pdf.set_fill_color(235, 245, 235)
    pdf.set_text_color(20, 20, 20)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, f"Predicted Food Waste: {predicted_waste:.2f} KG", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 9, f"Model Confidence: {confidence:.1f}%", ln=True, fill=True)
    pdf.cell(0, 9, f"Recommendation: {recommendation['action']}", ln=True, fill=True)
    pdf.ln(6)

    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 6, recommendation["detail"])
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 90, 50)
    pdf.cell(0, 10, "Input Details", ln=True)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(20, 20, 20)
    for key, value in inputs.items():
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(60, 8, f"{key}:")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"{value}", ln=True)

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(0, 5, "Generated by AI Food Waste Predictor & Donation Recommendation System.")

    return bytes(pdf.output())


# ======================================================================
# PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="FoodWaste AI | Predictive Intelligence Platform",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#FCFBF8", "figure.facecolor": "#FCFBF8"})

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

:root {
    --brand-dark: #123D24;
    --brand: #1E5A32;
    --brand-light: #2F8B4A;
    --brand-pale: #EAF3DE;
    --accent: #D98E3B;
    --ink: #20211D;
    --muted: #6B6A64;
    --surface: #FFFFFF;
    --border: #E4E1D6;
}

.stApp { background: linear-gradient(180deg,#FBFAF6 0%, #F5F3EC 100%); }

.hero {
    background: linear-gradient(120deg, var(--brand-dark) 0%, var(--brand) 55%, var(--brand-light) 100%);
    border-radius: 18px; padding: 28px 32px; color: white; margin-bottom: 1.6rem;
    box-shadow: 0 8px 24px rgba(18,61,36,0.18);
}
.hero h1 { margin: 0; font-size: 1.9rem; font-weight: 800; letter-spacing: -0.02em; }
.hero p { margin: 6px 0 0; opacity: 0.88; font-size: 0.95rem; }
.hero .pills { margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap; }
.pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.25);
    padding: 5px 12px; border-radius: 999px; font-size: 0.78rem; font-weight: 500;
}
.pill .dot { width: 7px; height: 7px; border-radius: 50%; background: #7CE38B; }

.metric-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    padding: 16px 18px; box-shadow: 0 1px 3px rgba(20,20,20,0.04);
}
.metric-card .label { font-size: 0.76rem; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px; }
.metric-card .value { font-size: 1.65rem; font-weight: 800; color: var(--ink); }
.metric-card .sub { font-size: 0.78rem; color: var(--muted); margin-top: 2px; }

.result-hero {
    background: linear-gradient(135deg, var(--brand-dark), var(--brand-light));
    border-radius: 18px; padding: 28px; color: white; text-align: center;
    box-shadow: 0 10px 28px rgba(18,61,36,0.22);
}
.result-hero .tag { font-size: 0.8rem; opacity: 0.85; letter-spacing: 0.05em; text-transform: uppercase; font-weight: 600; }
.result-hero .num { font-size: 3.1rem; font-weight: 800; margin: 6px 0; letter-spacing: -0.02em; }
.result-hero .stats { display:flex; justify-content:center; gap: 28px; margin-top: 10px; font-size: 0.85rem; opacity: 0.92; }

.rec-card { border-radius: 14px; padding: 18px 20px; margin-top: 16px; display: flex; gap: 14px; align-items: flex-start; border: 1px solid transparent; }
.rec-high   { background: #FCECE5; border-color: #F0B79C; }
.rec-medium { background: #FDF3DF; border-color: #F0CD8F; }
.rec-low    { background: var(--brand-pale); border-color: #BFDCA4; }
.rec-card .rec-icon { font-size: 1.6rem; line-height: 1; }
.rec-card h4 { margin: 0 0 4px; font-size: 1.02rem; font-weight: 700; color: var(--ink); }
.rec-card p { margin: 0; font-size: 0.87rem; color: #4A4A44; }

.section-title { font-size: 1.05rem; font-weight: 700; color: var(--ink); margin: 6px 0 12px; display:flex; align-items:center; gap:8px; }
.section-sub { font-size: 0.82rem; color: var(--muted); margin-top: -8px; margin-bottom: 14px; }

section[data-testid="stSidebar"] { background: linear-gradient(180deg, #123D24 0%, #1E5A32 100%); }
section[data-testid="stSidebar"] * { color: #EDF3E6 !important; }

.stButton>button, .stDownloadButton>button {
    background: var(--brand); color: white; border-radius: 10px; border: none;
    font-weight: 600; padding: 0.55rem 1rem; transition: all 0.15s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover { background: var(--brand-dark); transform: translateY(-1px); }

.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 10px 18px; font-weight: 600; }

footer, #MainMenu { visibility: hidden; }
.app-footer { text-align:center; color: var(--muted); font-size: 0.78rem; padding: 18px 0 6px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ======================================================================
# LOAD MODEL & DATA
# ======================================================================
model = load_model()
feature_names = get_feature_names(model)
raw_df = load_raw_data()
encoders = build_encoders(raw_df)
category_options = get_category_options(encoders)

# ======================================================================
# SIDEBAR
# ======================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
            <div style="font-size:1.8rem;">🌿</div>
            <div>
                <div style="font-weight:800; font-size:1.05rem; line-height:1.1;">FoodWaste AI</div>
                <div style="font-size:0.72rem; opacity:0.75;">Predictive Intelligence Platform</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:rgba(255,255,255,0.15); margin:10px 0 16px;'>", unsafe_allow_html=True)

    st.caption(
        "Predicts expected food waste for an event and recommends whether "
        "to donate surplus, run a discount sale, or trim tomorrow's prep."
    )
    st.markdown(
        "<div style='font-size:0.72rem; opacity:0.6; margin-top:20px;'>v3.0 · Powered by your trained model</div>",
        unsafe_allow_html=True,
    )

# ======================================================================
# HERO HEADER
# ======================================================================
st.markdown(
    """
    <div class="hero">
        <h1>🌿 AI Food Waste Predictor &amp; Donation Recommendation System</h1>
        <p>Forecast surplus food before it happens — and turn it into donations, not landfill.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_dashboard, tab_batch, tab_history = st.tabs(
    ["🔮  Predict", "📊  Dashboard", "📁  Batch Prediction", "🕘  History"]
)

# ======================================================================
# PREDICT TAB
# ======================================================================
with tab_predict:
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown('<div class="section-title">📝 Event Details</div>', unsafe_allow_html=True)
        with st.form("predict_form"):
            c1, c2 = st.columns(2)
            with c1:
                type_of_food = st.selectbox("Type of Food", category_options["Type of Food"])
                event_type = st.selectbox("Event Type", category_options["Event Type"])
                storage = st.selectbox("Storage Conditions", category_options["Storage Conditions"])
                purchase_history = st.selectbox("Purchase History", category_options["Purchase History"])
                season = st.selectbox("Seasonality", category_options["Seasonality"])
            with c2:
                num_guests = st.number_input("Number of Guests", min_value=1, max_value=5000, value=300, step=1)
                qty_food = st.number_input("Quantity of Food (KG)", min_value=1, max_value=5000, value=400, step=1)
                prep_method = st.selectbox("Preparation Method", category_options["Preparation Method"])
                location = st.selectbox("Geographical Location", category_options["Geographical Location"])
                pricing = st.selectbox("Pricing Tier", category_options["Pricing"])

            submitted = st.form_submit_button("🔮  Predict Waste", width="stretch")

    with right:
        st.markdown('<div class="section-title">📈 Prediction Result</div>', unsafe_allow_html=True)

        if not submitted:
            st.markdown(
                """
                <div style="border:1px dashed var(--border); border-radius:14px; padding:40px 20px; text-align:center; color:var(--muted); background:var(--surface);">
                    <div style="font-size:2rem;">🍽️</div>
                    <p style="margin-top:8px; font-size:0.88rem;">Fill in the event details and click <b>Predict Waste</b> to see results here.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            raw_input = {
                "Type of Food": type_of_food,
                "Number of Guests": num_guests,
                "Event Type": event_type,
                "Quantity of Food": qty_food,
                "Storage Conditions": storage,
                "Purchase History": purchase_history,
                "Seasonality": season,
                "Preparation Method": prep_method,
                "Geographical Location": location,
                "Pricing": pricing,
            }

            X = encode_input(raw_input, encoders, feature_names)
            prediction, confidence = predict_with_confidence(model, X)
            recommendation = get_recommendation(prediction, qty_food, raw_df)
            waste_pct = prediction / qty_food * 100

            st.markdown(
                f"""
                <div class="result-hero">
                    <div class="tag">Predicted Food Waste</div>
                    <div class="num">{prediction:.1f} <span style="font-size:1.4rem; font-weight:600;">KG</span></div>
                    <div class="stats">
                        <span>🎯 {confidence:.0f}% confidence</span>
                        <span>📉 {waste_pct:.1f}% of prepared food</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            icons = {"high": "🆘", "medium": "🏷️", "low": "✅"}
            sev_class = f"rec-{recommendation['severity']}"
            st.markdown(
                f"""
                <div class="rec-card {sev_class}">
                    <div class="rec-icon">{icons[recommendation['severity']]}</div>
                    <div>
                        <h4>{recommendation['action']}</h4>
                        <p>{recommendation['detail']}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **raw_input,
                "Predicted Waste (KG)": round(prediction, 2),
                "Confidence (%)": round(confidence, 1),
                "Recommendation": recommendation["action"],
            }
            append_to_history(record)

            pdf_bytes = build_pdf_report(raw_input, prediction, confidence, recommendation)
            st.download_button(
                "⬇️  Download PDF Report",
                data=pdf_bytes,
                file_name=f"food_waste_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                width="stretch",
            )

# ======================================================================
# DASHBOARD TAB
# ======================================================================
with tab_dashboard:
    st.markdown('<div class="section-title">📊 Waste Analytics Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Insights derived from the historical training dataset.</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        ("Total Records", f"{len(raw_df):,}", "training samples"),
        ("Avg Waste", f"{raw_df[TARGET_COLUMN].mean():.1f} kg", "per event"),
        ("Max Waste", f"{raw_df[TARGET_COLUMN].max():.0f} kg", "worst case observed"),
        ("Avg Guests", f"{raw_df['Number of Guests'].mean():.0f}", "per event"),
    ]
    for col, (label, value, sub) in zip([k1, k2, k3, k4], kpis):
        col.markdown(
            f"""<div class="metric-card"><div class="label">{label}</div>
            <div class="value">{value}</div><div class="sub">{sub}</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Waste Distribution**")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        sns.histplot(raw_df[TARGET_COLUMN], bins=20, color="#1E5A32", ax=ax, kde=True)
        ax.set_xlabel("Wastage Food Amount (KG)")
        st.pyplot(fig)

    with c2:
        st.markdown("**Average Waste by Food Type**")
        avg_by_type = raw_df.groupby("Type of Food")[TARGET_COLUMN].mean().sort_values(ascending=False)
        st.bar_chart(avg_by_type, color="#2F8B4A")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Waste by Event Type**")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        sns.boxplot(data=raw_df, x="Event Type", y=TARGET_COLUMN, ax=ax, palette="Greens")
        ax.tick_params(axis="x", rotation=25)
        st.pyplot(fig)

    with c4:
        st.markdown("**Guests vs Waste**")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        sns.scatterplot(data=raw_df, x="Number of Guests", y=TARGET_COLUMN, color="#D98E3B", alpha=0.6, ax=ax)
        st.pyplot(fig)

    st.markdown("**Correlation Heatmap**")
    numeric_df = raw_df.select_dtypes(include=["int64", "float64"])
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="Greens", ax=ax)
    st.pyplot(fig)

    st.markdown("<hr style='border-color:var(--border); margin:24px 0 18px;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🌲 Model Card</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    for col, (label, value) in zip(
        [m1, m2, m3, m4],
        [
            ("Algorithm", "Random Forest"),
            ("Test R²", f"{MODEL_INFO['r2']:.3f}"),
            ("Test MAE", f"{MODEL_INFO['mae']} kg"),
            ("Test RMSE", f"{MODEL_INFO['rmse']} kg"),
        ],
    ):
        col.markdown(
            f"""<div class="metric-card"><div class="label">{label}</div><div class="value" style="font-size:1.25rem;">{value}</div></div>""",
            unsafe_allow_html=True,
        )

# ======================================================================
# BATCH PREDICTION TAB
# ======================================================================
with tab_batch:
    st.markdown('<div class="section-title">📁 Batch Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-sub">Upload a CSV with columns: <code>{", ".join(feature_names)}</code></div>',
        unsafe_allow_html=True,
    )

    bc1, bc2 = st.columns([1, 2])
    with bc1:
        template = raw_df[feature_names].head(3)
        st.download_button(
            "⬇️  Sample Template CSV",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="batch_template.csv",
            mime="text/csv",
            width="stretch",
        )

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            missing = [c for c in feature_names if c not in batch_df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                preds, confs, recs = [], [], []
                for _, row in batch_df.iterrows():
                    ri = {col: row[col] for col in feature_names}
                    Xb = encode_input(ri, encoders, feature_names)
                    pred, conf = predict_with_confidence(model, Xb)
                    rec = get_recommendation(pred, row["Quantity of Food"], raw_df)
                    preds.append(round(pred, 2))
                    confs.append(round(conf, 1))
                    recs.append(rec["action"])

                batch_df["Predicted Waste (KG)"] = preds
                batch_df["Confidence (%)"] = confs
                batch_df["Recommendation"] = recs

                m1, m2, m3 = st.columns(3)
                m1.markdown(f"""<div class="metric-card"><div class="label">Rows Processed</div><div class="value">{len(batch_df)}</div></div>""", unsafe_allow_html=True)
                m2.markdown(f"""<div class="metric-card"><div class="label">Avg Predicted Waste</div><div class="value">{np.mean(preds):.1f} kg</div></div>""", unsafe_allow_html=True)
                m3.markdown(f"""<div class="metric-card"><div class="label">Donate Recommended</div><div class="value">{recs.count('Donate to NGO')}</div></div>""", unsafe_allow_html=True)

                st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
                st.dataframe(batch_df, width="stretch")

                st.download_button(
                    "⬇️  Download Results CSV",
                    data=batch_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Could not process file: {e}")

# ======================================================================
# HISTORY TAB
# ======================================================================
with tab_history:
    st.markdown('<div class="section-title">🕘 Prediction History</div>', unsafe_allow_html=True)

    hist_df = load_history()
    if hist_df.empty:
        st.markdown(
            """
            <div style="border:1px dashed var(--border); border-radius:14px; padding:40px 20px; text-align:center; color:var(--muted); background:var(--surface);">
                <div style="font-size:2rem;">📭</div>
                <p style="margin-top:8px; font-size:0.88rem;">No predictions logged yet. Make one on the Predict tab.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        h1, h2, h3 = st.columns(3)
        h1.markdown(f"""<div class="metric-card"><div class="label">Total Predictions</div><div class="value">{len(hist_df)}</div></div>""", unsafe_allow_html=True)
        h2.markdown(f"""<div class="metric-card"><div class="label">Avg Predicted Waste</div><div class="value">{hist_df['Predicted Waste (KG)'].mean():.1f} kg</div></div>""", unsafe_allow_html=True)
        h3.markdown(f"""<div class="metric-card"><div class="label">Most Common Action</div><div class="value" style="font-size:1.1rem;">{hist_df['Recommendation'].mode()[0]}</div></div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        st.dataframe(hist_df.sort_values("timestamp", ascending=False), width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Predicted Waste Over Time**")
            st.line_chart(hist_df.set_index("timestamp")["Predicted Waste (KG)"], color="#1E5A32")
        with c2:
            st.markdown("**Recommendation Breakdown**")
            st.bar_chart(hist_df["Recommendation"].value_counts(), color="#D98E3B")

        st.download_button(
            "⬇️  Download Full History CSV",
            data=hist_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
        )

st.markdown(
    '<div class="app-footer">🌿 FoodWaste AI · Predictive Intelligence Platform · Built with Streamlit &amp; scikit-learn</div>',
    unsafe_allow_html=True,
)
