import joblib
import streamlit as st
import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import seaborn as sns
import matplotlib.pyplot as plt


# PAGE CONFIG
st.set_page_config(page_title="EEG Emotion Classification", layout="wide")
st.markdown("""
    <style>
        /* Sidebar gradient */
        section[data-testid="stSidebar"] {
           background: linear-gradient(180deg, #E3F2FD, #BBDEFB);
        }

        /* Label umum */
        section[data-testid="stSidebar"] label {
            color: #1E3A5F !important;
        }

        /* FIX KUAT: Upload your CSV */
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] label p {
            color: #1E3A5F !important;
        }

        /* Header sidebar */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #1E3A5F !important;
        }

        /* Isi dropdown */
        section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
            color: #262730 !important;
        }

        /* Isi uploader */
        section[data-testid="stSidebar"] .stFileUploader div {
            color: #262730 !important;
        }

        /* Button */
        section[data-testid="stSidebar"] .stButton>button {
            background-color: #0D47A1;
            color: white;
            border-radius: 8px;
        }

        /* Hover button */
        section[data-testid="stSidebar"] .stButton>button:hover {
            background-color: #08306B;
        }
            
        /* DOWNLOAD BUTTON */
        section[data-testid="stSidebar"] .stDownloadButton>button {
            background-color: #0D47A1;
            color: white;
            border-radius: 8px;
        }

        /* HOVER DOWNLOAD BUTTON */
        section[data-testid="stSidebar"] .stDownloadButton>button:hover {
            background-color: #08306B;
        }
            
       /* Target tombol toggle sidebar */
        button[data-testid="collapsedControl"] [data-testid="stIconMaterial"] {
            color: #FFFFFF !important;  /* selalu putih */
        }

        /* Kasih background biar tetap keliatan di putih */
        button[data-testid="collapsedControl"] {
            background: rgba(0, 0, 0, 0.4) !important;
            border-radius: 8px;
            padding: 4px;
        }

        /* Hover biar interaktif */
        button[data-testid="collapsedControl"]:hover {
            background: rgba(0, 0, 0, 0.7) !important;
        }
            
        /* TABLE CONTAINER */
        .custom-table {
            background: white;
            padding: 18px;
            border-radius: 16px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.08);
            border: 1px solid #E3F2FD;
            overflow-x: auto;
            margin-bottom: 20px;
        }

        /* TABLE */
        .custom-table table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            border-radius: 12px;
            overflow: hidden;
        }

        /* HEADER PREVIEW */
        .custom-table thead tr {
            background: linear-gradient(90deg, #1E88E5, #1565C0);
            color: white;
            text-align: center;
        }

        /* HEADER SUMMARY */
        .summary-table thead tr {
            background: linear-gradient(90deg, #8E24AA, #4A148C);
            color: white;
            text-align: center;
        }

        /* HEADER CELL */
        .custom-table th,
        .summary-table th {
            padding: 12px;
            font-weight: 600;
            border: none;
            text-align: center;
        }

        /* BODY CELL */
        .custom-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #E3F2FD;
            color: #1E3A5F;
        }

        /* ZEBRA */
        .custom-table tbody tr:nth-child(even) {
            background-color: #F7FBFF;
        }

        /* HOVER */
        .custom-table tbody tr:hover {
            background-color: #E3F2FD;
            transition: 0.2s;
        }
        
        /* Isi tabel rata kanan */
        .custom-table td,
        .summary-table td {
            text-align: right;
        }
            
        /* RADIO CONTAINER */
        div[role="radiogroup"] {
            display: flex;
            gap: 10px;
        }

        /* RADIO LABEL (default state) */
        div[role="radiogroup"] label {
            background: #E3F2FD;
            padding: 8px 14px;
            border-radius: 20px;
            color: #0D47A1 !important;
            font-weight: 500;
            border: 1px solid #BBDEFB;
            transition: 0.2s ease-in-out;
        }

        /* HIDE RADIO DOT */
        div[role="radiogroup"] input {
            display: none;
        }

        /* SELECTED STATE */
        div[role="radiogroup"] label[data-checked="true"] {
            background: #0D47A1;
            color: white !important;
            border: 1px solid #0D47A1;
            box-shadow: 0 4px 10px rgba(13,71,161,0.3);
        }

        /* HOVER */
        div[role="radiogroup"] label:hover {
            transform: translateY(-1px);
            cursor: pointer;
        }
            
        div[role="radiogroup"] label {
            backdrop-filter: blur(10px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
            
        /* CLASSIFICATION REPORT CARD */
        .report-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-top: 10px;
        }

        .report-card {
            background: white;
            border-radius: 16px;
            padding: 18px;
            border: 1px solid #E3F2FD;
            box-shadow: 0 4px 14px rgba(0,0,0,0.06);
            transition: 0.2s ease;
        }

        .report-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(0,0,0,0.10);
        }

        .report-title {
            font-size: 16px;
            font-weight: 700;
            color: #0D47A1;
            margin-bottom: 14px;
        }

        .metric-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 14px;
        }

        .metric-label {
            color: #546E7A;
        }

        .metric-value {
            font-weight: 600;
            color: #1E3A5F;
        }
            
        /* EXPANDER */
        details {
            background: white;
            border-radius: 14px;
            border: 1px solid #E3F2FD;
            padding: 6px 12px;
            margin-top: 10px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.05);
        }

        /* Judul expander */
        details summary {
            font-size: 17px !important;
            font-weight: 700 !important;
            color: #0D47A1 !important;
            padding: 10px 4px;
        }

        /* Hover */
        details summary:hover {
            color: #1565C0 !important;
            cursor: pointer;
        }

        /* Isi expander */
        details[open] {
            padding-bottom: 14px;
        }
            
        /* Wrapper st.radio */
        div[data-testid="stRadio"] {
            margin-top: -12px;
        }
            
        /* SIDEBAR WARNING */
        section[data-testid="stSidebar"] div[data-baseweb="notification"] {
            background: rgba(13, 71, 161, 0.15) !important;
            border-radius: 10px !important;
        }

        /* Normal text */
        section[data-testid="stSidebar"] div[data-baseweb="notification"] p {
            color: #1E3A5F !important;
            font-weight: 500;
        }

        /* Bullet list */
        section[data-testid="stSidebar"] div[data-baseweb="notification"] li {
            color: #1E3A5F !important;
        }

        /* Icon */
        section[data-testid="stSidebar"] div[data-baseweb="notification"] svg {
            fill: #1E3A5F !important;
        }
            
</style>
""", unsafe_allow_html=True)

st.title("EEG Emotion Classification with Backpropagation Optimization")
st.caption("A comparative analysis of Backpropagation and PSO-optimized models for classifying emotional states from EEG signals")

# LOAD SAVED MODEL
bp_model = load_model(
    "saved_model/bp_model.keras"
)

bp_pso_model = load_model(
    "saved_model/bp_pso_model.keras"
)

scaler = joblib.load(
    "saved_model/scaler.pkl"
)

X_test = joblib.load(
    "saved_model/X_test.pkl"
)

y_test = joblib.load(
    "saved_model/y_test.pkl"
)

feature_columns = joblib.load(
    "saved_model/feature_columns.pkl"
)

label_encoder = joblib.load(
    "saved_model/label_encoder.pkl"
)

true_class = np.argmax(
    y_test,
    axis=1
)

# SIDEBAR
st.sidebar.header("⚙️ Settings")

st.sidebar.warning("""
Uploaded dataset must:

- use FFT-based EEG features
- contain identical feature columns
- match the training dataset structure
""")

# SAMPLE DATASET
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Example Dataset")

try:
    sample_df = pd.read_csv("sample_dataset/example_dataset.csv")

    st.sidebar.caption(
        "Download the example EEG dataset format used for testing."
    )

    st.sidebar.download_button(
        label="Download Example CSV",
        data=sample_df.to_csv(index=False),
        file_name="example_eeg_dataset.csv",
        mime="text/csv"
    )

except:
    st.sidebar.info(
        "Example dataset file not found."
    )

uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

if "last_file" not in st.session_state:
    st.session_state.last_file = None

current_file = uploaded_file.name if uploaded_file else None

if st.session_state.last_file != current_file:
    # 🔥 RESET SEMUA STATE
    for key in ["trained", "acc_1", "acc_2", "y_pred_1", "y_pred_2", "df", "summary", "preview"]:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state.last_file = current_file
    st.rerun()

run_model = st.sidebar.button("Run Classification")

# LOAD DATA
def load_data():
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df

    return None

if "df" not in st.session_state or st.session_state.last_file != current_file:
    st.session_state.df = load_data()

df = st.session_state.df

if df is None:
    st.info("Please upload an EEG dataset to begin classification.")
    st.stop()

# PREPROCESS

missing_cols = [
    col for col in feature_columns
    if col not in df.columns
]

if missing_cols:

    st.error(f"""
    Dataset is missing required columns.

    Missing columns:
    {missing_cols}
    """)

    st.stop()

# pastikan feature sesuai training
X = df[feature_columns]

# ambil label
y = df.iloc[:, -1]

# pastikan semua label jadi string dulu
y = y.astype(str)

# encode label
y = label_encoder.transform(y)

# SCALING
try:
    X_scaled = scaler.transform(X)

except Exception as e:
    st.error(f"""
    Dataset is not compatible with the trained model.

    Make sure:
    - feature columns are identical
    - column order is identical
    - preprocessing is identical

    Error:
    {e}
    """)

    st.stop()

# GLOBAL STATUS (PALING ATAS)
status_container = st.container()

if not st.session_state.get("trained", False):
    status_container.info(
        "Dataset loaded successfully. Click 'Run Classification' to evaluate the models."
    )

else:
    status_container.success(
        "✅ Classification completed successfully. Explore the evaluation results below."
    )

data_container = st.container()

with data_container: 
    st.subheader("📁 Dataset Preview")

    # simpan preview sekali
    if "preview" not in st.session_state: 
        st.session_state.preview = df.head()
    preview_html = st.session_state.preview.to_html(index=False)

    st.markdown(
        f"""
        <div class="custom-table">
            {preview_html}
        </div>
        """,
        unsafe_allow_html=True
    )

    stats_html = f"""
    <div style="
        display: flex;
        justify-content: center;
        gap: 18px;
        margin-top: 10px;
        margin-bottom: 10px;
    ">

        <div style="
            background: white;
            border: 1px solid #E3F2FD;
            border-radius: 14px;
            padding: 16px 28px;
            min-width: 170px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        ">
            <div style="
                font-size: 14px;
                color: #546E7A;
                margin-bottom: 6px;
                font-weight: 500;
            ">
                Rows
            </div>

            <div style="
                font-size: 30px;
                font-weight: 700;
                color: #1E3A5F;
            ">
                {df.shape[0]:,}
            </div>
        </div>

        <div style="
            background: white;
            border: 1px solid #E3F2FD;
            border-radius: 14px;
            padding: 16px 28px;
            min-width: 170px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        ">
            <div style="
                font-size: 14px;
                color: #546E7A;
                margin-bottom: 6px;
                font-weight: 500;
            ">
                Features
            </div>

            <div style="
                font-size: 30px;
                font-weight: 700;
                color: #1E3A5F;
            ">
                {df.shape[1] - 1}
            </div>
        </div>

        <div style="
            background: white;
            border: 1px solid #E3F2FD;
            border-radius: 14px;
            padding: 16px 28px;
            min-width: 170px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        ">
            <div style="
                font-size: 14px;
                color: #546E7A;
                margin-bottom: 6px;
                font-weight: 500;
            ">
                Classes
            </div>

            <div style="
                font-size: 30px;
                font-weight: 700;
                color: #1E3A5F;
            ">
                {len(pd.unique(y))}
            </div>
        </div>

    </div>
    """

    st.html(stats_html)

    st.subheader("📊 Data Summary")
    # simpan summary sekali
    if "summary" not in st.session_state: 
        st.session_state.summary = df.describe()

    with st.expander("Click to View", expanded=False):
        summary_html = st.session_state.summary.to_html()

        st.markdown(
            f"""
            <div class="custom-table summary-table">
                {summary_html}
            </div>
            """,
            unsafe_allow_html=True
        )

import time

if run_model:

    # tampilkan progress
    progress = status_container.progress(0, text="Starting training...")

    # BP
    progress.progress(30, text="Running Backpropagation...")
    y_pred_1 = bp_model.predict(
        X_test,
        verbose=0
    )

    y_pred_1 = np.argmax(
        y_pred_1,
        axis=1
    )

    acc_1 = accuracy_score(
        true_class,
        y_pred_1
    )

    # delay biar smooth
    time.sleep(0.5)

    # BP + PSO
    progress.progress(70, text="Running BP + PSO...")
    y_pred_2 = bp_pso_model.predict(
        X_test,
        verbose=0
    )

    y_pred_2 = np.argmax(
        y_pred_2,
        axis=1
    )

    acc_2 = accuracy_score(
        true_class,
        y_pred_2
    )

    time.sleep(0.5)

    progress.progress(100, text="Done!")

    time.sleep(0.5)  # biar user lihat "Done"

    # simpan state
    st.session_state.acc_1 = acc_1
    st.session_state.acc_2 = acc_2
    st.session_state.y_pred_1 = y_pred_1
    st.session_state.y_pred_2 = y_pred_2

    # simpan label test juga
    st.session_state.true_class = true_class

    st.session_state.trained = True

    # 🔥 HAPUS progress
    status_container.empty()

    # 🔥 PAKSA RERUN BIAR UI UPDATE SEKETIKA
    st.rerun()

# TAMPILKAN HASIL

def metric_card(title, value, gradient):
    return f"""
        <div style="
            background: {gradient};
            padding: 20px;
            border-radius: 12px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        ">
            <div style="font-size:18px; font-weight:600;">
                {title}
            </div>
            <div style="font-size:14px; opacity:0.8; margin-top:4px;">
                Accuracy
            </div>
            <div style="font-size:32px; font-weight:bold; margin-top:8px;">
                {f"{value * 100:.2f}%".replace(".", ",")}
            </div>
        </div>
    """

def performance_analysis_card(
    y,
    y_pred,
    gradient="linear-gradient(135deg, #0D47A1, #1565C0)"
):

    report = classification_report(
        y,
        y_pred,
        output_dict=True
    )

    overall_precision = report["weighted avg"]["precision"]
    overall_recall = report["weighted avg"]["recall"]
    overall_f1 = report["weighted avg"]["f1-score"]
    overall_support = report["weighted avg"]["support"]
    overall_accuracy = report["accuracy"]

    return f"""
    <div style="
        background: {gradient};
        color: white;
        border-radius: 18px;
        padding: 22px;
        margin-top: 10px;
        margin-bottom: 0px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.15);
        font-family: 'Segoe UI', sans-serif;
    ">

        <div style="
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 18px;
            letter-spacing: 0.3px;
        ">
            Performance Analysis
        </div>

        <div style="
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
            gap: 14px;
        ">

            <div style="
                background: rgba(255,255,255,0.12);
                padding: 14px;
                border-radius: 12px;
                text-align: center;
                backdrop-filter: blur(6px);
            ">
                <div style="font-size:12px; opacity:0.8; margin-bottom:6px;">
                    Accuracy
                </div>

                <div style="font-size:26px; font-weight:700;">
                    {f"{overall_accuracy * 100:.2f}%".replace(".", ",")}
                </div>
            </div>

            <div style="
                background: rgba(255,255,255,0.12);
                padding: 14px;
                border-radius: 12px;
                text-align: center;
                backdrop-filter: blur(6px);
            ">
                <div style="font-size:12px; opacity:0.8; margin-bottom:6px;">
                    Precision
                </div>

                <div style="font-size:26px; font-weight:700;">
                    {f"{overall_precision * 100:.2f}%".replace(".", ",")}
                </div>
            </div>

            <div style="
                background: rgba(255,255,255,0.12);
                padding: 14px;
                border-radius: 12px;
                text-align: center;
                backdrop-filter: blur(6px);
            ">
                <div style="font-size:12px; opacity:0.8; margin-bottom:6px;">
                    Recall
                </div>

                <div style="font-size:26px; font-weight:700;">
                    {f"{overall_recall * 100:.2f}%".replace(".", ",")}
                </div>
            </div>

            <div style="
                background: rgba(255,255,255,0.12);
                padding: 14px;
                border-radius: 12px;
                text-align: center;
                backdrop-filter: blur(6px);
            ">
                <div style="font-size:12px; opacity:0.8; margin-bottom:6px;">
                    F1-Score
                </div>

                <div style="font-size:26px; font-weight:700;">
                    {f"{overall_f1 * 100:.2f}%".replace(".", ",")}
                </div>
            </div>

            <div style="
                background: rgba(255,255,255,0.12);
                padding: 14px;
                border-radius: 12px;
                text-align: center;
                backdrop-filter: blur(6px);
            ">
                <div style="font-size:12px; opacity:0.8; margin-bottom:6px;">
                    Support
                </div>

                <div style="font-size:26px; font-weight:700;">
                    {overall_support:.0f}
                </div>
            </div>

        </div>
    </div>
    """
def performance_card_height(compare_mode=False):

    if compare_mode:
        return 300

    return 220

def classification_report_cards(y, y_pred):

    report = classification_report(
        y,
        y_pred,
        output_dict=True
    )

    exclude_keys = ["accuracy", "macro avg", "weighted avg"]

    html = """
    <style>
    .report-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        padding: 10px;
        font-family: Arial;
    }

    .report-card {
        background: white;
        border-radius: 16px;
        padding: 18px;
        border: 1px solid #E3F2FD;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
        transition: 0.2s ease;
    }

    .report-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.10);
    }

    .report-title {
        font-size: 16px;
        font-weight: 700;
        color: #0D47A1;
        margin-bottom: 14px;
    }

    .metric-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        font-size: 14px;
    }

    .metric-label {
        color: #546E7A;
    }

    .metric-value {
        font-weight: 600;
        color: #1E3A5F;
    }
    </style>

    <div class="report-container">
    """

    for label, metrics in report.items():

        if label in exclude_keys:
            continue

        html += f"""
        <div class="report-card">

            <div class="report-title">
                Class {label}
            </div>

            <div class="metric-row">
                <span class="metric-label">Precision</span>
                <span class="metric-value">{f"{metrics['precision'] * 100:.2f}%".replace(".", ",")}</span>
            </div>

            <div class="metric-row">
                <span class="metric-label">Recall</span>
                <span class="metric-value">{f"{metrics['recall'] * 100:.2f}%".replace(".", ",")}</span>
            </div>

            <div class="metric-row">
                <span class="metric-label">F1-Score</span>
                <span class="metric-value">{f"{metrics['f1-score'] * 100:.2f}%".replace(".", ",")}</span>
            </div>

            <div class="metric-row">
                <span class="metric-label">Support</span>
                <span class="metric-value">{metrics['support']}</span>
            </div>

        </div>
        """

    html += "</div>"

    return html

result_container = st.container()

with result_container:

    if st.session_state.get("trained", False):

        st.markdown("### 🔬 Model Evaluation Results")

        acc_1 = st.session_state.acc_1
        acc_2 = st.session_state.acc_2
        y_pred_1 = st.session_state.y_pred_1
        y_pred_2 = st.session_state.y_pred_2

        true_class = st.session_state.true_class

        st.markdown("""
        <div style="
            font-size: 17px;
            font-weight: 600;
            color: #1E3A5F;
            margin-bottom: 2px;
        ">
            Select Evaluation Mode
        </div>
        """, unsafe_allow_html=True)

        mode = st.radio(
            "",
            ["Backpropagation Model", "Optimized Model (BP + PSO)", "Model Comparison"],
            horizontal=True,
            key="mode"
        )

        # BP ONLY
        if mode == "Backpropagation Model":

            st.subheader("Backpropagation Result")

            st.markdown(
                metric_card(
                    "BP : Backpropagation",
                    acc_1,
                    "linear-gradient(135deg, #1E88E5, #0D47A1)"
                ),
                unsafe_allow_html=True
            )

            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(true_class, y_pred_1),
                        annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            st.html(
                performance_analysis_card(
                    true_class,
                    y_pred_1,
                    "linear-gradient(135deg, #0D47A1, #1565C0)"
                )
            )

            # JARAK TAMBAHAN
            st.markdown(
                "<div style='height:16px;'></div>",
                unsafe_allow_html=True
            )

            with st.expander("Detailed Classification Metrics"):

                st.html(
                    classification_report_cards(true_class, y_pred_1)
                )

        # BP + PSO
        elif mode == "Optimized Model (BP + PSO)":

            st.subheader("BP + PSO Result")

            st.markdown(
                metric_card(
                    "BP + PSO",
                    acc_2,
                    "linear-gradient(135deg, #8E24AA, #4A148C)"
                ),
                unsafe_allow_html=True
            )

            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(true_class, y_pred_2),
                        annot=True, fmt="d", cmap="Purples", ax=ax)
            st.pyplot(fig)

            st.html(
                performance_analysis_card(
                    true_class,
                    y_pred_2,
                    "linear-gradient(135deg, #8E24AA, #4A148C)"
                )
            )

            # JARAK TAMBAHAN
            st.markdown(
                "<div style='height:16px;'></div>",
                unsafe_allow_html=True
            )

            with st.expander("Detailed Classification Metrics"):

                st.html(
                    classification_report_cards(true_class, y_pred_2)
                )

        # COMPARE
        else:

            st.subheader("Model Comparison")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    metric_card(
                        "BP : Backpropagation",
                        acc_1,
                        "linear-gradient(135deg, #1E88E5, #0D47A1)"
                    ),
                    unsafe_allow_html=True
                )

            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

            with col2:
                st.markdown(
                    metric_card(
                        "BP + PSO",
                        acc_2,
                        "linear-gradient(135deg, #8E24AA, #4A148C)"
                    ),
                    unsafe_allow_html=True
                )

            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

            col3, col4 = st.columns(2)

            with col3:
                fig1, ax1 = plt.subplots()
                sns.heatmap(confusion_matrix(true_class, y_pred_1),
                            annot=True, fmt="d", cmap="Blues", ax=ax1)
                st.pyplot(fig1)

            with col4:
                fig2, ax2 = plt.subplots()
                sns.heatmap(confusion_matrix(true_class, y_pred_2),
                            annot=True, fmt="d", cmap="Purples", ax=ax2)
                st.pyplot(fig2)

            # PERFORMANCE ANALYSIS
            col5, col6 = st.columns(2)

            with col5:

                st.html(
                    performance_analysis_card(
                        true_class,
                        y_pred_1,
                        "linear-gradient(135deg, #0D47A1, #1565C0)"
                    )
                )

            with col6:

                st.html(
                    performance_analysis_card(
                        true_class,
                        y_pred_2,
                        "linear-gradient(135deg, #8E24AA, #4A148C)"
                    )
                )

            # DETAILED METRICS
            col7, col8 = st.columns(2)

            with col7:

                st.markdown(
                    "<div style='height:16px;'></div>",
                    unsafe_allow_html=True
                )

                with st.expander("BP Classification Metrics"):

                    st.html(
                        classification_report_cards(true_class, y_pred_1)
                    )

            with col8:

                st.markdown(
                    "<div style='height:16px;'></div>",
                    unsafe_allow_html=True
                )

                with st.expander("BP + PSO Classification Metrics"):

                    st.html(
                        classification_report_cards(true_class, y_pred_2)
                    )

# FOOTER
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Developed by Mareta Ananda Putri</p>",
    unsafe_allow_html=True
)