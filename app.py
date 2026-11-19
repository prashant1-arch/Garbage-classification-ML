
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="EcoVision AI | Smart Waste Intelligence",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #07111f 0%, #101c2c 55%, #0b1724 100%);
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background: #0b1624;
    border-right: 1px solid #243449;
}

.hero {
    padding: 35px;
    border-radius: 22px;
    background: linear-gradient(120deg, #123c36, #102b42, #172a42);
    border: 1px solid #28584e;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 38px;
    font-weight: 800;
    color: #e8fff6;
    margin-bottom: 8px;
}

.hero p {
    color: #a9c9c0;
    font-size: 15px;
}

.metric-card {
    background: linear-gradient(145deg, #17283a, #102033);
    border: 1px solid #2b4055;
    border-radius: 17px;
    padding: 20px;
    min-height: 125px;
}

.metric-label {
    color: #9aafc2;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #65e6b5;
    font-size: 29px;
    font-weight: 800;
    margin-top: 10px;
}

.section-title {
    color: #e6fff5;
    font-size: 22px;
    font-weight: 700;
    margin-top: 15px;
}

.info-card {
    background: #142538;
    padding: 18px;
    border-radius: 15px;
    border: 1px solid #2b4055;
}

div.stButton > button {
    background: linear-gradient(90deg, #13a879, #27c99b);
    color: #041b15;
    font-weight: 800;
    border: none;
    border-radius: 11px;
    padding: 12px 25px;
    width: 100%;
}

div.stButton > button:hover {
    background: #55e5b5;
    color: #071b15;
    border: none;
}

.stTabs [data-baseweb="tab"] {
    color: #c6d5e2;
}

.stTabs [aria-selected="true"] {
    color: #65e6b5 !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #2b4055;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD DATASET ----------------

st.sidebar.markdown("## ♻️ EcoVision AI")
st.sidebar.caption("Smart Waste Intelligence System")
st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    try:
        df = pd.read_csv("garbage_classification_dataset_5000.csv")
    except FileNotFoundError:
        st.warning(
            "Please upload garbage_classification_dataset_5000.csv "
            "using the sidebar."
        )
        st.stop()


# ---------------- DATA VALIDATION ----------------

required_columns = [
    "Weight_g",
    "Moisture_Percent",
    "Size_cm3",
    "Waste_Type"
]

if not all(col in df.columns for col in required_columns):
    st.error("Dataset columns do not match the required format.")
    st.write("Required columns:", required_columns)
    st.stop()

df = df[required_columns].dropna()

X = df[["Weight_g", "Moisture_Percent", "Size_cm3"]]
y = df["Waste_Type"].astype(str)

if y.nunique() < 2:
    st.error("Dataset must contain at least two waste categories.")
    st.stop()

# ---------------- MODEL TRAINING ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, average="weighted", zero_division=0
)
recall = recall_score(
    y_test, y_pred, average="weighted", zero_division=0
)
f1 = f1_score(
    y_test, y_pred, average="weighted", zero_division=0
)


# ---------------- SIDEBAR NAVIGATION ----------------

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Waste Prediction",
        "Data Analytics",
        "Model Performance",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("🟢 **System Status: Online**")
st.sidebar.caption("Powered by Machine Learning")


# ---------------- HERO SECTION ----------------

st.markdown("""
<div class="hero">
    <h1>♻️ EcoVision AI</h1>
    <p>Intelligent Waste Classification & Environmental Analytics</p>
    <p>Transforming waste data into smarter environmental decisions.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.markdown("## 📊 System Overview")
    st.caption("A real-time overview of your garbage classification dataset.")

    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        ("TOTAL RECORDS", f"{len(df):,}"),
        ("MODEL ACCURACY", f"{accuracy * 100:.2f}%"),
        ("WASTE CATEGORIES", f"{y.nunique()}"),
        ("TRAINING SAMPLES", f"{len(X_train):,}")
    ]

    for col, (label, value) in zip(
        [col1, col2, col3, col4], metrics
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Waste Distribution</div>',
                unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        counts = df["Waste_Type"].value_counts().reset_index()
        counts.columns = ["Waste Type", "Count"]

        fig = px.pie(
            counts,
            names="Waste Type",
            values="Count",
            hole=0.65,
            title="Category Distribution",
            color_discrete_sequence=["#27c99b", "#4389e8"]
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6fff5"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.histogram(
            df,
            x="Weight_g",
            color="Waste_Type",
            title="Waste Weight Analysis",
            barmode="overlay",
            opacity=0.75,
            color_discrete_sequence=["#27c99b", "#4389e8"]
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6fff5"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔍 Recent Dataset Records")
    st.dataframe(df.head(10), use_container_width=True)


# ---------------- WASTE PREDICTION ----------------

elif page == "Waste Prediction":

    st.markdown("## 🧠 AI Waste Prediction")
    st.caption("Enter waste characteristics to predict its category.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ⚙️ Input Parameters")

        weight = st.number_input(
            "Weight (grams)",
            min_value=0.0,
            max_value=10000.0,
            value=150.0,
            step=10.0
        )

        moisture = st.slider(
            "Moisture Percentage",
            min_value=0.0,
            max_value=100.0,
            value=45.0
        )

        size = st.number_input(
            "Size (cm³)",
            min_value=0.0,
            max_value=10000.0,
            value=250.0,
            step=10.0
        )

        predict_button = st.button(
            "🚀 Analyze Waste"
        )

    with col2:
        st.markdown("### 🌱 Prediction Result")

        if predict_button:

            input_data = pd.DataFrame(
                [[weight, moisture, size]],
                columns=["Weight_g", "Moisture_Percent", "Size_cm3"]
            )

            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            confidence = np.max(probabilities) * 100

            if prediction == "Biodegradable":
                st.success("🌿 Biodegradable Waste")
                st.markdown(
                    "This waste is classified as biodegradable "
                    "by the trained model."
                )
            else:
                st.warning("♻️ Non-Biodegradable Waste")
                st.markdown(
                    "This waste is classified as non-biodegradable "
                    "by the trained model."
                )

            st.metric("Prediction Confidence", f"{confidence:.2f}%")

            probability_df = pd.DataFrame({
                "Category": model.classes_,
                "Probability": probabilities
            })

            fig = px.bar(
                probability_df,
                x="Category",
                y="Probability",
                color="Category",
                title="Prediction Probability",
                color_discrete_sequence=["#27c99b", "#4389e8"]
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e6fff5",
                yaxis_range=[0, 1]
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.markdown("""
            <div class="info-card">
                <h3>🔬 Ready for Analysis</h3>
                <p>Enter the waste parameters and click Analyze Waste
                to receive an AI-based prediction.</p>
                <p>Our Logistic Regression model will analyze
                the input characteristics.</p>
            </div>
            """, unsafe_allow_html=True)


# ---------------- DATA ANALYTICS ----------------

elif page == "Data Analytics":

    st.markdown("## 📈 Data Intelligence Center")

    col1, col2 = st.columns(2)

    with col1:
        feature = st.selectbox(
            "Select Feature",
            ["Weight_g", "Moisture_Percent", "Size_cm3"]
        )

    with col2:
        chart_type = st.selectbox(
            "Select Chart",
            ["Histogram", "Box Plot", "Scatter Plot"]
        )

    if chart_type == "Histogram":
        fig = px.histogram(
            df,
            x=feature,
            color="Waste_Type",
            title=f"{feature} Distribution"
        )

    elif chart_type == "Box Plot":
        fig = px.box(
            df,
            x="Waste_Type",
            y=feature,
            color="Waste_Type",
            title=f"{feature} Comparison"
        )

    else:
        fig = px.scatter(
            df,
            x="Weight_g",
            y="Size_cm3",
            color="Waste_Type",
            size="Moisture_Percent",
            title="Waste Feature Relationship"
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e6fff5"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)


# ---------------- MODEL PERFORMANCE ----------------

elif page == "Model Performance":

    st.markdown("## 🎯 Machine Learning Performance")
    st.caption("Evaluation of the Logistic Regression classifier.")

    c1, c2, c3, c4 = st.columns(4)

    for col, label, value in zip(
        [c1, c2, c3, c4],
        ["Accuracy", "Precision", "Recall", "F1 Score"],
        [accuracy, precision, recall, f1]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value * 100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 🔲 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

    fig = px.imshow(
        cm,
        x=model.classes_,
        y=model.classes_,
        text_auto=True,
        color_continuous_scale="Teal",
        labels={
            "x": "Predicted Label",
            "y": "Actual Label",
            "color": "Count"
        }
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e6fff5"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📄 Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).transpose(),
        use_container_width=True
    )


# ---------------- ABOUT PROJECT ----------------

elif page == "About Project":

    st.markdown("## 🌍 About EcoVision AI")

    st.markdown("""
    <div class="info-card">
        <h2>♻️ Making Waste Management Smarter</h2>
        <p>
        EcoVision AI is a machine learning based garbage
        classification project. It predicts whether waste is
        biodegradable or non-biodegradable using numerical features.
        </p>
        <h3>🧠 Technology Stack</h3>
        <p>Python | Streamlit | Pandas | NumPy | Scikit-learn | Plotly</p>
        <h3>⚙️ Machine Learning Algorithm</h3>
        <p>Logistic Regression with StandardScaler</p>
        <h3>📊 Input Features</h3>
        <p>Weight, Moisture Percentage, Size</p>
        <h3>🎯 Project Objective</h3>
        <p>To demonstrate how machine learning can support
        intelligent waste classification and environmental awareness.</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "Note: This project uses numerical dataset features. "
        "It is not an image-recognition system. Predictions are "
        "based on the patterns learned from the uploaded dataset."
    )

# ---------------- FOOTER ----------------

st.markdown("---")
st.markdown(
    "<center style='color:#7893a8;'>"
    "♻️ EcoVision AI | Smart Waste Intelligence | Built with Python & Streamlit"
    "</center>",
    unsafe_allow_html=True
)
