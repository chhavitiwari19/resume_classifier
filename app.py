import streamlit as st
import joblib
import os
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="wide"
)

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #101827 0%, #172554 55%, #312e81 100%);
        color: #f8fafc;
    }
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #cbd5e1;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background: rgba(255,255,255,0.09);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 18px;
        padding: 22px;
        margin-top: 12px;
    }
    .result-label {
        color: #cbd5e1;
        font-size: 0.95rem;
    }
    .result-value {
        color: #a5f3fc;
        font-size: 1.8rem;
        font-weight: 800;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        padding: 14px;
        border-radius: 14px;
    }
    .stButton > button {
        width: 100%;
        border: 0;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        font-weight: 700;
        color: white;
        background: linear-gradient(90deg, #0891b2, #4f46e5);
    }
    .hint {
        color: #cbd5e1;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL FILES ----------------
@st.cache_resource
def load_artifacts():
    model_path = "resume_classifier_model.pkl"
    vectorizer_path = "resume_tfidf_vectorizer.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing {model_path}. Keep it in the same folder as app.py."
        )
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            f"Missing {vectorizer_path}. Keep it in the same folder as app.py."
        )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


# ---------------- HEADER ----------------
st.markdown('<div class="main-title">📄 AI Resume Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Analyze resume text and predict the most suitable job category using Machine Learning.</div>',
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("About the Project")
    st.write(
        "This application uses a TF-IDF vectorizer and a trained Logistic Regression "
        "model to classify resume text into one of the categories learned from the dataset."
    )
    st.divider()
    st.subheader("How it works")
    st.write("1. Enter or upload resume text.")
    st.write("2. TF-IDF converts text into numerical features.")
    st.write("3. Logistic Regression predicts the category.")
    st.write("4. Review the top predicted categories.")
    st.divider()
    st.caption("Model: Logistic Regression")
    st.caption("Text processing: TF-IDF")

# ---------------- MAIN INPUT ----------------
left, right = st.columns([1.25, 0.75], gap="large")

with left:
    st.subheader("Resume Details")
    uploaded_file = st.file_uploader(
        "Upload a plain-text resume (.txt)",
        type=["txt"],
        help="Upload a .txt file containing the resume text."
    )

    uploaded_text = ""
    if uploaded_file is not None:
        try:
            uploaded_text = uploaded_file.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            uploaded_text = uploaded_file.getvalue().decode("latin-1")

    resume_text = st.text_area(
        "Paste resume text here",
        value=uploaded_text,
        height=320,
        placeholder="Paste the candidate's resume, including skills, education, experience and projects..."
    )

    classify_clicked = st.button("Classify Resume", type="primary")

with right:
    st.subheader("Tips for better results")
    st.info(
        "Use the full resume text where possible. Include skills, work experience, "
        "education, projects and responsibilities. The prediction depends on the "
        "information present in the text."
    )
    st.markdown("**Supported input**")
    st.write("• Paste resume text")
    st.write("• Upload a `.txt` file")
    st.markdown("**Note**")
    st.caption(
        "This model was trained on resume text from the supplied dataset. "
        "Its output is a prediction, not a hiring recommendation."
    )

# ---------------- PREDICTION ----------------
if classify_clicked:
    cleaned_text = resume_text.strip()

    if not cleaned_text:
        st.warning("Please paste resume text or upload a .txt file before classifying.")
    else:
        try:
            model, vectorizer = load_artifacts()
            resume_vector = vectorizer.transform([cleaned_text])
            predicted_category = model.predict(resume_vector)[0]

            st.divider()
            st.subheader("Classification Result")
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted Resume Category</div>
                    <div class="result-value">{predicted_category}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Show the top category scores when supported by the model.
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(resume_vector)[0]
                classes = model.classes_
                top_indices = probabilities.argsort()[::-1][:5]

                top_results = pd.DataFrame({
                    "Category": [classes[i] for i in top_indices],
                    "Model score (%)": [round(float(probabilities[i]) * 100, 2) for i in top_indices]
                })

                st.markdown("### Top 5 predicted categories")
                st.dataframe(top_results, use_container_width=True, hide_index=True)

                st.caption(
                    "Scores are model-estimated class probabilities; they are not a guarantee "
                    "that the candidate is suitable for a role."
                )

        except FileNotFoundError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Something went wrong while classifying the resume: {error}")

# ---------------- FOOTER ----------------
st.divider()
st.caption("Resume Classifier | Built with Python, Scikit-learn, TF-IDF and Streamlit")
