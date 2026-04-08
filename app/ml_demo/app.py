import os
import re
import sys
import pandas as pd
import streamlit as st
import joblib
from pycaret.classification import load_model, predict_model

# === KONFIGURASI PATH ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Stopwords List Asli
STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "itu", "ini", "dengan", "adalah",
    "ada", "tidak", "juga", "untuk", "pada", "dalam", "sudah", "atau",
    "saya", "aku", "kamu", "dia", "mereka", "kita", "kami", "akan",
    "bisa", "telah", "bahwa", "karena", "oleh", "jadi", "lagi", "ya",
    "jangan", "tapi", "kalau", "mau", "aja", "deh",
    "sih", "lah", "dong", "nih", "kan", "nya", "yg", "dgn", "utk",
    "rt", "user", "url"
}

# === HELPER FUNCTIONS (PREPROCESSING) ===
@st.cache_data
def load_dictionaries():
    """Load slang and abusive dictionaries with caching for Speed."""
    slang_path = os.path.join(DATA_RAW_DIR, "new_kamusalay.csv")
    abusive_path = os.path.join(DATA_RAW_DIR, "abusive.csv")
    
    # Load Slang
    df_slang = pd.read_csv(slang_path, header=None, names=["slang", "formal"], encoding="latin-1")
    slang_dict = dict(zip(df_slang["slang"].str.lower(), df_slang["formal"].str.lower()))
    
    # Load Abusive
    df_abusive = pd.read_csv(abusive_path, encoding="latin-1")
    col = df_abusive.columns[0]
    abusive_set = set(df_abusive[col].str.lower().str.strip().tolist())
    
    return slang_dict, abusive_set

@st.cache_resource
def load_ml_components():
    """Load TFIDF Vectorizer & PyCaret Model."""
    # 1. Load TFIDF
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    tfidf = joblib.load(tfidf_path)
    
    # 2. Load PyCaret Model (PyCaret implicitly adds .pkl, so we strip it if present)
    model_path = os.path.join(MODELS_DIR, "best_model_HS")
    # if path not found, maybe predict_model needs exact name, load_model handles it
    pycaret_model = load_model(model_path)
    
    return tfidf, pycaret_model

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\buser\b|\burl\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text(text: str, slang_dict: dict, abusive_set: set):
    """Full preprocessing pipeline yielding the final feature DataFrame."""
    # 1. Clean
    text_clean = clean_text(text)
    
    # 2. Normalize Slang
    tokens = text_clean.split()
    tokens = [slang_dict.get(tok, tok) for tok in tokens]
    
    # 3. Stopword Removal
    tokens = [tok for tok in tokens if tok not in STOPWORDS_ID]
    final_text = " ".join(tokens)
    
    # 4. Abusive Count
    abusive_count = sum(1 for tok in tokens if tok in abusive_set)
    
    return final_text, abusive_count

st.set_page_config(page_title="Hate Speech Detector", page_icon="🛡️", layout="centered")

# === STYLES (Aesthetics) ===
st.markdown("""
    <style>
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #ffffff;
        background: linear-gradient(135deg, #6B73FF 0%, #000DFF 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .result-box {
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .result-hate {
        background-color: #ffebee;
        color: #c62828;
        border-left: 5px solid #c62828;
    }
    .result-safe {
        background-color: #e8f5e9;
        color: #2e7d32;
        border-left: 5px solid #2e7d32;
    }
    </style>
""", unsafe_allow_html=True)

# === UI COMPONENTS ===
st.markdown("<div class='main-header'><h1>🛡️ Indonesian Hate Speech Detector</h1><p>Analyze Tweets / Comments for Hate Speech using Machine Learning</p></div>", unsafe_allow_html=True)

with st.spinner("Memuat model dan data kamus... (ini mungkin butuh beberapa detik)"):
    # Load assets
    slang_dict, abusive_set = load_dictionaries()
    tfidf, pycaret_model = load_ml_components()

# User Input
user_input = st.text_area("Masukkan teks yang ingin dianalisis (Bahasa Indonesia):", height=150, placeholder="Contoh: dasar cowok bego ga tau diri...")

if st.button("🔍 Analisis Teks", use_container_width=True):
    if user_input.strip() == "":
        st.warning("⚠️ Masukkan teks terlebih dahulu!")
    else:
        with st.spinner("Sedang memproses teks..."):
            # 1. Preprocesing Text
            cleaned_text, abusive_amount = preprocess_text(user_input, slang_dict, abusive_set)
            
            # 2. Extract Features
            tfidf_matrix = tfidf.transform([cleaned_text])
            feature_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
            feature_df["abusive_count"] = abusive_amount
            
            # 3. Predict via PyCaret
            predictions = predict_model(pycaret_model, data=feature_df)
            
            # Label Output
            pred_label = predictions['prediction_label'].iloc[0]
            pred_score = predictions['prediction_score'].iloc[0]
            
            # Output Display
            st.markdown("### Hasil Analisis:")
            if pred_label == 1:
                st.markdown(f"<div class='result-box result-hate'>🚨 Deteksi: HATE SPEECH (Confidence: {pred_score*100:.1f}%)</div>", unsafe_allow_html=True)
                st.error("Teks ini teridentifikasi mengandung ujaran kebencian.")
            else:
                st.markdown(f"<div class='result-box result-safe'>✅ Deteksi: AMAN (Confidence: {pred_score*100:.1f}%)</div>", unsafe_allow_html=True)
                st.success("Teks ini teridentifikasi bersih dari ujaran kebencian.")
            
            with st.expander("Detail Proses (Debug)"):
                st.write(f"- **Teks Mentah:** `{user_input}`")
                st.write(f"- **Teks Bersih & Normal:** `{cleaned_text}`")
                st.write(f"- **Jumlah Kata Abusif:** `{abusive_amount}`")
                
st.markdown("---")
st.markdown("<div style='text-align:center; color:grey; font-size:0.8rem;'>Powered by Hugging Face Spaces & Streamlit • Model: LightGBM (via PyCaret)</div>", unsafe_allow_html=True)
