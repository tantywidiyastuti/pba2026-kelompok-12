import os
import re
import sys
import pandas as pd
import gradio as gr
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
def load_dictionaries():
    """Load slang and abusive dictionaries."""
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

def load_ml_components():
    """Load TFIDF Vectorizer & PyCaret Model."""
    # 1. Load TFIDF
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    tfidf = joblib.load(tfidf_path)
    
    # 2. Load PyCaret Model 
    model_path = os.path.join(MODELS_DIR, "best_model_HS")
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

# Load once, Gradio will keep this in memory across calls
try:
    slang_dict, abusive_set = load_dictionaries()
    tfidf, pycaret_model = load_ml_components()
    models_loaded = True
except Exception as e:
    models_loaded = False
    error_message = str(e)

def analyze_text(user_input):
    if not models_loaded:
        return f"Error: Model or Data not successfully loaded.\nDetails: {error_message}", ""

    if not user_input or user_input.strip() == "":
        return "⚠️ Masukkan teks terlebih dahulu!", ""
    
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
    
    debug_info = f"- Teks Mentah: {user_input}\n- Teks Bersih & Normal: {cleaned_text}\n- Jumlah Kata Abusif: {abusive_amount}"

    if pred_label == 1:
        result_text = f"🚨 Deteksi: HATE SPEECH (Confidence: {pred_score*100:.1f}%)\n\nTeks ini teridentifikasi mengandung ujaran kebencian."
    else:
        result_text = f"✅ Deteksi: AMAN (Confidence: {pred_score*100:.1f}%)\n\nTeks ini teridentifikasi bersih dari ujaran kebencian."

    return result_text, debug_info

custom_css = """
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
.gradio-container {
    font-family: 'Inter', sans-serif;
}
"""

with gr.Blocks(title="Hate Speech Detector", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.HTML("<div class='main-header'><h1>🛡️ Indonesian Hate Speech Detector</h1><p>Analyze Tweets / Comments for Hate Speech using Machine Learning</p></div>")
    
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(lines=5, placeholder="Contoh: dasar cowok bego ga tau diri...", label="Masukkan teks yang ingin dianalisis (Bahasa Indonesia):")
            analyze_btn = gr.Button("🔍 Analisis Teks", variant="primary")
            
        with gr.Column(scale=1):
            result_output = gr.Textbox(label="Hasil Analisis")
            debug_output = gr.Textbox(lines=4, label="Detail Proses (Debug)", interactive=False)
            
    analyze_btn.click(fn=analyze_text, inputs=input_text, outputs=[result_output, debug_output])
    
    gr.HTML("<hr><div style='text-align:center; color:grey; font-size:0.8rem;'>Powered by Hugging Face Spaces & Gradio • Model: LightGBM (via PyCaret)</div>")

if __name__ == "__main__":
    demo.launch()
