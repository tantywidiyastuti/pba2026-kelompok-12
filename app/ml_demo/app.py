"""
app.py — HuggingFace Spaces: PyCaret Hate Speech Detector (ML)
==================================================================
Deploy-ready Gradio app untuk mendeteksi hate speech pada teks 
bahasa Indonesia menggunakan Machine Learning (PyCaret).

Struktur folder yang di-upload ke Space:
    /
    ├── app.py                       ← file ini
    ├── requirements.txt
    ├── README.md
    ├── models/
    │   ├── best_model_HS.pkl
    │   └── tfidf_vectorizer.pkl
    └── data/
        ├── new_kamusalay.csv
        └── abusive.csv
"""

import os
import re
import joblib
import pandas as pd
import gradio as gr
from pycaret.classification import load_model, predict_model

# ── Path ──────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "models")
DATA_DIR    = os.path.join(BASE_DIR, "data")

SLANG_PATH      = os.path.join(DATA_DIR,  "new_kamusalay.csv")
ABUSIVE_PATH    = os.path.join(DATA_DIR,  "abusive.csv")
TFIDF_PATH      = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
MODEL_HS_PATH   = os.path.join(MODEL_DIR, "best_model_HS") # Jangan tambahkan .pkl, dimuat oleh PyCaret otomatis

STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "itu", "ini", "dengan", "adalah",
    "ada", "tidak", "juga", "untuk", "pada", "dalam", "sudah", "atau",
    "saya", "aku", "kamu", "dia", "mereka", "kita", "kami", "akan",
    "bisa", "telah", "bahwa", "karena", "oleh", "jadi", "lagi", "ya",
    "jangan", "tapi", "kalau", "mau", "aja", "deh", "sih", "lah",
    "dong", "nih", "kan", "nya", "yg", "dgn", "utk", "rt", "user", "url"
}

# =============================================================
# Load komponen saat startup
# =============================================================
print("[+] Loading komponen model PyCaret...")

def load_dictionaries():
    df_slang = pd.read_csv(SLANG_PATH, header=None, names=["slang", "formal"], encoding="latin-1")
    slang_dict = dict(zip(df_slang["slang"].str.lower(), df_slang["formal"].str.lower()))
    
    df_abusive = pd.read_csv(ABUSIVE_PATH, encoding="latin-1")
    abusive_set = set(df_abusive.iloc[:, 0].str.lower().str.strip().tolist())
    return slang_dict, abusive_set

try:
    slang_dict, abusive_set = load_dictionaries()
    print(f"    Slang dict  : {len(slang_dict):,} entri ✅")
except Exception as e:
    slang_dict, abusive_set = {}, set()

try:
    tfidf = joblib.load(TFIDF_PATH)
    model_hs = load_model(MODEL_HS_PATH)
    models_ready = True
    print(f"    Model/TFIDF : loaded ✅")
except Exception as e:
    tfidf = model_hs = None
    models_ready = False
    print(f"    Model/TFIDF : GAGAL — {e}")

if models_ready:
    print("[+] Startup selesai ✅")
else:
    print("[!] STARTUP GAGAL — periksa file PyCaret dan TFIDF.")

# =============================================================
# Preprocessing
# =============================================================
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\buser\b|\burl\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text(text: str):
    text_clean = clean_text(text)
    tokens = text_clean.split()
    tokens = [slang_dict.get(tok, tok) for tok in tokens]
    tokens = [tok for tok in tokens if tok not in STOPWORDS_ID]
    final_text = " ".join(tokens)
    abusive_count = sum(1 for tok in tokens if tok in abusive_set)
    return final_text, abusive_count

# =============================================================
# Prediksi
# =============================================================
def predict_with_detail(raw_text: str):
    if not models_ready or tfidf is None:
        msg = (
            "❌ **Model belum siap.** Pastikan file berikut ada:\n"
            "- `models/best_model_HS.pkl`\n"
            "- `models/tfidf_vectorizer.pkl`"
        )
        return msg, "", "", ""

    if not raw_text or not raw_text.strip():
        return "⚠️ Masukkan teks terlebih dahulu.", "", "", ""

    clean, abu_count = preprocess_text(raw_text)
    if not clean:
        return "⚠️ Teks tidak valid setelah preprocessing.", "", "", ""

    # TF-IDF Feature Extraction
    tfidf_matrix = tfidf.transform([clean])
    feature_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    feature_df["abusive_count"] = abu_count
    
    predictions = predict_model(model_hs, data=feature_df)
    
    pred_label = int(predictions['prediction_label'].iloc[0])
    pred_score = float(predictions['prediction_score'].iloc[0])

    label_hs = "🚨 HATE SPEECH" if pred_label == 1 else "✅ Bukan Hate Speech"

    summary = (
        "⚠️ Teks terdeteksi mengandung: **Hate Speech**"
        if pred_label == 1 else
        "✅ Teks ini **aman** — tidak mengandung hate speech."
    )

    detail = (
        f"**Teks asli:** {raw_text}\n\n"
        f"**Teks setelah preprocessing:** `{clean}`\n\n"
        f"**Jumlah Kata Abusif:** `{abu_count}`\n\n"
        f"**Dimensi Fitur (TFIDF Matrix):** `(1, {tfidf_matrix.shape[1]})`"
    )

    return (
        summary,
        label_hs,
        f"{'🔴' if pred_label == 1 else '🟢'} {pred_score*100:.1f}%",
        detail,
    )

# =============================================================
# Gradio UI
# =============================================================
CSS = """
.main-header {
    background: linear-gradient(135deg, #10b981 0%, #059669 50%, #047857 100%);
    padding: 28px 24px; border-radius: 16px; text-align: center;
    margin-bottom: 24px; border: 1px solid rgba(255,255,255,0.08);
}
.main-header h1 { color: #ffffff; font-size: 1.9rem; margin: 0 0 6px 0; }
.main-header p  { color: #d1fae5; font-size: 0.95rem; margin: 0; }
.badge {
    display: inline-block; background: rgba(255,255,255,0.2);
    color: #ffffff; padding: 3px 10px; border-radius: 20px;
    font-size: 0.8rem; border: 1px solid rgba(255,255,255,0.4); margin-top: 8px;
}
footer { display: none !important; }
"""

EXAMPLES = [
    ["dasar kafir lu semua, pergi dari negeri ini!"],
    ["hari ini cuaca bagus, semangat belajar semua!"],
    ["anjing lu brengsek, ga tau diri!"],
    ["selamat pagi, semoga harimu menyenangkan"],
    ["si bego itu emang ga pantes hidup, buang-buang oksigen aja"],
]

with gr.Blocks(
    title="🛡️ ML Hate Speech Detector",
    css=CSS,
    theme=gr.themes.Base(
        primary_hue="emerald",
        secondary_hue="zinc",
        neutral_hue="zinc",
    )
) as demo:

    gr.HTML("""
    <div class="main-header">
        <h1>🛡️ Indonesian Hate Speech Detector</h1>
        <p>Deteksi ujaran kebencian pada teks Bahasa Indonesia berbasis Machine Learning</p>
        <span class="badge">PyCaret · TF-IDF · LightGBM</span>
    </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            input_text  = gr.Textbox(
                label="📝 Masukkan Teks (Bahasa Indonesia)",
                placeholder="Contoh: dasar brengsek, pergi dari sini!",
                lines=5,
            )
            with gr.Row():
                clear_btn   = gr.Button("🗑️ Bersihkan", variant="secondary", size="sm")
                analyze_btn = gr.Button("🔍 Analisis",  variant="primary",   size="lg")
            gr.Examples(examples=EXAMPLES, inputs=input_text, label="💡 Contoh Kalimat")

        with gr.Column(scale=4):
            summary_out = gr.Markdown(label="Hasil")
            gr.Markdown("#### 🏷️ Hate Speech Klasifikasi")
            hs_label = gr.Textbox(label="Prediksi",   interactive=False)
            hs_conf  = gr.Textbox(label="Confidence", interactive=False)

    with gr.Accordion("🔬 Detail Preprocessing", open=False):
        detail_out = gr.Markdown()

    with gr.Accordion("ℹ️ Tentang Model", open=False):
        gr.Markdown("""
| Properti | Detail |
|---|---|
| **Pipeline** | TF-IDF Vectorizer + Machine Learning |
| **Tool** | PyCaret AutoML |
| **Model Terbaik** | LightGBM (dijadikan referensi utama) |
| **Dataset** | Indonesian Hate Speech Twitter (~13.169 tweet) |
| **Labels Digunakan** | HS (Hate Speech) |
| **Fitur Tambahan** | Ekstraksi `abusive_count` dari kamus abusive.csv |

> **Kelompok 12 — PBA 2026 | Institut Teknologi Sumatera**
        """)

    gr.HTML("""
    <div style="text-align:center;color:#71717a;font-size:.8rem;margin-top:20px;
                padding-top:16px;border-top:1px solid rgba(0,0,0,.06)">
        Powered by Hugging Face Spaces · Gradio · PyCaret
        &nbsp;|&nbsp; Model: Machine Learning · Dataset: id-multi-label-hate-speech
    </div>
    """)

    OUTPUTS = [summary_out, hs_label, hs_conf, detail_out]
    analyze_btn.click(fn=predict_with_detail, inputs=input_text, outputs=OUTPUTS)
    input_text.submit(fn=predict_with_detail, inputs=input_text, outputs=OUTPUTS)
    clear_btn.click(
        fn=lambda: tuple([""] * (len(OUTPUTS) + 1)),
        outputs=[input_text] + OUTPUTS,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
