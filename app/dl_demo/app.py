"""
app.py — HuggingFace Spaces: CNN-BiLSTM Hate Speech Detector (DL)
==================================================================
Deploy-ready Gradio app untuk mendeteksi hate speech dan abusive
language pada teks bahasa Indonesia menggunakan CNN-BiLSTM.

Struktur folder yang di-upload ke Space:
    /
    ├── app.py                       ← file ini
    ├── requirements.txt
    ├── models/
    │   ├── cnn_bilstm_HS_best.h5
    │   ├── cnn_bilstm_Abusive_best.h5
    │   └── cnn_bilstm_tokenizer.pkl
    └── data/
        └── new_kamusalay.csv
"""

import os
import re
import sys
import types
import pickle
import warnings

import numpy as np
import pandas as pd
import gradio as gr

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Path ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")

SLANG_PATH     = os.path.join(DATA_DIR,  "new_kamusalay.csv")
TOK_PATH       = os.path.join(MODEL_DIR, "cnn_bilstm_tokenizer.pkl")
MODEL_HS_PATH  = os.path.join(MODEL_DIR, "cnn_bilstm_HS_best.h5")
MODEL_ABU_PATH = os.path.join(MODEL_DIR, "cnn_bilstm_Abusive_best.h5")

MAX_SEQ_LEN = 50

STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "itu", "ini", "dengan", "adalah",
    "ada", "tidak", "juga", "untuk", "pada", "dalam", "sudah", "atau",
    "saya", "aku", "kamu", "dia", "mereka", "kita", "kami", "akan",
    "bisa", "telah", "bahwa", "karena", "oleh", "jadi", "lagi", "ya",
    "jangan", "tapi", "kalau", "mau", "aja", "deh", "sih", "lah",
    "dong", "nih", "kan", "nya", "yg", "dgn", "utk", "rt", "user", "url"
}


# =============================================================
# Compatibility Patches
# =============================================================
def _patch_keras_compat():
    """
    Tambahkan shim modul agar tokenizer .pkl yang disimpan dengan
    Keras 3.x (path: keras.src.legacy.*) bisa dimuat di Keras 2.x,
    dan sebaliknya.

    Masalah: pickle menyimpan path class secara literal.
    Jika versi Keras berubah, path kelas berubah → ModuleNotFoundError.
    """
    try:
        import keras

        # ── keras.src.legacy.preprocessing.text ──────────────
        # Dibutuhkan saat pkl disimpan dengan Keras 3.x
        def _ensure_module(parent, name, full_name):
            if not hasattr(parent, name):
                mod = types.ModuleType(full_name)
                setattr(parent, name, mod)
                sys.modules[full_name] = mod
            return getattr(parent, name)

        src_mod  = _ensure_module(keras,     "src",          "keras.src")
        leg_mod  = _ensure_module(src_mod,   "legacy",       "keras.src.legacy")
        prep_mod = _ensure_module(leg_mod,   "preprocessing","keras.src.legacy.preprocessing")

        if not hasattr(prep_mod, "text"):
            import tensorflow as tf
            prep_mod.text = tf.keras.preprocessing.text
            sys.modules["keras.src.legacy.preprocessing.text"] = tf.keras.preprocessing.text

        # ── keras.preprocessing.text ──────────────────────────
        # Dibutuhkan saat pkl disimpan dengan Keras 2.x
        if not hasattr(keras, "preprocessing"):
            import tensorflow as tf
            keras.preprocessing = tf.keras.preprocessing
            sys.modules["keras.preprocessing"]      = tf.keras.preprocessing
            sys.modules["keras.preprocessing.text"] = tf.keras.preprocessing.text

    except Exception as exc:
        print(f"    [warn] keras compat patch: {exc}")


def _load_model_compat(path: str):
    """
    MEM-BYPASS KESALAHAN DESERIALISASI JSON ANTAR VERSI KERAS.
    Daripada memuat config model `.h5` yang strukturnya sering konflik antara Keras 3 dan 2,
    kita bangun ulang arsitektur Keras 2 secara native dan tempelkan weights-nya.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input
    
    # Harus SAMA PERSIS dengan architecture di train_dl.py
    inp = Input(shape=(MAX_SEQ_LEN,), name="input")
    x = layers.Embedding(input_dim=15001, output_dim=128, name="embedding")(inp)
    x = layers.SpatialDropout1D(0.3, name="spatial_dropout")(x)
    
    x = layers.Conv1D(128, 3, activation="relu", padding="same", name="conv1d")(x)
    x = layers.MaxPooling1D(pool_size=2, name="maxpool")(x)
    
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, name="lstm1"), name="bilstm1")(x)
    x = layers.Bidirectional(layers.LSTM(32, name="lstm2"), name="bilstm2")(x)
    
    x = layers.Dense(128, activation="relu", name="dense")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    
    model = Model(inp, out)
    model.load_weights(path)
    return model


def _load_tokenizer_compat(path: str):
    """
    Muat tokenizer .pkl dengan fallback multi-tahap:
      1. Pickle biasa
      2. Setelah keras compat patch
      3. Rekonstruksi manual dari word_index (last resort)
    """
    # Langkah 1: coba langsung
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (ModuleNotFoundError, ImportError):
        pass

    # Langkah 2: terapkan patch lalu coba lagi
    _patch_keras_compat()
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass

    # Langkah 3: baca raw bytes, rekonstruksi Tokenizer kosong + word_index
    # (jika kelas sudah tidak ada tapi data word_index masih ada di pickle)
    try:
        import copyreg, io, pickle as _pkl
        import tensorflow as tf

        # Paksa register Tokenizer ke path lama agar unpickling berhasil
        TokenizerClass = tf.keras.preprocessing.text.Tokenizer

        # Buka dengan encoding 'latin-1' agar byte raw tetap terbaca
        with open(path, "rb") as f:
            raw = f.read()

        # Ganti path class di byte stream (bytes manipulation)
        # keras.src.legacy.preprocessing.text → keras.preprocessing.text
        raw = raw.replace(
            b"keras.src.legacy.preprocessing.text",
            b"keras.preprocessing.text"
        )
        tokenizer = pickle.loads(raw)
        print("    [info] Tokenizer dimuat via byte-patch.")
        return tokenizer
    except Exception as e3:
        raise RuntimeError(
            f"Semua metode load tokenizer gagal.\n"
            f"Pastikan versi TensorFlow di Space sama dengan saat training.\n"
            f"Detail: {e3}"
        )


# =============================================================
# Load komponen saat startup
# =============================================================
print("[+] Loading komponen model...")

def _load_slang(path):
    df = pd.read_csv(path, header=None, names=["slang", "formal"], encoding="latin-1")
    return dict(zip(df["slang"].str.lower(), df["formal"].str.lower()))

try:
    slang_dict = _load_slang(SLANG_PATH)
    print(f"    Slang dict  : {len(slang_dict):,} entri ✅")
except Exception as e:
    slang_dict = {}
    print(f"    Slang dict  : tidak dimuat ({e})")

try:
    tokenizer = _load_tokenizer_compat(TOK_PATH)
    print(f"    Tokenizer   : loaded ✅")
except Exception as e:
    tokenizer = None
    print(f"    Tokenizer   : GAGAL — {e}")

try:
    import tensorflow as tf
    model_hs  = _load_model_compat(MODEL_HS_PATH)
    model_abu = _load_model_compat(MODEL_ABU_PATH)
    print(f"    Model HS    : ✅  ({model_hs.count_params():,} params)")
    print(f"    Model ABU   : ✅  ({model_abu.count_params():,} params)")
    models_ready = True
except Exception as e:
    model_hs = model_abu = None
    models_ready = False
    print(f"    Model       : GAGAL — {e}")

if models_ready and tokenizer:
    print("[+] Startup selesai ✅")
else:
    print("[!] STARTUP GAGAL — periksa versi TensorFlow dan file model.")


# =============================================================
# Preprocessing
# =============================================================
def preprocess(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"\buser\b|\burl\b|\brt\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [slang_dict.get(t, t) for t in text.split()]
    tokens = [t for t in tokens if t not in STOPWORDS_ID]
    return " ".join(tokens)


def text_to_sequence(text: str) -> np.ndarray:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq    = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
    return padded


# =============================================================
# Prediksi
# =============================================================
def predict_with_detail(raw_text: str):
    if not models_ready or tokenizer is None:
        msg = (
            "❌ **Model belum siap.** Pastikan file berikut ada:\n"
            "- `models/cnn_bilstm_HS_best.h5`\n"
            "- `models/cnn_bilstm_Abusive_best.h5`\n"
            "- `models/cnn_bilstm_tokenizer.pkl`"
        )
        return msg, "", "", "", "", ""

    if not raw_text or not raw_text.strip():
        return "⚠️ Masukkan teks terlebih dahulu.", "", "", "", "", ""

    clean = preprocess(raw_text)
    if not clean:
        return "⚠️ Teks tidak valid setelah preprocessing.", "", "", "", "", ""

    X = text_to_sequence(clean)

    prob_hs  = float(model_hs.predict(X,  verbose=0)[0][0])
    prob_abu = float(model_abu.predict(X, verbose=0)[0][0])

    label_hs  = "🚨 HATE SPEECH"      if prob_hs  >= 0.5 else "✅ Bukan Hate Speech"
    label_abu = "🚨 ABUSIVE LANGUAGE" if prob_abu >= 0.5 else "✅ Tidak Abusive"

    flags = []
    if prob_hs  >= 0.5: flags.append("Hate Speech")
    if prob_abu >= 0.5: flags.append("Abusive Language")

    summary = (
        f"⚠️ Teks terdeteksi mengandung: **{' & '.join(flags)}**"
        if flags else
        "✅ Teks ini **aman** — tidak mengandung hate speech atau abusive language."
    )

    n_token = len(tokenizer.texts_to_sequences([clean])[0])
    detail  = (
        f"**Teks asli:** {raw_text}\n\n"
        f"**Teks setelah preprocessing:** `{clean}`\n\n"
        f"**Panjang token** (sebelum/sesudah padding): `{n_token}` → `{MAX_SEQ_LEN}`"
    )

    return (
        summary,
        f"{label_hs}",
        f"{'🔴' if prob_hs >= 0.5 else '🟢'} {prob_hs*100:.1f}%",
        f"{label_abu}",
        f"{'🔴' if prob_abu >= 0.5 else '🟢'} {prob_abu*100:.1f}%",
        detail,
    )


# =============================================================
# Gradio UI
# =============================================================
CSS = """
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 28px 24px; border-radius: 16px; text-align: center;
    margin-bottom: 24px; border: 1px solid rgba(255,255,255,0.08);
}
.main-header h1 { color: #e2e8f0; font-size: 1.9rem; margin: 0 0 6px 0; }
.main-header p  { color: #94a3b8; font-size: 0.95rem; margin: 0; }
.badge {
    display: inline-block; background: rgba(99,102,241,0.2);
    color: #a5b4fc; padding: 3px 10px; border-radius: 20px;
    font-size: 0.8rem; border: 1px solid rgba(99,102,241,0.4); margin-top: 8px;
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
    title="🛡️ CNN-BiLSTM Hate Speech Detector",
    css=CSS,
    theme=gr.themes.Base(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
    )
) as demo:

    gr.HTML("""
    <div class="main-header">
        <h1>🛡️ Indonesian Hate Speech Detector</h1>
        <p>Deteksi ujaran kebencian pada teks Bahasa Indonesia berbasis Deep Learning</p>
        <span class="badge">CNN-BiLSTM · ~2.12 juta parameter · TensorFlow/Keras</span>
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

        with gr.Column(scale=5):
            summary_out = gr.Markdown(label="Hasil")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🏷️ Hate Speech")
                    hs_label = gr.Textbox(label="Prediksi",   interactive=False)
                    hs_conf  = gr.Textbox(label="Confidence", interactive=False)
                with gr.Column():
                    gr.Markdown("#### 🏷️ Abusive Language")
                    abu_label = gr.Textbox(label="Prediksi",   interactive=False)
                    abu_conf  = gr.Textbox(label="Confidence", interactive=False)

    with gr.Accordion("🔬 Detail Preprocessing", open=False):
        detail_out = gr.Markdown()

    with gr.Accordion("ℹ️ Tentang Model", open=False):
        gr.Markdown("""
| Properti | Detail |
|---|---|
| **Arsitektur** | CNN-BiLSTM Hybrid |
| **Layer** | Embedding(vocab,128) → Conv1D(128,k=3) → MaxPool → BiLSTM(64) → BiLSTM(32) → Dense(128) → Sigmoid |
| **Parameter** | ~2.12 juta ✅ |
| **Dataset** | Indonesian Hate Speech Twitter (~13.169 tweet) |
| **Labels** | HS (Hate Speech), Abusive |
| **Strategi** | Binary Relevance — 1 model per label |
| **Framework** | TensorFlow 2.15 / Keras |

> **Kelompok 12 — PBA 2026 | Institut Teknologi Sumatera**
        """)

    gr.HTML("""
    <div style="text-align:center;color:#475569;font-size:.8rem;margin-top:20px;
                padding-top:16px;border-top:1px solid rgba(255,255,255,.06)">
        Powered by Hugging Face Spaces · Gradio · TensorFlow/Keras
        &nbsp;|&nbsp; Model: CNN-BiLSTM · Dataset: id-multi-label-hate-speech
    </div>
    """)

    OUTPUTS = [summary_out, hs_label, hs_conf, abu_label, abu_conf, detail_out]
    analyze_btn.click(fn=predict_with_detail, inputs=input_text, outputs=OUTPUTS)
    input_text.submit(fn=predict_with_detail, inputs=input_text, outputs=OUTPUTS)
    clear_btn.click(
        fn=lambda: tuple([""] * (len(OUTPUTS) + 1)),
        outputs=[input_text] + OUTPUTS,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
