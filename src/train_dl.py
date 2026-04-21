"""
train_dl.py
===========
Deep Learning — Hate Speech Detection (Indonesian Twitter)

Model   : CNN-BiLSTM (Hybrid) — ~2.12 juta parameter ✅ < 10 juta
Dataset : data/raw/re_dataset.csv  (~13.169 tweet, multi-label)
Labels  : HS, Abusive (Binary Relevance: satu model per label)

Mengapa CNN-BiLSTM?
  ├── Data berupa teks pendek Twitter (~17 kata/tweet)
  ├── CNN 1D → menangkap pola lokal: frasa / n-gram khas hate speech
  ├── BiLSTM → membaca konteks dua arah (kiri→kanan & kanan→kiri)
  └── Kombinasi keduanya unggul secara empiris pada teks media sosial pendek

Alur:
  1. Load & preprocess data (normalisasi slang, stopword removal)
  2. Tokenisasi & padding sekuens
  3. Train CNN-BiLSTM untuk label HS dan Abusive
  4. Evaluasi (Accuracy, AUC, F1, Precision, Recall)
  5. Simpan model (.h5) + tokenizer (.pkl) ke OUTPUT_DIR
"""

import os
import re
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(dotenv_path=os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", ".env.config")
))

# ============================================================
# Konfigurasi
# ============================================================
DATASET_DIR  = os.getenv("DATASET_DIR",     "D:/NLP/data/raw")
DATASET_FILE = os.getenv("DATASET_FILE",    "re_dataset.csv")
SLANG_FILE   = os.getenv("SLANG_DICT_FILE", "new_kamusalay.csv")
OUTPUT_DIR   = os.getenv("OUTPUT_DIR",      "D:/NLP/data/processed")
TARGET_LABELS = os.getenv("TARGET_LABELS",  "HS,Abusive").split(",")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Hyperparameter CNN-BiLSTM ─────────────────────────────────
MAX_VOCAB        = 15000  # Ukuran vocabulary
MAX_SEQ_LEN      = 50     # Panjang sekuens (avg tweet ~17 kata, max 52)
EMBEDDING_DIM    = 128    # Dimensi embedding
CNN_FILTERS      = 128    # Jumlah filter Conv1D
CNN_KERNEL       = 3      # Kernel size (trigram)
LSTM_UNITS       = 64     # Unit BiLSTM per arah
DENSE_UNITS      = 128    # Unit Dense layer
DROPOUT_RATE     = 0.3    # Dropout untuk regularisasi
LEARNING_RATE    = 1e-3   # Adam learning rate
EPOCHS           = 20     # Max epoch (EarlyStopping aktif)
BATCH_SIZE       = 64     # Batch size
VALIDATION_SPLIT = 0.15   # Proporsi validasi dari data train
RANDOM_SEED      = 42
MAX_PARAMS       = 10_000_000  # Batas parameter model

np.random.seed(RANDOM_SEED)


# ============================================================
# Preprocessing
# ============================================================
STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "itu", "ini", "dengan", "adalah",
    "ada", "tidak", "juga", "untuk", "pada", "dalam", "sudah", "atau",
    "saya", "aku", "kamu", "dia", "mereka", "kita", "kami", "akan",
    "bisa", "telah", "bahwa", "karena", "oleh", "jadi", "lagi", "ya",
    "jangan", "tapi", "kalau", "mau", "aja", "deh", "sih", "lah",
    "dong", "nih", "kan", "nya", "yg", "dgn", "utk", "rt", "user", "url"
}


def _load_slang(path: str) -> dict:
    df = pd.read_csv(path, header=None, names=["slang", "formal"], encoding="latin-1")
    return dict(zip(df["slang"].str.lower(), df["formal"].str.lower()))


def _clean(text: str, slang_dict: dict) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\buser\b|\burl\b|\brt\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [slang_dict.get(t, t) for t in text.split()]
    tokens = [t for t in tokens if t not in STOPWORDS_ID]
    return " ".join(tokens)


def load_data() -> pd.DataFrame:
    """Load & preprocess dataset. Return DataFrame dengan kolom 'clean_text'."""
    print("=" * 60)
    print("  PREPROCESSING PIPELINE")
    print("=" * 60)

    dataset_path = os.path.join(DATASET_DIR, DATASET_FILE)
    slang_path   = os.path.join(DATASET_DIR, SLANG_FILE)

    print(f"[+] Load dataset   : {dataset_path}")
    df = pd.read_csv(dataset_path, encoding="latin-1")
    print(f"    Baris total    : {len(df):,}")
    print(f"    Kolom          : {list(df.columns)}")

    print(f"[+] Load kamus slang: {slang_path}")
    slang_dict = _load_slang(slang_path)
    print(f"    Entri slang    : {len(slang_dict):,}")

    tweet_col = df.columns[0]
    print("[+] Cleaning & normalisasi teks...")
    df["clean_text"] = df[tweet_col].apply(lambda t: _clean(t, slang_dict))

    before = len(df)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    print(f"    Data bersih    : {len(df):,} (hapus {before - len(df)} baris kosong)")
    print("=" * 60)
    return df


# ============================================================
# Tokenisasi & Padding
# ============================================================
def build_sequences(texts: list, tokenizer=None, fit: bool = True):
    """Tokenisasi ke sekuens integer lalu padding. Return (X, tokenizer)."""
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    if fit or tokenizer is None:
        tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)

    seqs   = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
    return padded, tokenizer


# ============================================================
# Arsitektur Model: CNN-BiLSTM
# ============================================================
def _make_metrics():
    """
    Buat objek metrik BARU setiap kali dipanggil.
    WAJIB: jangan share satu list metrik antar compile() karena Keras
    akan otomatis menambah suffix _1, _2, … → KeyError saat evaluate().
    """
    import tensorflow as tf
    return [
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]


def _get_metric(d: dict, name: str, default: float = 0.0) -> float:
    """Cari metrik di dict; fallback prefix-matching agar tahan suffix _1, _2, …"""
    if name in d:
        return d[name]
    for k in d:
        if k.startswith(name):
            return d[k]
    return default


def build_cnn_bilstm(vocab_size: int, name: str = "CNN_BiLSTM"):
    """
    CNN-BiLSTM Hybrid — model terpilih untuk data ini.

    Arsitektur:
        Embedding(vocab_size, 128)
            → SpatialDropout1D(0.3)
            → Conv1D(128, kernel=3, relu, same)   ← tangkap trigram lokal
            → MaxPooling1D(2)                      ← kompres fitur
            → BiLSTM(64, return_sequences=True)   ← konteks dua arah layer 1
            → BiLSTM(32)                           ← ringkas representasi
            → Dense(128, relu)
            → Dropout(0.3)
            → Dense(1, sigmoid)                   ← output biner

    Parameter: ~2.12 juta ✅ < 10 juta
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input

    inp = Input(shape=(MAX_SEQ_LEN,), name="input")

    # Embedding
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        mask_zero=False,          # Conv1D tidak mendukung masking
        name="embedding"
    )(inp)
    x = layers.SpatialDropout1D(DROPOUT_RATE, name="spatial_dropout")(x)

    # CNN: ekstraksi fitur lokal (n-gram / frasa)
    x = layers.Conv1D(
        CNN_FILTERS, CNN_KERNEL,
        activation="relu", padding="same",
        name="conv1d"
    )(x)
    x = layers.MaxPooling1D(pool_size=2, name="maxpool")(x)

    # BiLSTM: konteks sekuensial dua arah
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS, return_sequences=True, name="lstm1"),
        name="bilstm1"
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS // 2, name="lstm2"),
        name="bilstm2"
    )(x)

    # Klasifikasi
    x = layers.Dense(DENSE_UNITS, activation="relu", name="dense")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inp, out, name=name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=_make_metrics()   # objek baru setiap compile
    )
    return model


# ============================================================
# Training & Evaluasi
# ============================================================
def train(X: np.ndarray, y: np.ndarray, label: str, vocab_size: int):
    """
    Train CNN-BiLSTM untuk satu label.

    Return:
        model   : Model Keras terlatih
        history : Training history
        metrics : Dict metrik evaluasi pada test set
    """
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import f1_score, classification_report

    tf.random.set_seed(RANDOM_SEED)

    print(f"\n{'='*60}")
    print(f"  TRAINING — Label: {label}  |  Model: CNN-BiLSTM")
    print(f"{'='*60}")

    # Distribusi label
    pos = int(y.sum()); neg = len(y) - pos
    print(f"  Distribusi  : 0={neg:,}, 1={pos:,} ({pos/len(y)*100:.1f}% positif)")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Train/Test  : {len(X_train):,} / {len(X_test):,}")

    # Class weight untuk imbalance
    cw  = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(np.unique(y_train), cw)}
    print(f"  Class weight: {class_weight}")

    # Bangun model
    model = build_cnn_bilstm(vocab_size, name=f"cnn_bilstm_{label}")

    # Validasi parameter
    n_params = model.count_params()
    print(f"\n  📐 Total parameter : {n_params:,}")
    print(f"     Batas maksimum  : {MAX_PARAMS:,}")
    if n_params > MAX_PARAMS:
        raise RuntimeError(
            f"Model memiliki {n_params:,} param, melebihi batas {MAX_PARAMS:,}. "
            "Kurangi LSTM_UNITS, CNN_FILTERS, atau EMBEDDING_DIM."
        )
    print(f"     Status          : ✅ Di bawah batas 10 juta parameter")

    model.summary(print_fn=lambda s: print("  " + s))

    # Callbacks
    ckpt_path = os.path.join(OUTPUT_DIR, f"cnn_bilstm_{label}_best.h5")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=5, mode="max",
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_auc", save_best_only=True,
            mode="max", verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-6, verbose=1
        ),
    ]

    print(f"\n[+] Mulai training (max_epochs={EPOCHS}, batch={BATCH_SIZE})...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluasi
    print("\n[+] Evaluasi pada test set...")
    raw = dict(zip(model.metrics_names, model.evaluate(X_test, y_test, verbose=0)))
    metrics = {
        "accuracy" : _get_metric(raw, "accuracy"),
        "auc"      : _get_metric(raw, "auc"),
        "precision": _get_metric(raw, "precision"),
        "recall"   : _get_metric(raw, "recall"),
    }
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    metrics["f1"] = f1_score(y_test, y_pred)

    print(f"\n  📊 Hasil — {label} (CNN-BiLSTM):")
    print(f"     Accuracy  : {metrics['accuracy']:.4f}")
    print(f"     AUC       : {metrics['auc']:.4f}")
    print(f"     F1-Score  : {metrics['f1']:.4f}")
    print(f"     Precision : {metrics['precision']:.4f}")
    print(f"     Recall    : {metrics['recall']:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Non', label])}")
    print(f"  💾 Best model: {ckpt_path}")

    return model, history, metrics


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  CNN-BiLSTM — HATE SPEECH DETECTION")
    print(f"  Dataset : {DATASET_FILE}")
    print(f"  Labels  : {TARGET_LABELS}")
    print("=" * 60)

    # Step 1: Load & preprocess
    df = load_data()
    available_labels = [l for l in TARGET_LABELS if l in df.columns]
    if not available_labels:
        raise ValueError(f"Tidak ada label valid. Kolom: {list(df.columns)}")

    # Step 2: Tokenisasi
    print("\n[+] Tokenisasi & padding...")
    X, tokenizer = build_sequences(df["clean_text"].tolist(), fit=True)
    vocab_size = min(len(tokenizer.word_index) + 1, MAX_VOCAB + 1)
    print(f"    Vocabulary : {vocab_size:,}")
    print(f"    Shape X    : {X.shape}")

    tok_path = os.path.join(OUTPUT_DIR, "cnn_bilstm_tokenizer.pkl")
    with open(tok_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"    💾 Tokenizer: {tok_path}")

    # Step 3: Train per label
    all_metrics = {}
    for label in available_labels:
        y = df[label].astype(int).values
        _, _, metrics = train(X, y, label, vocab_size)
        all_metrics[label] = metrics

    # Step 4: Ringkasan
    print("\n" + "=" * 60)
    print("  RINGKASAN AKHIR — CNN-BiLSTM")
    print("=" * 60)
    rows = []
    for label, m in all_metrics.items():
        rows.append({
            "Label"    : label,
            "Accuracy" : round(m["accuracy"],  4),
            "AUC"      : round(m["auc"],       4),
            "F1-Score" : round(m["f1"],        4),
            "Precision": round(m["precision"], 4),
            "Recall"   : round(m["recall"],    4),
        })
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))

    out_path = os.path.join(OUTPUT_DIR, "cnn_bilstm_results.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\n💾 Ringkasan disimpan: {out_path}")
    print("=" * 60)
    print("  SELESAI!")
    print("=" * 60)


if __name__ == "__main__":
    main()
