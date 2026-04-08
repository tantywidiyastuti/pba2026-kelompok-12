"""
preprocess.py
=============
Modul preprocessing teks untuk dataset hate speech bahasa Indonesia.

Langkah-langkah:
1. Lowercase & pembersihan teks
2. Normalisasi kata slang/typo via new_kamusalay.csv
3. Feature extraction via TF-IDF + abusive word count
"""

import os
import re
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment variables dari .env.config di folder root
load_dotenv(dotenv_path=os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".env.config")))

# === Konfigurasi dari .env ===
DATASET_DIR = os.getenv("DATASET_DIR")
DATASET_FILE = os.getenv("DATASET_FILE")
SLANG_DICT_FILE = os.getenv("SLANG_DICT_FILE")
ABUSIVE_FILE = os.getenv("ABUSIVE_FILE")
TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", 5000))
TFIDF_NGRAM_MIN = int(os.getenv("TFIDF_NGRAM_MIN", 1))
TFIDF_NGRAM_MAX = int(os.getenv("TFIDF_NGRAM_MAX", 2))
TARGET_LABELS = os.getenv("TARGET_LABELS", "HS,Abusive").split(",")


# === Stopwords bahasa Indonesia (built-in, tanpa library tambahan) ===
STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "itu", "ini", "dengan", "adalah",
    "ada", "tidak", "juga", "untuk", "pada", "dalam", "sudah", "atau",
    "saya", "aku", "kamu", "dia", "mereka", "kita", "kami", "akan",
    "bisa", "telah", "bahwa", "karena", "oleh", "jadi", "lagi", "ya",
    "jangan", "tapi", "tapi", "kalau", "mau", "bisa", "aja", "deh",
    "sih", "lah", "dong", "nih", "kan", "nya", "yg", "dgn", "utk",
    "rt", "user", "url"
}


def load_slang_dict(path: str) -> dict:
    """Muat kamus slang dari new_kamusalay.csv."""
    df = pd.read_csv(path, header=None, names=["slang", "formal"], encoding="latin-1")
    return dict(zip(df["slang"].str.lower(), df["formal"].str.lower()))


def load_abusive_words(path: str) -> set:
    """Muat daftar kata abusif dari abusive.csv."""
    df = pd.read_csv(path, encoding="latin-1")
    col = df.columns[0]
    return set(df[col].str.lower().str.strip().tolist())


def clean_text(text: str) -> str:
    """Bersihkan teks: lowercase, hapus karakter non-alfabet."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Hapus mention USER dan URL placeholder
    text = re.sub(r"\buser\b|\burl\b", " ", text)
    # Hapus karakter non-alfabet dan non-spasi
    text = re.sub(r"[^a-z\s]", " ", text)
    # Hapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_slang(text: str, slang_dict: dict) -> str:
    """Normalisasi kata slang/typo ke bentuk formal."""
    tokens = text.split()
    tokens = [slang_dict.get(tok, tok) for tok in tokens]
    return " ".join(tokens)


def remove_stopwords(text: str) -> str:
    """Hapus stopwords bahasa Indonesia."""
    tokens = [tok for tok in text.split() if tok not in STOPWORDS_ID]
    return " ".join(tokens)


def count_abusive_words(text: str, abusive_set: set) -> int:
    """Hitung jumlah kata abusif dalam teks."""
    tokens = text.split()
    return sum(1 for tok in tokens if tok in abusive_set)


def preprocess_pipeline(text: str, slang_dict: dict) -> str:
    """Pipeline lengkap preprocessing teks."""
    text = clean_text(text)
    text = normalize_slang(text, slang_dict)
    text = remove_stopwords(text)
    return text


def load_and_preprocess():
    """
    Load dataset dan jalankan full preprocessing.

    Returns:
        df_processed (pd.DataFrame): DataFrame dengan kolom 'clean_text',
                                     'abusive_count', dan semua kolom label.
        tfidf (TfidfVectorizer): Fitted TfidfVectorizer.
        feature_df (pd.DataFrame): DataFrame fitur numerik (TF-IDF + abusive_count).
    """
    print("=" * 60)
    print("  PREPROCESSING PIPELINE")
    print("=" * 60)

    # --- Load data ---
    dataset_path = os.path.normpath(os.path.join(DATASET_DIR, DATASET_FILE))
    slang_path = os.path.normpath(os.path.join(DATASET_DIR, SLANG_DICT_FILE))
    abusive_path = os.path.normpath(os.path.join(DATASET_DIR, ABUSIVE_FILE))

    print(f"[+] Load dataset: {dataset_path}")
    df = pd.read_csv(dataset_path, encoding="latin-1")
    print(f"    Total baris: {len(df)}")

    print(f"[+] Load kamus slang: {slang_path}")
    slang_dict = load_slang_dict(slang_path)
    print(f"    Total entri slang: {len(slang_dict)}")

    print(f"[+] Load leksikon abusif: {abusive_path}")
    abusive_set = load_abusive_words(abusive_path)
    print(f"    Total kata abusif: {len(abusive_set)}")

    # --- Preprocessing teks ---
    print("\n[+] Preprocessing teks...")
    tweet_col = df.columns[0]  # Kolom pertama = Tweet
    df["clean_text"] = df[tweet_col].apply(
        lambda t: preprocess_pipeline(t, slang_dict)
    )

    # --- Fitur tambahan: abusive word count ---
    df["abusive_count"] = df["clean_text"].apply(
        lambda t: count_abusive_words(t, abusive_set)
    )

    # Hapus baris dengan teks kosong setelah preprocessing
    before = len(df)
    df = df[df["clean_text"].str.strip().str.len() > 0].reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"    [!] Hapus {before - after} baris dengan teks kosong")

    print(f"    Selesai! Total data bersih: {len(df)}")

    # --- TF-IDF Vectorization ---
    print(f"\n[+] TF-IDF Vectorization (max_features={TFIDF_MAX_FEATURES}, ngram=({TFIDF_NGRAM_MIN},{TFIDF_NGRAM_MAX}))...")
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(TFIDF_NGRAM_MIN, TFIDF_NGRAM_MAX),
        sublinear_tf=True,
        min_df=2
    )
    tfidf_matrix = tfidf.fit_transform(df["clean_text"])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=tfidf.get_feature_names_out()
    )
    print(f"    Dimensi TF-IDF: {tfidf_df.shape}")

    # --- Gabungkan TF-IDF + fitur tambahan ---
    feature_df = pd.concat(
        [tfidf_df, df[["abusive_count"]].reset_index(drop=True)],
        axis=1
    )
    print(f"    Total fitur: {feature_df.shape[1]}")
    print("=" * 60)

    return df, tfidf, feature_df


if __name__ == "__main__":
    df, tfidf, feature_df = load_and_preprocess()
    print("\nContoh teks bersih (5 baris pertama):")
    print(df[["clean_text", "abusive_count", "HS", "Abusive"]].head())
