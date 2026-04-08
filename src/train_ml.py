"""
main.py
=======
Script utama untuk menemukan model terbaik menggunakan PyCaret.

Strategi: Binary Relevance
  - Karena PyCaret tidak mendukung multi-label secara native,
    kita train satu model PyCaret per label (HS, Abusive).

Alur:
  1. Load & preprocess data (via preprocess.py)
  2. Untuk setiap label target:
     a. setup() — inisialisasi eksperimen PyCaret
     b. compare_models() — bandingkan semua classifier
     c. Ambil best model & metriknya
  3. Simpan ringkasan hasil ke output/results.csv
"""

import os
import sys
import warnings
import pandas as pd
from dotenv import load_dotenv

# Tambahkan folder root ke path agar bisa import config jika diperlukan
# Namun sekarang script ada di src/, jadi kita sesuaikan
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Suppress verbose warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Load env
load_dotenv(dotenv_path=os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".env.config")))

# === Konfigurasi dari .env ===
PYCARET_SESSION_ID = int(os.getenv("PYCARET_SESSION_ID", 42))
PYCARET_TRAIN_SIZE = float(os.getenv("PYCARET_TRAIN_SIZE", 0.8))
PYCARET_FOLD = int(os.getenv("PYCARET_FOLD", 5))
TARGET_LABELS = os.getenv("TARGET_LABELS", "HS,Abusive").split(",")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "D:/NLP/output")
RESULTS_FILE = os.getenv("RESULTS_FILE", "results.csv")

# Buat output directory jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_pycaret_for_label(feature_df: pd.DataFrame, labels_df: pd.DataFrame, label: str):
    """
    Jalankan PyCaret untuk satu label tertentu.

    Args:
        feature_df: DataFrame berisi fitur numerik (TF-IDF + abusive_count)
        labels_df: DataFrame berisi semua kolom label
        label: nama kolom label yang diproses (mis. 'HS' atau 'Abusive')

    Returns:
        best_model: Best model dari compare_models()
        results_df: DataFrame hasil compare_models
    """
    from pycaret.classification import (
        setup, compare_models, pull,
        get_config, save_model
    )

    print(f"\n{'='*60}")
    print(f"  LABEL: {label}")
    print(f"{'='*60}")

    # Gabungkan fitur + label target
    data = feature_df.copy()
    data[label] = labels_df[label].astype(int).values

    # Distribusi label
    vc = data[label].value_counts()
    print(f"  Distribusi: 0={vc.get(0,0)}, 1={vc.get(1,0)} "
          f"(rasio positif: {vc.get(1,0)/len(data)*100:.1f}%)")

    # Setup PyCaret
    print(f"\n[+] Setup PyCaret...")
    clf = setup(
        data=data,
        target=label,
        session_id=PYCARET_SESSION_ID,
        train_size=PYCARET_TRAIN_SIZE,
        fold=PYCARET_FOLD,
        n_jobs=1,                     # Mencegah MemoryError akibat multiprocessing
        verbose=False,
        html=False,
        fix_imbalance=True,          # Handle class imbalance dengan SMOTE
        normalize=True,               # Normalisasi fitur
        feature_selection=False,      # Sudah TF-IDF, skip feature selection
        remove_multicollinearity=False,
    )

    # Compare semua model
    print(f"[+] Membandingkan model (fold={PYCARET_FOLD})...")
    print("    Harap tunggu, ini mungkin butuh beberapa menit...\n")

    best_model = compare_models(
        sort="AUC",
        n_select=3,
        verbose=True,
        exclude=["catboost"],    # Exclude catboost agar lebih cepat
    )

    # Ambil hasil tabel
    results_df = pull()

    # Simpan best model
    model_path = os.path.join(OUTPUT_DIR, f"best_model_{label}")
    if isinstance(best_model, list):
        save_model(best_model[0], model_path)
        top_model = best_model[0]
    else:
        save_model(best_model, model_path)
        top_model = best_model

    print(f"\n  [OK] Model terbaik: {type(top_model).__name__}")
    print(f"  💾 Model disimpan ke: {model_path}.pkl")

    return top_model, results_df


def main():
    print("=" * 60)
    print("  PYCARET HATE SPEECH MODEL COMPARISON")
    print("  Label: HS, Abusive")
    print("=" * 60)

    # --- Step 1: Load & Preprocessing ---
    import preprocessing as pp
    df, tfidf, feature_df = pp.load_and_preprocess()

    # Pastikan semua target labels ada di dataframe
    for lbl in TARGET_LABELS:
        if lbl not in df.columns:
            raise ValueError(f"Label '{lbl}' tidak ditemukan di dataset! Kolom ada: {list(df.columns)}")

    labels_df = df[TARGET_LABELS].copy()

    # --- Step 2: PyCaret per label ---
    all_results = {}
    summary_rows = []

    for label in TARGET_LABELS:
        best_model, results_df = run_pycaret_for_label(feature_df, labels_df, label)

        # Simpan detail hasil
        results_df.insert(0, "Label", label)
        all_results[label] = results_df

        # Ambil baris pertama (best model)
        best_row = results_df.iloc[0].copy()
        summary_rows.append({
            "Label": label,
            "Best_Model": results_df.index[0],
            "AUC": round(best_row.get("AUC", float("nan")), 4),
            "F1": round(best_row.get("F1", float("nan")), 4),
            "Accuracy": round(best_row.get("Accuracy", float("nan")), 4),
            "Precision": round(best_row.get("Prec.", float("nan")), 4),
            "Recall": round(best_row.get("Recall", float("nan")), 4),
        })

    # --- Step 3: Simpan hasil ---
    print(f"\n{'='*60}")
    print("  RINGKASAN HASIL")
    print(f"{'='*60}")

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # Simpan ke CSV
    results_path = os.path.join(OUTPUT_DIR, RESULTS_FILE)
    summary_df.to_csv(results_path, index=False)
    print(f"\n💾 Ringkasan hasil disimpan ke: {results_path}")

    # Simpan detail semua model per label
    detail_path = os.path.join(OUTPUT_DIR, "results_detail.csv")
    all_results_combined = pd.concat(all_results.values(), ignore_index=True)
    all_results_combined.to_csv(detail_path, index=False)
    print(f"💾 Detail semua model disimpan ke: {detail_path}")

    print(f"\n{'='*60}")
    print("  SELESAI! Semua model telah dibandingkan.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
