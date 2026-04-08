import os
import sys
import joblib

# Tambahkan path root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import load_and_preprocess

def main():
    print("[+] Mengekstrak TF-IDF dari dataset...")
    df, tfidf, feature_df = load_and_preprocess()
    
    output_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models", "tfidf_vectorizer.pkl"))
    
    print(f"\n[+] Menyimpan TF-IDF vectorizer ke: {output_path}")
    joblib.dump(tfidf, output_path)
    print("[+] Selesai! Model TF-IDF siap digunakan oleh Streamlit.")

if __name__ == "__main__":
    main()
