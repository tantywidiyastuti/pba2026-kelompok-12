---
title: ML Hate Speech Detector 
emoji: 🛡️
colorFrom: emerald
colorTo: green
sdk: gradio
sdk_version: 4.20.0
app_file: app.py
pinned: false
python_version: "3.10"
---

# 🛡️ Indonesian Hate Speech Detector — PyCaret (Machine Learning)

Deteksi **hate speech** pada teks Bahasa Indonesia menggunakan model Machine Learning **(PyCaret AutoML / LightGBM)**.

**Kelompok 12 — PBA 2026 | Institut Teknologi Sumatera**

---

## 📁 Struktur File (upload semua ke Space)

```
/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── best_model_HS.pkl            ← hasil training PyCaret
│   └── tfidf_vectorizer.pkl         ← hasil TF-IDF
└── data/
    ├── new_kamusalay.csv            ← kamus slang Indonesia
    └── abusive.csv                  ← kamus kata abusif (untuk ekstraksi fitur)
```

## 🧠 Model

| Properti | Detail |
|---|---|
| Pipeline | TF-IDF + Machine Learning |
| Framework | PyCaret / LightGBM |
| Labels | HS (Hate Speech) |
