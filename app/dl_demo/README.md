---
title: CNN-BiLSTM Hate Speech Detector (DL)
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.20.0
app_file: app.py
pinned: false
python_version: "3.10"
---

# 🛡️ Indonesian Hate Speech Detector — CNN-BiLSTM (Deep Learning)

Deteksi **hate speech** dan **abusive language** pada teks Bahasa Indonesia
menggunakan model Deep Learning **CNN-BiLSTM**.

**Kelompok 12 — PBA 2026 | Institut Teknologi Sumatera**

---

## 📁 Struktur File (upload semua ke Space)

```
/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── cnn_bilstm_HS_best.h5        ← hasil training
│   ├── cnn_bilstm_Abusive_best.h5   ← hasil training
│   └── cnn_bilstm_tokenizer.pkl     ← hasil training
└── data/
    └── new_kamusalay.csv            ← kamus slang Indonesia
```

## 🧠 Model

| Properti | Detail |
|---|---|
| Arsitektur | CNN-BiLSTM Hybrid |
| Parameter | ~2.12 juta |
| Framework | TensorFlow / Keras |
| Labels | HS (Hate Speech), Abusive |
