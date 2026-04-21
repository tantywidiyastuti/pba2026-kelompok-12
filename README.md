# NLP Text Classification Benchmark

Natural Language Processing Project — Institut Teknologi Sumatera (ITERA)

## Project Description

This project aims to compare the performance of **Machine Learning (ML)** and **Deep Learning (DL)** approaches for Natural Language Processing (NLP) tasks, specifically for identifying Hate Speech and Abusive Language in Indonesian Tweets.

Machine Learning models are implemented using **PyCaret AutoML**, while the Deep Learning models are implemented using **TensorFlow/Keras** with a custom CNN-BiLSTM architecture.

Both approaches are evaluated and compared on the same dataset to determine which method performs better for the selected text classification task.

---

## Team Members

| Name                     | NIM       | GitHub Username |
| ------------------------ | --------- | --------------- |
| Tanty Widiyastuti        | 123450094 | tantywidiyastuti|
| Mayada                   | 121450145 | -               |
| Adisty Syawalda Ariyanto | 121450136 | adistyS         |

---

## Dataset

Dataset used in this project is sourced from a public NLP open repository focusing on Indonesian multi-label hate speech.

Dataset Link: [id-multi-label-hate-speech-and-abusive-language-detection](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection?tab=readme-ov-file)

---

## Project Objectives

The objectives of this project are:

* Perform **Exploratory Data Analysis (EDA)** on the selected dataset
* Implement **Machine Learning models using PyCaret AutoML**
* Implement **Deep Learning models using TensorFlow / Keras (CNN-BiLSTM)**
* Compare the performance between ML and DL models
* Deploy interactive demos using **Hugging Face Spaces**
* Publish a scientific report in **ArXiv format**

---

## Repository Structure

```
pba2026-kelompok-12/
│
├── data/
│   ├── raw/                 # Base datasets and dictionaries (e.g., new_kamusalay.csv)
│   └── processed/           # Cleaned data ready for modeling
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_pycaret_model.ipynb
│
├── src/
│   ├── preprocessing.py     # Central preprocessing scripts
│   ├── train_ml.py          # PyCaret model training pipeline
│   └── train_dl.py          # TensorFlow / Keras model training pipeline
│
├── models/
│   ├── model_ml/            # Saved PyCaret models (.pkl)
│   └── model_dl/            # Saved TensorFlow models (.h5) & Tokenizer (.pkl)
│
├── results_dl/              # Output charts, confusion matrices, and training curves
│
├── app/
│   ├── ml_demo/             # Gradio App bundle for Machine Learning HF Space
│   └── dl_demo/             # Gradio App bundle for Deep Learning HF Space
│
├── paper/                   # Standardized scientific report (ArXiv format)
│
├── requirements.txt         # Dependencies for local development & training
└── README.md
```

---

## Machine Learning Approach

Machine Learning models are developed using **PyCaret AutoML**.
Several algorithms are compared automatically, with the **LightGBM** classifier yielding the best performance based on evaluation metrics for tabular TF-IDF representations.

* **Pipeline:** TF-IDF Vectorizer + Abusive Word Count Feature
* **Best Model:** LightGBM Node
* **Strategy:** Binary Relevance

---

## Deep Learning Approach

The Deep Learning model is implemented using **TensorFlow/Keras**.

We designed a hybrid **CNN-BiLSTM** architecture specifically suited for short social media texts.
* **Conv1D** is utilized to capture local spatial features (n-grams/discriminative phrases).
* **BiLSTM** captures bidirectional sequential contexts.
* **Size:** Parameter count is explicitly restricted to under 10 million (~2.12M params).

Model checkpoints are logged automatically via callbacks and saved as `.h5` files.

---

## Deployment

Two interactive Application User Interfaces (UI) will be deployed using **Hugging Face Spaces** built upon the **Gradio** framework:

1. **Machine Learning Model (PyCaret UI)**
   [Live Demo: NLP-Hate-Speech-Detection](https://huggingface.co/spaces/TantyWidiyastuti/NLP-Hate-Speech-Detection)

2. **Deep Learning Model (TensorFlow/Keras UI)** 
   [Live Demo: NLP-Hate-speech-Detection-DL](https://huggingface.co/spaces/TantyWidiyastuti/NLP-Hate-speech-Detection-DL)

---

## Scientific Paper

The final project report will be written in **LaTeX using ArXiv format** and will include:

* Dataset description
* Methodology
* Experiment setup
* Benchmark results
* Comparative analysis

ArXiv Link: *(to be added)*

---

## Pemanfaatan Penggunaan AI

Dalam penyelesaian proyek ini, tim kami memanfaatkan teknologi *Generative AI* (Kecerdasan Buatan Generatif) sebagai asisten pengembangan utama. Pemanfaatan AI dalam pengerjaan proyek ini mencakup:
* **Pengembangan dan Refactoring Kode**: Membantu merancang arsitektur Deep Learning (CNN-BiLSTM < 10 juta parameter), mengkonversi Streamlit ke **Gradio**, serta men-setup integrasi model `tf.keras` dengan `PyCaret AutoML`.
* **Debugging Secara Mandiri**: Mengidentifikasi konfigurasi konflik versi Keras (DTypePolicy/batch_shape metadata), memperbaiki `KeyError` pada array evaluasi metric Keras, serta mencari pemecahan dependencies konflik tingkat lanjut (*dependency hell*) pada container Hugging Face Spaces.
* **Penyusunan Dokumentasi & Struktur Direktori**: Merapikan *repository* sesuai standar rekayasa perangkat lunak (pemisahan _Local Training_ dan _HuggingFace Bundle_), dan merapikan deskripsi *README*.

*Penting dicatat: Semua hasil rumusan kode dan ide (output) yang diberikan oleh asisten AI telah ditinjau per baris, dipahami algoritmanya, dan divalidasi secara manual oleh anggota tim untuk memastikan model beroperasi sesuai dengan tujuan pembelajaran NLP yang diharapkan.*

**Lihat beberapa histori prompt di sini:**

1. *"ganti saja streamlit, jadi Gradio yang akan di deploy ke hunggingface-nya"*
2. *"buatkan kode untuk train_dl di src/train_dl.py dengan mengarah pada data data/raw, berikan rekomendasi model yang fit"*
3. *"saya ingin model yang kurang dari 10 juta parameter"*
4. *"note: This error originates from a subprocess... error: metadata-generation-failed... pandas"*
5. *"Exit code: 1. Reason: ImportError: cannot import name 'HfFolder' from 'huggingface_hub'"*
6. *"buatkan satu folder yang mana isi dari folder tersebut bisa saya masukkan ke dalam space hunggingface saya untuk deploy khusus dl, telusuri semua file dan buatkan app.py- sekalian khusus deep learning ini"*
7. *"Model: GAGAL — cnn_bilstm_HS_best.h5: Error when deserializing class 'Embedding'... Unknown dtype policy: 'DTypePolicy'"*

---

## Course Information

Course: **Pemrosesan Bahasa Alami**
Program: **Sains Data — Institut Teknologi Sumatera**
Semester: **Genap 2025/2026**

Instructor:
Martin C.T. Manullang
