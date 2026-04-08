---
title: ITera NLP Hate Speech Detection
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.20.0
app_file: app/ml_demo/app.py
pinned: false
---

# NLP Text Classification Benchmark

Natural Language Processing Project — Institut Teknologi Sumatera (ITERA)

## Project Description

This project aims to compare the performance of **Machine Learning (ML)** and **Deep Learning (DL)** approaches for Natural Language Processing (NLP) tasks.

Machine Learning models will be implemented using **PyCaret AutoML**, while the Deep Learning model will be implemented using **PyTorch**.

Both approaches will be evaluated and compared on the same dataset to determine which method performs better for the selected text classification task.

---

## Team Members

| Name                     | NIM       | GitHub Username |
| ------------------------ | --------- | --------------- |
| Tanty Widiyastuti        | 123450094 | tantywidiyastuti|
| Mayada                   | 121450145 | -               |
| Adisty Syawalda Ariyanto | 121450136 | adistyS         |

---

## Dataset

Dataset used in this project will be sourced from public NLP datasets such as Kaggle, Hugging Face Datasets, or other open repositories.

Dataset Link:
(https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection?tab=readme-ov-file)

---

## Project Objectives

The objectives of this project are:

* Perform **Exploratory Data Analysis (EDA)** on the selected dataset
* Implement **Machine Learning models using PyCaret AutoML**
* Implement **Deep Learning models using PyTorch**
* Compare the performance between ML and DL models
* Deploy interactive demos using **Hugging Face Spaces**
* Publish a scientific report in **ArXiv format**

---

## Repository Structure

```
pba2026-nama-kelompok 12
│
├── data
│   ├── raw
│   └── processed
│
├── notebooks
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_pycaret_model.ipynb
│   └── 04_deep_learning.ipynb
│
├── src
│   ├── preprocessing.py
│   ├── train_ml.py
│   ├── train_dl.py
│   └── utils.py
│
├── models
│
├── app
│   ├── ml_demo
│   └── dl_demo
│
├── paper
│
└── README.md
```

---

## Machine Learning Approach

Machine Learning models will be developed using **PyCaret AutoML**.
Several algorithms will be compared automatically, and the best-performing model will be selected based on evaluation metrics.

Examples of algorithms evaluated:

* Logistic Regression
* Random Forest
* Support Vector Machine
* Gradient Boosting

---

## Deep Learning Approach

The Deep Learning model will be implemented using **PyTorch**.

Possible architectures include:

* LSTM
* GRU
* CNN for text classification
* Lightweight Transformer models

The model will be trained and evaluated using standard NLP evaluation metrics.

---

## Deployment

Two interactive demos will be deployed using **Hugging Face Spaces**:

* **Machine Learning Model (PyCaret)**
  
  [deploy](https://huggingface.co/spaces/TantyWidiyastuti/NLP-Hate-Speech-Detection)

* **Deep Learning Model (PyTorch)**
  *(link will be added later)*

---

## Scientific Paper

The final project report will be written in **LaTeX using ArXiv format** and will include:

* Dataset description
* Methodology
* Experiment setup
* Benchmark results
* Comparative analysis

ArXiv Link:
*(to be added)*

---

## Pemanfaatan Penggunaan AI

Dalam penyelesaian proyek ini, tim kami memanfaatkan teknologi *Generative AI* (Kecerdasan Buatan Generatif) sebagai asisten pengembangan utama. Pemanfaatan AI dalam pengerjaan proyek ini mencakup:
* **Pengembangan dan Refactoring Kode**: Membantu menulis kode dan skrip yang efisien, termasuk menata ulang *framework* dari Streamlit ke Gradio untuk kebutuhan antarmuka interaktif.
* **Debugging dan Resolusi Konflik Server**: Mengidentifikasi berbagai solusi secara presisi untuk konflik *library* tingkat lanjut (*dependency hell*) pada saat mengatur repositori untuk *deployment* di Hugging Face Spaces.
* **Penyusunan Dokumentasi**: Membantu dalam penambahan komentar pada kode, merapikan deskripsi *README*, dan penataan format.

*Penting dicatat: Semua hasil rumusan kode dan ide (output) yang diberikan oleh asisten AI telah ditinjau per baris, dipahami algoritmanya, dan divalidasi secara manual oleh anggota tim untuk memastikan aplikasi beroperasi sesuai dengan tujuan pembelajaran NLP yang diharapkan.*

**Lihat histori prompt di sini:**

1. *"ganti saja streamlit , jadi Gradio yang akan di deploy ke hunggingface-nya"*
2. *"saya ingin memasukkan file secara manuall ke dalam hunggingfacenya , file apa saja yang harus saya masukkan"*
3. *"note: This error originates from a subprocess, and is likely not a problem with pip. error: metadata-generation-failed × Encountered error while generating package metadata. ╰─> pandas"*
4. *"--> ERROR: process "/bin/sh -c pip install --no-cache-dir -r /tmp/requirements.txt streamlit==1.32.0 spaces" did not complete successfully: exit code: 1"*
5. *"--> ERROR: docker.io/library/python:3.1: not found"*
6. *"Exit code: 1. Reason: Traceback (most recent call last): File "/app/app/ml_demo/app.py", line 5 ... ImportError: cannot import name 'HfFolder' from 'huggingface_hub'"*
7. *"TypeError: unhashable type: 'dict' ... ValueError: When localhost is not accessible, a shareable link must be created."*
---

## Course Information

Course: **Pemrosesan Bahasa Alami**
Program: **Sains Data — Institut Teknologi Sumatera**
Semester: **Genap 2025/2026**

Instructor:
Martin C.T. Manullang
