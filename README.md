# NLP Text Classification Benchmark

Natural Language Processing Project — Institut Teknologi Sumatera (ITERA)

---

## Project Description

This project aims to conduct a systematic comparison between **Machine Learning (ML)** and **Deep Learning (DL)** approaches for text classification tasks in Natural Language Processing (NLP), specifically focusing on hate speech detection in Indonesian Twitter data.

The dataset used in this study is derived from the paper:

**"Multi-label Hate Speech and Abusive Language Detection in Indonesian Twitter"** by Ibrohim & Budi (2019).

Although the dataset provides multi-label annotations (HS, Abusive, Target, Category, Level), this project strategically reformulates the task into a **binary classification problem (Hate Speech vs Non-Hate Speech)**. This decision is motivated by the need to improve model stability, reduce label complexity, and enable clearer benchmarking between ML and DL approaches.

Machine Learning models are implemented using **PyCaret AutoML** with TF-IDF-based feature engineering, while the Deep Learning model is developed using **PyTorch**, leveraging sequence-based architectures such as LSTM.

Both approaches are evaluated under the same experimental setup to determine the most effective and robust method for hate speech detection in the Indonesian social media context.

---

## Team Members

| Name                     | NIM       | GitHub Username |
| ------------------------ | --------- | --------------- |
| Tanty Widiyastuti        | 123450094 | tantywidiyastuti |
| Mayada                   | 121450145 | iterastudent |
| Adisty Syawalda Ariyanto | 121450136 | adistyS |

---

## Dataset

The dataset used in this project is publicly available and originates from:

https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection

### Dataset Characteristics

- Language: Indonesian (informal, social media text)
- Source: Twitter
- Annotation types:
  - Hate Speech (HS)
  - Abusive Language
  - Target
  - Category
  - Level

### Task Definition

This project focuses on:

> Binary classification: Hate Speech (HS) vs Non-Hate Speech

This formulation is chosen to:
- improve model robustness  
- simplify evaluation  
- enable consistent comparison across modeling approaches  

---

## Project Objectives

The objectives of this project are:

- Perform **Exploratory Data Analysis (EDA)** on the dataset  
- Develop a comprehensive **text preprocessing pipeline**  
- Implement **Machine Learning models using PyCaret AutoML**  
- Implement **Deep Learning models using PyTorch**  
- Address **class imbalance issues** to reduce false negatives  
- Compare performance between ML and DL approaches  
- Deploy interactive demos using **Hugging Face Spaces**  
- Produce a scientific report in **LaTeX (ArXiv format)**  

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

Machine Learning models are developed using **PyCaret AutoML**. The pipeline includes preprocessing, feature extraction, model training, and evaluation.

### Algorithms Evaluated

- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest  
- Gradient Boosting (optional extension)  

### Methodology

- Text representation using **TF-IDF (unigram and bigram)**  
- Integration of **abusive lexicon features**  
- Handling class imbalance using **SMOTE**  
- Model comparison using cross-validation  
- Selection of the best model based on **F1-score**  

---

## Deep Learning Approach

The Deep Learning model is implemented using **PyTorch**.

### Candidate Architectures

- LSTM (primary model)  
- GRU  
- CNN for text classification  
- Lightweight Transformer (optional)  

### Methodology

- Tokenization and sequence processing  
- Word embedding representation  
- Model training using backpropagation  
- Evaluation on validation and test sets  

---

## Evaluation Metrics

All models are evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score (primary metric)  
- Confusion Matrix  

Special attention is given to minimizing **false negative predictions**, as they are critical in hate speech detection tasks.

---

## Deployment

Two interactive applications will be deployed using **Hugging Face Spaces**:

- Machine Learning Model (PyCaret)  
  *(link to be added)*  

- Deep Learning Model (PyTorch)  
  *(link to be added)*  

Each deployment provides an interface for real-time text classification.

---

## Scientific Paper

The final report will be written in **LaTeX using ArXiv format**, including:

- Dataset description  
- Methodology  
- Experimental setup  
- Benchmark results  
- Comparative analysis  

ArXiv Link:  
*(to be added)*  

---

## Course Information

Course: **Pemrosesan Bahasa Alami**  
Program: **Sains Data — Institut Teknologi Sumatera**  
Semester: **Genap 2025/2026**

Instructor:  
Martin C.T. Manullang
