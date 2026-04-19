# 🏦 Loan Repayment Prediction — Neural Network (TensorFlow/Keras)

A binary classification neural network trained on 300,000+ real LendingClub loan records to predict whether a borrower will repay their loan or default.

---

## 📌 Project Overview

LendingClub is a peer-to-peer lending platform that connects borrowers with investors. The goal of this project is to build a model that, given a loan applicant's historical financial data, predicts whether they will **fully repay** or **charge off** (default) on their loan.

For this project, We will be using a subset of the LendingClub DataSet obtained from Kaggle: https://www.kaggle.com/wordsforthewise/lending-club

This kind of model is at the core of real-world **credit risk systems**.

---

## 📊 Dataset

- **Source:** [LendingClub Dataset — Kaggle](https://www.kaggle.com/wordsforthewise/lending-club)
- **Size:** ~300,000 loan records
- **Target column:** `loan_status` → converted to binary `loan_repaid` (1 = Fully Paid, 0 = Charged Off)
- **Features:** 28 original features covering loan amount, interest rate, employment, credit history, and more

---

## 🔧 Project Pipeline

### 1. Exploratory Data Analysis (EDA)
- Countplots, histograms, and boxplots to understand distributions
- Correlation heatmap across all numeric features
- Subgrade analysis — identified F and G grades as highest default risk
- Bar chart of feature correlations with `loan_repaid`

### 2. Data Preprocessing
- Dropped high-cardinality / redundant columns (`emp_title`, `title`, `grade`, `issue_d`)
- Filled missing `mort_acc` values using mean per `total_acc` group
- Dropped rows with <0.5% missing data (`revol_util`, `pub_rec_bankruptcies`)
- Feature engineered:
  - Extracted zip codes from address strings
  - Converted `earliest_cr_line` to numeric year (`earliest_cr_year`)
  - Converted `term` to integer (36 or 60)
- One-hot encoded categorical columns: `sub_grade`, `verification_status`, `application_type`, `initial_list_status`, `purpose`, `home_ownership`, `zip_code`
- Normalized features using `MinMaxScaler` (fit on train set only — no data leakage)

### 3. Model Architecture

```
Input (78 features)
    ↓
Dense(78, activation='relu')
Dropout(0.4)
    ↓
Dense(39, activation='relu')
Dropout(0.4)
    ↓
Dense(21, activation='relu')
Dropout(0.4)
    ↓
Dense(1, activation='sigmoid')   ← Binary output
```

- **Loss:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Early Stopping:** Monitored `val_loss`, patience=3, restored best weights  

### 4. Results

| Metric | Score |
|---|---|
| Overall Accuracy | **88%** |
| Precision (Default) | **93%** |
| Recall (Fully Paid) | **99%** |
| Epochs trained | ~11 (early stopping) |

> The model is intentionally conservative — when it flags someone as likely to default, it is correct 93% of the time.

---

## 📉 Training Loss Curve

Training and validation loss tracked closely throughout, with early stopping preventing overfitting at the right moment.

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- TensorFlow / Keras

---

## 💡 Key Learnings

- Good **data preprocessing is 80% of the work** — most of the complexity was in cleaning and encoding, not the model itself
- **Early Stopping** is essential — the first run without it showed clear overfitting after epoch 5
- **Dropout(0.4)** significantly tightened the gap between training and validation loss
- Subtle bugs in data pipelines (like sampling after features are already extracted) can silently affect results — always verify shapes and ordering

---

## 🚀 How to Run

1. Clone the repository
2. Download the dataset from Kaggle and place it in a `/DATA` folder
3. Open `03-Keras-Project-Exercise.ipynb` in Jupyter Notebook
4. Run all cells in order

---

## 📁 File Structure

```
├── 03-Keras-Project-Exercise.ipynb   # Main notebook
├── README.md                         # This file
└── DATA/
    ├── lending_club_loan_two.csv     # Main dataset (download from Kaggle)
    └── lending_club_info.csv         # Feature descriptions
```

---

*Built as part of a deep learning project using real-world financial data.*
