
# Fraud Detection in Banking

## Project Overview
This project demonstrates how machine learning can help identify
potentially fraudulent credit card transactions.

## Features
- Upload a transaction dataset
- View transaction statistics
- Train a Random Forest classifier
- Review classification metrics and a confusion matrix
- Test individual transactions

## Technologies
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn

## Dataset
Use a publicly available credit card fraud dataset with a `Class`
column, where 0 represents legitimate transactions and 1 represents
fraudulent transactions.

A commonly used source is the Credit Card Fraud Detection dataset
from the Machine Learning Group at ULB, available on Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download the CSV and upload it through the application. The dataset
is not included in this repository by default.

## Installation
Install Python, then run:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## Disclaimer
This is an educational internship project, not a production banking
fraud detection system. Predictions can be wrong. Do not use it to
make real banking decisions without further validation and security
controls.