# Insurance Payment Risk Scoring System

A machine learning project for developing an interpretable credit risk scorecard to assess the payment default risk of insurance policyholders requesting payment facilities.

---

## Overview

This project focuses on building a **risk scoring system** capable of estimating the probability that an insurance customer will default on a payment facility.

The solution combines a **traditional credit scorecard methodology** based on Weight of Evidence (WOE) and Logistic Regression with a modern **XGBoost** model to compare predictive performance and interpretability.

The project follows a complete data science workflow, from raw data preprocessing to model evaluation and comparison.

---

## Project Objectives

* Clean and preprocess raw insurance data
* Perform Exploratory Data Analysis (EDA)
* Identify the most predictive variables
* Engineer features suitable for credit risk modeling
* Develop an interpretable WOE-based scorecard
* Train an XGBoost classifier
* Compare both approaches using industry-standard evaluation metrics
* Produce a reproducible and well-documented modeling pipeline

---

## Methodology

The project is organized into the following stages:

1. **Data Cleaning**

   * Data quality assessment
   * Missing value analysis
   * Duplicate handling
   * Data type correction
   * Feature standardization

2. **Exploratory Data Analysis**

   * Target distribution
   * Feature distributions
   * Correlation analysis
   * Missing data visualization
   * Business insights

3. **Feature Engineering**

   * Variable transformation
   * Categorical encoding
   * Creation of derived variables
   * Feature selection

4. **Scorecard Development**

   * Weight of Evidence (WOE)
   * Information Value (IV)
   * Optimal binning
   * Logistic Regression

5. **Machine Learning**

   * XGBoost training
   * Hyperparameter tuning
   * Feature importance analysis

6. **Model Evaluation**

   * ROC-AUC
   * Gini Coefficient
   * KS Statistic
   * Precision
   * Recall
   * F1 Score
   * Confusion Matrix

---

## Repository Structure

```text
insurance-risk-scoring/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── notebooks/
│
├── src/
│
├── reports/
│
├── figures/
│
├── results/
│
├── requirements.txt
└── README.md
```

### Folder Description

| Folder         | Description                                                          |
| -------------- | -------------------------------------------------------------------- |
| **data/**      | Dataset placeholders (raw data are not publicly available).          |
| **notebooks/** | Jupyter notebooks documenting each step of the analysis.             |
| **src/**       | Reusable Python scripts for preprocessing, modeling, and evaluation. |
| **reports/**   | Project documentation and technical reports.                         |
| **figures/**   | Charts, plots, and visualizations generated throughout the project.  |
| **results/**   | Model outputs, evaluation metrics, and comparison tables.            |

---

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Jupyter Notebook

---

## Dataset

The dataset used in this project contains confidential business information and cannot be publicly shared.

To respect data privacy and company policies, this repository includes only the project code, documentation, and reproducible workflow. No customer or company-sensitive data are distributed.

---

## Current Progress

* Data Cleaning ✔
* Exploratory Data Analysis ✔
* Feature Engineering ✔
* WOE & Information Value Analysis ⏳
* Logistic Regression Scorecard ⏳
* XGBoost Model ⏳
* Model Evaluation ⏳

---

## Future Improvements

* Incorporate additional business variables when available
* Improve feature engineering with temporal variables
* Implement probability calibration
* Deploy the model as an interactive web application
* Automate the end-to-end scoring pipeline

---

## Author

**Elaa Benabdallah**

Business Analytics Student | Data Analytics Intern

Special interests:

* Credit Risk Modeling
* Machine Learning
* Business Analytics
* Data Science

---

## License

This repository is intended for educational and portfolio purposes.

The original dataset belongs to the company and is not redistributed.

