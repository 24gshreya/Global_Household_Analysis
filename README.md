# 🏠 Global Household Financial Health Analysis

A full end-to-end data science project exploring how income, debt, savings, and employment shape financial health across 5 global regions — from exploratory analysis through to supervised and unsupervised machine learning.

---

## 📌 Project Overview

This project analyses a global household financial dataset to answer 14 business questions spanning descriptive statistics, hypothesis testing, regression, classification, explainability, and clustering.

The pipeline covers:
- **Exploratory Data Analysis (EDA)** — income distribution, savings behaviour, and regional patterns
- **Hypothesis Testing** — ANOVA, Pearson Correlation, Mann-Whitney U
- **OLS Regression** — quantifying the drivers of monthly savings
- **Supervised ML (Classification)** — predicting household financial health category (Poor / Fair / Good)
- **Supervised ML (Regression)** — predicting monthly savings
- **SHAP Explainability** — understanding what causes a household to fall into poor financial health
- **Unsupervised ML (K-Means Clustering)** — discovering natural financial profiles across the global population

---

## ❓ Business Questions Answered

| # | Question | Method |
|---|----------|--------|
| 1 | How is household income distributed globally, and which countries earn the most/least? | EDA |
| 2 | What are the savings rate and expense rate distributions? | EDA |
| 3 | Does having more earners in a household affect income per capita? | EDA |
| 4 | Is there a statistically significant difference in savings rates across countries? | One-Way ANOVA |
| 5 | Do larger families have a lower standard of living? | Pearson Correlation |
| 6 | Do multi-earner households have different spending rates? | Mann-Whitney U |
| 7 | What drives monthly savings — income, family size, earners, expenses? | OLS Regression |
| 8 | Can we predict whether a household will fall into Low, Medium or High financial health? | Classification ML |
| 9 | Which household characteristics are the strongest predictors of financial health? | Feature Importance |
| 10 | What causes a household to fall into poor health? | SHAP Explainability |
| 11 | How accurately can we predict a household's monthly savings? | Regression ML |
| 12 | Do households naturally cluster into distinct financial profiles? | K-Means + Silhouette |
| 13 | What share of the global population is at financial risk? | Cluster Profiling |
| 14 | Which countries and family structures are most over-represented in the vulnerable segment? | Cluster Analysis |

---

## 📂 Repository Structure

```
Global_Household_Analysis/
│
├── dataset/
│   └── Global_Household_Financial_Dynamics.csv
│
├── GlobalHouseholdAnalysis.ipynb    # Main analysis notebook
└── README.md
```

---

## 🔧 Feature Engineering

The following features were derived from the raw dataset to power downstream modelling:

| Feature | Definition |
|---------|------------|
| `net_income` | Total Household Income − Estimated Taxes |
| `disposable_income` | (Net Income − Monthly Expenses) × 12 |
| `savings_rate` | Monthly Savings ÷ Monthly Net Income |
| `expense_ratio` | Monthly Expenses ÷ Monthly Net Income |
| `income_per_capita` | Total Income ÷ Family Size |
| `income_per_earner` | Total Income ÷ Number of Earners |
| `primary_income_share` | Primary Income ÷ Total Income |
| `tax_rate` | Estimated Taxes ÷ Total Income |
| `financial_health_score` | Savings Rate − Expense Ratio |
| `health_category` | Binned health score → Poor / Fair / Good |

---

## 🤖 Machine Learning Models

### Classification — Predicting Financial Health Category

Three models were trained and evaluated with 5-fold cross-validation:

| Model | CV Accuracy | ROC-AUC |
|-------|-------------|---------|
| Logistic Regression | — | — |
| Random Forest | — | — |
| **XGBoost** ✅ | **~0.97** | **~0.9991** |

**Winner: XGBoost** — highest and most stable cross-validation accuracy, near-perfect ROC-AUC, and best generalisation to unseen data.

### Regression — Predicting Monthly Savings

Three models were compared on MAE, RMSE, and R²:

| Model | Result |
|-------|--------|
| **Linear Regression** ✅ | R² ≈ 1.00 |
| Ridge Regression | High R² |
| Random Forest Regressor | High R² |

**Winner: Linear Regression** — Monthly Savings is a near-deterministic function of Income, Taxes, and Expenses, making the linear model optimal.

### Clustering — Discovering Financial Profiles (K-Means, k=7)

| Cluster | Segment Label |
|---------|--------------|
| 0 | Low-Income Surviving Households |
| 1 | Affluent Mid-Savers |
| 2 | High-Income Stretched Spenders |
| 3 | Financially Distressed Households |
| 4 | Low-Income Large Families |
| 5 | High-Income Low-Efficiency Savers |
| 6 | Emerging Market Middle Households |

Optimal k selected using the **Elbow Method** + **Silhouette Score Analysis**. Clusters visualised with **PCA (2 components)**.

---

## 📊 Key Findings

- **Income** is the strongest protective factor against poor financial health
- **Monthly savings** is the primary risk indicator — low savings reliably predicts poor household health (SHAP)
- **Savings rates differ significantly across countries** (One-Way ANOVA, p < 0.05)
- **Larger families have lower income per capita** (negative Pearson correlation)
- **OLS model explains ~89.9% of variation** in monthly savings; family size and expense ratio are the biggest negative drivers
- **Germany, UK, and USA** households save significantly less relative to other regions after controlling for income
- Country-cluster heatmap reveals which regions are most over-represented in financially distressed segments

---

## 🛠️ Tech Stack

```
Python 3.x
│
├── Data        pandas, numpy
├── Viz         matplotlib, seaborn
├── Stats       scipy, pingouin, statsmodels
├── ML          scikit-learn, xgboost
├── Explain     shap
└── Reduction   PCA (sklearn)
```

---

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/24gshreya/Global_Household_Analysis.git
cd Global_Household_Analysis
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scipy pingouin statsmodels scikit-learn xgboost shap
```

**3. Add the dataset**

Place `Global_Household_Financial_Dynamics.csv` inside the `dataset/` folder. Update the file path in the notebook's import cell if needed.

**4. Run the notebook**
```bash
jupyter notebook GlobalHouseholdAnalysis.ipynb
```

---

## 📄 Dataset

The dataset contains one row per household with the following raw fields:

`Household_ID` · `Country` · `City` · `Family_Size` · `Num_Earners` · `Primary_Income` · `Total_Household_Income` · `Estimated_Taxes` · `Monthly_Expenses` · `Monthly_Savings`

Covers 5 global regions with no missing values in the source data.

---

## 👤 Author

**24gshreya** — Data Science Personal Project  
Feel free to raise an issue if you have questions.

---

## ⭐ If you found this useful

Give the repo a star — it helps others discover the project!
