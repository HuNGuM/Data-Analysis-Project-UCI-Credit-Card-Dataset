# 📊 Data Analysis Project – UCI Credit Card Dataset

## 📁 Project Description

This project focuses on analyzing a credit card dataset from Taiwan. The main objective is to understand which customer attributes are associated with the risk of defaulting on a payment in the next month.

The analysis includes:
- Descriptive statistics for numerical and categorical variables
- Various data visualizations
- Correlation analysis
- Linear regression analysis

---

## 🔧 Setup and Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # for Linux/macOS
.venv\Scripts\activate   # for Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

1. **Load the dataset:**
   ```bash
   python load_data.py
   ```

2. **Generate summary statistics:**
   ```bash
   python analyze_data.py
   ```

3. **Generate all visualizations and reports:**
   ```bash
   python data_analysis_main.py
   ```

> All results will be saved in the `plots/` folder and in `.csv` summary files.

---

## 📂 Project Structure

```
project/
│
├── UCI_Credit_Card.csv           # Dataset
├── requirements.txt              # Dependencies
│
├── analyze_data.py               # Summary statistics
├── load_data.py                  # Dataset loader
├── visualize_data.py             # Visualizations
├── data_analysis_main.py         # Main runner
│
├── plots/                        # All generated plots
│   ├── catplot/
│   ├── errorbars/
│   ├── histograms/
│   ├── regression/
│   └── heatmap_corr.png
│
├── numeric_summary.csv           # Stats for numeric features
└── categorical_summary.csv       # Stats for categorical features
```

---

## 📈 Methods Used

- Boxplot, Violinplot, Countplot
- Error bars (mean ± std deviation)
- Conditional histograms with hue
- Correlation heatmap
- Linear regression with hue
- Categorical feature analysis (gender, education, marriage)

---

## 📌 Key Questions Answered

- Which features are most informative for predicting default?
- Are there strong correlations between variables?
- Are there outliers affecting the analysis?
- What is the profile of a typical defaulter?

---


## 📚 References

- [UCI ML Repository – Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
