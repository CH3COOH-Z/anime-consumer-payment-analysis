# Anime Consumer Payment Intention Analysis

An applied data science project analyzing consumer payment intention in the anime/ACG market using survey data, PCA and logistic regression.

## Overview

This project investigates the following question:

> Which behavioral and preference factors are associated with consumers' willingness to pay for anime-related content, products, games, and offline experiences?

The dataset contains **495 valid survey responses**. The original coursework used survey coding, PCA and logistic regression. This repository reorganizes the analysis into a reproducible Python workflow and adds a leakage-aware predictive benchmark.

## Why leakage matters in this survey

The questionnaire asks about payment intention and then asks several questions about past spending and purchase decisions.

Those later questions should **not** be used to predict payment intention because they are conceptually too close to the target, and some were skipped by respondents who never pay. Using them would make model performance look unrealistically strong.

For that reason, the predictive model in this repository excludes:

- past spending type;
- past spending amount;
- purchase-decision questions that were conditionally skipped by non-paying respondents.

This produces a more conservative and methodologically defensible benchmark.

## Data privacy

The raw questionnaire file is **not included** in this repository because the original export contains metadata such as submission timestamps and IP-related information.

Store the private source file locally as:

```text
data/raw/survey.xlsx
```

or:

```text
data/raw/survey.csv
```

The `data/raw/` directory is excluded by `.gitignore`.

## Target definition

The questionnaire records payment willingness in four ordered levels:

1. Never pay
2. Occasional payment
3. Regular small payment
4. Willing to pay more for high-quality content or products

For the binary benchmark:

```text
Y_pay = 0  -> never pay
Y_pay = 1  -> levels 2–4
```

About **90.3%** of respondents are in the positive class, so raw accuracy is not an appropriate metric by itself.

## Features used in the predictive benchmark

The model uses demographic and engagement controls together with PCA-reduced survey blocks that are not direct measures of past payment behavior.

### Control variables

- gender;
- age;
- education;
- occupation;
- city tier;
- years of exposure to anime/ACG content;
- weekly exposure time;
- community participation;
- offline-activity frequency.

### PCA feature blocks

- content-access channels;
- preferred content types;
- preferred IP types;
- offline-activity motives;
- perceived market problems;
- expectations for future market development.

## Methodology

```text
Raw survey
    ↓
Privacy-aware local loading
    ↓
Target construction
    ↓
Train/test split or cross-validation
    ↓
Preprocessing fitted only on training folds
    ├── categorical encoding
    ├── binary multi-select cleaning
    ├── standardization
    └── PCA within each survey block
    ↓
Class-weighted Logistic Regression
    ↓
Evaluation
```

A key design choice is that PCA and scaling are fitted **inside the scikit-learn pipeline**, so validation data is not used to learn preprocessing parameters.

## Evaluation

Because the target is highly imbalanced, the project reports:

- balanced accuracy;
- ROC-AUC;
- precision;
- recall;
- F1-score;
- average precision;
- confusion matrix.

A majority-class baseline is also reported for context.

## Cross-validation results

Using 5-fold stratified cross-validation on the leakage-reduced feature set:

| Metric | Logistic Regression | Majority Baseline |
|---|---:|---:|
| Balanced accuracy | ~0.62 | 0.50 |
| ROC-AUC | ~0.61 | 0.50 |
| F1-score | ~0.82 | ~0.95 |
| Average precision | ~0.92 | ~0.90 |

The F1-score of the majority baseline is high because roughly 90% of observations belong to the positive class. This is exactly why balanced accuracy and ROC-AUC are more informative here.

The modest leakage-free performance is more credible than the near-perfect scores obtained when post-payment variables are included.

## Visual outputs

The analysis script generates:

- target class distribution;
- holdout confusion matrix;
- PCA variance summary;
- logistic-regression coefficient plot.

Generated figures are stored in:

```text
reports/figures/
```

Generated tables are stored in:

```text
reports/tables/
```

## Project structure

```text
anime-consumer-payment-analysis/
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   └── run_analysis.py
│
├── notebooks/
│   └── 01_walkthrough.ipynb
│
├── data/
│   ├── README.md
│   └── raw/
│
└── reports/
    ├── figures/
    └── tables/
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

With an Excel file:

```bash
python src/run_analysis.py --data data/raw/survey.xlsx
```

With a CSV file:

```bash
python src/run_analysis.py --data data/raw/survey.csv
```

## Limitations

- The survey used convenience / targeted sampling rather than probability sampling.
- The positive class is highly imbalanced.
- Responses are self-reported.
- Some questionnaire branches were conditional.
- PCA improves dimensionality management but makes individual survey-item interpretation less direct.
- Predictive association does not imply causation.
- With only 48 negative-class observations, model estimates are sensitive to the exact split.

## Future work

Useful extensions include:

- ordinal modeling using the original four-level payment-intention variable;
- repeated stratified cross-validation;
- calibration analysis;
- comparison with Random Forest and Gradient Boosting;
- SQL-based preprocessing;
- a small interactive dashboard for exploratory analysis.
