# Non-Alcoholic Fatty Liver Disease Detection using ML

A comprehensive machine learning pipeline for predicting Non-Alcoholic Fatty Liver Disease (NAFLD) using multiple classification algorithms. This project evaluates 27 different machine learning models.

## Overview

This project implements a complete ML pipeline that:
- Preprocesses clinical data with proper handling of missing values
- Trains and evaluates 27 different classification models
- Performs stratified cross-validation for robust performance estimation
- Generates detailed performance reports in multiple formats (CSV, Excel, Markdown, LaTeX)

## Models Implemented

The pipeline includes the following model families:

### Decision Trees
- Fine Tree, Medium Tree, Coarse Tree

### Linear Models
- Linear Discriminant Analysis
- Logistic Regression

### Naive Bayes
- Gaussian Naive Bayes
- Kernel Naive Bayes (custom implementation)

### Support Vector Machines (SVM)
- Linear, Quadratic, Cubic kernels
- Gaussian kernels (Fine, Medium, Coarse)

### K-Nearest Neighbors (KNN)
- Fine, Medium, Coarse variants
- Cosine, Cubic, and Weighted distance metrics

### Ensemble Methods
- Bagged Trees
- Boosted Trees (AdaBoost)
- RUSBoosted Trees
- Subspace Discriminant
- Subspace KNN

## Project Structure

```
. 
├── paper_table_pipeline. py      # Main training pipeline with all models
├── run_crossval.py               # Cross-validation script
├── preprocess_data. py            # Data preprocessing utilities
├── export_excel_report.py        # Excel report generation with charts
├── feature_names.csv             # Processed feature names
├── missing_counts.csv            # Missing value analysis
├── y_train.csv / y_test.csv      # Train/test target splits
├── model_performance_comparison_cv.md  # Cross-validation results (Markdown)
├── model_performance_comparison_cv.tex # Cross-validation results (LaTeX)
├── model_performance_report.xlsx # Comprehensive Excel report
└── TEAM.md                       # Project team information
```

## Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn openpyxl
pip install imbalanced-learn  # Optional, for RUSBoost
```

### Running the Pipeline

1. **Preprocess the data:**
```bash
python preprocess_data. py
```

2. **Train all models and generate comparison:**
```bash
python paper_table_pipeline.py
```

3. **Run cross-validation:**
```bash
python run_crossval.py
```

4. **Generate Excel report:**
```bash
python export_excel_report.py
```

## Evaluation Metrics

The pipeline computes comprehensive metrics for each model:
- **Accuracy (%)**: Overall classification accuracy
- **AUC**: Area Under the ROC Curve
- **Precision (PPV %)**: Positive Predictive Value
- **NPV (%)**: Negative Predictive Value
- **Recall/Sensitivity (%)**: True Positive Rate
- **Specificity (%)**: True Negative Rate

## Features

### Data Preprocessing
- Automatic handling of missing values (median for numeric, mode for categorical)
- Standard scaling for numeric features
- One-hot encoding for categorical features
- Stratified train-test split (80/20)

### Cross-Validation
- 5-fold stratified cross-validation
- Mean and standard deviation metrics across folds
- Robust performance estimation

### Reporting
- CSV export for programmatic analysis
- Excel reports with conditional formatting and charts
- Markdown and LaTeX tables for publications
- Color-coded performance visualization

## Output Files

- **model_performance_comparison.csv**: Single train/test run results
- **model_performance_comparison_cv.csv**: Cross-validation results with mean±std
- **model_performance_report.xlsx**: Formatted Excel report with charts
- **X_train_processed.npy / X_test_processed.npy**: Preprocessed feature matrices

## Team

See [TEAM.md](TEAM.md) for the list of contributors. 

## Notes

- The target variable is binary: `Status == 'D'` (died) as positive class
- Missing values are handled during preprocessing
- Random state is set to 0 for reproducibility
- Dataset path needs to be configured in each script (currently set to `/Users/dikshadamahe/Downloads/cirrhosis.csv`)

## Configuration

Update the dataset path in the following files:
- `paper_table_pipeline.py` (line 203)
- `run_crossval.py` (line 118)
- `preprocess_data.py` (line 17)

```python
src = Path('/path/to/your/cirrhosis.csv')
```

## Custom Implementation

The project includes a custom **Kernel Naive Bayes** classifier that uses kernel density estimation for continuous features, providing an alternative to Gaussian Naive Bayes assumptions.

## License

This project is part of academic research on NAFLD detection.
