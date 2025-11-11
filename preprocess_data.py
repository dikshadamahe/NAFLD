"""Preprocess cirrhosis.csv using the pipeline helpers and save outputs.

Saves:
- X_train_processed.npy, X_test_processed.npy
- y_train.csv, y_test.csv
- missing_counts.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

from paper_table_pipeline import load_and_clean, prepare_features
from sklearn.model_selection import train_test_split


def main():
    src = Path('/Users/dikshadamahe/Downloads/cirrhosis.csv')
    if not src.exists():
        raise FileNotFoundError(src)

    df = load_and_clean(str(src))
    print('Dataset shape:', df.shape)
    print('Columns:', list(df.columns))

    X, y, preprocessor, num_cols, cat_cols = prepare_features(df)
    print('Numeric cols:', num_cols)
    print('Categorical cols:', cat_cols)

    # missing value counts
    miss = df.isnull().sum().sort_values(ascending=False)
    miss.to_csv('missing_counts.csv')
    print('Saved missing_counts.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    np.save('X_train_processed.npy', X_train_proc)
    np.save('X_test_processed.npy', X_test_proc)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

    print('Saved processed arrays: X_train_processed.npy (shape', X_train_proc.shape, '), X_test_processed.npy (shape', X_test_proc.shape, ')')


if __name__ == '__main__':
    main()
