"""Run stratified cross-validation for all models and save mean±std metrics.

Produces: model_performance_comparison_cv.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix

from paper_table_pipeline import load_and_clean, prepare_features, build_models, KernelNB


def fold_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = np.nan
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {
        'Accuracy (%)': acc * 100,
        'AUC': auc,
        'Precision (PPV %)': prec * 100,
        'NPV (%)': npv * 100,
        'Recall/Sensitivity (%)': rec * 100,
        'Specificity (%)': spec * 100,
    }


def run_cv(df, n_splits=5, random_state=0):
    X, y, preprocessor, num_cols, cat_cols = prepare_features(df)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    models = build_models()

    records = []

    for name, estimator in models:
        print('CV', name)
        fold_res = {k: [] for k in ['Accuracy (%)', 'AUC', 'Precision (PPV %)', 'NPV (%)', 'Recall/Sensitivity (%)', 'Specificity (%)']}

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Fit preprocessor on fold train
            X_train_proc = preprocessor.fit_transform(X_train)
            X_val_proc = preprocessor.transform(X_val)

            if isinstance(estimator, KernelNB):
                est = estimator
                est.fit(X_train_proc, y_train.values)
                y_pred = est.predict(X_val_proc)
                proba = est.predict_proba(X_val_proc)
                # pick positive class prob
                if proba.shape[1] == 1:
                    y_proba_pos = proba[:, 0]
                else:
                    idx = list(est.classes_).index(1) if 1 in est.classes_ else 0
                    y_proba_pos = proba[:, idx]
            else:
                # pipeline
                from sklearn.pipeline import Pipeline
                pipe = Pipeline([('pre', preprocessor), ('clf', estimator)])
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_val)
                # probabilities
                if hasattr(pipe, 'predict_proba'):
                    proba = pipe.predict_proba(X_val)
                    if proba.shape[1] == 1:
                        y_proba_pos = proba[:, 0]
                    else:
                        classes = pipe.named_steps['clf'].classes_
                        idx = list(classes).index(1) if 1 in classes else 0
                        y_proba_pos = proba[:, idx]
                elif hasattr(pipe.named_steps['clf'], 'decision_function'):
                    try:
                        dec = pipe.decision_function(X_val)
                        y_proba_pos = 1 / (1 + np.exp(-dec))
                    except Exception:
                        y_proba_pos = np.zeros(len(y_val))
                else:
                    y_proba_pos = np.zeros(len(y_val))

            m = fold_metrics(y_val, y_pred, y_proba_pos)
            for k, v in m.items():
                fold_res[k].append(v)

        # compute mean and std
        rec = {'Model': name}
        for k, vals in fold_res.items():
            arr = np.array(vals, dtype=float)
            rec[f'{k} Mean'] = np.nanmean(arr)
            rec[f'{k} Std'] = np.nanstd(arr, ddof=1)

        records.append(rec)

    df_out = pd.DataFrame(records)
    # order columns nicely
    cols = ['Model']
    metrics = ['Accuracy (%)', 'AUC', 'Precision (PPV %)', 'NPV (%)', 'Recall/Sensitivity (%)', 'Specificity (%)']
    for m in metrics:
        cols += [f'{m} Mean', f'{m} Std']

    df_out = df_out[cols]
    out = Path.cwd() / 'model_performance_comparison_cv.csv'
    df_out.to_csv(out, index=False)
    print('Saved CV table to', out)
    return df_out


if __name__ == '__main__':
    src = Path('/Users/dikshadamahe/Downloads/cirrhosis.csv')
    df = load_and_clean(str(src))
    run_cv(df, n_splits=5)
