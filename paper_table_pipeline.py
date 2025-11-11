"""Train many classifiers on the cirrhosis dataset and produce a comparison table.

Produces `model_performance_comparison.csv` in the current directory.

Assumptions:
- Binary target: Status == 'D' (died) is positive class, all others negative.
- Categorical 'NA' and literal 'NA' strings are treated as missing.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
	accuracy_score,
	roc_auc_score,
	precision_score,
	recall_score,
	confusion_matrix,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

import math


def load_and_clean(path: str) -> pd.DataFrame:
	df = pd.read_csv(path, na_values=["NA", "NaN", "nan", ""] )
	return df


def prepare_features(df: pd.DataFrame):
	# Target: assume Status == 'D' is positive; others negative
	if 'Status' not in df.columns:
		raise RuntimeError('Expected column Status in the dataset')
	y = (df['Status'] == 'D').astype(int)

	# Drop ID and outcome-ish columns
	X = df.drop(columns=['ID', 'Status'], errors='ignore')

	# Identify numeric and categorical
	numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
	categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

	# Basic pipelines
	numeric_pipeline = Pipeline([
		('imputer', SimpleImputer(strategy='median')),
		('scaler', StandardScaler())
	])

	categorical_pipeline = Pipeline([
		('imputer', SimpleImputer(strategy='most_frequent')),
		('onehot', OneHotEncoder(handle_unknown='ignore'))
	])

	preprocessor = ColumnTransformer([
		('num', numeric_pipeline, numeric_cols),
		('cat', categorical_pipeline, categorical_cols),
	])

	return X, y, preprocessor, numeric_cols, categorical_cols


class KernelNB:
	"""A small Kernel Density based Naive Bayes for continuous features.

	Works only on dense numeric arrays after preprocessing.
	"""
	def __init__(self, bandwidth=1.0):
		self.bandwidth = bandwidth

	def fit(self, X, y):
		# X: numpy array
		self.classes_ = np.unique(y)
		self.kdes_ = {}
		self.priors_ = {}
		for c in self.classes_:
			Xc = X[y == c]
			# if only one sample, create small gaussian about it
			if Xc.shape[0] < 2:
				# store mean and var
				self.kdes_[c] = ('single', Xc.mean(axis=0), max(1e-3, Xc.var(axis=0)))
			else:
				kde = KernelDensity(bandwidth=self.bandwidth)
				kde.fit(Xc)
				self.kdes_[c] = ('kde', kde)
			self.priors_[c] = Xc.shape[0] / X.shape[0]
		return self

	def predict_proba(self, X):
		# compute unnormalized posterior p(x|c)*p(c)
		probs = np.zeros((X.shape[0], len(self.classes_)))
		for i, c in enumerate(self.classes_):
			entry = self.kdes_[c]
			if entry[0] == 'single':
				mean = entry[1]
				var = entry[2]
				# gaussian density (product over dims)
				coef = 1.0 / np.sqrt(2 * np.pi * var)
				exponent = -0.5 * ((X - mean) ** 2) / var
				dens = np.prod(coef * np.exp(exponent), axis=1)
				probs[:, i] = dens * self.priors_[c]
			else:
				kde = entry[1]
				log_dens = kde.score_samples(X)
				probs[:, i] = np.exp(log_dens) * self.priors_[c]
		# normalize
		s = probs.sum(axis=1, keepdims=True)
		s[s == 0] = 1e-12
		return probs / s

	def predict(self, X):
		p = self.predict_proba(X)
		return self.classes_[np.argmax(p, axis=1)]


def compute_metrics(y_true, y_pred, y_proba):
	acc = accuracy_score(y_true, y_pred)
	try:
		auc = roc_auc_score(y_true, y_proba)
	except Exception:
		auc = np.nan
	prec = precision_score(y_true, y_pred, zero_division=0)
	rec = recall_score(y_true, y_pred, zero_division=0)
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
	specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
	npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
	return {
		'Accuracy (%)': acc * 100,
		'AUC': auc,
		'Precision (PPV %)': prec * 100,
		'NPV (%)': npv * 100,
		'Recall/Sensitivity (%)': rec * 100,
		'Specificity (%)': specificity * 100,
	}


def build_models():
	models = []

	# Decision Trees: Fine, Medium, Coarse
	models.append(('Fine Tree', DecisionTreeClassifier(min_samples_leaf=1, random_state=0)))
	models.append(('Medium Tree', DecisionTreeClassifier(min_samples_leaf=5, random_state=0)))
	models.append(('Coarse Tree', DecisionTreeClassifier(min_samples_leaf=20, random_state=0)))

	# Discriminant
	models.append(('Linear Discriminant', LinearDiscriminantAnalysis()))

	# Logistic Regression
	models.append(('Logistic Regression', LogisticRegression(max_iter=2000)))

	# Naive Bayes
	models.append(('Gaussian NB', GaussianNB()))
	models.append(('Kernel NB', KernelNB(bandwidth=1.0)))

	# SVMs
	models.append(('SVM Linear', SVC(kernel='linear', probability=True)))
	models.append(('SVM Quadratic', SVC(kernel='poly', degree=2, probability=True)))
	models.append(('SVM Cubic', SVC(kernel='poly', degree=3, probability=True)))
	models.append(('SVM Fine Gaussian', SVC(kernel='rbf', gamma=1.0, probability=True)))
	models.append(('SVM Medium Gaussian', SVC(kernel='rbf', gamma=0.1, probability=True)))
	models.append(('SVM Coarse Gaussian', SVC(kernel='rbf', gamma=0.01, probability=True)))

	# KNN variants
	models.append(('KNN Fine', KNeighborsClassifier(n_neighbors=1)))
	models.append(('KNN Medium', KNeighborsClassifier(n_neighbors=10)))
	models.append(('KNN Coarse', KNeighborsClassifier(n_neighbors=50)))
	models.append(('KNN Cosine', KNeighborsClassifier(n_neighbors=10, metric='cosine')))
	models.append(('KNN Cubic', KNeighborsClassifier(n_neighbors=3, p=3)))
	models.append(('KNN Weighted', KNeighborsClassifier(n_neighbors=10, weights='distance')))

	# Ensembles
	models.append(('Bagged Trees', BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=0)))
	models.append(('Boosted Trees', AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=0)))
	# Subspace: use Bagging with max_features fraction
	models.append(('Subspace Discriminant', BaggingClassifier(estimator=LinearDiscriminantAnalysis(), n_estimators=30, max_features=0.7, bootstrap=False, random_state=0)))
	models.append(('Subspace KNN', BaggingClassifier(estimator=KNeighborsClassifier(n_neighbors=10), n_estimators=30, max_features=0.7, bootstrap=False, random_state=0)))

	# RUSBoosted Trees: attempt to import imblearn implementation; fallback to AdaBoost
	try:
		from imblearn.ensemble import RUSBoostClassifier
		models.append(('RUSBoosted Trees', RUSBoostClassifier(n_estimators=100, random_state=0)))
	except Exception:
		models.append(('RUSBoosted Trees (approx AdaBoost)', AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=0)))

	return models


def main():
	src = Path('/Users/dikshadamahe/Downloads/cirrhosis.csv')
	if not src.exists():
		raise FileNotFoundError(f"Expected dataset at {src}")

	df = load_and_clean(str(src))
	X, y, preprocessor, num_cols, cat_cols = prepare_features(df)

	# Split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

	# --- Persist preprocessing artifacts so preprocessing is reproducible and inspectable ---
	# fit the preprocessor once on the training data and save processed arrays + feature names
	X_train_proc = preprocessor.fit_transform(X_train)
	X_test_proc = preprocessor.transform(X_test)

	# missing value counts
	missing = df.isnull().sum()
	missing.to_csv(Path.cwd() / 'missing_counts.csv')

	# try to recover one-hot names for categorical features
	try:
		cat_ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
		cat_feat_names = list(cat_ohe.get_feature_names_out(cat_cols))
	except Exception:
		cat_feat_names = []

	feature_names = list(num_cols) + cat_feat_names

	# save processed arrays and metadata

	np.save(Path.cwd() / 'X_train_processed.npy', X_train_proc)
	np.save(Path.cwd() / 'X_test_processed.npy', X_test_proc)
	# save y as simple csvs
	pd.Series(y_train.values).to_csv(Path.cwd() / 'y_train.csv', index=False, header=['target'])
	pd.Series(y_test.values).to_csv(Path.cwd() / 'y_test.csv', index=False, header=['target'])
	pd.Series(feature_names).to_csv(Path.cwd() / 'feature_names.csv', index=False, header=['feature'])

	print(f"Saved preprocessed data: X_train {X_train_proc.shape}, X_test {X_test_proc.shape}")
	# -------------------------------------------------------------------------------

	models = build_models()

	results = []
	for name, estimator in models:
		print('Training', name)
		# For custom KernelNB, we need to fit preprocessor then the custom model manually
		if isinstance(estimator, KernelNB):
			X_train_proc = preprocessor.fit_transform(X_train)
			X_test_proc = preprocessor.transform(X_test)
			estimator.fit(X_train_proc, y_train.values)
			y_pred = estimator.predict(X_test_proc)
			# get probability for positive class
			proba = estimator.predict_proba(X_test_proc)
			if proba.shape[1] == 1:
				y_proba_pos = proba[:, 0]
			else:
				# classes_ order
				idx = list(estimator.classes_).index(1) if 1 in estimator.classes_ else 0
				y_proba_pos = proba[:, idx]
		else:
			# Wrap estimator in pipeline with preprocessor (so categorical handling and scaling applied)
			pipe = Pipeline([('pre', preprocessor), ('clf', estimator)])
			pipe.fit(X_train, y_train)
			y_pred = pipe.predict(X_test)
			# probability for positive class
			if hasattr(pipe, 'predict_proba'):
				proba = pipe.predict_proba(X_test)
				# find index of positive class
				if proba.shape[1] == 1:
					y_proba_pos = proba[:, 0]
				else:
					# classes_ attribute inside clf
					classes = pipe.named_steps['clf'].classes_
					idx = list(classes).index(1) if 1 in classes else 0
					y_proba_pos = proba[:, idx]
			elif hasattr(pipe.named_steps['clf'], 'decision_function'):
				try:
					dec = pipe.decision_function(X_test)
					# map to probability-ish via sigmoid
					y_proba_pos = 1 / (1 + np.exp(-dec))
				except Exception:
					y_proba_pos = np.zeros(len(y_test))
			else:
				y_proba_pos = np.zeros(len(y_test))

		m = compute_metrics(y_test, y_pred, y_proba_pos)
		m['Model'] = name
		results.append(m)

	df_res = pd.DataFrame(results)
	# keep column order: Model, Accuracy, AUC, Precision (PPV %), NPV (%), Recall/Sensitivity (%), Specificity (%)
	cols = ['Model', 'Accuracy (%)', 'AUC', 'Precision (PPV %)', 'NPV (%)', 'Recall/Sensitivity (%)', 'Specificity (%)']
	df_res = df_res[cols]
	df_res = df_res.sort_values(by='Accuracy (%)', ascending=False).reset_index(drop=True)

	out = Path.cwd() / 'model_performance_comparison.csv'
	df_res.to_csv(out, index=False)
	print('\nSaved performance table to', out)


if __name__ == '__main__':
	main()

