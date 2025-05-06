# -*- coding: utf-8 -*-
"""
Created on Thu May  1 12:57:16 2025

@author: Tiffany
"""
import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc, average_precision_score, make_scorer, recall_score, precision_recall_curve
from imblearn.pipeline import Pipeline

# Read the data
df = pd.read_csv("cleaned_and_encoded_dataset(in).csv")

# train-test-split
X = df.drop(columns=['bad_flag', 'customer_id']).copy()
y = df['bad_flag'].copy()
ids = df['customer_id'].copy()

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, ids, test_size=0.2, random_state=42, stratify=y
)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Recall for bad customers (label 1)
recall_bad = make_scorer(recall_score, pos_label=1)
    
# Specificity (true-negative rate for good customers)
def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)
specificity_good = make_scorer(specificity, greater_is_better=True)

# Build SVM model
svm_model = SVC(kernel='linear', random_state=42, probability=True)

# Create pipeline with no resampling, just SVM model
pipe = Pipeline(steps=[ 
    ('clf', svm_model)  # Model training step without resampling
])

# Define the parameter grid
param_grid = {
    'clf__C': [0.1, 1, 10, 100],
    'clf__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'clf__kernel': ['rbf']
}

# GridSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=1, scoring={
    "recall_bad": recall_bad,
    "specificity_good": specificity_good,
}, refit="recall_bad")
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters and best cross-validation recall score
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Cross-validation Recall (bad customers): {grid_search.best_score_:.4f}')

# Evaluate the model accuracy on both training and test data
best_svm_model = grid_search.best_estimator_
y_train_pred = best_svm_model.predict(X_train_scaled)
y_test_pred = best_svm_model.predict(X_test_scaled)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f'Training Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Check for overfitting: Compare training and test accuracy
if train_accuracy > test_accuracy:
    print("Warning: Possible overfitting! The model performs better on the training set.")
else:
    print("The model's performance on the training and test sets are relatively close, indicating no overfitting.")

# Print classification report for the test set
print(classification_report(y_test, y_test_pred, digits=3))

# Compute confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

# Extract True Negative (TN) and False Positive (FP)
TN = cm[0, 0]
FP = cm[0, 1]

# Calculate specificity
specificity_value = TN / (TN + FP)
print(f'Specificity: {specificity_value:.4f}')

# Use predict_proba to get predicted probabilities for class 1
y_prob = best_svm_model.predict_proba(X_test_scaled)[:, 1]

# Threshold optimisation
optimal_threshold = np.arange(0.01, 1.0, 0.01)  # Set the range of thresholds to test
results_opt = []
for t in optimal_threshold:
    y_pred_thresh = (y_prob >= t).astype(int)  # Get predictions based on current threshold
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()  # Confusion matrix
    recall = tp / (tp + fn)  # Calculate recall
    specificity = tn / (tn + fp)  # Calculate specificity
    if recall >= 0.85 and specificity >= 0.70:  # Apply constraints
        results_opt.append({
            'threshold': t,
            'recall': recall,
            'specificity': specificity,
            'false_negatives': fn,
            'false_positives': fp
        })

# Create DataFrame from results
opt_df = pd.DataFrame(results_opt)

if not opt_df.empty:
    best_threshold = opt_df.sort_values(by='false_negatives').iloc[0]['threshold']  # Best threshold with least false negatives
    print(f"\nBest threshold under constraints: {best_threshold:.3f}")
    print(opt_df.sort_values(by='false_negatives').head())  # Display top 5 results
else:
    # Default to threshold 0.5 if no threshold meets the criteria
    best_threshold = 0.5
    print("\nNo threshold satisfies both recall ≥ 85% and specificity ≥ 70%. Defaulting to threshold = 0.50")

# Adjust predictions based on the optimal threshold
y_pred_adjusted = (y_prob >= best_threshold).astype(int)

# Print the classification report for adjusted predictions
print(f"Classification report for threshold {best_threshold:.3f}:")
print(classification_report(y_test, y_pred_adjusted, digits=3))

# Specificity after threshold adjustment
cm = confusion_matrix(y_test, y_pred_adjusted)
TN = cm[0, 0]
FP = cm[0, 1]
specificity_value = TN / (TN + FP)
print(f'Specificity for threshold {best_threshold:.3f}: {specificity_value:.4f}')

